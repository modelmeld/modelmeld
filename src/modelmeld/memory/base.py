# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Memory schema + MemoryStore ABC — the contract every backend honors.

Tiered memory:
    L0 — raw turn log (everything that happened, in order)
    L1 — persistent facts (durable key/values declared or extracted)
    L2 — evolving summary (built async by the summarizer worker)
    L3 — recent verbatim window (covers the L2 update gap)

The schema + L0/L1 ship with an in-memory backend. L2, L3, the summarizer
worker, cross-tokenizer accounting, and multi-tenant Qdrant layer on top.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Sentinel used for unauthenticated requests so anonymous traffic doesn't
# bleed into a real tenant's memory. Production sets auth_enabled=True;
# this sentinel is mostly a dev / local-testing convenience.
ANONYMOUS_TENANT_ID = "__anonymous__"

# Tenant IDs are user-controllable in some flows; lock the shape down to
# prevent shenanigans like newline injection in log lines, sentinel
# impersonation, or absurdly long values blowing out storage keys.
# `\A...\Z` (NOT `^...$`) anchors absolute start/end — `$` would silently
# allow a trailing newline (Python's regex default).
_TENANT_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_.-]{1,128}\Z")


class Role(str, Enum):
    """Role of a turn participant — mirrors the OpenAI message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Session:
    """One conversational session. Identified by (tenant_id, session_id)."""

    session_id: str
    tenant_id: str
    user_id: str | None
    created_at: datetime
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    """One message exchange — user prompt OR assistant response OR tool result."""

    turn_id: str
    session_id: str
    tenant_id: str
    role: Role
    content: str
    token_count: int
    model_used: str | None
    timestamp: datetime


@dataclass(frozen=True)
class Fact:
    """One persistent L1 fact about the session.

    `key` is human-readable (e.g. "user_name", "preferred_language", "open_pr_id").
    `value` is a free-form string — structure is the caller's responsibility.
    `source` is "declared" (set by the framework / user) or "extracted"
    (inferred by the summarizer worker).
    """

    fact_id: str
    session_id: str
    tenant_id: str
    key: str
    value: str
    source: str
    confidence: float
    timestamp: datetime


@dataclass(frozen=True)
class Summary:
    """L2 — the evolving per-session summary.

    Updated asynchronously by the summarizer worker. One Summary
    per session at any time; the worker upserts new versions in-place.

    `text`                    — the actual summary, written by the worker.
    `last_applied_turn_id`    — high-water mark; turns AFTER this one are
                                not yet reflected in `text`. L3
                                covers them verbatim until the next update.
    `version`                 — monotonic counter for optimistic concurrency.
                                Two workers writing simultaneously: the one
                                with stale `expected_version` loses.
    `source_model`            — which model produced this summary (audit).
    """

    summary_id: str
    session_id: str
    tenant_id: str
    text: str
    last_applied_turn_id: str | None
    version: int
    source_model: str | None
    created_at: datetime
    updated_at: datetime


class SummaryVersionMismatch(RuntimeError):
    """Raised on optimistic-concurrency conflict during summary upsert.

    The summarizer worker should refetch the current summary, regenerate
    against the now-fresher turn log, and retry.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Summary version mismatch: caller expected v{expected}, store has v{actual}"
        )
        self.expected = expected
        self.actual = actual


# ---------------------------------------------------------------------------
# MemoryStore ABC
# ---------------------------------------------------------------------------

class MemoryStore(ABC):
    """Persistence boundary for the tiered memory system.

    The in-memory backend (`InMemoryMemoryStore`) is for dev/tests.
    Enterprise-control plugs in a Postgres-backed store for L0/L1 and Qdrant
    for L2/L3 vectors.

    All methods are tenant-scoped: callers MUST pass tenant_id and the store
    MUST refuse to serve rows from another tenant.
    """

    # ----- Session lifecycle ----------------------------------------------

    @abstractmethod
    async def get_or_create_session(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Session:
        """Return an existing session or create a new one with these attributes.

        Idempotent: calling twice with the same (tenant_id, session_id) returns
        the first session unchanged (metadata and user_id from later calls are
        ignored — declare them at creation time).
        """

    @abstractmethod
    async def get_session(
        self, session_id: str, tenant_id: str
    ) -> Session | None:
        """Look up a session without creating one. Returns None when not found."""

    # ----- L0: raw turn log -----------------------------------------------

    @abstractmethod
    async def append_turn(
        self,
        session_id: str,
        tenant_id: str,
        role: Role,
        content: str,
        token_count: int,
        model_used: str | None = None,
    ) -> Turn:
        """Append a turn to the L0 log. Session must exist."""

    @abstractmethod
    async def list_turns(
        self,
        session_id: str,
        tenant_id: str,
        limit: int | None = None,
    ) -> list[Turn]:
        """Return turns in append order, oldest first. `limit` keeps the LAST n."""

    @abstractmethod
    async def turn_count(self, session_id: str, tenant_id: str) -> int:
        """Total number of turns logged for this session."""

    # ----- L1: persistent facts -------------------------------------------

    @abstractmethod
    async def set_fact(
        self,
        session_id: str,
        tenant_id: str,
        key: str,
        value: str,
        source: str = "declared",
        confidence: float = 1.0,
    ) -> Fact:
        """Upsert a fact. Same `key` overwrites; bump `fact_id` to the new row."""

    @abstractmethod
    async def get_facts(
        self, session_id: str, tenant_id: str
    ) -> list[Fact]:
        """All facts for this session. Order is implementation-defined."""

    @abstractmethod
    async def delete_fact(
        self, session_id: str, tenant_id: str, key: str
    ) -> bool:
        """Remove a fact by key. Returns True if a row was removed."""

    # ----- L2: evolving summary -------------------------------------------

    @abstractmethod
    async def get_summary(
        self, session_id: str, tenant_id: str
    ) -> Summary | None:
        """Return the current summary, or None if one hasn't been written yet."""

    @abstractmethod
    async def upsert_summary(
        self,
        session_id: str,
        tenant_id: str,
        text: str,
        last_applied_turn_id: str | None,
        source_model: str | None = None,
        expected_version: int | None = None,
    ) -> Summary:
        """Write a new summary version. Session must exist.

        If `expected_version` is given and doesn't match the current stored
        version, raises `SummaryVersionMismatch`. Pass None to force-overwrite
        (initial write or admin reset).
        """

    @abstractmethod
    async def clear_summary(
        self, session_id: str, tenant_id: str
    ) -> bool:
        """Drop the summary entirely (next read returns None). Returns True if one existed."""

    # ----- Lifecycle ------------------------------------------------------

    async def close(self) -> None:
        """Release any held resources. Default no-op."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex[:24]}"


def new_fact_id() -> str:
    return f"fact_{uuid.uuid4().hex[:24]}"


def new_summary_id() -> str:
    return f"sum_{uuid.uuid4().hex[:24]}"


class InvalidTenantIdError(ValueError):
    """Tenant ID failed validation (empty, reserved sentinel, malformed)."""


def validate_tenant_id(tenant_id: str) -> None:
    """Raise InvalidTenantIdError if `tenant_id` violates the storage contract.

    Permits `ANONYMOUS_TENANT_ID` only when it's the exact sentinel — external
    callers can't construct a `tenant_id` that collides with the sentinel.
    All other values must match `^[A-Za-z0-9_.-]{1,128}$`.

    Called at every InMemoryMemoryStore entry point. Defense-in-depth: the
    (tenant_id, session_id) tuple key already prevents cross-tenant lookups,
    but this catches misuse early with a clear error.
    """
    if not isinstance(tenant_id, str):
        raise InvalidTenantIdError(
            f"tenant_id must be str, got {type(tenant_id).__name__}"
        )
    if not tenant_id:
        raise InvalidTenantIdError("tenant_id must be non-empty")
    if tenant_id == ANONYMOUS_TENANT_ID:
        return  # exact sentinel value — internal use only, but legal here
    if not _TENANT_ID_PATTERN.match(tenant_id):
        raise InvalidTenantIdError(
            f"tenant_id {tenant_id!r} fails validation; "
            f"must match {_TENANT_ID_PATTERN.pattern}"
        )


# ---------------------------------------------------------------------------
# Qdrant collection naming — forward-prep for vector retrieval.
# Even without a Qdrant impl in core-engine, downstream backends use this
# helper so collection names are validated identically.
# ---------------------------------------------------------------------------

QDRANT_COLLECTION_PREFIX = "tksaver_"
_QDRANT_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_QDRANT_MAX_NAME_LEN = 60   # well under Qdrant's 255 limit


def tenant_collection_name(
    tenant_id: str,
    prefix: str = QDRANT_COLLECTION_PREFIX,
) -> str:
    """Map `tenant_id` to a Qdrant-safe collection name.

    Non-ASCII chars are stripped; long ids are truncated and disambiguated
    with a SHA-256 prefix. Always validates `tenant_id` first so callers can't
    smuggle in malformed values. Same tenant_id always → same collection name
    (deterministic; no random salt).
    """
    validate_tenant_id(tenant_id)
    safe = _QDRANT_SAFE_CHARS.sub("_", tenant_id)
    if len(safe) <= _QDRANT_MAX_NAME_LEN - len(prefix):
        return f"{prefix}{safe}"
    # Truncate + append a short hash so different long tenant_ids stay distinct
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:12]
    head_len = _QDRANT_MAX_NAME_LEN - len(prefix) - len(digest) - 1
    return f"{prefix}{safe[:head_len]}_{digest}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

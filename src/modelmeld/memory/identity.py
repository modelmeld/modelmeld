# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Session/tenant identity + memory-mode extraction from request headers.

A request is in **memory-active** mode when both:
  - the client supplied `x-modelmeld-session-id`, AND
  - we have a tenant_id (from enterprise auth) OR the gateway is running in
    anonymous mode (which uses the `ANONYMOUS_TENANT_ID` sentinel).

Anonymous + session_id mode is fine for dev/local but writes everything to
the shared anonymous namespace. Production gateways enable auth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from modelmeld.memory.base import (
    ANONYMOUS_TENANT_ID,
    InvalidTenantIdError,
    validate_tenant_id,
)

HEADER_SESSION_ID = "x-modelmeld-session-id"
HEADER_USER_ID_OVERRIDE = "x-modelmeld-user-id"   # rare: agent frameworks override
HEADER_MEMORY_MODE = "x-modelmeld-memory-mode"


class MemoryMode(str, Enum):
    """How much memory to inject into the outgoing request.

    AUGMENT (default) — prepend L1 facts + L2 summary as a synthetic system
        message. The framework's existing message history is preserved. Safe
        default: no duplication risk when the framework sends full history.

    FULL — same as AUGMENT plus L3 (recent verbatim turns that aren't yet in
        L2) replayed as user/assistant messages BEFORE the framework's
        messages. Use when the framework only sends the latest user message
        and expects the gateway to remember the rest.

    OFF — write turns but don't inject any context. Useful for debugging or
        for sessions where you only want the audit trail, not augmentation.
    """

    AUGMENT = "augment"
    FULL = "full"
    OFF = "off"

    def __str__(self) -> str:
        return self.value


class MemoryHeaderError(ValueError):
    """Malformed memory header. Chat route maps this to 400."""


@dataclass(frozen=True)
class MemoryIdentity:
    """The tuple required to read/write memory for a request.

    `active` is False when we don't have enough info to attach memory (no
    session_id). The chat route skips memory ops in that case.
    """

    tenant_id: str
    session_id: str | None
    user_id: str | None
    mode: MemoryMode = MemoryMode.AUGMENT

    @property
    def active(self) -> bool:
        return self.session_id is not None


def extract_memory_identity(
    headers: Mapping[str, str],
    auth_tenant_id: str | None,
    auth_user_id: str | None,
) -> MemoryIdentity:
    """Combine request headers + auth state into a MemoryIdentity.

    - `auth_tenant_id` / `auth_user_id` come from the enterprise auth
      middleware via request.state. May be None for anonymous requests.
    - Returns identity with `active=False` when session_id is missing.
    - Malformed `x-modelmeld-memory-mode` raises `MemoryHeaderError`.
    """
    lower = {k.lower(): v for k, v in headers.items()}
    raw_session = lower.get(HEADER_SESSION_ID, "").strip()
    session_id = raw_session or None

    tenant_id = auth_tenant_id or ANONYMOUS_TENANT_ID
    # Defense-in-depth: validate the tenant_id BEFORE using it as a storage key
    # so a misconfigured auth middleware can't poison the memory namespace
    # with newlines, control chars, or sentinel-impersonation attempts.
    try:
        validate_tenant_id(tenant_id)
    except InvalidTenantIdError as e:
        raise MemoryHeaderError(f"invalid_tenant_id: {e}") from e

    # Frameworks like AutoGen / LangGraph use their own per-agent user ID;
    # the header override lets them propagate it without overhauling auth.
    user_id_override = lower.get(HEADER_USER_ID_OVERRIDE, "").strip() or None
    user_id = user_id_override or auth_user_id

    mode = _parse_memory_mode(lower.get(HEADER_MEMORY_MODE))

    return MemoryIdentity(
        tenant_id=tenant_id,
        session_id=session_id,
        user_id=user_id,
        mode=mode,
    )


def _parse_memory_mode(raw: str | None) -> MemoryMode:
    if raw is None:
        return MemoryMode.AUGMENT
    value = raw.strip().lower()
    if not value:
        return MemoryMode.AUGMENT
    try:
        return MemoryMode(value)
    except ValueError as e:
        valid = [m.value for m in MemoryMode]
        raise MemoryHeaderError(
            f"{HEADER_MEMORY_MODE!r}: '{raw}' not in {valid}"
        ) from e

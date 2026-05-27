# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""In-memory MemoryStore backend — dev/tests only.

Thread-safe via an asyncio.Lock per session. Tenant isolation enforced at
every method: requests for one tenant cannot see another tenant's data.

NOT suitable for production: process-local, lost on restart. Enterprise
ships a Postgres-backed implementation.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping

from modelmeld.memory.base import (
    Fact,
    MemoryStore,
    Role,
    Session,
    Summary,
    SummaryVersionMismatch,
    Turn,
    new_fact_id,
    new_summary_id,
    new_turn_id,
    utc_now,
    validate_tenant_id,
)


class TenantMismatchError(LookupError):
    """Raised when the requested (tenant_id, session_id) doesn't match the stored row.

    Distinguished from "not found" so callers can detect and audit cross-tenant
    access attempts.
    """


class InMemoryMemoryStore(MemoryStore):
    """Dict-of-dicts MemoryStore. Thread-safe within one event loop."""

    def __init__(self) -> None:
        # All maps keyed on (tenant_id, session_id) so cross-tenant access
        # is structurally impossible (the lookup just misses).
        self._sessions: dict[tuple[str, str], Session] = {}
        self._turns: dict[tuple[str, str], list[Turn]] = defaultdict(list)
        # facts: per-session dict[key, Fact] so upserts overwrite by key
        self._facts: dict[tuple[str, str], dict[str, Fact]] = defaultdict(dict)
        # L2 evolving summary — at most one row per session
        self._summaries: dict[tuple[str, str], Summary] = {}
        # Per-session lock to serialize writes within a session
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # ----- Session lifecycle ---------------------------------------------

    async def get_or_create_session(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Session:
        validate_tenant_id(tenant_id)
        key = (tenant_id, session_id)
        async with self._lock_for(key):
            existing = self._sessions.get(key)
            if existing is not None:
                return existing
            session = Session(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                created_at=utc_now(),
                metadata=dict(metadata) if metadata else {},
            )
            self._sessions[key] = session
            return session

    async def get_session(
        self, session_id: str, tenant_id: str
    ) -> Session | None:
        validate_tenant_id(tenant_id)
        return self._sessions.get((tenant_id, session_id))

    # ----- L0: raw turn log ----------------------------------------------

    async def append_turn(
        self,
        session_id: str,
        tenant_id: str,
        role: Role,
        content: str,
        token_count: int,
        model_used: str | None = None,
    ) -> Turn:
        validate_tenant_id(tenant_id)
        key = (tenant_id, session_id)
        async with self._lock_for(key):
            if key not in self._sessions:
                raise LookupError(
                    f"Session not found: tenant_id={tenant_id!r} session_id={session_id!r}"
                )
            turn = Turn(
                turn_id=new_turn_id(),
                session_id=session_id,
                tenant_id=tenant_id,
                role=role,
                content=content,
                token_count=token_count,
                model_used=model_used,
                timestamp=utc_now(),
            )
            self._turns[key].append(turn)
            return turn

    async def list_turns(
        self,
        session_id: str,
        tenant_id: str,
        limit: int | None = None,
    ) -> list[Turn]:
        validate_tenant_id(tenant_id)
        turns = self._turns.get((tenant_id, session_id), [])
        if limit is None or limit <= 0:
            return list(turns)
        return list(turns[-limit:])

    async def turn_count(self, session_id: str, tenant_id: str) -> int:
        validate_tenant_id(tenant_id)
        return len(self._turns.get((tenant_id, session_id), []))

    # ----- L1: persistent facts ------------------------------------------

    async def set_fact(
        self,
        session_id: str,
        tenant_id: str,
        key: str,
        value: str,
        source: str = "declared",
        confidence: float = 1.0,
    ) -> Fact:
        validate_tenant_id(tenant_id)
        sess_key = (tenant_id, session_id)
        async with self._lock_for(sess_key):
            if sess_key not in self._sessions:
                raise LookupError(
                    f"Session not found: tenant_id={tenant_id!r} session_id={session_id!r}"
                )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be in [0,1], got {confidence}")
            fact = Fact(
                fact_id=new_fact_id(),
                session_id=session_id,
                tenant_id=tenant_id,
                key=key,
                value=value,
                source=source,
                confidence=confidence,
                timestamp=utc_now(),
            )
            self._facts[sess_key][key] = fact   # upsert overwrites
            return fact

    async def get_facts(
        self, session_id: str, tenant_id: str
    ) -> list[Fact]:
        validate_tenant_id(tenant_id)
        return list(self._facts.get((tenant_id, session_id), {}).values())

    async def delete_fact(
        self, session_id: str, tenant_id: str, key: str
    ) -> bool:
        validate_tenant_id(tenant_id)
        sess_key = (tenant_id, session_id)
        async with self._lock_for(sess_key):
            facts = self._facts.get(sess_key)
            if facts is None or key not in facts:
                return False
            del facts[key]
            return True

    # ----- L2: evolving summary ------------------------------------------

    async def get_summary(
        self, session_id: str, tenant_id: str
    ) -> Summary | None:
        validate_tenant_id(tenant_id)
        return self._summaries.get((tenant_id, session_id))

    async def upsert_summary(
        self,
        session_id: str,
        tenant_id: str,
        text: str,
        last_applied_turn_id: str | None,
        source_model: str | None = None,
        expected_version: int | None = None,
    ) -> Summary:
        validate_tenant_id(tenant_id)
        sess_key = (tenant_id, session_id)
        async with self._lock_for(sess_key):
            if sess_key not in self._sessions:
                raise LookupError(
                    f"Session not found: tenant_id={tenant_id!r} session_id={session_id!r}"
                )
            existing = self._summaries.get(sess_key)
            current_version = existing.version if existing else 0
            if expected_version is not None and expected_version != current_version:
                raise SummaryVersionMismatch(
                    expected=expected_version, actual=current_version
                )
            now = utc_now()
            new_version = current_version + 1
            updated = Summary(
                summary_id=existing.summary_id if existing else new_summary_id(),
                session_id=session_id,
                tenant_id=tenant_id,
                text=text,
                last_applied_turn_id=last_applied_turn_id,
                version=new_version,
                source_model=source_model,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._summaries[sess_key] = updated
            return updated

    async def clear_summary(
        self, session_id: str, tenant_id: str
    ) -> bool:
        validate_tenant_id(tenant_id)
        sess_key = (tenant_id, session_id)
        async with self._lock_for(sess_key):
            if sess_key not in self._summaries:
                return False
            del self._summaries[sess_key]
            return True

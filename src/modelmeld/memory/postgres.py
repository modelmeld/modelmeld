# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""SQL-backed MemoryStore skeleton for Postgres deployments."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

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


class MissingPostgresDependencyError(RuntimeError):
    """Raised when the postgres extra is not installed."""


def _sqlalchemy() -> tuple[Any, Any]:
    try:
        from sqlalchemy import text  # type: ignore[reportMissingImports]
        from sqlalchemy.ext.asyncio import create_async_engine  # type: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise MissingPostgresDependencyError(
            "PostgresMemoryStore requires SQLAlchemy. Install with "
            "`pip install modelmeld[postgres]`."
        ) from exc
    return text, create_async_engine


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def _loads(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class PostgresMemoryStore(MemoryStore):
    """Async SQL MemoryStore.

    The table stores typed JSON documents. Turns are keyed by
    `(tenant_id, session_id, turn_idx)` through `item_key`/`sort_idx`; tests pass
    a SQLite async engine, while production uses a Postgres URL and JSONB.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: Any | None = None,
        create_schema: bool = True,
    ) -> None:
        if engine is None:
            if not database_url:
                raise ValueError(
                    "PostgresMemoryStore requires MODELMELD_MEMORY_DATABASE_URL"
                )
            _, create_async_engine = _sqlalchemy()
            self._engine = create_async_engine(_normalize_url(database_url))
            self._owns_engine = True
        else:
            self._engine = engine
            self._owns_engine = False
        self._create_schema = create_schema
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    @property
    def _json_expr(self) -> str:
        return "CAST(:payload AS jsonb)" if self._engine.dialect.name == "postgresql" else ":payload"

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            if self._create_schema:
                text, _ = _sqlalchemy()
                json_type = "JSONB" if self._engine.dialect.name == "postgresql" else "TEXT"
                async with self._engine.begin() as conn:
                    await conn.execute(text(f"""
                        CREATE TABLE IF NOT EXISTS modelmeld_memory_docs (
                            kind TEXT NOT NULL,
                            tenant_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            item_key TEXT NOT NULL,
                            sort_idx INTEGER NOT NULL,
                            payload {json_type} NOT NULL,
                            PRIMARY KEY (kind, tenant_id, session_id, item_key)
                        )
                    """))
            self._initialized = True

    async def _put(
        self,
        kind: str,
        tenant_id: str,
        session_id: str,
        item_key: str,
        sort_idx: int,
        payload: dict[str, Any],
    ) -> None:
        await self.initialize()
        text, _ = _sqlalchemy()
        async with self._engine.begin() as conn:
            await conn.execute(text(f"""
                INSERT INTO modelmeld_memory_docs (
                    kind, tenant_id, session_id, item_key, sort_idx, payload
                )
                VALUES (
                    :kind, :tenant_id, :session_id, :item_key, :sort_idx,
                    {self._json_expr}
                )
                ON CONFLICT (kind, tenant_id, session_id, item_key) DO UPDATE SET
                    sort_idx = excluded.sort_idx,
                    payload = excluded.payload
            """), {
                "kind": kind,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "item_key": item_key,
                "sort_idx": sort_idx,
                "payload": json.dumps(payload, separators=(",", ":"), sort_keys=True),
            })

    async def _get(self, kind: str, tenant_id: str, session_id: str, item_key: str) -> dict[str, Any] | None:
        await self.initialize()
        text, _ = _sqlalchemy()
        async with self._engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT payload FROM modelmeld_memory_docs
                WHERE kind = :kind AND tenant_id = :tenant_id
                  AND session_id = :session_id AND item_key = :item_key
            """), {
                "kind": kind, "tenant_id": tenant_id,
                "session_id": session_id, "item_key": item_key,
            })
            row = result.first()
            return _loads(row._mapping["payload"]) if row is not None else None

    async def _list(self, kind: str, tenant_id: str, session_id: str) -> list[dict[str, Any]]:
        await self.initialize()
        text, _ = _sqlalchemy()
        async with self._engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT payload FROM modelmeld_memory_docs
                WHERE kind = :kind AND tenant_id = :tenant_id AND session_id = :session_id
                ORDER BY sort_idx ASC, item_key ASC
            """), {"kind": kind, "tenant_id": tenant_id, "session_id": session_id})
            return [_loads(row._mapping["payload"]) for row in result.all()]

    async def _delete(self, kind: str, tenant_id: str, session_id: str, item_key: str) -> bool:
        await self.initialize()
        text, _ = _sqlalchemy()
        async with self._engine.begin() as conn:
            result = await conn.execute(text("""
                DELETE FROM modelmeld_memory_docs
                WHERE kind = :kind AND tenant_id = :tenant_id
                  AND session_id = :session_id AND item_key = :item_key
            """), {
                "kind": kind, "tenant_id": tenant_id,
                "session_id": session_id, "item_key": item_key,
            })
            return bool(result.rowcount)

    async def _require_session(self, session_id: str, tenant_id: str) -> None:
        if await self.get_session(session_id, tenant_id) is None:
            raise LookupError(
                f"Session not found: tenant_id={tenant_id!r} session_id={session_id!r}"
            )

    def _session(self, data: dict[str, Any]) -> Session:
        return Session(
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            user_id=data["user_id"],
            created_at=_dt(data["created_at"]),
            metadata=data["metadata"],
        )

    def _turn(self, data: dict[str, Any]) -> Turn:
        return Turn(
            turn_id=data["turn_id"],
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            role=Role(data["role"]),
            content=data["content"],
            token_count=data["token_count"],
            model_used=data["model_used"],
            timestamp=_dt(data["timestamp"]),
        )

    def _fact(self, data: dict[str, Any]) -> Fact:
        return Fact(
            fact_id=data["fact_id"],
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            key=data["key"],
            value=data["value"],
            source=data["source"],
            confidence=data["confidence"],
            timestamp=_dt(data["timestamp"]),
        )

    def _summary(self, data: dict[str, Any]) -> Summary:
        return Summary(
            summary_id=data["summary_id"],
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            text=data["text"],
            last_applied_turn_id=data["last_applied_turn_id"],
            version=data["version"],
            source_model=data["source_model"],
            created_at=_dt(data["created_at"]),
            updated_at=_dt(data["updated_at"]),
        )

    async def get_or_create_session(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Session:
        validate_tenant_id(tenant_id)
        existing = await self.get_session(session_id, tenant_id)
        if existing is not None:
            return existing
        data = {
            "session_id": session_id, "tenant_id": tenant_id, "user_id": user_id,
            "created_at": utc_now().isoformat(),
            "metadata": dict(metadata) if metadata else {},
        }
        async with self._write_lock:
            existing = await self.get_session(session_id, tenant_id)
            if existing is not None:
                return existing
            await self._put("session", tenant_id, session_id, "_", 0, data)
            return self._session(data)

    async def get_session(self, session_id: str, tenant_id: str) -> Session | None:
        validate_tenant_id(tenant_id)
        data = await self._get("session", tenant_id, session_id, "_")
        return self._session(data) if data is not None else None

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
        async with self._write_lock:
            await self._require_session(session_id, tenant_id)
            turn_idx = await self.turn_count(session_id, tenant_id)
            data = {
                "turn_id": new_turn_id(), "session_id": session_id,
                "tenant_id": tenant_id, "role": role.value, "content": content,
                "token_count": token_count, "model_used": model_used,
                "timestamp": utc_now().isoformat(),
            }
            await self._put("turn", tenant_id, session_id, f"{turn_idx:020d}", turn_idx, data)
            return self._turn(data)

    async def list_turns(
        self,
        session_id: str,
        tenant_id: str,
        limit: int | None = None,
    ) -> list[Turn]:
        validate_tenant_id(tenant_id)
        turns = [self._turn(data) for data in await self._list("turn", tenant_id, session_id)]
        return turns if limit is None or limit <= 0 else turns[-limit:]

    async def turn_count(self, session_id: str, tenant_id: str) -> int:
        validate_tenant_id(tenant_id)
        return len(await self._list("turn", tenant_id, session_id))

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
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {confidence}")
        async with self._write_lock:
            await self._require_session(session_id, tenant_id)
            data = {
                "fact_id": new_fact_id(), "session_id": session_id,
                "tenant_id": tenant_id, "key": key, "value": value,
                "source": source, "confidence": confidence,
                "timestamp": utc_now().isoformat(),
            }
            await self._put("fact", tenant_id, session_id, key, 0, data)
            return self._fact(data)

    async def get_facts(self, session_id: str, tenant_id: str) -> list[Fact]:
        validate_tenant_id(tenant_id)
        return [self._fact(data) for data in await self._list("fact", tenant_id, session_id)]

    async def delete_fact(self, session_id: str, tenant_id: str, key: str) -> bool:
        validate_tenant_id(tenant_id)
        return await self._delete("fact", tenant_id, session_id, key)

    async def get_summary(self, session_id: str, tenant_id: str) -> Summary | None:
        validate_tenant_id(tenant_id)
        data = await self._get("summary", tenant_id, session_id, "_")
        return self._summary(data) if data is not None else None

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
        async with self._write_lock:
            await self._require_session(session_id, tenant_id)
            existing = await self.get_summary(session_id, tenant_id)
            version = existing.version if existing is not None else 0
            if expected_version is not None and expected_version != version:
                raise SummaryVersionMismatch(expected=expected_version, actual=version)
            now = utc_now().isoformat()
            data = {
                "summary_id": existing.summary_id if existing else new_summary_id(),
                "session_id": session_id, "tenant_id": tenant_id, "text": text,
                "last_applied_turn_id": last_applied_turn_id, "version": version + 1,
                "source_model": source_model,
                "created_at": existing.created_at.isoformat() if existing else now,
                "updated_at": now,
            }
            await self._put("summary", tenant_id, session_id, "_", 0, data)
            return self._summary(data)

    async def clear_summary(self, session_id: str, tenant_id: str) -> bool:
        validate_tenant_id(tenant_id)
        return await self._delete("summary", tenant_id, session_id, "_")

    async def close(self) -> None:
        if self._owns_engine:
            await self._engine.dispose()

"""PostgresMemoryStore contract coverage via SQLite-backed SQLAlchemy."""

from __future__ import annotations

import importlib.util

import pytest

from modelmeld.config import GatewaySettings
from modelmeld.memory import PostgresMemoryStore, Role, SummaryVersionMismatch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sqlalchemy") is None
    or importlib.util.find_spec("aiosqlite") is None,
    reason="SQLAlchemy + aiosqlite test dependencies are not installed",
)


async def _make_store(tmp_path) -> PostgresMemoryStore:
    from sqlalchemy.ext.asyncio import create_async_engine  # type: ignore[reportMissingImports]

    db_path = tmp_path / "memory.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    store = PostgresMemoryStore(engine=engine)
    await store.initialize()
    return store


async def test_postgres_memory_store_round_trips_session_turn_fact_summary(tmp_path) -> None:
    store = await _make_store(tmp_path)
    try:
        session = await store.get_or_create_session(
            "s-1",
            "acme",
            user_id="alice",
            metadata={"workspace": "eng"},
        )
        assert session.metadata == {"workspace": "eng"}

        first = await store.append_turn(
            "s-1",
            "acme",
            Role.USER,
            "remember blue",
            3,
            model_used="gpt-5-mini",
        )
        await store.append_turn("s-1", "acme", Role.ASSISTANT, "noted", 2)

        assert await store.turn_count("s-1", "acme") == 2
        assert [turn.content for turn in await store.list_turns("s-1", "acme")] == [
            "remember blue",
            "noted",
        ]
        assert [
            turn.content for turn in await store.list_turns("s-1", "acme", limit=1)
        ] == ["noted"]

        fact = await store.set_fact("s-1", "acme", "favorite_color", "blue")
        assert fact.key == "favorite_color"
        await store.set_fact("s-1", "acme", "favorite_color", "green")
        facts = await store.get_facts("s-1", "acme")
        assert len(facts) == 1
        assert facts[0].value == "green"

        summary = await store.upsert_summary(
            "s-1",
            "acme",
            "The user likes green.",
            first.turn_id,
            source_model="gpt-5-mini",
            expected_version=0,
        )
        assert summary.version == 1
        updated = await store.upsert_summary(
            "s-1",
            "acme",
            "The user prefers green.",
            first.turn_id,
            source_model="gpt-5-mini",
            expected_version=1,
        )
        assert updated.version == 2
        current = await store.get_summary("s-1", "acme")
        assert current is not None
        assert current.text == "The user prefers green."
    finally:
        await store.close()


async def test_postgres_memory_store_preserves_tenant_isolation(tmp_path) -> None:
    store = await _make_store(tmp_path)
    try:
        await store.get_or_create_session("shared", "tenant-a")
        await store.get_or_create_session("shared", "tenant-b")
        await store.append_turn("shared", "tenant-a", Role.USER, "a-only", 1)
        await store.append_turn("shared", "tenant-b", Role.USER, "b-only", 1)

        a_turns = await store.list_turns("shared", "tenant-a")
        b_turns = await store.list_turns("shared", "tenant-b")
        assert [turn.content for turn in a_turns] == ["a-only"]
        assert [turn.content for turn in b_turns] == ["b-only"]
    finally:
        await store.close()


async def test_postgres_memory_store_enforces_summary_versions(tmp_path) -> None:
    store = await _make_store(tmp_path)
    try:
        await store.get_or_create_session("s-1", "acme")
        await store.upsert_summary("s-1", "acme", "v1", None, expected_version=0)
        with pytest.raises(SummaryVersionMismatch):
            await store.upsert_summary("s-1", "acme", "stale", None, expected_version=0)
    finally:
        await store.close()


def test_build_app_uses_postgres_memory_backend_from_settings(tmp_path) -> None:
    from modelmeld.api.server import build_app

    settings = GatewaySettings(
        memory_backend="postgres",
        memory_database_url=f"sqlite+aiosqlite:///{tmp_path / 'server.sqlite3'}",
    )
    app = build_app(settings)
    assert isinstance(app.state.memory_store, PostgresMemoryStore)

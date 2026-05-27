"""InMemoryMemoryStore — schema correctness + tenant isolation + concurrency."""

from __future__ import annotations

import asyncio

import pytest

from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    Fact,
    InMemoryMemoryStore,
    Role,
    Session,
    Turn,
)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

async def test_get_or_create_session_returns_new_session() -> None:
    store = InMemoryMemoryStore()
    sess = await store.get_or_create_session(
        session_id="s-1", tenant_id="acme", user_id="alice",
        metadata={"workspace": "eng"},
    )
    assert isinstance(sess, Session)
    assert sess.session_id == "s-1"
    assert sess.tenant_id == "acme"
    assert sess.user_id == "alice"
    assert sess.metadata == {"workspace": "eng"}


async def test_get_or_create_session_is_idempotent() -> None:
    store = InMemoryMemoryStore()
    first = await store.get_or_create_session("s-1", "acme", user_id="alice")
    second = await store.get_or_create_session("s-1", "acme", user_id="bob")  # different user
    # Same session returned — second call's user_id is ignored
    assert first is second
    assert second.user_id == "alice"


async def test_get_session_returns_none_when_missing() -> None:
    store = InMemoryMemoryStore()
    assert await store.get_session("never-existed", "acme") is None


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

async def test_sessions_isolated_per_tenant() -> None:
    """Same session_id under different tenants → distinct sessions."""
    store = InMemoryMemoryStore()
    a = await store.get_or_create_session("shared-id", "tenant-a", user_id="alice")
    b = await store.get_or_create_session("shared-id", "tenant-b", user_id="bob")
    assert a is not b
    assert a.tenant_id == "tenant-a"
    assert b.tenant_id == "tenant-b"


async def test_turns_isolated_per_tenant() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "tenant-a", user_id="alice")
    await store.get_or_create_session("s", "tenant-b", user_id="bob")
    await store.append_turn("s", "tenant-a", Role.USER, "alice's data", 5)
    await store.append_turn("s", "tenant-b", Role.USER, "bob's data", 5)

    a_turns = await store.list_turns("s", "tenant-a")
    b_turns = await store.list_turns("s", "tenant-b")
    assert len(a_turns) == 1 and a_turns[0].content == "alice's data"
    assert len(b_turns) == 1 and b_turns[0].content == "bob's data"
    # Querying tenant-a with tenant-b's session_id MUST NOT return tenant-b's data
    # (this is structurally enforced by the (tenant_id, session_id) key tuple)


async def test_facts_isolated_per_tenant() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "tenant-a")
    await store.get_or_create_session("s", "tenant-b")
    await store.set_fact("s", "tenant-a", "k", "alice-secret")
    await store.set_fact("s", "tenant-b", "k", "bob-secret")

    a_facts = await store.get_facts("s", "tenant-a")
    b_facts = await store.get_facts("s", "tenant-b")
    assert len(a_facts) == 1 and a_facts[0].value == "alice-secret"
    assert len(b_facts) == 1 and b_facts[0].value == "bob-secret"


# ---------------------------------------------------------------------------
# L0: turn append + list + count
# ---------------------------------------------------------------------------

async def test_append_turn_requires_session() -> None:
    store = InMemoryMemoryStore()
    with pytest.raises(LookupError):
        await store.append_turn("missing", "acme", Role.USER, "hi", 1)


async def test_turns_returned_in_append_order() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    for i in range(5):
        await store.append_turn("s", "acme", Role.USER if i % 2 == 0 else Role.ASSISTANT,
                                f"turn {i}", 1)
    turns = await store.list_turns("s", "acme")
    assert [t.content for t in turns] == [f"turn {i}" for i in range(5)]
    assert isinstance(turns[0], Turn)
    assert turns[0].role == Role.USER
    assert turns[1].role == Role.ASSISTANT


async def test_list_turns_limit_returns_most_recent() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    for i in range(10):
        await store.append_turn("s", "acme", Role.USER, f"t{i}", 1)
    last3 = await store.list_turns("s", "acme", limit=3)
    assert [t.content for t in last3] == ["t7", "t8", "t9"]


async def test_turn_count() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    assert await store.turn_count("s", "acme") == 0
    await store.append_turn("s", "acme", Role.USER, "x", 1)
    await store.append_turn("s", "acme", Role.ASSISTANT, "y", 1)
    assert await store.turn_count("s", "acme") == 2


# ---------------------------------------------------------------------------
# L1: facts set/get/upsert/delete
# ---------------------------------------------------------------------------

async def test_set_fact_creates_new_row() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    fact = await store.set_fact("s", "acme", "user_name", "Kevin",
                                source="declared", confidence=1.0)
    assert isinstance(fact, Fact)
    assert fact.key == "user_name"
    assert fact.value == "Kevin"
    assert fact.source == "declared"
    assert fact.confidence == 1.0


async def test_set_fact_upserts_same_key() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    first = await store.set_fact("s", "acme", "favorite_lang", "Python")
    second = await store.set_fact("s", "acme", "favorite_lang", "Rust")
    facts = await store.get_facts("s", "acme")
    assert len(facts) == 1
    assert facts[0].value == "Rust"
    assert facts[0].fact_id != first.fact_id   # new row id on upsert


async def test_invalid_confidence_rejected() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    with pytest.raises(ValueError):
        await store.set_fact("s", "acme", "k", "v", confidence=1.5)
    with pytest.raises(ValueError):
        await store.set_fact("s", "acme", "k", "v", confidence=-0.1)


async def test_set_fact_requires_session() -> None:
    store = InMemoryMemoryStore()
    with pytest.raises(LookupError):
        await store.set_fact("missing", "acme", "k", "v")


async def test_delete_fact() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    await store.set_fact("s", "acme", "k1", "v1")
    await store.set_fact("s", "acme", "k2", "v2")
    assert await store.delete_fact("s", "acme", "k1") is True
    assert await store.delete_fact("s", "acme", "k1") is False  # already gone
    remaining = await store.get_facts("s", "acme")
    assert {f.key for f in remaining} == {"k2"}


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------

async def test_concurrent_turn_appends_preserved() -> None:
    """50 concurrent appends to one session → all 50 turns present, no loss."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    tasks = [
        store.append_turn("s", "acme", Role.USER, f"t{i}", 1)
        for i in range(50)
    ]
    await asyncio.gather(*tasks)
    turns = await store.list_turns("s", "acme")
    assert len(turns) == 50
    # All turn IDs unique
    assert len({t.turn_id for t in turns}) == 50


async def test_concurrent_session_creates_share_one_row() -> None:
    """Two parallel get_or_create on the same key → same Session object."""
    store = InMemoryMemoryStore()
    results = await asyncio.gather(
        *[store.get_or_create_session("s", "acme") for _ in range(10)]
    )
    # All ten calls return the SAME session (idempotent)
    first = results[0]
    for s in results[1:]:
        assert s is first


# ---------------------------------------------------------------------------
# Anonymous tenant
# ---------------------------------------------------------------------------

async def test_anonymous_tenant_works_like_any_other() -> None:
    store = InMemoryMemoryStore()
    sess = await store.get_or_create_session("s-1", ANONYMOUS_TENANT_ID)
    assert sess.tenant_id == ANONYMOUS_TENANT_ID
    await store.append_turn("s-1", ANONYMOUS_TENANT_ID, Role.USER, "hi", 1)
    turns = await store.list_turns("s-1", ANONYMOUS_TENANT_ID)
    assert len(turns) == 1

"""Property-based memory isolation.

Hand-written cross-tenant tests live in `test_memory_tenant_isolation.py`
and exercise many specific scenarios. This file augments them with
hypothesis-generated random write sequences across multiple (tenant,
session) pairs and asserts the read invariant: a (tenant, session) sees
exactly the writes it received, no more.

Two properties:

  1. Facts: for any sequence of `set_fact(t, s, k, v)` writes, calling
     `get_facts(t, s)` returns exactly the writes addressed to (t, s),
     with the latest value per key.

  2. Turns: for any sequence of `append_turn(t, s, role, content)`
     writes, calling `list_turns(t, s)` returns exactly the turns
     appended to (t, s) in order.

The point is to catch namespacing bugs the hand-written tests would
miss — empty strings, lookalike unicode, edge-case ordering — without
spelling out every combination manually.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelmeld.memory import InMemoryMemoryStore, Role
from modelmeld.memory.base import (
    ANONYMOUS_TENANT_ID,
    InvalidTenantIdError,
    validate_tenant_id,
)

# Tenant-id strategy: produce values likely to pass `validate_tenant_id`.
# Using a small pool of pre-validated ids gives hypothesis more freedom
# to find isolation bugs without burning iterations on input validation.
_VALID_TENANT_IDS = [
    "tenant-a",
    "tenant-b",
    "tenant-c",
    ANONYMOUS_TENANT_ID,
]


@st.composite
def _safe_tenant_id(draw) -> str:
    return draw(st.sampled_from(_VALID_TENANT_IDS))


@st.composite
def _safe_session_id(draw) -> str:
    # Mix of common shapes + some edge cases that should still be legal.
    return draw(st.sampled_from([
        "s1", "s2", "s3",
        "session-with-dashes",
        "UPPER",
        "with_underscores",
        "1234567890",
        "x",   # single char
    ]))


@st.composite
def _safe_fact_key(draw) -> str:
    return draw(st.sampled_from([
        "user_name", "topic", "lang", "k1", "k2",
    ]))


# Values are unicode-friendly but short. Avoid control characters that the
# storage layer doesn't promise to round-trip cleanly.
@st.composite
def _safe_fact_value(draw) -> str:
    return draw(st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, max_size=40,
    ))


_FACT_WRITE = st.tuples(
    _safe_tenant_id(),
    _safe_session_id(),
    _safe_fact_key(),
    _safe_fact_value(),
)

_TURN_WRITE = st.tuples(
    _safe_tenant_id(),
    _safe_session_id(),
    st.sampled_from([Role.USER, Role.ASSISTANT, Role.SYSTEM]),
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1, max_size=80,
    ),
)


# Sanity: the tenant id pool we're using is actually accepted by the validator.
def test_tenant_id_pool_is_valid() -> None:
    for tid in _VALID_TENANT_IDS:
        try:
            validate_tenant_id(tid)
        except InvalidTenantIdError as e:  # pragma: no cover
            pytest.fail(f"pool contains invalid tenant id {tid!r}: {e}")


# ---------------------------------------------------------------------------
# Facts: writes addressed to (T,S) appear only when reading (T,S)
# ---------------------------------------------------------------------------


@given(writes=st.lists(_FACT_WRITE, min_size=0, max_size=30))
@settings(max_examples=75, deadline=None)
def test_facts_isolation_under_arbitrary_interleavings(
    writes: list[tuple[str, str, str, str]],
) -> None:
    """Apply every write; assert each (tenant, session) reads back exactly
    its own writes (latest-wins per key)."""

    async def _run() -> None:
        store = InMemoryMemoryStore()
        # Ground-truth map: (tenant, session) -> {key: latest_value}
        expected: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

        for tenant_id, session_id, key, value in writes:
            await store.get_or_create_session(session_id, tenant_id)
            await store.set_fact(session_id, tenant_id, key, value)
            expected[(tenant_id, session_id)][key] = value

        # For each addressed (tenant, session), the store must return exactly
        # the expected key→value mapping. No extra keys (would mean cross-
        # leak); no missing keys (would mean write swallowed). get_facts()
        # returns list[Fact]; project into a dict for comparison.
        for (tenant_id, session_id), kvs in expected.items():
            got_list = await store.get_facts(session_id, tenant_id)
            got = {f.key: f.value for f in got_list}
            assert got == kvs, (
                f"isolation failure for ({tenant_id!r}, {session_id!r}):\n"
                f"  expected: {kvs!r}\n"
                f"  got:      {got!r}"
            )

        # Every (tenant, session) we did NOT address must return empty facts.
        for tid in _VALID_TENANT_IDS:
            for sid in ("never-written-1", "never-written-2"):
                if (tid, sid) in expected:
                    continue
                got = await store.get_facts(sid, tid)
                assert got == [], (
                    f"non-addressed ({tid!r}, {sid!r}) saw facts: {got!r}"
                )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Turns: per-(tenant, session) order preserved; no cross-session contamination
# ---------------------------------------------------------------------------


@given(writes=st.lists(_TURN_WRITE, min_size=0, max_size=30))
@settings(max_examples=75, deadline=None)
def test_turns_isolation_under_arbitrary_interleavings(
    writes: list[tuple[str, str, Role, str]],
) -> None:
    """Apply every turn write; assert each (tenant, session) reads back
    exactly its own turns in insertion order."""

    async def _run() -> None:
        store = InMemoryMemoryStore()
        expected: dict[tuple[str, str], list[tuple[Role, str]]] = defaultdict(list)

        for tenant_id, session_id, role, content in writes:
            await store.get_or_create_session(session_id, tenant_id)
            # token_count is bookkeeping for downstream summarizer budgeting;
            # use a cheap estimate so the test doesn't depend on a real
            # tokenizer being installed.
            await store.append_turn(
                session_id, tenant_id, role, content,
                token_count=max(1, len(content) // 4),
            )
            expected[(tenant_id, session_id)].append((role, content))

        for (tenant_id, session_id), expected_turns in expected.items():
            stored_turns = await store.list_turns(session_id, tenant_id)
            got = [(t.role, t.content) for t in stored_turns]
            assert got == expected_turns, (
                f"turn isolation failure for ({tenant_id!r}, {session_id!r}):\n"
                f"  expected: {expected_turns!r}\n"
                f"  got:      {got!r}"
            )

        # Non-addressed sessions return empty turn lists.
        for tid in _VALID_TENANT_IDS:
            for sid in ("never-written-1", "never-written-2"):
                if (tid, sid) in expected:
                    continue
                got = await store.list_turns(sid, tid)
                assert got == [], (
                    f"non-addressed ({tid!r}, {sid!r}) saw turns: {got!r}"
                )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Same session_id under two distinct tenants — content never crosses
# ---------------------------------------------------------------------------


@given(
    session_id=_safe_session_id(),
    writes=st.lists(
        st.tuples(
            st.sampled_from(["tenant-a", "tenant-b"]),
            _safe_fact_key(),
            _safe_fact_value(),
        ),
        min_size=0, max_size=20,
    ),
)
@settings(max_examples=50, deadline=None)
def test_same_session_id_distinct_tenants_never_share_facts(
    session_id: str,
    writes: list[tuple[str, str, str]],
) -> None:
    """Two different tenants both use the same session_id literal.
    Their facts must never appear in each other's reads."""

    async def _run() -> None:
        store = InMemoryMemoryStore()
        expected: dict[str, dict[str, str]] = {"tenant-a": {}, "tenant-b": {}}

        for tenant_id, key, value in writes:
            await store.get_or_create_session(session_id, tenant_id)
            await store.set_fact(session_id, tenant_id, key, value)
            expected[tenant_id][key] = value

        got_a_list = await store.get_facts(session_id, "tenant-a")
        got_b_list = await store.get_facts(session_id, "tenant-b")
        got_a = {f.key: f.value for f in got_a_list}
        got_b = {f.key: f.value for f in got_b_list}
        assert got_a == expected["tenant-a"], (
            f"tenant-a read leaked under shared session_id={session_id!r}:\n"
            f"  expected: {expected['tenant-a']!r}\n"
            f"  got:      {got_a!r}"
        )
        assert got_b == expected["tenant-b"], (
            f"tenant-b read leaked under shared session_id={session_id!r}:\n"
            f"  expected: {expected['tenant-b']!r}\n"
            f"  got:      {got_b!r}"
        )

    asyncio.run(_run())

"""Adversarial cross-tenant isolation tests.

The (tenant_id, session_id) tuple key in InMemoryMemoryStore makes
cross-tenant access structurally impossible. These tests prove it
across every public method, against several attacker scenarios:

  - Tenant A tries to read tenant B's session
  - Anonymous tries to read an authenticated tenant's session
  - Authenticated tenant tries to read anonymous data
  - Same session_id under two tenants doesn't collide
  - Concurrent multi-tenant writes don't leak via shared locks
  - Summarizer worker run by tenant A on B's session_id is inert
  - Context injection respects tenant boundary
  - tenant_id validation catches sentinel impersonation + malformed values

This file is the security-critical regression net for the memory layer.
"""

from __future__ import annotations

import asyncio

import pytest

from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    QDRANT_COLLECTION_PREFIX,
    InMemoryMemoryStore,
    InvalidTenantIdError,
    MemoryIdentity,
    MemoryMode,
    Role,
    SummarizerConfig,
    SummarizerWorker,
    assemble_context,
    tenant_collection_name,
    validate_tenant_id,
)

# ===========================================================================
# tenant_id validation
# ===========================================================================

@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "\n",
        "tenant\nwith\nnewlines",
        "tenant id with spaces",
        "tenant/with/slashes",
        "tenant:with:colons",
        "../../etc/passwd",
        "a" * 129,                 # too long
        "tenant\x00null",
    ],
)
def test_malformed_tenant_id_rejected(bad: str) -> None:
    with pytest.raises(InvalidTenantIdError):
        validate_tenant_id(bad)


@pytest.mark.parametrize(
    "good",
    [
        "acme",
        "tenant-1",
        "tenant_1",
        "Tenant.Inc",
        "a",
        "a" * 128,                 # max length
        "1234567890",
        "TENANT-WITH-UPPER",
    ],
)
def test_valid_tenant_ids_accepted(good: str) -> None:
    validate_tenant_id(good)   # no raise


def test_anonymous_sentinel_doesnt_collide_with_similar_names() -> None:
    """Exact ANONYMOUS_TENANT_ID is the only string that maps to the anonymous
    namespace. Tenants whose ID happens to look similar (`__anonymous`,
    `_anonymous_`, `__ANONYMOUS__`) get their OWN distinct storage namespace —
    no collision, no privilege escalation. Whitespace-padded impostors that
    fail the regex are rejected outright.
    """
    validate_tenant_id(ANONYMOUS_TENANT_ID)   # exact sentinel: OK
    # These pass the regex → they ARE valid tenant_ids. Critically they map
    # to namespaces DISTINCT from ANONYMOUS_TENANT_ID.
    for similar in ["__anonymous", "_anonymous_", "__ANONYMOUS__"]:
        validate_tenant_id(similar)
        assert similar != ANONYMOUS_TENANT_ID
    # But values with disallowed characters (spaces, control chars) are
    # rejected — they can't be used to confuse logs or shells.
    for invalid in ["__anonymous__ ", " __anonymous__", "__anonymous__\n"]:
        with pytest.raises(InvalidTenantIdError):
            validate_tenant_id(invalid)


async def test_similar_named_tenants_have_distinct_storage() -> None:
    """End-to-end proof that `__anonymous` cannot read `__anonymous__` data."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", ANONYMOUS_TENANT_ID)
    await store.set_fact("s", ANONYMOUS_TENANT_ID, "k", "real-anonymous-data")
    # A tenant_id that LOOKS like the sentinel but isn't gets its own namespace
    assert await store.get_facts("s", "__anonymous") == []
    assert await store.get_facts("s", "_anonymous_") == []


def test_non_string_tenant_id_rejected() -> None:
    with pytest.raises(InvalidTenantIdError):
        validate_tenant_id(None)   # type: ignore[arg-type]
    with pytest.raises(InvalidTenantIdError):
        validate_tenant_id(123)    # type: ignore[arg-type]


# ===========================================================================
# Cross-tenant lookups: every method must return "not found" semantics
# ===========================================================================

async def test_get_session_cross_tenant_returns_none() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    assert await store.get_session("s-1", "tenant-B") is None


async def test_list_turns_cross_tenant_returns_empty() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.append_turn("s-1", "tenant-A", Role.USER, "secret-A", 1)
    # Tenant B queries with the same session_id → empty
    assert await store.list_turns("s-1", "tenant-B") == []
    assert await store.turn_count("s-1", "tenant-B") == 0


async def test_get_facts_cross_tenant_returns_empty() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.set_fact("s-1", "tenant-A", "k", "secret-A")
    assert await store.get_facts("s-1", "tenant-B") == []


async def test_get_summary_cross_tenant_returns_none() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.upsert_summary("s-1", "tenant-A", text="A's summary",
                                last_applied_turn_id=None)
    assert await store.get_summary("s-1", "tenant-B") is None


async def test_delete_fact_cross_tenant_is_noop() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.set_fact("s-1", "tenant-A", "k", "v")
    # Tenant B "deleting" → returns False, A's fact untouched
    assert await store.delete_fact("s-1", "tenant-B", "k") is False
    assert len(await store.get_facts("s-1", "tenant-A")) == 1


async def test_clear_summary_cross_tenant_is_noop() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.upsert_summary("s-1", "tenant-A", text="x", last_applied_turn_id=None)
    assert await store.clear_summary("s-1", "tenant-B") is False
    assert await store.get_summary("s-1", "tenant-A") is not None


# ===========================================================================
# Mutations under wrong tenant: session lookup fails
# ===========================================================================

async def test_append_turn_under_wrong_tenant_raises_not_creates() -> None:
    """Critical: another tenant CANNOT create rows in a session id they don't own."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    # Tenant B tries to append using A's session_id → fails because no such
    # session exists in B's namespace. (It does NOT silently bleed into A's row.)
    with pytest.raises(LookupError):
        await store.append_turn("s-1", "tenant-B", Role.USER, "B's intrusion", 1)
    # A's data still pristine
    assert await store.list_turns("s-1", "tenant-A") == []


async def test_set_fact_under_wrong_tenant_raises() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    with pytest.raises(LookupError):
        await store.set_fact("s-1", "tenant-B", "intrusion", "value")
    assert await store.get_facts("s-1", "tenant-A") == []


async def test_upsert_summary_under_wrong_tenant_raises() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    with pytest.raises(LookupError):
        await store.upsert_summary("s-1", "tenant-B", text="forged",
                                    last_applied_turn_id=None)


# ===========================================================================
# Same session_id under two tenants doesn't collide
# ===========================================================================

async def test_same_session_id_two_tenants_distinct_rows() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("user-thread-42", "tenant-A", user_id="alice")
    await store.get_or_create_session("user-thread-42", "tenant-B", user_id="bob")

    await store.append_turn("user-thread-42", "tenant-A", Role.USER, "A's data", 1)
    await store.append_turn("user-thread-42", "tenant-B", Role.USER, "B's data", 1)
    await store.set_fact("user-thread-42", "tenant-A", "owner", "alice")
    await store.set_fact("user-thread-42", "tenant-B", "owner", "bob")
    await store.upsert_summary("user-thread-42", "tenant-A", text="A-summary",
                                last_applied_turn_id=None)
    await store.upsert_summary("user-thread-42", "tenant-B", text="B-summary",
                                last_applied_turn_id=None)

    # Every dimension stays isolated
    a_turns = await store.list_turns("user-thread-42", "tenant-A")
    b_turns = await store.list_turns("user-thread-42", "tenant-B")
    assert [t.content for t in a_turns] == ["A's data"]
    assert [t.content for t in b_turns] == ["B's data"]

    a_facts = await store.get_facts("user-thread-42", "tenant-A")
    b_facts = await store.get_facts("user-thread-42", "tenant-B")
    assert a_facts[0].value == "alice"
    assert b_facts[0].value == "bob"

    a_sum = await store.get_summary("user-thread-42", "tenant-A")
    b_sum = await store.get_summary("user-thread-42", "tenant-B")
    assert a_sum is not None and a_sum.text == "A-summary"
    assert b_sum is not None and b_sum.text == "B-summary"


# ===========================================================================
# Anonymous boundary
# ===========================================================================

async def test_anonymous_cannot_read_authenticated_tenant() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "tenant-A")
    await store.set_fact("s", "tenant-A", "k", "secret")
    assert await store.get_facts("s", ANONYMOUS_TENANT_ID) == []


async def test_authenticated_tenant_cannot_read_anonymous_data() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", ANONYMOUS_TENANT_ID)
    await store.set_fact("s", ANONYMOUS_TENANT_ID, "k", "dev-only")
    assert await store.get_facts("s", "tenant-A") == []


# ===========================================================================
# Concurrent multi-tenant writes don't leak via shared locks
# ===========================================================================

async def test_concurrent_writes_across_tenants_isolate() -> None:
    """Tenants writing in parallel must not bleed into each other's rows."""
    store = InMemoryMemoryStore()
    for tenant in ("tenant-A", "tenant-B", "tenant-C"):
        await store.get_or_create_session("shared-sid", tenant)

    async def writer(tenant: str, count: int) -> None:
        for i in range(count):
            await store.append_turn(
                "shared-sid", tenant, Role.USER, f"{tenant}-msg-{i}", 1,
            )

    await asyncio.gather(
        writer("tenant-A", 30),
        writer("tenant-B", 30),
        writer("tenant-C", 30),
    )

    for tenant in ("tenant-A", "tenant-B", "tenant-C"):
        turns = await store.list_turns("shared-sid", tenant)
        assert len(turns) == 30
        # Every turn's content tagged with its own tenant
        for t in turns:
            assert t.content.startswith(tenant)


# ===========================================================================
# Higher-level: assemble_context respects tenant boundary
# ===========================================================================

async def test_assemble_context_does_not_leak_other_tenants_facts() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "tenant-A")
    await store.set_fact("s-1", "tenant-A", "secret_a", "alpha-keys-3xK")

    # Tenant B asks for context with the SAME session_id
    identity = MemoryIdentity(
        tenant_id="tenant-B", session_id="s-1", user_id=None, mode=MemoryMode.FULL,
    )
    ctx = await assemble_context(store, identity)
    assert ctx.has_content() is False
    assert all("alpha-keys-3xK" not in f.value for f in ctx.facts)


# ===========================================================================
# Summarizer worker can't be tricked into writing cross-tenant
# ===========================================================================

async def test_summarizer_on_nonexistent_session_in_other_tenant_is_inert() -> None:
    """Worker run by tenant B on tenant A's session_id has nothing to do."""
    store = InMemoryMemoryStore()
    # Tenant A populates a session with 25 turns
    await store.get_or_create_session("s", "tenant-A")
    for i in range(25):
        await store.append_turn("s", "tenant-A", Role.USER, f"A-turn-{i}", 1)

    summarize_call_count = {"n": 0}

    async def stub(_messages):
        summarize_call_count["n"] += 1
        return "cross-tenant summary"

    worker = SummarizerWorker(
        memory=store, summarize_call=stub,
        config=SummarizerConfig(turn_threshold=20),
    )
    # Tenant B's worker run finds 0 unsummarized turns in B's namespace
    result = await worker.run_once("s", "tenant-B")
    assert result is None
    # The LLM was never called
    assert summarize_call_count["n"] == 0
    # And tenant A's data is still un-summarized + intact
    assert await store.get_summary("s", "tenant-A") is None
    assert await store.turn_count("s", "tenant-A") == 25


# ===========================================================================
# tenant_collection_name (Qdrant naming)
# ===========================================================================

def test_qdrant_name_prefix_always_applied() -> None:
    name = tenant_collection_name("acme")
    assert name.startswith(QDRANT_COLLECTION_PREFIX)


def test_qdrant_name_deterministic() -> None:
    """Same tenant_id always maps to the same collection name."""
    assert tenant_collection_name("acme") == tenant_collection_name("acme")


def test_qdrant_name_distinct_tenants_distinct_collections() -> None:
    assert tenant_collection_name("tenant-A") != tenant_collection_name("tenant-B")


def test_qdrant_name_short_tenant_id_inlined() -> None:
    name = tenant_collection_name("acme")
    assert name == f"{QDRANT_COLLECTION_PREFIX}acme"


def test_qdrant_name_long_tenant_id_hashed() -> None:
    long_valid = "tenant-" + "a" * 100   # 107 chars, passes regex
    name = tenant_collection_name(long_valid)
    assert name.startswith(QDRANT_COLLECTION_PREFIX)
    # Length capped near Qdrant safety limit
    assert len(name) <= 70


def test_qdrant_name_two_long_tenant_ids_distinct() -> None:
    """Same length-truncated prefix + different hash = distinct collections."""
    a = "tenant-A-" + "x" * 80
    b = "tenant-A-" + "x" * 79 + "y"
    name_a = tenant_collection_name(a)
    name_b = tenant_collection_name(b)
    assert name_a != name_b


def test_qdrant_name_rejects_malformed_tenant_id() -> None:
    with pytest.raises(InvalidTenantIdError):
        tenant_collection_name("bad/tenant")


def test_qdrant_name_handles_anonymous_sentinel() -> None:
    name = tenant_collection_name(ANONYMOUS_TENANT_ID)
    # Underscore sentinel name is reformatted but kept stable
    assert name.startswith(QDRANT_COLLECTION_PREFIX)
    assert tenant_collection_name(ANONYMOUS_TENANT_ID) == name   # deterministic

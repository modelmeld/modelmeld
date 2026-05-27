"""L2 evolving summary — storage + optimistic concurrency + cross-tier helpers."""

from __future__ import annotations

import asyncio

import pytest

from modelmeld.memory import (
    InMemoryMemoryStore,
    Role,
    Summary,
    SummaryVersionMismatch,
    needs_summary_refresh,
    summary_freshness,
    turns_since_summary,
)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

async def test_get_summary_returns_none_when_unset() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    assert await store.get_summary("s", "acme") is None


async def test_upsert_summary_creates_first_version() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    summary = await store.upsert_summary(
        "s", "acme",
        text="User is debugging an auth flow.",
        last_applied_turn_id="turn-7",
        source_model="claude-sonnet-4-6",
    )
    assert isinstance(summary, Summary)
    assert summary.version == 1
    assert summary.text == "User is debugging an auth flow."
    assert summary.last_applied_turn_id == "turn-7"
    assert summary.source_model == "claude-sonnet-4-6"
    assert summary.created_at == summary.updated_at


async def test_upsert_summary_increments_version_and_preserves_created_at() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    first = await store.upsert_summary("s", "acme", text="v1", last_applied_turn_id="t1")
    await asyncio.sleep(0.001)
    second = await store.upsert_summary(
        "s", "acme", text="v2", last_applied_turn_id="t5",
        expected_version=1,
    )
    assert second.version == 2
    assert second.text == "v2"
    # created_at is preserved across versions; updated_at moves forward
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at
    # Same summary_id across versions (it's the row id, not the version id)
    assert second.summary_id == first.summary_id


async def test_upsert_requires_session() -> None:
    store = InMemoryMemoryStore()
    with pytest.raises(LookupError):
        await store.upsert_summary("missing", "acme", text="x", last_applied_turn_id=None)


# ---------------------------------------------------------------------------
# Optimistic concurrency
# ---------------------------------------------------------------------------

async def test_stale_expected_version_raises_mismatch() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    await store.upsert_summary("s", "acme", text="v1", last_applied_turn_id="t1")
    # Caller thinks version is still 0, but it's now 1 → mismatch
    with pytest.raises(SummaryVersionMismatch) as exc_info:
        await store.upsert_summary(
            "s", "acme", text="v2", last_applied_turn_id="t2",
            expected_version=0,
        )
    assert exc_info.value.expected == 0
    assert exc_info.value.actual == 1


async def test_force_overwrite_when_expected_version_omitted() -> None:
    """`expected_version=None` is the admin/initial-write escape hatch."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    await store.upsert_summary("s", "acme", text="v1", last_applied_turn_id="t1")
    forced = await store.upsert_summary(
        "s", "acme", text="forced", last_applied_turn_id="t99",
    )
    assert forced.version == 2
    assert forced.text == "forced"


async def test_concurrent_summarizers_one_wins() -> None:
    """Two workers race; the loser sees SummaryVersionMismatch and can retry."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    # Initial summary at v1
    await store.upsert_summary("s", "acme", text="seed", last_applied_turn_id="t0")

    async def writer(text: str):
        try:
            return await store.upsert_summary(
                "s", "acme", text=text, last_applied_turn_id="tX",
                expected_version=1,
            )
        except SummaryVersionMismatch:
            return "mismatch"

    results = await asyncio.gather(writer("A"), writer("B"))
    successes = [r for r in results if isinstance(r, Summary)]
    failures = [r for r in results if r == "mismatch"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert successes[0].version == 2


# ---------------------------------------------------------------------------
# clear_summary
# ---------------------------------------------------------------------------

async def test_clear_summary_removes_row() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    await store.upsert_summary("s", "acme", text="x", last_applied_turn_id="t1")
    assert await store.clear_summary("s", "acme") is True
    assert await store.get_summary("s", "acme") is None


async def test_clear_summary_when_none_existed() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    assert await store.clear_summary("s", "acme") is False


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

async def test_summaries_isolated_per_tenant() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "tenant-a")
    await store.get_or_create_session("s", "tenant-b")
    await store.upsert_summary("s", "tenant-a", text="A's secrets", last_applied_turn_id="t1")
    await store.upsert_summary("s", "tenant-b", text="B's secrets", last_applied_turn_id="t1")
    a = await store.get_summary("s", "tenant-a")
    b = await store.get_summary("s", "tenant-b")
    assert a is not None and a.text == "A's secrets"
    assert b is not None and b.text == "B's secrets"


# ---------------------------------------------------------------------------
# Cross-tier helpers
# ---------------------------------------------------------------------------

async def test_turns_since_summary_empty_log() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    assert await turns_since_summary(store, "s", "acme") == []


async def test_turns_since_summary_with_no_summary_returns_all() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    for i in range(3):
        await store.append_turn("s", "acme", Role.USER, f"t{i}", 1)
    after = await turns_since_summary(store, "s", "acme")
    assert [t.content for t in after] == ["t0", "t1", "t2"]


async def test_turns_since_summary_after_high_water_mark() -> None:
    """Turns AFTER the last_applied_turn_id are the L3 hot-zone candidates."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    turns = []
    for i in range(5):
        turns.append(await store.append_turn("s", "acme", Role.USER, f"t{i}", 1))
    # Summary applied through turn[2]; expect t3, t4 unsummarized
    await store.upsert_summary("s", "acme", text="...", last_applied_turn_id=turns[2].turn_id)
    after = await turns_since_summary(store, "s", "acme")
    assert [t.content for t in after] == ["t3", "t4"]


async def test_turns_since_summary_unknown_high_water_returns_all() -> None:
    """Recorded high-water mark missing from log → conservative: re-fold everything."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    for i in range(3):
        await store.append_turn("s", "acme", Role.USER, f"t{i}", 1)
    await store.upsert_summary("s", "acme", text="x", last_applied_turn_id="never-existed")
    after = await turns_since_summary(store, "s", "acme")
    assert len(after) == 3


async def test_turns_since_summary_respects_cap() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    for i in range(10):
        await store.append_turn("s", "acme", Role.USER, f"t{i}", 1)
    after = await turns_since_summary(store, "s", "acme", cap=3)
    assert [t.content for t in after] == ["t7", "t8", "t9"]


async def test_summary_freshness_counts_behind() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s", "acme")
    turns = []
    for i in range(8):
        turns.append(await store.append_turn("s", "acme", Role.USER, f"t{i}", 1))
    await store.upsert_summary("s", "acme", text="...", last_applied_turn_id=turns[4].turn_id)
    # 3 turns appended after the high-water mark (t5, t6, t7)
    assert await summary_freshness(store, "s", "acme") == 3


def test_needs_summary_refresh_threshold() -> None:
    assert needs_summary_refresh(behind=5, turn_threshold=20) is False
    assert needs_summary_refresh(behind=20, turn_threshold=20) is True
    assert needs_summary_refresh(behind=100, turn_threshold=20) is True


def test_needs_summary_refresh_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        needs_summary_refresh(behind=10, turn_threshold=0)

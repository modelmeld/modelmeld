"""InMemoryCompletionCache — LRU + TTL + concurrency + corruption safety."""

from __future__ import annotations

import asyncio
import time

import pytest

from modelmeld.api.schemas import (
    ChatCompletion,
    Choice,
    ResponseMessage,
    Usage,
)
from modelmeld.cache import InMemoryCompletionCache


def _completion(content: str = "hi") -> ChatCompletion:
    return ChatCompletion(
        model="test-model",
        choices=[Choice(index=0, message=ResponseMessage(content=content),
                        finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


# ---------------------------------------------------------------------------
# Basic put/get
# ---------------------------------------------------------------------------

async def test_miss_on_unknown_key() -> None:
    cache = InMemoryCompletionCache()
    result = await cache.get("never-set")
    assert result.hit is False
    assert result.value is None


async def test_set_then_get_returns_completion() -> None:
    cache = InMemoryCompletionCache()
    c = _completion("cached-answer")
    await cache.set("k1", c, ttl_seconds=60)
    result = await cache.get("k1")
    assert result.hit is True
    assert result.value is c
    assert result.value.choices[0].message.content == "cached-answer"


async def test_overwriting_key_keeps_latest() -> None:
    cache = InMemoryCompletionCache()
    await cache.set("k", _completion("v1"), ttl_seconds=60)
    await cache.set("k", _completion("v2"), ttl_seconds=60)
    result = await cache.get("k")
    assert result.value.choices[0].message.content == "v2"


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------

async def test_expired_entry_is_dropped(monkeypatch) -> None:
    """Mock the clock: set at t=0 with TTL=10, then jump to t=100 → miss."""
    cache = InMemoryCompletionCache()
    import modelmeld.cache.in_memory as mod
    state = {"now": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: state["now"])

    await cache.set("k", _completion(), ttl_seconds=10)
    # Confirm it's there at t≈now
    assert (await cache.get("k")).hit is True
    # Jump past expiry
    state["now"] += 100
    assert (await cache.get("k")).hit is False


async def test_zero_or_negative_ttl_is_noop() -> None:
    """Setting with TTL ≤ 0 shouldn't store the entry (defensive)."""
    cache = InMemoryCompletionCache()
    await cache.set("k", _completion(), ttl_seconds=0)
    assert (await cache.get("k")).hit is False
    await cache.set("k", _completion(), ttl_seconds=-5)
    assert (await cache.get("k")).hit is False


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

async def test_lru_evicts_when_capacity_exceeded() -> None:
    cache = InMemoryCompletionCache(max_entries=3)
    await cache.set("a", _completion("A"), ttl_seconds=60)
    await cache.set("b", _completion("B"), ttl_seconds=60)
    await cache.set("c", _completion("C"), ttl_seconds=60)
    # Touch "a" so it's most-recently-used
    await cache.get("a")
    # Insert "d" → evicts "b" (oldest unaccessed)
    await cache.set("d", _completion("D"), ttl_seconds=60)

    assert (await cache.get("a")).hit is True
    assert (await cache.get("b")).hit is False   # evicted
    assert (await cache.get("c")).hit is True
    assert (await cache.get("d")).hit is True
    assert cache._size() == 3


async def test_invalid_max_entries_rejected() -> None:
    with pytest.raises(ValueError):
        InMemoryCompletionCache(max_entries=0)
    with pytest.raises(ValueError):
        InMemoryCompletionCache(max_entries=-1)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

async def test_concurrent_set_then_get_doesnt_lose_writes() -> None:
    cache = InMemoryCompletionCache()
    # 50 concurrent writes to distinct keys
    await asyncio.gather(*[
        cache.set(f"k{i}", _completion(f"v{i}"), ttl_seconds=60)
        for i in range(50)
    ])
    # All readable
    for i in range(50):
        r = await cache.get(f"k{i}")
        assert r.hit is True
        assert r.value.choices[0].message.content == f"v{i}"


# ---------------------------------------------------------------------------
# Error swallowing — contract says cache MUST NEVER raise
# ---------------------------------------------------------------------------

async def test_get_does_not_raise_on_corrupt_state(monkeypatch) -> None:
    """Defensive: corrupted internal state should degrade to a miss, not 500."""
    cache = InMemoryCompletionCache()
    # Break the store deliberately
    cache._store = None  # type: ignore[assignment]
    # Must NOT raise; must return a miss
    result = await cache.get("anything")
    assert result.hit is False
    assert result.value is None

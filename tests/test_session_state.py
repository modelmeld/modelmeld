# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Unit tests for the per-session stall state store (scout/session_state.py)."""

from __future__ import annotations

from modelmeld.scout.session_state import SessionStallState, SessionStallStore


class _Clock:
    """Deterministic monotonic-style clock for TTL tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_get_or_create_is_idempotent() -> None:
    store = SessionStallStore()
    a = store.get_or_create("k")
    a.turns_seen += 1
    b = store.get_or_create("k")
    assert a is b
    assert b.turns_seen == 1
    assert len(store) == 1


def test_default_state_shape() -> None:
    s = SessionStallState()
    assert s.turns_seen == 0
    assert s.last_signals == ()
    assert s.shadow_fired_turn is None


def test_ttl_eviction() -> None:
    clock = _Clock()
    store = SessionStallStore(ttl_sec=100.0, clock=clock)
    store.get_or_create("old")
    clock.advance(101.0)
    # Touching a different key triggers lazy eviction of the expired one.
    store.get_or_create("fresh")
    assert len(store) == 1


def test_active_session_not_evicted() -> None:
    clock = _Clock()
    store = SessionStallStore(ttl_sec=100.0, clock=clock)
    store.get_or_create("k")
    clock.advance(50.0)
    store.get_or_create("k")  # refreshes last_seen
    clock.advance(60.0)       # 60 < ttl since last touch
    store.get_or_create("k")
    assert len(store) == 1


def test_max_entries_cap_evicts_oldest() -> None:
    clock = _Clock()
    store = SessionStallStore(max_entries=2, clock=clock)
    store.get_or_create("a")
    clock.advance(1.0)
    store.get_or_create("b")
    clock.advance(1.0)
    store.get_or_create("c")  # over cap → oldest ("a") evicted
    assert len(store) == 2
    keys = set(store._entries)  # type: ignore[attr-defined]
    assert keys == {"b", "c"}

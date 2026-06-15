# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Per-session stall state for reactive escalation.

A small TTL-bounded store, keyed by the session key from
[[scout.session_key.derive_session_key]], holding the running stall-detection
state for each in-flight agentic session. Lives on `app.state.session_stall`
for the gateway instance's lifetime.

In this shadow/observe-only increment the state's only behavioural job is to
de-duplicate telemetry — emit the "would escalate" signal ONCE per session
(`shadow_fired_turn`) rather than on every subsequent turn. The reactive
increment reuses the same store to carry the sticky escalation decision across
turns, so the shape is built out now.

The TTL/eviction idiom mirrors `CapabilityRouter._health_cache`: a monotonic
clock, lazy expiry on access, no lock (the gateway runs each route coroutine on
one event loop and these mutations have no `await` between read and write).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

# Default TTL: a session goes idle and is evicted after this many seconds with
# no turns. 30 min comfortably spans a coding session's think-time gaps.
_DEFAULT_TTL_SEC = 1800.0
# Hard cap on tracked sessions so a flood of distinct implicit keys can't grow
# the store unbounded; oldest-by-last_seen are evicted past this.
_DEFAULT_MAX_ENTRIES = 10_000


@dataclass
class SessionStallState:
    """Mutable per-session stall-detection state."""

    turns_seen: int = 0
    last_signals: tuple[str, ...] = ()
    # Turn number at which the shadow detector first fired for this session, or
    # None if it never has. Used to emit telemetry exactly once per session.
    shadow_fired_turn: int | None = None
    last_seen: float = 0.0


class SessionStallStore:
    """TTL-bounded `session_key -> SessionStallState` map.

    `clock` is injectable so tests can advance time deterministically; it
    defaults to `time.monotonic` (never goes backwards, immune to wall-clock
    adjustment — the same choice the router's health cache makes).
    """

    def __init__(
        self,
        *,
        ttl_sec: float = _DEFAULT_TTL_SEC,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_sec = ttl_sec
        self._max_entries = max_entries
        self._clock = clock
        self._entries: dict[str, SessionStallState] = {}

    def get_or_create(self, key: str) -> SessionStallState:
        """Return the state for `key`, creating it if absent.

        Evicts expired entries first; enforces the size cap after insert. The
        returned state's `last_seen` is refreshed to now so the caller doesn't
        have to remember to (it's about to be used).
        """
        now = self._clock()
        self._evict_expired(now)
        state = self._entries.get(key)
        if state is None:
            state = SessionStallState(last_seen=now)
            self._entries[key] = state
            # Stamp last_seen BEFORE enforcing the cap so the just-created entry
            # isn't mistaken for the oldest and immediately evicted.
            self._enforce_cap()
        else:
            state.last_seen = now
        return state

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_expired(self, now: float) -> None:
        if not self._entries:
            return
        cutoff = now - self._ttl_sec
        expired = [k for k, s in self._entries.items() if s.last_seen < cutoff]
        for k in expired:
            del self._entries[k]

    def _enforce_cap(self) -> None:
        overflow = len(self._entries) - self._max_entries
        if overflow <= 0:
            return
        # Drop the oldest-by-last_seen entries.
        oldest = sorted(self._entries.items(), key=lambda kv: kv[1].last_seen)
        for k, _ in oldest[:overflow]:
            del self._entries[k]


__all__ = ["SessionStallState", "SessionStallStore"]

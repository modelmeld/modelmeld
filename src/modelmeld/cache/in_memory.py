# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""In-memory completion cache — dev/tests + single-worker fallback.

Bounded LRU with per-entry TTL. Eviction order: expired first, then LRU.
Not safe across processes; production multi-worker deployments use the
Redis backend in enterprise-control.

`get` / `set` NEVER raise — even on corrupt internal state, the worst case
is a miss. The chat route relies on this.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from modelmeld.api.schemas import ChatCompletion
from modelmeld.cache.base import CacheLookup, CompletionCache

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 10_000


class InMemoryCompletionCache(CompletionCache):
    """Bounded LRU + per-entry TTL. Lock-protected for asyncio safety."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")
        self._max = max_entries
        # OrderedDict preserves insertion order → we move-to-end on access for LRU.
        # Value is (completion, expires_at_monotonic).
        self._store: OrderedDict[str, tuple[ChatCompletion, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheLookup:
        try:
            async with self._lock:
                entry = self._store.get(key)
                if entry is None:
                    return CacheLookup(hit=False, value=None)
                completion, expires_at = entry
                if expires_at <= time.monotonic():
                    # Expired — drop it.
                    del self._store[key]
                    return CacheLookup(hit=False, value=None)
                # Refresh LRU position.
                self._store.move_to_end(key)
                return CacheLookup(hit=True, value=completion)
        except Exception:
            logger.exception("in-memory cache get failed for key %r", key)
            return CacheLookup(hit=False, value=None)

    async def set(
        self, key: str, completion: ChatCompletion, ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            return  # negative/zero TTLs are no-ops — never store stale data
        try:
            async with self._lock:
                expires_at = time.monotonic() + ttl_seconds
                if key in self._store:
                    self._store.move_to_end(key)
                self._store[key] = (completion, expires_at)
                # Evict from the front (oldest) until under cap.
                while len(self._store) > self._max:
                    self._store.popitem(last=False)
        except Exception:
            logger.exception("in-memory cache set failed for key %r", key)

    # --- test/debug helpers ---------------------------------------------

    def _size(self) -> int:
        return len(self._store)

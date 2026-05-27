# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Completion cache — exact-match + (later) semantic caching.

Ships the ABC + an in-memory backend + the request-normalization
function used to compute cache keys. Redis backend lives in
enterprise-control. Semantic (Qdrant) backend layers on top.

Cache invariant: a hit returns the SAME `ChatCompletion` that the original
request produced, byte-for-byte. The wrapping route re-stamps the response
id / created timestamp via the existing OpenAI schemas, but the choices /
usage / model fields come back unchanged.

Uncacheable requests bypass the cache entirely:
  - `stream=True`        — caching SSE streams is hard; not worth it for v1
  - `tools=[...]`        — tool calls are context-dependent (output references
                            the conversation history)
  - `n > 1`              — would need to cache the whole choice list
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modelmeld.api.schemas import ChatCompletion, ChatCompletionRequest

CACHE_KEY_VERSION = "v1"

# Default TTL is generous: a chat about static code or a fixed doc can
# legitimately be cached for hours. Operators tighten this in settings.
DEFAULT_CACHE_TTL_SECONDS = 3600


@dataclass(frozen=True)
class CacheLookup:
    """Result of a cache lookup. `hit=False` with `value=None` is a miss."""

    hit: bool
    value: ChatCompletion | None


class CompletionCache(ABC):
    """Pluggable cache for `ChatCompletion` responses keyed by request body."""

    @abstractmethod
    async def get(self, key: str) -> CacheLookup:
        """Return `CacheLookup(hit=True, value=…)` on hit, `(False, None)` on miss.

        Implementation MUST NEVER raise — backend failures degrade to a miss.
        """

    @abstractmethod
    async def set(
        self, key: str, completion: ChatCompletion, ttl_seconds: int,
    ) -> None:
        """Store `completion` under `key` with the given TTL.

        MUST NEVER raise — backend failures are logged but never propagate.
        """

    async def close(self) -> None:
        """Release held resources. Default no-op."""


# ---------------------------------------------------------------------------
# Request normalization → stable cache key
# ---------------------------------------------------------------------------

# Fields that vary per-call without changing the semantic answer.
# Stripping them before hashing means a `user` header or one-off `metadata`
# blob doesn't fracture the cache.
_NON_SEMANTIC_FIELDS = frozenset({
    "stream",       # bypassed entirely upstream; defensive
    "stream_options",
    "user",
    "metadata",
})


def cache_key_for_request(
    request: ChatCompletionRequest,
    *,
    tenant_id: str | None,
    served_model: str | None = None,
) -> str | None:
    """Return a stable cache key or None if the request is uncacheable.

    `served_model` — when capability routing rewrote `request.model`, pass
        the post-override id. The cache MUST key on the served model: two
        users asking for `claude-opus-4-7` who both get routed to
        `qwen3-coder` share a cache entry; if either gets routed elsewhere
        the entry is a separate row.

    `tenant_id` — included in the key as defense-in-depth even before the
        per-tenant namespacing. None bucketed under `__anon__`.
    """
    if request.stream:
        return None
    if request.tools:
        return None
    if request.n is not None and request.n > 1:
        return None

    canonical = _canonicalize(request, served_model=served_model)
    payload_bytes = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    bucket = tenant_id or "__anon__"
    return f"gateway:cache:{CACHE_KEY_VERSION}:{bucket}:{digest}"


def _canonicalize(
    request: ChatCompletionRequest, *, served_model: str | None,
) -> dict[str, Any]:
    """`request` → dict with non-semantic fields stripped, served model pinned."""
    raw = request.model_dump(exclude_none=True)
    for field in _NON_SEMANTIC_FIELDS:
        raw.pop(field, None)
    if served_model:
        raw["model"] = served_model
    return raw

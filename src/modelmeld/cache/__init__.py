# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Completion cache.

Exact-match cache by SHA-256 of the canonicalized request, plus a
semantic cache (Qdrant) layered on top of the same ABC, plus per-tenant
TTL + per-dev-tool cache-hit analytics.

Public surface:
    CompletionCache              — abstract base class
    CacheLookup                  — frozen result of `get`
    InMemoryCompletionCache      — dev/test backend (bounded LRU + TTL)
    cache_key_for_request        — normalize request → stable key (or None)
    CACHE_KEY_VERSION            — bump to invalidate everyone
    DEFAULT_CACHE_TTL_SECONDS    — 1 hour
"""

from __future__ import annotations

from modelmeld.cache.base import (
    CACHE_KEY_VERSION,
    DEFAULT_CACHE_TTL_SECONDS,
    CacheLookup,
    CompletionCache,
    cache_key_for_request,
)
from modelmeld.cache.embedding import (
    EmbeddingClient,
    HashedBagOfWordsEmbedder,
    cosine_similarity,
)
from modelmeld.cache.in_memory import (
    DEFAULT_MAX_ENTRIES,
    InMemoryCompletionCache,
)
from modelmeld.cache.semantic import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SemanticCompletionCache,
    canonicalize_request_text,
    is_request_semantically_cacheable,
)

__all__ = [
    "CACHE_KEY_VERSION",
    "CacheLookup",
    "CompletionCache",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "EmbeddingClient",
    "HashedBagOfWordsEmbedder",
    "InMemoryCompletionCache",
    "SemanticCompletionCache",
    "cache_key_for_request",
    "canonicalize_request_text",
    "cosine_similarity",
    "is_request_semantically_cacheable",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Semantic completion cache — fuzzy match by embedding similarity.

Different interface from the exact-match `CompletionCache`: the semantic
cache works in PROMPT-TEXT space (embed → vector search) rather than
canonical-key space (SHA-256). The chat route consults exact-first, then
semantic, then falls through to the adapter on miss.

Same uncacheable-request gates apply (tools, streaming, n>1 bypass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionRequest,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
)
from modelmeld.cache.base import CacheLookup

# Default similarity threshold. Tuned against the hashed-BOW test embedder
# where paraphrases land around 0.93–1.00 and unrelated prompts at 0.2–0.5.
# Real embeddings (OpenAI text-embedding-3-small) score paraphrases higher
# (~0.95+); operators tune this per-deployment.
DEFAULT_SIMILARITY_THRESHOLD = 0.92


class SemanticCompletionCache(ABC):
    """Vector-search backed cache. Tenant-scoped at the backend layer."""

    @abstractmethod
    async def search(
        self,
        prompt_text: str,
        *,
        tenant_id: str | None,
        served_model: str | None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> CacheLookup:
        """Find the most-similar prior request whose similarity ≥ threshold.

        Returns `(hit=False, value=None)` on miss OR backend failure — same
        never-raise contract as the exact-match cache.
        """

    @abstractmethod
    async def store(
        self,
        prompt_text: str,
        completion: ChatCompletion,
        *,
        tenant_id: str | None,
        served_model: str | None,
        ttl_seconds: int,
    ) -> None:
        """Persist an embedding + completion. Never raises."""

    async def close(self) -> None:
        """Release held resources. Default no-op."""


# ---------------------------------------------------------------------------
# Prompt-text canonicalization for embedding input
# ---------------------------------------------------------------------------

def canonicalize_request_text(
    request: ChatCompletionRequest,
    *,
    served_model: str | None = None,
) -> str:
    """Flatten a request to the text we embed for similarity search.

    The served model is prefixed so prompts directed at different models
    don't accidentally pool (a "summarize this" answer from claude-haiku
    is different shape from one from qwen3-coder-next, even if the prompt
    is identical). System + tool messages contribute; assistant messages
    don't (they're prior context that varies per session).
    """
    parts: list[str] = []
    if served_model:
        parts.append(f"model={served_model}")

    for msg in request.messages:
        if isinstance(msg, SystemMessage):
            parts.append("[system] " + _text_of(msg))
        elif isinstance(msg, UserMessage):
            parts.append("[user] " + _text_of(msg))
        elif isinstance(msg, ToolMessage):
            parts.append("[tool] " + _text_of(msg))
        # AssistantMessage content is prior turns — skip; they vary per
        # session and would hurt cache hit rates.

    # Include hyperparameters that genuinely change the answer.
    for field in ("temperature", "top_p", "max_completion_tokens", "max_tokens",
                  "response_format", "seed"):
        value = getattr(request, field, None)
        if value is not None:
            parts.append(f"{field}={value}")

    return "\n".join(parts)


def _text_of(msg: Any) -> str:
    """Pull plain text from any OpenAI-shaped message."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            getattr(p, "text", "") for p in content if isinstance(p, TextPart)
        )
    return ""


def is_request_semantically_cacheable(request: ChatCompletionRequest) -> bool:
    """Same gates as exact-match: no streaming, no tools, no n>1.

    The chat route also consults this before reaching the semantic cache.
    """
    if request.stream:
        return False
    if request.tools:
        return False
    if request.n is not None and request.n > 1:
        return False
    return True

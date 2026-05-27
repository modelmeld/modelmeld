# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""TokenCounter — pluggable token counting across providers.

Memory budgets, prompt sizing, and FinOps reporting all need an honest token
count. A 100-char message can be 25 tokens of English, 50+ tokens of code, or
200+ tokens of CJK text. Char/4 is fine for a rough estimate but lies enough
to matter at scale.

Two backends:
  - `CharBasedTokenCounter`  — heuristic (1 token ≈ 4 chars), no deps.
  - `LiteLLMTokenCounter`    — accurate via `litellm.token_counter()`.
    Picks the right tokenizer (tiktoken / anthropic / huggingface) per model.
    Optional extra: `pip install modelmeld[tokenizer]`.

Falls back to char-based silently when `model` is unknown or the upstream
tokenizer errors — degraded accuracy, never a 500.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TokenCounter(ABC):
    """Pluggable per-model token counting."""

    name: str

    @abstractmethod
    def count_text(self, text: str, model: str | None = None) -> int:
        """Count tokens in a single string. `model` selects the tokenizer."""

    @abstractmethod
    def count_messages(self, messages: list, model: str | None = None) -> int:
        """Count tokens across OpenAI-shaped messages (pydantic or dict)."""


class CharBasedTokenCounter(TokenCounter):
    """1 token ≈ 4 chars. Fast, no deps, ~20% off for English on average.

    Acceptable for low-latency paths (streaming chunk accounting) and as a
    fallback when `litellm` isn't installed. Production should prefer
    `LiteLLMTokenCounter` for L0 turn writes and budget enforcement.
    """

    name = "char"
    CHARS_PER_TOKEN = 4

    def count_text(self, text: str, model: str | None = None) -> int:
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def count_messages(self, messages: list, model: str | None = None) -> int:
        return sum(self.count_text(_extract_text(m)) for m in messages)


class LiteLLMTokenCounter(TokenCounter):
    """Accurate counts via `litellm.token_counter()`.

    Routes to the right per-model tokenizer:
      - tiktoken for OpenAI (cl100k_base / o200k_base / etc.)
      - anthropic SDK's count_tokens for Claude
      - huggingface tokenizers for open-weight models
    Falls back to char-based when `model` is None or the lookup errors.
    """

    name = "litellm"

    def __init__(self) -> None:
        try:
            import litellm  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "LiteLLMTokenCounter requires `litellm`. "
                "Install with: pip install modelmeld[tokenizer]"
            ) from e
        self._litellm = litellm
        self._fallback = CharBasedTokenCounter()

    def count_text(self, text: str, model: str | None = None) -> int:
        if not text:
            return 0
        if not model:
            return self._fallback.count_text(text)
        try:
            return int(self._litellm.token_counter(model=model, text=text))
        except Exception:
            logger.debug(
                "litellm.token_counter(text) failed for model=%s; falling back to char-based",
                model,
            )
            return self._fallback.count_text(text)

    def count_messages(self, messages: list, model: str | None = None) -> int:
        if not model:
            return self._fallback.count_messages(messages)
        try:
            return int(self._litellm.token_counter(
                model=model, messages=_to_dicts(messages)
            ))
        except Exception:
            logger.debug(
                "litellm.token_counter(messages) failed for model=%s; falling back",
                model,
            )
            return self._fallback.count_messages(messages, model)


def build_token_counter(settings: Any) -> TokenCounter:
    """Factory keyed on `GatewaySettings.token_counter_backend`.

    `litellm` selection falls back to char-based with a logged warning when
    the optional `litellm` dep isn't installed — boot succeeds with degraded
    accuracy rather than failing.
    """
    backend = getattr(settings, "token_counter_backend", "char")
    if backend == "litellm":
        try:
            return LiteLLMTokenCounter()
        except ImportError:
            logger.warning(
                "settings.token_counter_backend='litellm' but the litellm extra "
                "isn't installed; using char-based fallback. "
                "Install: pip install modelmeld[tokenizer]"
            )
            return CharBasedTokenCounter()
    return CharBasedTokenCounter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(message: Any) -> str:
    """Pull plain text from any OpenAI-shaped message (pydantic or dict)."""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            text = (
                part.get("text") if isinstance(part, dict)
                else getattr(part, "text", "")
            )
            if isinstance(text, str):
                pieces.append(text)
        return "".join(pieces)
    return ""


def _to_dicts(messages: list) -> list[dict]:
    """Pydantic messages → dicts for litellm's API."""
    out: list[dict] = []
    for m in messages:
        if hasattr(m, "model_dump"):
            out.append(m.model_dump(exclude_none=True))
        else:
            out.append(m)
    return out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Provider adapter abstract base class.

`ProviderAdapter` is the extension point through which the gateway forwards
OpenAI-shaped requests to a concrete upstream (OpenAI cloud, Anthropic cloud,
local vLLM, etc.). Implementations live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
)


class AdapterError(Exception):
    """Raised when an adapter fails to fulfill a request.

    Network failures, upstream 5xx responses, schema-translation errors, and
    misconfiguration (missing API key, etc.) all surface as this exception.

    Subclasses `TransientAdapterError` and `PermanentAdapterError` carry the
    retry-ability signal so the TieredRouter can decide whether to fail over
    to the other tier or bubble the error up to the caller.
    """


class TransientAdapterError(AdapterError):
    """Adapter failed in a way that may succeed on retry / failover.

    Examples: HTTP 5xx, 429 rate limit, 529 overloaded, network blip,
    timeout. Routers should attempt the other tier; callers should treat
    repeated occurrences as a real outage.
    """


class PermanentAdapterError(AdapterError):
    """Adapter failed in a way that retry won't fix.

    Examples: HTTP 401/403 auth failure, HTTP 404 model-not-found,
    schema-translation errors, misconfiguration. Routers should NOT fail
    over — surface the error so the caller sees the real cause instead of
    a misleading fallback response.
    """


class ProviderAdapter(ABC):
    """Translate an OpenAI-shaped request to a concrete upstream provider."""

    name: str
    # True when this adapter sends traffic outside the customer's network.
    # Used by the chat route to gate PII scrubbing.
    is_egress: bool = False
    # F-8: operator-configured model this adapter actually serves upstream.
    # When set, the adapter substitutes `request.model` with this value on
    # outbound calls — the client can send any model name (or none) and the
    # gateway routes them based on the scout's tier decision while the
    # adapter uses its configured upstream model.
    # When None, the adapter passes the client's model name through unchanged
    # (default for adapters that proxy to multi-model providers).
    served_model: str | None = None

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        """Non-streaming chat completion."""

    @abstractmethod
    def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming chat completion. Implementations are async generators."""

    @abstractmethod
    async def health(self) -> bool:
        """Cheap upstream reachability check. Returns False on failure."""

    def serves_model(self, model_id: str) -> bool:  # noqa: ARG002 — base default
        """Whether this adapter can serve the given model id (F-8).

        Default returns True for both pinned and pass-through configurations:
        - `served_model=None` → pass-through; we don't know what upstream
          supports, so we assume failover is safe.
        - `served_model="X"`  → substitution; `_apply_served_model()` will
          rewrite `request.model` to X on the outbound call, so the
          adapter will serve any request regardless of the client's
          model id. Setting `served_model` is opting into substitution.

        TieredRouter consults this before failover. Subclasses can
        override for stricter behavior (e.g. compliance-mode adapter
        that rejects non-matching model ids outright).
        """
        return True

    def _apply_served_model(
        self, request: ChatCompletionRequest,
    ) -> ChatCompletionRequest:
        """Return a request with `model` substituted to `served_model` if set.

        Returns the original request when `served_model` is None — no copy
        on the hot path for the pass-through case. Adapters call this at
        the top of `chat()` / `stream_chat()` before delegating upstream.
        """
        if self.served_model is None or request.model == self.served_model:
            return request
        # Pydantic model_copy is shallow + cheap; preserves all other fields.
        return request.model_copy(update={"model": self.served_model})

    async def close(self) -> None:
        """Release any held resources. Default no-op; override if needed."""

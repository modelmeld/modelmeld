# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""AnthropicAdapter — pass-through to Anthropic Messages API with schema translation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.adapters.retry import (
    RetryConfig,
    retry_async,
    wrap_as_adapter_error,
)
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
)
from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest
from modelmeld.translation import (
    AnthropicStreamTranslator,
    from_anthropic_response,
    to_anthropic_params,
)


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
        served_model: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise AdapterError(
                "AnthropicAdapter requires the `anthropic` package. "
                "Install with: pip install 'modelmeld[anthropic]'"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AdapterError(
                "AnthropicAdapter requires an API key "
                "(pass api_key= or set ANTHROPIC_API_KEY / MODELMELD_ANTHROPIC_API_KEY)."
            )

        # Disable the SDK's built-in retry; we own retry policy via retry_async.
        # Stacking SDK retries on top of ours wastes time and rate limit.
        kwargs: dict = {"api_key": key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._retry_config = retry_config or RetryConfig()
        # F-8: operator-pinned upstream model (overrides request.model).
        self.served_model = served_model

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        native_request: object | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Non-streaming chat.

        `extra_headers` is the optional /v1/messages escape hatch for
        forwarding caller-supplied Anthropic protocol headers
        (`anthropic-beta`, `anthropic-version`, etc.) verbatim to the
        upstream. Without this, beta features the customer activates
        silently fall back at our gateway.
        """
        params = self._build_params(request, native_request)
        if extra_headers:
            params["extra_headers"] = dict(extra_headers)

        async def _call():
            return await self._client.messages.create(**params)

        try:
            sdk_message = await retry_async(
                _call, self._retry_config, label="anthropic.chat",
            )
        except Exception as e:
            raise wrap_as_adapter_error(e, "Anthropic chat call failed") from e
        return from_anthropic_response(sdk_message.model_dump())

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        native_request: object | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        params = self._build_params(request, native_request)
        params["stream"] = True
        if extra_headers:
            params["extra_headers"] = dict(extra_headers)

        async def _open_stream():
            return await self._client.messages.create(**params)

        try:
            stream = await retry_async(
                _open_stream, self._retry_config, label="anthropic.stream_chat",
            )
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "Anthropic stream_chat call failed",
            ) from e

        translator = AnthropicStreamTranslator()
        async for event in stream:
            chunk = translator.translate_event(event.model_dump())
            if chunk is not None:
                yield chunk

    def _build_params(
        self,
        request: ChatCompletionRequest,
        native_request: object | None,
    ) -> dict:
        """Construct the Anthropic SDK params from either the native
        Anthropic request (preserving cache_control + tool schemas +
        image content blocks intact) or, when not available, by
        round-tripping through the OpenAI internal shape.

        Native-passthrough is the path /v1/messages takes when routing
        to an Anthropic upstream — without it, cache_control breakpoints
        get silently dropped and customers pay ~5x more on what would
        otherwise be cache hits (the failure mode musistudio/claude-code-router
        ships today). /v1/chat/completions callers don't supply
        native_request and use the translation path.
        """
        if isinstance(native_request, AnthropicMessagesRequest):
            # Native passthrough — preserve the customer's exact request
            # shape. Apply F-8 served_model substitution at this layer so
            # operators can still pin the upstream model regardless of
            # what the customer asked for.
            params = native_request.model_dump(exclude_none=True)
            if self.served_model is not None:
                params["model"] = self.served_model
            elif params.get("model") is None:
                # Defense in depth — Anthropic SDK requires `model`.
                params["model"] = request.model
            return params
        # Translation path (the existing /v1/chat/completions behavior).
        request = self._apply_served_model(request)
        return to_anthropic_params(request)

    async def health(self) -> bool:
        # Anthropic has no cheap public health endpoint; consider the client
        # configured-and-imported as healthy. Real check happens on first call.
        return True

    async def close(self) -> None:
        await self._client.close()

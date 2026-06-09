# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""OpenAIAdapter — pass-through to OpenAI's cloud API via the official SDK.

Named `openai_adapter` (not `openai`) to avoid shadowing the upstream package
when read by humans.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

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


class OpenAIAdapter(ProviderAdapter):
    name = "openai"
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        retry_config: RetryConfig | None = None,
        served_model: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise AdapterError(
                "OpenAIAdapter requires the `openai` package. "
                "Install with: pip install 'modelmeld[openai]'"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AdapterError(
                "OpenAIAdapter requires an API key "
                "(pass api_key= or set OPENAI_API_KEY / MODELMELD_OPENAI_API_KEY)."
            )
        # Disable the SDK's built-in retry; retry policy lives in retry_async.
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )
        self._retry_config = retry_config or RetryConfig()
        # F-8: operator-pinned upstream model (overrides request.model).
        self.served_model = served_model

    def _to_params(
        self, request: ChatCompletionRequest, *, stream: bool
    ) -> dict[str, Any]:
        # exclude_none keeps optional fields off the wire so we don't override the
        # upstream's defaults; we set `stream` explicitly per call.
        excluded: set[str] = {"stream"}
        if not stream:
            excluded.add("stream_options")
        params = request.model_dump(exclude_none=True, exclude=excluded)
        params["stream"] = stream
        return params

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        request = self._apply_served_model(request)

        async def _call():
            return await self._client.chat.completions.create(
                **self._to_params(request, stream=False)
            )

        try:
            sdk_response = await retry_async(
                _call, self._retry_config, label="openai.chat",
            )
        except Exception as e:
            raise wrap_as_adapter_error(e, "OpenAI chat call failed") from e
        return ChatCompletion.model_validate(sdk_response.model_dump())

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        request = self._apply_served_model(request)

        async def _open_stream():
            return await self._client.chat.completions.create(
                **self._to_params(request, stream=True)
            )

        try:
            stream = await retry_async(
                _open_stream, self._retry_config, label="openai.stream_chat",
            )
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "OpenAI stream_chat call failed",
            ) from e
        # Errors raised DURING iteration (e.g. a provider that opens the stream
        # 200 then injects an SSE error event, which the SDK re-raises as a bare
        # APIError) must also be wrapped as AdapterError. Otherwise they escape
        # uncaught past `_try_open_stream_native` (which only catches
        # AdapterError), bypassing the router's failover and surfacing as a 500
        # instead of a clean 502/failover. GeneratorExit / CancelledError are
        # BaseException, not Exception, so consumer-side close/cancel passes
        # through untouched.
        try:
            async for chunk in stream:
                yield ChatCompletionChunk.model_validate(chunk.model_dump())
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "OpenAI stream_chat stream interrupted",
            ) from e

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""AnthropicAdapter — pass-through to Anthropic Messages API with schema translation."""

from __future__ import annotations

import json
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
from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest
from modelmeld.translation import (
    AnthropicStreamTranslator,
    from_anthropic_response,
    to_anthropic_params,
)

# Anthropic Messages API version. Same value Claude Code sends today —
# preserved so OAuth-passthrough requests look indistinguishable on the
# wire from a direct Claude Code call.
_ANTHROPIC_API_VERSION = "2023-06-01"
_DEFAULT_BASE_URL = "https://api.anthropic.com"


def _block_text(content: object) -> str:
    """Flatten a message content (str or list of blocks) to its text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _hoist_system_messages(params: dict) -> None:
    """Move any `role: "system"` entries out of `messages[]` into the top-level
    `system` field, in place.

    Anthropic's API rejects system-role messages in the array ("role 'system'
    is not supported on this model") — system must be top-level. Real clients
    (Claude Code) put system-role messages in the array, and native passthrough
    forwards them verbatim, so hoist them here. The existing top-level `system`
    is preserved structurally (incl. cache_control on its blocks); hoisted text
    is appended.
    """
    msgs = params.get("messages")
    if not isinstance(msgs, list):
        return
    hoisted: list[str] = []
    kept: list = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            text = _block_text(m.get("content"))
            if text:
                hoisted.append(text)
        else:
            kept.append(m)
    if not hoisted:
        return
    params["messages"] = kept
    existing = params.get("system")
    if existing is None:
        params["system"] = "\n\n".join(hoisted)
    elif isinstance(existing, str):
        params["system"] = "\n\n".join([existing, *hoisted])
    elif isinstance(existing, list):
        # Append as text blocks; existing blocks (with any cache_control) intact.
        params["system"] = [*existing, *({"type": "text", "text": t} for t in hoisted)]


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        oauth_bearer: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        retry_config: RetryConfig | None = None,
        served_model: str | None = None,
    ) -> None:
        """Two construction modes, mutually exclusive:

        - **API key mode** (`api_key=...` or `ANTHROPIC_API_KEY` env):
          uses the official `anthropic` SDK with `x-api-key` auth.
          The historical default.

        - **OAuth-bearer mode** (`oauth_bearer=...`): subscription
          passthrough for Claude Max users. Bypasses the SDK
          (which doesn't speak OAuth) and uses raw `httpx` to POST
          to `/v1/messages` with `Authorization: Bearer <jwt>`.
          ToS posture: self-host only, single-user-per-instance.
          See `docs/subscription-passthrough.md` (Sprint 5).
        """
        if oauth_bearer and api_key:
            raise AdapterError(
                "AnthropicAdapter: pass api_key OR oauth_bearer, not both. "
                "API-key and subscription-OAuth modes are mutually exclusive."
            )

        self._retry_config = retry_config or RetryConfig()
        # F-8: operator-pinned upstream model (overrides request.model).
        self.served_model = served_model

        if oauth_bearer:
            # OAuth-bearer mode — raw HTTP, no SDK.
            self._oauth_bearer: str | None = oauth_bearer
            self._base_url: str = (base_url or _DEFAULT_BASE_URL).rstrip("/")
            self._http: httpx.AsyncClient | None = (
                http_client or httpx.AsyncClient(timeout=120.0)
            )
            self._owns_http = http_client is None
            self._client = None
            return

        # API-key mode — existing SDK path.
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
                "AnthropicAdapter requires either an API key "
                "(api_key= / ANTHROPIC_API_KEY / MODELMELD_ANTHROPIC_API_KEY) "
                "or an OAuth bearer (oauth_bearer=) for subscription passthrough."
            )

        # Disable the SDK's built-in retry; we own retry policy via retry_async.
        # Stacking SDK retries on top of ours wastes time and rate limit.
        kwargs: dict = {"api_key": key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._oauth_bearer = None
        self._http = None
        self._owns_http = False
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

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
        if self._oauth_bearer is not None:
            return await self._chat_via_oauth(
                request, native_request=native_request, extra_headers=extra_headers,
            )
        params = self._build_params(request, native_request)
        if extra_headers:
            params["extra_headers"] = dict(extra_headers)

        # Type narrow: oauth_bearer is None here, so api-key path was taken
        # and self._client is the AsyncAnthropic instance. Capture in a
        # local so the closure below sees the narrowed type.
        assert self._client is not None
        client = self._client

        async def _call():
            return await client.messages.create(**params)

        try:
            sdk_message = await retry_async(
                _call, self._retry_config, label="anthropic.chat",
            )
        except Exception as e:
            raise wrap_as_adapter_error(e, "Anthropic chat call failed") from e
        return from_anthropic_response(sdk_message.model_dump())

    async def _chat_via_oauth(
        self,
        request: ChatCompletionRequest,
        *,
        native_request: object | None,
        extra_headers: dict[str, str] | None,
    ) -> ChatCompletion:
        """Sprint 5 OAuth-bearer path. Raw httpx POST to /v1/messages
        with Authorization: Bearer instead of x-api-key. Uses the same
        translation helpers as the SDK path so response shapes match."""
        params = self._build_params(request, native_request)
        # SDK-specific "extra_headers" wrapper isn't sent on the wire —
        # for raw HTTP we merge the caller's headers directly.
        params.pop("extra_headers", None)
        params["stream"] = False
        headers = _oauth_request_headers(self._oauth_bearer, extra_headers)
        # Capture the http client in a local; the closure below can't
        # narrow `self._http` because pyright treats attribute access
        # across awaits as possibly-mutated.
        assert self._http is not None
        http = self._http

        async def _call() -> dict[str, Any]:
            response = await http.post(
                f"{self._base_url}/v1/messages",
                content=json.dumps(params).encode("utf-8"),
                headers=headers,
            )
            if response.status_code >= 400:
                raise AdapterError(
                    f"Anthropic OAuth {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response.json()

        try:
            response_dict = await retry_async(
                _call, self._retry_config, label="anthropic.chat.oauth",
            )
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "Anthropic OAuth passthrough chat failed",
            ) from e
        return from_anthropic_response(response_dict)

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        *,
        native_request: object | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if self._oauth_bearer is not None:
            async for chunk in self._stream_chat_via_oauth(
                request, native_request=native_request, extra_headers=extra_headers,
            ):
                yield chunk
            return

        params = self._build_params(request, native_request)
        params["stream"] = True
        if extra_headers:
            params["extra_headers"] = dict(extra_headers)

        # Type narrow: oauth_bearer is None here (oauth branch returned
        # above), so the api-key path was taken and self._client exists.
        assert self._client is not None
        client = self._client

        async def _open_stream():
            return await client.messages.create(**params)

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

    async def _stream_chat_via_oauth(
        self,
        request: ChatCompletionRequest,
        *,
        native_request: object | None,
        extra_headers: dict[str, str] | None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Sprint 5 OAuth-bearer streaming. Parses Anthropic SSE format
        manually (the SDK's stream helpers don't apply on the OAuth
        path) and feeds each event dict to AnthropicStreamTranslator."""
        params = self._build_params(request, native_request)
        params.pop("extra_headers", None)
        params["stream"] = True
        headers = _oauth_request_headers(self._oauth_bearer, extra_headers)
        # SSE upstream — accept text/event-stream
        headers["accept"] = "text/event-stream"
        assert self._http is not None

        translator = AnthropicStreamTranslator()
        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/v1/messages",
                content=json.dumps(params).encode("utf-8"),
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise AdapterError(
                        f"Anthropic OAuth stream {response.status_code}: "
                        f"{body.decode('utf-8', 'replace')[:200]}"
                    )
                async for event_dict in _parse_anthropic_sse(response):
                    chunk = translator.translate_event(event_dict)
                    if chunk is not None:
                        yield chunk
        except AdapterError:
            raise
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "Anthropic OAuth passthrough stream_chat failed",
            ) from e

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
            extras = dict(native_request.model_extra or {})
            # Model substitution: capability/alias routing rewrote the internal
            # request.model to the scout's pick, while native_request.model is
            # still what the client asked for. When they differ, drop client
            # `thinking` config — it was tuned for the requested model and may be
            # unsupported on the one we routed to (Anthropic 400 "adaptive
            # thinking is not supported on this model"). See projectplan backlog
            # B-3 for the capability-aware refinement (forward when supported).
            if request.model and native_request.model != request.model:
                params.pop("thinking", None)
                extras.pop("thinking", None)
            # Fields the client sent that aren't declared on our schema
            # (extra="allow") — e.g. Claude Code's `context_management` — are NOT
            # valid keyword args for the SDK's create(); passing them raises
            # "unexpected keyword argument". Route them through `extra_body` so
            # the SDK still forwards them to the API verbatim (passthrough
            # intent preserved) without choking on unknown kwargs.
            if extras:
                for key in extras:
                    params.pop(key, None)
                params["extra_body"] = {**(params.get("extra_body") or {}), **extras}
            _hoist_system_messages(params)
            return params
        # Translation path (the existing /v1/chat/completions behavior).
        request = self._apply_served_model(request)
        return to_anthropic_params(request)

    async def health(self) -> bool:
        # Anthropic has no cheap public health endpoint; consider the client
        # configured-and-imported as healthy. Real check happens on first call.
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._http is not None and self._owns_http:
            await self._http.aclose()


def _oauth_request_headers(
    oauth_bearer: str | None, extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    """Build the header set for an OAuth-bearer request to api.anthropic.com.

    Required: Authorization (Bearer JWT), anthropic-version, content-type.
    `extra_headers` (typically forwarded from the inbound /v1/messages
    request) merges last so customer-provided beta flags / version
    overrides win — same precedence as the SDK path's extra_headers.
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {oauth_bearer}",
        "anthropic-version": _ANTHROPIC_API_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


async def _parse_anthropic_sse(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """Parse Anthropic's Messages API SSE stream into event dicts.

    Format:
        event: <event_type>
        data: <json>
        <blank line>

    Yields a dict per event with `type` set to the SSE `event:` line
    plus all fields from the parsed JSON `data:` payload —
    matching the shape AnthropicStreamTranslator expects.
    """
    current_event: str | None = None
    async for line in response.aiter_lines():
        if not line:
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                if current_event and "type" not in payload:
                    payload["type"] = current_event
                yield payload

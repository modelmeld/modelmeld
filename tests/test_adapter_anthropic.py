"""AnthropicAdapter unit tests + gated integration test."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from modelmeld.adapters.anthropic_adapter import AnthropicAdapter
from modelmeld.adapters.base import AdapterError
from modelmeld.api.schemas import ChatCompletionRequest

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="API key"):
        AnthropicAdapter()


def test_accepts_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    adapter = AnthropicAdapter()
    assert adapter.name == "anthropic"


def test_accepts_api_key_from_constructor() -> None:
    adapter = AnthropicAdapter(api_key="ctor-key")
    assert adapter.name == "anthropic"


def test_is_egress_true() -> None:
    adapter = AnthropicAdapter(api_key="x")
    assert adapter.is_egress is True


# ---------------------------------------------------------------------------
# chat() — mocked SDK response
# ---------------------------------------------------------------------------

def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Capital of France?"},
        ],
        max_completion_tokens=200,
        temperature=0.0,
    )


def _fake_sdk_message() -> SimpleNamespace:
    """Returns a stand-in for anthropic.types.Message with model_dump()."""
    payload = {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Paris."}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 2},
    }
    return SimpleNamespace(model_dump=lambda: payload)


async def test_chat_translates_request_and_response() -> None:
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    response = await adapter.chat(_request())

    assert response.id == "msg_test_1"
    assert response.choices[0].message.content == "Paris."
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 2

    # Verify the call into the SDK was translated correctly
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["system"] == "You are concise."
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["temperature"] == 0.0
    # System extracted out of messages
    assert all(m["role"] != "system" for m in call_kwargs["messages"])


async def test_chat_wraps_upstream_error_in_adapter_error() -> None:
    adapter = AnthropicAdapter(api_key="test-key")
    adapter._client.messages.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("anthropic 500")
    )
    with pytest.raises(AdapterError, match="Anthropic chat call failed"):
        await adapter.chat(_request())


# ---------------------------------------------------------------------------
# stream_chat() — mocked SDK event stream
# ---------------------------------------------------------------------------

def _event(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: payload)


class _FakeStream:
    def __init__(self, events: list) -> None:
        self._events = events

    def __aiter__(self) -> _FakeStream:
        self._i = iter(self._events)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._i)
        except StopIteration as e:
            raise StopAsyncIteration from e


async def test_stream_chat_translates_event_stream() -> None:
    adapter = AnthropicAdapter(api_key="test-key")
    events = [
        _event({
            "type": "message_start",
            "message": {"id": "msg_s_1", "model": "claude-sonnet-4-6"},
        }),
        _event({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }),
        _event({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        }),
        _event({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 10},
        }),
        _event({"type": "message_stop"}),
    ]
    adapter._client.messages.create = AsyncMock(return_value=_FakeStream(events))  # type: ignore[method-assign]

    chunks = []
    async for chunk in adapter.stream_chat(_request()):
        chunks.append(chunk)

    # role chunk + 2 content deltas + 1 finish chunk = 4 chunks
    assert len(chunks) == 4
    assert chunks[0].choices[0].delta.role == "assistant"
    reconstructed = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert reconstructed == "Hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_stream_chat_wraps_upstream_error() -> None:
    adapter = AnthropicAdapter(api_key="test-key")
    adapter._client.messages.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("anthropic stream 500")
    )
    with pytest.raises(AdapterError, match="Anthropic stream_chat"):
        async for _ in adapter.stream_chat(_request()):
            pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def test_health_returns_true() -> None:
    """Anthropic has no cheap health endpoint; configured client is assumed healthy."""
    adapter = AnthropicAdapter(api_key="test-key")
    assert await adapter.health() is True


# ---------------------------------------------------------------------------
# OAuth-bearer mode (Sprint 5 subscription passthrough)
# ---------------------------------------------------------------------------

import json

import httpx

from modelmeld.api.schemas import UserMessage


def test_rejects_both_api_key_and_oauth_bearer() -> None:
    """The two auth modes are mutually exclusive — silent precedence
    would surprise operators about which mode is active."""
    with pytest.raises(AdapterError, match="mutually exclusive"):
        AnthropicAdapter(api_key="sk-ant-x", oauth_bearer="eyJtest")


def test_oauth_bearer_mode_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OAuth-bearer mode must work even when ANTHROPIC_API_KEY is unset."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MODELMELD_ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicAdapter(oauth_bearer="eyJfake_jwt")
    assert adapter.name == "anthropic"


def _oauth_response_handler(captured: dict) -> httpx.MockTransport:
    """MockTransport that records the request + returns canned non-stream JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        if request.content:
            captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_oauth_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Hello from OAuth path"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 5},
            },
        )

    return httpx.MockTransport(handler)


async def test_chat_via_oauth_sends_bearer_authorization_header() -> None:
    """OAuth path: POST hits /v1/messages with Authorization: Bearer <jwt>
    (NOT x-api-key, which is the SDK/API-key-mode auth header)."""
    captured: dict = {}
    http = httpx.AsyncClient(transport=_oauth_response_handler(captured))
    adapter = AnthropicAdapter(
        oauth_bearer="eyJtest_subscription_jwt",
        http_client=http,
    )
    try:
        result = await adapter.chat(ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[UserMessage(role="user", content="hello")],
            max_tokens=64,
        ))
    finally:
        await adapter.close()

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/v1/messages"), captured["url"]
    assert captured["headers"].get("authorization") == "Bearer eyJtest_subscription_jwt"
    assert "x-api-key" not in captured["headers"]
    assert captured["headers"].get("anthropic-version") == "2023-06-01"
    # Response translation roundtrip — sanity check
    assert result.choices[0].message.content == "Hello from OAuth path"


async def test_chat_via_oauth_forwards_extra_headers() -> None:
    """`extra_headers` (typically forwarded from the inbound /v1/messages
    request) merge AFTER the OAuth defaults — customer beta flags win."""
    captured: dict = {}
    http = httpx.AsyncClient(transport=_oauth_response_handler(captured))
    adapter = AnthropicAdapter(oauth_bearer="eyJjwt", http_client=http)
    try:
        await adapter.chat(
            ChatCompletionRequest(
                model="claude-sonnet-4-6",
                messages=[UserMessage(role="user", content="hi")],
                max_tokens=32,
            ),
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31",
                "user-agent": "Claude-Code/1.0",
            },
        )
    finally:
        await adapter.close()

    assert captured["headers"].get("anthropic-beta") == "prompt-caching-2024-07-31"
    assert captured["headers"].get("user-agent") == "Claude-Code/1.0"


async def test_chat_via_oauth_wraps_upstream_error() -> None:
    """4xx from api.anthropic.com (e.g., expired token) surfaces as AdapterError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"type": "authentication_error", "message": "Invalid bearer"}},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(oauth_bearer="eyJexpired", http_client=http)
    try:
        with pytest.raises(AdapterError, match="Anthropic OAuth"):
            await adapter.chat(ChatCompletionRequest(
                model="claude-sonnet-4-6",
                messages=[UserMessage(role="user", content="hi")],
                max_tokens=32,
            ))
    finally:
        await adapter.close()


async def test_stream_chat_via_oauth_parses_sse_events() -> None:
    """Streaming OAuth path: parses Anthropic SSE format manually and
    yields ChatCompletionChunks for each text delta."""
    sse_body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":4,"output_tokens":0}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(oauth_bearer="eyJjwt", http_client=http)
    try:
        chunks: list = []
        async for chunk in adapter.stream_chat(ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[UserMessage(role="user", content="hi")],
            max_tokens=32,
            stream=True,
        )):
            chunks.append(chunk)
    finally:
        await adapter.close()

    # The translator yields at least one chunk per content_block_delta
    # event with text. We expect "Hello" then " world" to surface.
    texts: list[str] = []
    for chunk in chunks:
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                texts.append(choice.delta.content)
    joined = "".join(texts)
    assert "Hello" in joined and "world" in joined


async def test_close_releases_http_client_in_oauth_mode() -> None:
    """OAuth-mode adapter owns its httpx client when one wasn't passed
    in. close() should aclose() it — defense against client leak."""
    adapter = AnthropicAdapter(oauth_bearer="eyJjwt")
    # When http_client wasn't passed, adapter owns it
    assert adapter._owns_http is True
    # close() must not raise even though no actual HTTP traffic flowed.
    await adapter.close()


# ---------------------------------------------------------------------------
# Gated integration test against the real Anthropic API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="integration test — set ANTHROPIC_API_KEY to run",
)
# ---------------------------------------------------------------------------
# native_request passthrough — Anthropic compliance
# ---------------------------------------------------------------------------
# When /v1/messages forwards to AnthropicAdapter, the original Anthropic
# request body is passed as `native_request=` so the adapter preserves
# cache_control, image content blocks, etc. — fields that the OpenAI
# internal shape drops. Without this passthrough, our gateway has the
# same `cache_control`-stripping bug as musistudio/claude-code-router
# and customers pay ~5x more on cache misses.

def _anthropic_request_with_cache_control() -> dict:
    """A realistic Claude-Code-shaped request with cache_control breakpoints
    on the system message and an early user message."""
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "system": [
            {
                "type": "text",
                "text": "You are a coding assistant. Be terse.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Refactor my Python codebase.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "assistant", "content": "Sure, show me a file."},
            {"role": "user", "content": "Here's main.py: ..."},
        ],
    }


async def test_chat_preserves_cache_control_when_native_request_supplied() -> None:
    """The smoking-gun test — when messages route passes native_request,
    cache_control breakpoints reach the upstream Anthropic call verbatim.
    """
    from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest

    body = AnthropicMessagesRequest.model_validate(
        _anthropic_request_with_cache_control()
    )
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    # The internal OpenAI-shape request is what /v1/messages would build
    # by translating body. It does NOT contain cache_control (that's the
    # whole point of native_request).
    internal_req = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ignored"}],
    )

    await adapter.chat(internal_req, native_request=body)

    call_kwargs = mock_create.call_args.kwargs
    # System came through as the native list form (not flattened to string)
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # User-message cache_control breakpoint preserved
    first_user_content = call_kwargs["messages"][0]["content"]
    assert isinstance(first_user_content, list)
    assert first_user_content[0]["cache_control"] == {"type": "ephemeral"}


async def test_native_passthrough_routes_unknown_fields_to_extra_body() -> None:
    """Fields the client sends that aren't on our schema (extra="allow") — e.g.
    Claude Code's `context_management` — must go via `extra_body`, not as
    top-level kwargs the SDK rejects ("unexpected keyword argument"). Regression
    for the streaming 502 the dogfooding loop surfaced.
    """
    from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest

    body = AnthropicMessagesRequest.model_validate({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
    })
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(
        ChatCompletionRequest(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ignored"}],
        ),
        native_request=body,
    )

    call_kwargs = mock_create.call_args.kwargs
    # Not a top-level kwarg (that's what raised the SDK TypeError) ...
    assert "context_management" not in call_kwargs
    # ... forwarded via extra_body instead, preserving passthrough intent.
    assert call_kwargs["extra_body"]["context_management"] == {
        "edits": [{"type": "clear_tool_uses_20250919"}]
    }


async def test_thinking_dropped_when_model_substituted() -> None:
    """Capability/alias routing serves a different model than the client asked
    for. Client `thinking` config is tuned for the requested model and may be
    unsupported on the routed one ("adaptive thinking is not supported on this
    model"), so it's dropped on substitution. Regression for the loop's 400.
    """
    from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest

    body = AnthropicMessagesRequest.model_validate({
        "model": "anthropic/modelmeld-auto",   # alias the client requested
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
    })
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(
        ChatCompletionRequest(  # routing substituted the model
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
        ),
        native_request=body,
    )

    ck = mock_create.call_args.kwargs
    assert "thinking" not in ck
    assert "thinking" not in (ck.get("extra_body") or {})


async def test_thinking_preserved_when_no_substitution() -> None:
    """When the served model equals what the client requested (no substitution),
    the client's thinking config is forwarded verbatim via extra_body."""
    from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest

    thinking = {"type": "enabled", "budget_tokens": 1024}
    body = AnthropicMessagesRequest.model_validate({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": thinking,
    })
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(
        ChatCompletionRequest(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
        ),
        native_request=body,
    )

    ck = mock_create.call_args.kwargs
    assert (ck.get("extra_body") or {}).get("thinking") == thinking


async def test_chat_falls_back_to_translation_when_no_native_request() -> None:
    """/v1/chat/completions callers don't supply native_request; the adapter
    must still work via the translation path (existing behavior unchanged).
    """
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(_request())  # no native_request kwarg

    call_kwargs = mock_create.call_args.kwargs
    # System is the flattened-from-OpenAI-shape string
    assert call_kwargs["system"] == "You are concise."


async def test_chat_forwards_extra_headers_to_upstream() -> None:
    """anthropic-beta and anthropic-version headers from the caller must
    reach the upstream Anthropic call verbatim. Without this, beta features
    silently fall back at our gateway boundary."""
    adapter = AnthropicAdapter(api_key="test-key")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(
        _request(),
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31",
            "anthropic-version": "2023-06-01",
        },
    )

    call_kwargs = mock_create.call_args.kwargs
    assert "extra_headers" in call_kwargs
    assert call_kwargs["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert call_kwargs["extra_headers"]["anthropic-version"] == "2023-06-01"


async def test_chat_served_model_override_wins_over_native_request_model() -> None:
    """F-8 served_model substitution still applies on the native path."""
    from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest

    body = AnthropicMessagesRequest.model_validate(
        _anthropic_request_with_cache_control()
    )
    # Adapter pinned to a different model
    adapter = AnthropicAdapter(api_key="test-key", served_model="claude-opus-4-7")
    mock_create = AsyncMock(return_value=_fake_sdk_message())
    adapter._client.messages.create = mock_create  # type: ignore[method-assign]

    await adapter.chat(_request(), native_request=body)

    call_kwargs = mock_create.call_args.kwargs
    # served_model wins, not body.model
    assert call_kwargs["model"] == "claude-opus-4-7"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY env var; gated integration test against the real Anthropic API.",
)
async def test_round_trip_against_real_anthropic() -> None:
    adapter = AnthropicAdapter()
    try:
        response = await adapter.chat(
            ChatCompletionRequest(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_completion_tokens=10,
                temperature=0.0,
            )
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason in ("stop", "length")
    finally:
        await adapter.close()

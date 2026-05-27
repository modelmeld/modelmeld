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

"""OpenAIAdapter unit tests (mocked SDK) + optional integration test (gated)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion as SDKChatCompletion
from openai.types.chat import ChatCompletionChunk as SDKChatCompletionChunk

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter
from modelmeld.api.schemas import ChatCompletionRequest
from tests.fixtures.openai_responses import SIMPLE_TEXT


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="API key"):
        OpenAIAdapter()


def test_accepts_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    adapter = OpenAIAdapter()
    assert adapter.name == "openai"


def test_accepts_api_key_from_constructor() -> None:
    adapter = OpenAIAdapter(api_key="ctor-key")
    assert adapter.name == "openai"


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )


async def test_chat_translates_request_and_response() -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    sdk_response = SDKChatCompletion.model_validate(SIMPLE_TEXT)

    mock_create = AsyncMock(return_value=sdk_response)
    adapter._client.chat.completions.create = mock_create  # type: ignore[method-assign]

    response = await adapter.chat(_request())

    assert response.id == SIMPLE_TEXT["id"]
    assert response.model == SIMPLE_TEXT["model"]
    # adapter must have called with stream=False
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["stream"] is False
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_wraps_upstream_errors_in_adapter_error() -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("upstream 500")
    )

    with pytest.raises(AdapterError, match="OpenAI chat call failed"):
        await adapter.chat(_request())


class _FakeAsyncStream:
    def __init__(self, chunks: list[SDKChatCompletionChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeAsyncStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> SDKChatCompletionChunk:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


async def test_stream_chat_yields_converted_chunks() -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    sdk_chunks = [
        SDKChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        SDKChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        ),
    ]
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_FakeAsyncStream(sdk_chunks)
    )

    collected = []
    async for c in adapter.stream_chat(_request()):
        collected.append(c)

    assert len(collected) == 2
    assert collected[0].choices[0].delta.role == "assistant"
    assert collected[-1].choices[0].finish_reason == "stop"


async def test_health_true_on_success() -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    adapter._client.models.list = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    assert await adapter.health() is True


async def test_health_false_on_failure() -> None:
    adapter = OpenAIAdapter(api_key="test-key")
    adapter._client.models.list = AsyncMock(side_effect=RuntimeError("net"))  # type: ignore[method-assign]
    assert await adapter.health() is False


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="integration test — set OPENAI_API_KEY to run",
)
async def test_round_trip_against_real_openai() -> None:
    """Gated integration test. Requires OPENAI_API_KEY in env."""
    adapter = OpenAIAdapter()
    try:
        response = await adapter.chat(
            ChatCompletionRequest(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Reply with exactly: OK"},
                ],
                max_completion_tokens=10,
                temperature=0.0,
            )
        )
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason in ("stop", "length")
    finally:
        await adapter.close()

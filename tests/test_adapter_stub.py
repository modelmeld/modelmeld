"""StubAdapter unit tests."""

from __future__ import annotations

from modelmeld.adapters.stub import StubAdapter
from modelmeld.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    StreamOptions,
)


async def test_stub_chat_returns_valid_completion() -> None:
    adapter = StubAdapter()
    request = ChatCompletionRequest(
        model="any-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    response = await adapter.chat(request)

    assert response.model == "any-model"
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content is not None
    assert response.usage is not None
    assert response.usage.total_tokens > 0


async def test_stub_stream_emits_role_then_content_then_finish() -> None:
    adapter = StubAdapter()
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks: list[ChatCompletionChunk] = []
    async for c in adapter.stream_chat(request):
        chunks.append(c)

    assert len(chunks) >= 3
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"
    # all chunks share the same id
    assert len({c.id for c in chunks}) == 1


async def test_stub_stream_with_include_usage() -> None:
    adapter = StubAdapter()
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )
    chunks: list[ChatCompletionChunk] = []
    async for c in adapter.stream_chat(request):
        chunks.append(c)

    with_usage = [c for c in chunks if c.usage is not None]
    assert len(with_usage) == 1


async def test_stub_health_returns_true() -> None:
    assert await StubAdapter().health() is True

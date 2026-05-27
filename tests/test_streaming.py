"""SSE streaming tests for /v1/chat/completions with stream=true."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk as OpenAIChunk

from modelmeld.api.schemas import ChatCompletionChunk
from modelmeld.api.server import build_app


def _parse_sse_events(body: str) -> list[str]:
    """Split an SSE body into the data payloads (excluding [DONE])."""
    events: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        events.append(payload)
    return events


def test_streaming_response_content_type() -> None:
    with TestClient(build_app()) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


def test_streaming_emits_role_then_content_then_finish() -> None:
    with TestClient(build_app()) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode("utf-8")

    assert "data: [DONE]" in body
    events = _parse_sse_events(body)
    assert len(events) >= 3  # role + at least 1 content + finish

    chunks = [ChatCompletionChunk.model_validate_json(e) for e in events]
    assert all(c.object == "chat.completion.chunk" for c in chunks)
    # First chunk carries the role
    assert chunks[0].choices[0].delta.role == "assistant"
    # Final chunk carries finish_reason
    assert chunks[-1].choices[0].finish_reason == "stop"
    # All chunks share the same id
    assert len({c.id for c in chunks}) == 1


def test_streaming_with_include_usage_emits_usage_chunk() -> None:
    with TestClient(build_app()) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode("utf-8")

    events = _parse_sse_events(body)
    chunks = [ChatCompletionChunk.model_validate_json(e) for e in events]
    # Exactly one chunk should carry usage when include_usage is set.
    with_usage = [c for c in chunks if c.usage is not None]
    assert len(with_usage) == 1
    assert with_usage[0].usage is not None
    assert with_usage[0].usage.total_tokens == 25


def test_streaming_chunks_parse_with_openai_sdk() -> None:
    with TestClient(build_app()) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = b"".join(response.iter_bytes()).decode("utf-8")

    events = _parse_sse_events(body)
    sdk_chunks = [OpenAIChunk.model_validate_json(e) for e in events]
    # Reconstruct content from deltas the way the SDK consumer would
    reconstructed = "".join(
        (c.choices[0].delta.content or "") for c in sdk_chunks if c.choices
    )
    assert "ModelMeld stub adapter" in reconstructed


async def test_openai_sdk_streams_via_asgi_transport() -> None:
    """End-to-end: official openai SDK against our app via ASGI transport, streaming.

    Uses AsyncOpenAI + httpx.AsyncClient because httpx.ASGITransport only implements
    the async transport interface.
    """
    transport = httpx.ASGITransport(app=build_app())
    http_client = httpx.AsyncClient(transport=transport, base_url="http://gateway.test")
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://gateway.test/v1",
        http_client=http_client,
    )
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = [c async for c in stream]
    await http_client.aclose()

    assert len(chunks) >= 3
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_non_stream_path_unaffected() -> None:
    with TestClient(build_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"


def test_invalid_request_returns_422() -> None:
    """The new strict schema rejects malformed requests."""
    with TestClient(build_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini"},  # missing required `messages`
        )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "bad_role",
    ["nonsense", "developer", ""],
)
def test_invalid_message_role_rejected(bad_role: str) -> None:
    with TestClient(build_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": bad_role, "content": "hi"}],
            },
        )
    assert response.status_code == 422

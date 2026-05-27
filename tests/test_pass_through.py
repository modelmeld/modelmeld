"""End-to-end pass-through tests: route → configured adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app


class RecordingAdapter(ProviderAdapter):
    """Test double — captures requests and returns canned responses."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[ChatCompletionRequest] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.calls.append(request)
        return ChatCompletion(
            id="chatcmpl-recorded",
            created=1715000000,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=f"recorded reply for {request.model}"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        self.calls.append(request)
        yield ChatCompletionChunk(
            id="chatcmpl-recorded",
            created=1715000000,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=""))],
        )
        yield ChatCompletionChunk(
            id="chatcmpl-recorded",
            created=1715000000,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="hello"))],
        )
        yield ChatCompletionChunk(
            id="chatcmpl-recorded",
            created=1715000000,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )

    async def health(self) -> bool:
        return True


def test_route_delegates_to_configured_adapter() -> None:
    adapter = RecordingAdapter()
    with TestClient(build_app(adapter=adapter)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chatcmpl-recorded"
    assert body["model"] == "gpt-4o-mini"
    assert len(adapter.calls) == 1
    assert adapter.calls[0].messages[0].content == "ping"  # type: ignore[union-attr]


def test_route_delegates_streaming_to_configured_adapter() -> None:
    adapter = RecordingAdapter()
    with TestClient(build_app(adapter=adapter)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")
    assert "data: [DONE]" in body
    assert '"id":"chatcmpl-recorded"' in body
    assert len(adapter.calls) == 1


def test_default_app_uses_single_stub_router() -> None:
    from modelmeld.router import SingleAdapterRouter

    app = build_app()
    assert isinstance(app.state.router, SingleAdapterRouter)
    assert app.state.router.adapter.name == "stub"


def test_openai_provider_setting_constructs_openai_adapter(
    monkeypatch,
) -> None:
    from modelmeld.config import GatewaySettings
    from modelmeld.router import SingleAdapterRouter

    monkeypatch.setenv("MODELMELD_UPSTREAM_PROVIDER", "openai")
    monkeypatch.setenv("MODELMELD_OPENAI_API_KEY", "test-key")
    app = build_app(GatewaySettings())
    assert isinstance(app.state.router, SingleAdapterRouter)
    assert app.state.router.adapter.name == "openai"


def test_adapter_error_surfaces_as_502() -> None:
    class FailingAdapter(ProviderAdapter):
        name = "failing"

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            from modelmeld.adapters.base import AdapterError

            raise AdapterError("upstream timeout")

        async def stream_chat(
            self, request: ChatCompletionRequest
        ) -> AsyncIterator[ChatCompletionChunk]:
            if False:  # pragma: no cover
                yield

        async def health(self) -> bool:
            return False

    with TestClient(build_app(adapter=FailingAdapter())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 502
    assert "upstream timeout" in response.json()["detail"]

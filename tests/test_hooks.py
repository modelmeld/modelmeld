"""HookRegistry firing tests + chat-route integration."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from modelmeld.adapters.base import AdapterError, ProviderAdapter
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
from modelmeld.hooks import HookRegistry, RequestCompletedEvent


class CapturingAdapter(ProviderAdapter):
    name = "capture"
    is_egress = False

    def __init__(self, fail_chat: bool = False) -> None:
        self.fail_chat = fail_chat

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        if self.fail_chat:
            raise AdapterError("simulated upstream error")
        return ChatCompletion(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(
            id="chatcmpl-stream",
            created=1,
            model=request.model,
            choices=[
                ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"))
            ],
        )
        yield ChatCompletionChunk(
            id="chatcmpl-stream",
            created=1,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        )

    async def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# HookRegistry plumbing
# ---------------------------------------------------------------------------

def test_default_app_state_has_empty_registry() -> None:
    app = build_app()
    assert isinstance(app.state.hooks, HookRegistry)
    assert app.state.hooks.subscriber_count == 0


async def test_registry_invokes_handlers_in_order() -> None:
    registry = HookRegistry()
    log: list[str] = []

    async def h1(event: RequestCompletedEvent) -> None:
        log.append("h1")

    async def h2(event: RequestCompletedEvent) -> None:
        log.append("h2")

    registry.register_on_request_complete(h1)
    registry.register_on_request_complete(h2)
    assert registry.subscriber_count == 2

    event = _dummy_event()
    await registry.fire_on_request_complete(event)
    assert log == ["h1", "h2"]


async def test_handler_failure_logged_not_raised() -> None:
    registry = HookRegistry()
    succeeded: list[str] = []

    async def boom(event: RequestCompletedEvent) -> None:
        raise RuntimeError("handler exploded")

    async def ok(event: RequestCompletedEvent) -> None:
        succeeded.append("ok")

    registry.register_on_request_complete(boom)
    registry.register_on_request_complete(ok)

    # Must not propagate; subsequent handlers still fire.
    await registry.fire_on_request_complete(_dummy_event())
    assert succeeded == ["ok"]


# ---------------------------------------------------------------------------
# End-to-end: chat route fires the hook
# ---------------------------------------------------------------------------

def _payload() -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_non_stream_success_fires_hook_with_token_counts() -> None:
    captured: list[RequestCompletedEvent] = []

    async def capture(event: RequestCompletedEvent) -> None:
        captured.append(event)

    hooks = HookRegistry()
    hooks.register_on_request_complete(capture)

    with TestClient(build_app(adapter=CapturingAdapter(), hooks=hooks)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    assert len(captured) == 1
    event = captured[0]
    assert event.requested_model == "gpt-4o-mini"
    assert event.routed_to == "capture"
    assert event.input_tokens == 12
    assert event.output_tokens == 8
    assert event.total_tokens == 20
    assert event.error is None
    assert event.error_type is None
    assert event.latency_ms > 0
    assert event.prompt_hash and len(event.prompt_hash) == 64
    assert event.request_id.startswith("req_")


def test_non_stream_failure_fires_hook_with_error() -> None:
    captured: list[RequestCompletedEvent] = []

    async def capture(event: RequestCompletedEvent) -> None:
        captured.append(event)

    hooks = HookRegistry()
    hooks.register_on_request_complete(capture)

    with TestClient(
        build_app(adapter=CapturingAdapter(fail_chat=True), hooks=hooks)
    ) as client:
        resp = client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 502
    assert len(captured) == 1
    event = captured[0]
    assert event.error is not None
    assert event.error_type == "adapter_error"
    assert event.input_tokens == 0
    assert event.output_tokens == 0


def test_stream_success_fires_hook_after_done() -> None:
    captured: list[RequestCompletedEvent] = []

    async def capture(event: RequestCompletedEvent) -> None:
        captured.append(event)

    hooks = HookRegistry()
    hooks.register_on_request_complete(capture)

    with TestClient(build_app(adapter=CapturingAdapter(), hooks=hooks)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={**_payload(), "stream": True},
        ) as resp:
            body = b"".join(resp.iter_bytes()).decode("utf-8")

    assert "data: [DONE]" in body
    assert len(captured) == 1
    event = captured[0]
    # Stream used last-usage chunk to populate token counts.
    assert event.input_tokens == 12
    assert event.output_tokens == 8
    assert event.error is None


def test_zero_subscribers_means_no_overhead() -> None:
    """When no enterprise plugin is installed, the route shouldn't try to fire."""
    # Smoke: default app has 0 subscribers; request still succeeds.
    with TestClient(build_app(adapter=CapturingAdapter())) as client:
        resp = client.post("/v1/chat/completions", json=_payload())
    assert resp.status_code == 200


def test_prompt_hash_deterministic_across_requests() -> None:
    captured: list[RequestCompletedEvent] = []

    async def capture(event: RequestCompletedEvent) -> None:
        captured.append(event)

    hooks = HookRegistry()
    hooks.register_on_request_complete(capture)

    with TestClient(build_app(adapter=CapturingAdapter(), hooks=hooks)) as client:
        client.post("/v1/chat/completions", json=_payload())
        client.post("/v1/chat/completions", json=_payload())

    assert len(captured) == 2
    assert captured[0].prompt_hash == captured[1].prompt_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_event() -> RequestCompletedEvent:
    from datetime import datetime, timezone

    return RequestCompletedEvent(
        request_id="req_test",
        timestamp=datetime.now(timezone.utc),
        requested_model="m",
        devtool="unknown",
        devtool_confidence=0.0,
        prompt_hash="0" * 64,
        routed_to="capture",
        tier="local",
        failover_from=None,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        latency_ms=1.0,
    )

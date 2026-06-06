# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""End-to-end /v1/responses (Phase 1, non-streaming)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    FunctionCall,
    ResponseMessage,
    SystemMessage,
    ToolCall,
    Usage,
    UserMessage,
)
from modelmeld.api.server import build_app


class _StubAdapter(ProviderAdapter):
    """Returns a fixed ChatCompletion and records the request it received."""

    name = "stub-responses"
    is_egress = False

    def __init__(self, completion: ChatCompletion) -> None:
        self._completion = completion
        self.received: ChatCompletionRequest | None = None

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.received = request
        return self._completion

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


def _text_completion(text: str, *, model: str = "qwen3-coder-next") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x", model=model,
        choices=[Choice(index=0, message=ResponseMessage(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
    )


async def _post(app, body: dict) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        return await client.post("/v1/responses", json=body)


async def test_text_request_round_trips_to_responses_shape() -> None:
    adapter = _StubAdapter(_text_completion("the answer", model="qwen3-coder-next"))
    app = build_app(adapter=adapter)

    resp = await _post(app, {"model": "anthropic/modelmeld-auto", "input": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "qwen3-coder-next"   # the routed/served model
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output"][0]["content"][0]["text"] == "the answer"
    assert body["usage"]["input_tokens"] == 9
    # Audit-header parity with the other surfaces.
    assert resp.headers["x-modelmeld-routed-to"] == "stub-responses"
    # The adapter saw the translated Chat-shape request.
    assert isinstance(adapter.received.messages[0], UserMessage)
    assert adapter.received.messages[0].content == "hello"


async def test_instructions_become_a_system_message() -> None:
    adapter = _StubAdapter(_text_completion("ok"))
    app = build_app(adapter=adapter)

    resp = await _post(app, {
        "model": "m", "instructions": "Be terse.", "input": "hi",
    })

    assert resp.status_code == 200
    msgs = adapter.received.messages
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "Be terse."
    assert isinstance(msgs[1], UserMessage)


async def test_tool_calls_surface_as_function_call_items() -> None:
    completion = ChatCompletion(
        id="chatcmpl-y", model="qwen3-coder-next",
        choices=[Choice(
            index=0,
            message=ResponseMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(
                    id="call_42", type="function",
                    function=FunctionCall(name="search", arguments='{"q":"x"}'),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    app = build_app(adapter=_StubAdapter(completion))

    resp = await _post(app, {"model": "m", "input": "find x"})

    assert resp.status_code == 200
    fcs = [o for o in resp.json()["output"] if o["type"] == "function_call"]
    assert len(fcs) == 1
    assert fcs[0]["name"] == "search"
    assert fcs[0]["arguments"] == '{"q":"x"}'
    assert fcs[0]["call_id"] == "call_42"


async def test_streaming_not_yet_supported_returns_501() -> None:
    app = build_app(adapter=_StubAdapter(_text_completion("x")))
    resp = await _post(app, {"model": "m", "input": "hi", "stream": True})
    assert resp.status_code == 501

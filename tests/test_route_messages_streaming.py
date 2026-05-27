# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Integration tests for the /v1/messages streaming path.

Validates end-to-end SSE flow: client POSTs stream=true, mock adapter
yields OpenAI-shape ChatCompletionChunks, route translates to Anthropic
SSE events, client receives the raw event stream and parses it back.

Verifies:
  - Correct SSE wire format (`event: <type>\\ndata: <json>\\n\\n`)
  - Event sequence: message_start → block(s) → message_delta → message_stop
  - Anthropic streaming uses NO `[DONE]` terminator (vs. OpenAI)
  - Memory write-back happens after stream completion
  - Routing headers present on streaming response
  - Tool-use streaming round-trips correctly (text + tool_use blocks)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChoiceDelta,
    ChunkChoice,
    FunctionCallDelta,
    ToolCallDelta,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    HEADER_SESSION_ID,
    InMemoryMemoryStore,
    Role,
)

# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    """Parse Anthropic SSE wire format → list of event dicts.

    Each event is `event: <type>\\ndata: <json>\\n\\n`. Blank lines
    between events. Returns events in order, each as a parsed-JSON dict.
    Adds the `_event_name` synthetic field for easy assertion of the
    `event:` header line.
    """
    events = []
    blocks = raw.split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.split("\n")
        event_name = None
        data_str = None
        for line in lines:
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_str = line[len("data: "):]
        if data_str is None:
            continue
        payload = json.loads(data_str)
        payload["_event_name"] = event_name
        events.append(payload)
    return events


# ---------------------------------------------------------------------------
# Mock streaming adapters
# ---------------------------------------------------------------------------

class _TextStreamAdapter(ProviderAdapter):
    """Streams a fixed text response split across chunks."""
    name = "mock-stream-text"
    is_egress = False

    def __init__(self, text: str = "Hello there!") -> None:
        self.text = text

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        raise NotImplementedError("text-stream adapter is streaming-only")

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        cid = "chatcmpl-stream-fixture"
        # Split into 3 pieces
        pieces = [self.text[i:i+5] for i in range(0, len(self.text), 5)] or [self.text]
        first = True
        for piece in pieces:
            delta = ChoiceDelta(role="assistant" if first else None, content=piece)
            first = False
            yield ChatCompletionChunk(
                id=cid, object="chat.completion.chunk", created=1,
                model=request.model,
                choices=[ChunkChoice(index=0, delta=delta)],
            )
        # Final usage chunk
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1,
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=15,
                completion_tokens=len(self.text),
                total_tokens=15 + len(self.text),
            ),
        )

    async def health(self) -> bool:
        return True


class _ToolStreamAdapter(ProviderAdapter):
    """Streams: text intro + one tool_use with arguments split across chunks."""
    name = "mock-stream-tools"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        raise NotImplementedError("tool-stream adapter is streaming-only")

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        cid = "chatcmpl-stream-tools"
        # 1. text intro
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="I'll search."))],
        )
        # 2. tool call begin (name + empty args)
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
                ToolCallDelta(index=0, id="tu_stream_1", type="function",
                              function=FunctionCallDelta(name="search", arguments="")),
            ]))],
        )
        # 3. arg piece 1
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
                ToolCallDelta(index=0, function=FunctionCallDelta(arguments='{"q":')),
            ]))],
        )
        # 4. arg piece 2
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
                ToolCallDelta(index=0, function=FunctionCallDelta(arguments='"hello"}')),
            ]))],
        )
        # 5. finish + usage
        yield ChatCompletionChunk(
            id=cid, object="chat.completion.chunk", created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="tool_calls")],
            usage=Usage(prompt_tokens=30, completion_tokens=12, total_tokens=42),
        )

    async def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _stream_post(app, body: dict, headers: dict | None = None) -> tuple[int, dict, str]:
    """POST stream=true, collect the full body. Returns (status, headers, raw_text)."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client, client.stream("POST", "/v1/messages",
                             json=body, headers=headers or {}) as resp:
        text = ""
        async for piece in resp.aiter_text():
            text += piece
        return resp.status_code, dict(resp.headers), text


# ---------------------------------------------------------------------------
# Pure text streaming
# ---------------------------------------------------------------------------

async def test_text_only_stream_emits_correct_event_sequence() -> None:
    app = build_app(adapter=_TextStreamAdapter(text="Hi there"))
    status, headers, raw = await _stream_post(app, {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "ping"}],
    })

    assert status == 200
    # Anthropic streaming uses text/event-stream
    assert "text/event-stream" in headers.get("content-type", "")
    # No [DONE] terminator (that's OpenAI's convention)
    assert "[DONE]" not in raw

    events = _parse_sse(raw)
    event_types = [e["type"] for e in events]
    # First and last events
    assert event_types[0] == "message_start"
    assert event_types[-1] == "message_stop"
    # Must contain at least one text block start/stop and one delta
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types


async def test_text_stream_reassembles_to_full_response() -> None:
    """All text_delta payloads concatenated should equal the original text."""
    app = build_app(adapter=_TextStreamAdapter(text="The quick brown fox"))
    status, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    assert status == 200

    events = _parse_sse(raw)
    text_pieces = [
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    ]
    assert "".join(text_pieces) == "The quick brown fox"


async def test_message_start_carries_model_and_input_tokens() -> None:
    app = build_app(adapter=_TextStreamAdapter(text="ok"))
    status, _, raw = await _stream_post(app, {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "some short text"}],
    })
    assert status == 200
    events = _parse_sse(raw)
    msg_start = events[0]
    assert msg_start["type"] == "message_start"
    assert msg_start["_event_name"] == "message_start"
    assert msg_start["message"]["model"] == "claude-haiku-4-5-20251001"
    # input_tokens is the pre-counted estimate; must be >0 since the
    # user message wasn't empty
    assert msg_start["message"]["usage"]["input_tokens"] > 0
    assert msg_start["message"]["usage"]["output_tokens"] == 0


async def test_message_delta_carries_stop_reason_and_output_tokens() -> None:
    app = build_app(adapter=_TextStreamAdapter(text="hi"))
    status, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    assert status == 200
    events = _parse_sse(raw)
    msg_delta = next(e for e in events if e["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert msg_delta["usage"]["output_tokens"] > 0


# ---------------------------------------------------------------------------
# Tool-use streaming
# ---------------------------------------------------------------------------

async def test_tool_use_stream_emits_text_then_tool_use_blocks() -> None:
    app = build_app(adapter=_ToolStreamAdapter())
    status, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "search for hi"}],
        "tools": [{
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object",
                              "properties": {"q": {"type": "string"}}},
        }],
    })
    assert status == 200

    events = _parse_sse(raw)
    block_starts = [e for e in events if e["type"] == "content_block_start"]
    # Two blocks: one text, one tool_use
    assert len(block_starts) == 2
    assert block_starts[0]["content_block"]["type"] == "text"
    assert block_starts[1]["content_block"]["type"] == "tool_use"
    assert block_starts[1]["content_block"]["name"] == "search"
    assert block_starts[1]["content_block"]["id"] == "tu_stream_1"


async def test_tool_use_stream_reassembles_arguments_as_valid_json() -> None:
    app = build_app(adapter=_ToolStreamAdapter())
    status, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "search", "input_schema": {"type": "object"}}],
    })
    assert status == 200

    events = _parse_sse(raw)
    # Find the tool_use block index
    tool_block = next(
        e for e in events
        if e["type"] == "content_block_start"
        and e["content_block"]["type"] == "tool_use"
    )
    tool_idx = tool_block["index"]
    # Concatenate all input_json_delta pieces on that block
    json_pieces = [
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["index"] == tool_idx
        and e["delta"]["type"] == "input_json_delta"
    ]
    reassembled = "".join(json_pieces)
    assert json.loads(reassembled) == {"q": "hello"}


async def test_tool_call_stream_has_tool_use_stop_reason() -> None:
    app = build_app(adapter=_ToolStreamAdapter())
    status, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "search", "input_schema": {"type": "object"}}],
    })
    assert status == 200
    events = _parse_sse(raw)
    msg_delta = next(e for e in events if e["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Headers + observability
# ---------------------------------------------------------------------------

async def test_streaming_response_has_routing_headers() -> None:
    """x-modelmeld-* headers must appear on the streaming response too."""
    app = build_app(adapter=_TextStreamAdapter(text="ok"))
    status, headers, _ = await _stream_post(app, {
        "model": "m", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    assert status == 200
    assert "x-modelmeld-routed-to" in headers
    assert "x-modelmeld-tier" in headers


# ---------------------------------------------------------------------------
# Memory write-back after stream completion
# ---------------------------------------------------------------------------

async def test_session_header_writes_memory_after_stream_completes() -> None:
    """Session-tagged streaming request must still record user + assistant turns."""
    store = InMemoryMemoryStore()
    app = build_app(adapter=_TextStreamAdapter(text="echo: hello"), memory_store=store)
    status, _, raw = await _stream_post(
        app,
        {
            "model": "m", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={HEADER_SESSION_ID: "stream-sess-1"},
    )
    assert status == 200
    # Verify stream was actually consumed (sanity)
    assert "message_stop" in raw

    turns = await store.list_turns("stream-sess-1", ANONYMOUS_TENANT_ID)
    # 1 user + 1 assistant
    assert len(turns) == 2
    assert turns[0].role == Role.USER
    assert turns[0].content == "hello"
    assert turns[1].role == Role.ASSISTANT
    assert "echo" in turns[1].content


# ---------------------------------------------------------------------------
# SSE wire-format invariants
# ---------------------------------------------------------------------------

async def test_sse_format_uses_event_and_data_lines() -> None:
    """Anthropic SSE format: each event has an `event: <type>` header line
    AND a `data: <json>` body line, separated by `\\n\\n`. This differs
    from OpenAI's stream which uses only `data:` lines."""
    app = build_app(adapter=_TextStreamAdapter(text="x"))
    _, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 8, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    # At least one of each line type
    assert "event: message_start" in raw
    assert "event: content_block_delta" in raw
    assert "event: message_stop" in raw
    assert "data: {" in raw
    # No OpenAI-style [DONE] sentinel
    assert "[DONE]" not in raw


async def test_each_event_has_matching_event_and_data_lines() -> None:
    """Each Anthropic event is exactly 2 lines (event: + data:) followed
    by blank. Parser should produce events where _event_name agrees with
    the inner `type` field."""
    app = build_app(adapter=_TextStreamAdapter(text="ok"))
    _, _, raw = await _stream_post(app, {
        "model": "m", "max_tokens": 8, "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    })
    events = _parse_sse(raw)
    assert events  # non-empty
    for e in events:
        assert e["_event_name"] == e["type"], (
            f"event header / data type mismatch: {e['_event_name']!r} vs {e['type']!r}"
        )

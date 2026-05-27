# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for `OpenAIToAnthropicStreamTranslator` + `format_anthropic_sse`.

Validates the streaming-direction translation: internal OpenAI
ChatCompletionChunk stream → Anthropic SSE event sequence. The state
machine has to:

  - Emit `message_start` exactly once
  - Track which content block is currently open (text or tool_use)
  - Close + reopen blocks at the right transitions (text → tool_use,
    new parallel tool_use, etc.)
  - Emit `content_block_stop` for every opened block
  - Emit `message_delta` with stop_reason + final output_tokens
  - Emit `message_stop` to terminate
  - Guarantee Anthropic's "≥1 content block per response" invariant
    even for empty completions
"""

from __future__ import annotations

import json

import pytest

from modelmeld.api.schemas import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    FunctionCallDelta,
    ToolCallDelta,
    Usage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicMessageDeltaEvent,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
)
from modelmeld.translation import (
    OpenAIToAnthropicStreamTranslator,
    format_anthropic_sse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    *,
    role: str | None = None,
    content: str | None = None,
    tool_calls: list[ToolCallDelta] | None = None,
    finish_reason: str | None = None,
    usage: Usage | None = None,
    id: str = "chatcmpl-stream-test",
    model: str = "internal-m",
) -> ChatCompletionChunk:
    delta = ChoiceDelta(role=role, content=content, tool_calls=tool_calls)  # type: ignore[arg-type]
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=1,
        model=model,
        choices=[ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,  # type: ignore[arg-type]
        )],
        usage=usage,
    )


def _run(t: OpenAIToAnthropicStreamTranslator, *chunks: ChatCompletionChunk) -> list:
    events = []
    for c in chunks:
        events.extend(t.translate_chunk(c))
    events.extend(t.finalize())
    return events


def _types(events) -> list[str]:
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# Text-only stream
# ---------------------------------------------------------------------------

def test_text_only_stream_emits_correct_event_sequence() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="claude-haiku-4-5-20251001", input_tokens=5)
    events = _run(
        t,
        _chunk(role="assistant", content="Hello"),
        _chunk(content=" world"),
        _chunk(finish_reason="stop",
               usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)),
    )
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_message_start_carries_input_tokens_and_request_model() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="claude-haiku-4-5-20251001", input_tokens=42)
    events = _run(t, _chunk(role="assistant", content="x"), _chunk(finish_reason="stop"))
    start = events[0]
    assert isinstance(start, AnthropicMessageStartEvent)
    assert start.message.model == "claude-haiku-4-5-20251001"
    assert start.message.usage.input_tokens == 42
    assert start.message.usage.output_tokens == 0
    assert start.message.role == "assistant"
    assert start.message.content == []


def test_text_block_open_emits_empty_text() -> None:
    """content_block_start for a text block carries text='' (Anthropic spec)."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="hi"), _chunk(finish_reason="stop"))
    block_start = events[1]
    assert isinstance(block_start, AnthropicContentBlockStartEvent)
    assert block_start.index == 0
    assert block_start.content_block.type == "text"
    assert block_start.content_block.text == ""  # type: ignore[union-attr]


def test_multiple_text_deltas_accumulate_in_same_block() -> None:
    """Many text deltas → many content_block_delta events, all index=0."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="The "),
        _chunk(content="quick "),
        _chunk(content="brown "),
        _chunk(content="fox."),
        _chunk(finish_reason="stop"),
    )
    delta_events = [e for e in events if isinstance(e, AnthropicContentBlockDeltaEvent)]
    assert len(delta_events) == 4
    assert all(d.index == 0 for d in delta_events)
    assert all(d.delta.type == "text_delta" for d in delta_events)
    reassembled = "".join(d.delta.text for d in delta_events)  # type: ignore[union-attr]
    assert reassembled == "The quick brown fox."


def test_finish_reason_maps_to_stop_reason_on_message_delta() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="x"), _chunk(finish_reason="length"))
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.delta.stop_reason == "max_tokens"


def test_output_tokens_from_final_usage_chunk_lands_in_message_delta() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="x"),
        _chunk(finish_reason="stop",
               usage=Usage(prompt_tokens=10, completion_tokens=77, total_tokens=87)),
    )
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.usage.output_tokens == 77


# ---------------------------------------------------------------------------
# ID handling (parallels to_anthropic_response behavior)
# ---------------------------------------------------------------------------

def test_anthropic_shaped_id_in_first_chunk_passes_through() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="x", id="msg_real_anthropic_id_xyz"),
        _chunk(finish_reason="stop"),
    )
    start = events[0]
    assert isinstance(start, AnthropicMessageStartEvent)
    assert start.message.id == "msg_real_anthropic_id_xyz"


def test_openai_shaped_id_in_first_chunk_gets_rewritten() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="x", id="chatcmpl-from-vllm"),
        _chunk(finish_reason="stop"),
    )
    start = events[0]
    assert isinstance(start, AnthropicMessageStartEvent)
    assert start.message.id.startswith("msg_")
    assert start.message.id != "chatcmpl-from-vllm"


# ---------------------------------------------------------------------------
# Tool-use streams
# ---------------------------------------------------------------------------

def test_tool_only_stream_no_text_block() -> None:
    """If the LLM produces ONLY a tool call (no text), we should NOT emit
    an empty text block ahead of the tool_use block."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant"),  # role only, no content
        _chunk(tool_calls=[ToolCallDelta(
            index=0, id="tu_1", type="function",
            function=FunctionCallDelta(name="ping", arguments="{}"),
        )]),
        _chunk(finish_reason="tool_calls"),
    )
    types = _types(events)
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",  # the partial_json="{}"
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # The single content_block_start is tool_use, not text
    start = events[1]
    assert isinstance(start, AnthropicContentBlockStartEvent)
    assert start.content_block.type == "tool_use"
    assert start.content_block.name == "ping"  # type: ignore[union-attr]
    assert start.content_block.id == "tu_1"  # type: ignore[union-attr]


def test_text_then_tool_use_closes_text_block_first() -> None:
    """Critical transition: a text block must be closed before a tool_use
    block is opened — Anthropic spec requires non-overlapping blocks."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="Let me check."),
        _chunk(tool_calls=[ToolCallDelta(
            index=0, id="tu_1", type="function",
            function=FunctionCallDelta(name="read_file", arguments='{"path":"/etc/hosts"}'),
        )]),
        _chunk(finish_reason="tool_calls"),
    )
    assert _types(events) == [
        "message_start",
        "content_block_start",   # text block opens
        "content_block_delta",   # "Let me check."
        "content_block_stop",    # text block closes (transition!)
        "content_block_start",   # tool_use block opens
        "content_block_delta",   # full JSON args
        "content_block_stop",    # tool_use block closes (finalize)
        "message_delta",
        "message_stop",
    ]
    # Verify the block indices: text=0, tool_use=1
    block_starts = [e for e in events if isinstance(e, AnthropicContentBlockStartEvent)]
    assert block_starts[0].index == 0 and block_starts[0].content_block.type == "text"
    assert block_starts[1].index == 1 and block_starts[1].content_block.type == "tool_use"


def test_streaming_tool_call_arguments_emit_input_json_deltas() -> None:
    """OpenAI streams tool_call arguments as partial JSON strings;
    we emit one input_json_delta per chunk that carries arguments."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", tool_calls=[ToolCallDelta(
            index=0, id="tu_1", type="function",
            function=FunctionCallDelta(name="search", arguments=""),
        )]),
        _chunk(tool_calls=[ToolCallDelta(index=0, function=FunctionCallDelta(arguments='{"q"'))]),
        _chunk(tool_calls=[ToolCallDelta(index=0, function=FunctionCallDelta(arguments=':"test"}'))]),
        _chunk(finish_reason="tool_calls"),
    )
    deltas = [e for e in events if isinstance(e, AnthropicContentBlockDeltaEvent)]
    # First chunk's empty arguments → no delta emitted; the other two each emit one
    assert len(deltas) == 2
    assert all(d.delta.type == "input_json_delta" for d in deltas)
    pieces = [d.delta.partial_json for d in deltas]  # type: ignore[union-attr]
    assert "".join(pieces) == '{"q":"test"}'


def test_parallel_tool_calls_emit_separate_blocks_in_order() -> None:
    """Two distinct OpenAI tool_call indices → two separate Anthropic
    tool_use blocks, opened/closed in the order they first appeared."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant"),
        # First tool call (index=0)
        _chunk(tool_calls=[ToolCallDelta(
            index=0, id="tu_a", type="function",
            function=FunctionCallDelta(name="read", arguments='{"p":"a"}'),
        )]),
        # Second tool call (index=1) — closes the previous block, opens a new one
        _chunk(tool_calls=[ToolCallDelta(
            index=1, id="tu_b", type="function",
            function=FunctionCallDelta(name="read", arguments='{"p":"b"}'),
        )]),
        _chunk(finish_reason="tool_calls"),
    )
    block_starts = [e for e in events if isinstance(e, AnthropicContentBlockStartEvent)]
    block_stops = [e for e in events if isinstance(e, AnthropicContentBlockStopEvent)]
    assert len(block_starts) == 2
    assert len(block_stops) == 2
    # Block 0 = first tool call
    assert block_starts[0].index == 0
    assert block_starts[0].content_block.id == "tu_a"  # type: ignore[union-attr]
    # Block 1 = second tool call
    assert block_starts[1].index == 1
    assert block_starts[1].content_block.id == "tu_b"  # type: ignore[union-attr]
    # Stops also in order
    assert block_stops[0].index == 0
    assert block_stops[1].index == 1


def test_tool_call_without_id_synthesizes_one() -> None:
    """Some local adapters may omit the tool_call id. We synthesize toolu_*
    so Claude Code doesn't choke on an empty id."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", tool_calls=[ToolCallDelta(
            index=0, id=None, type="function",  # type: ignore[arg-type]
            function=FunctionCallDelta(name="fn", arguments="{}"),
        )]),
        _chunk(finish_reason="tool_calls"),
    )
    block_start = next(
        e for e in events if isinstance(e, AnthropicContentBlockStartEvent)
    )
    assert block_start.content_block.id  # non-empty  # type: ignore[union-attr]
    assert block_start.content_block.id.startswith("toolu_")  # type: ignore[union-attr]


def test_tool_calls_finish_reason_maps_to_tool_use_stop_reason() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", tool_calls=[ToolCallDelta(
            index=0, id="tu_1", type="function",
            function=FunctionCallDelta(name="fn", arguments="{}"),
        )]),
        _chunk(finish_reason="tool_calls"),
    )
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.delta.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_stream_yields_well_formed_event_sequence() -> None:
    """No chunks at all — finalize() must still produce a valid sequence:
    synthetic message_start + empty text block + message_delta + message_stop."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = t.finalize()
    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # The synthesized block is an empty text block
    block_start = events[1]
    assert isinstance(block_start, AnthropicContentBlockStartEvent)
    assert block_start.content_block.type == "text"


def test_role_only_chunk_followed_by_finish_still_emits_text_block() -> None:
    """If the only chunks were role announcement + finish_reason (no actual
    content), Anthropic still needs ≥1 block. Emit an empty text block."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant"),
        _chunk(finish_reason="stop"),
    )
    assert _types(events) == [
        "message_start",
        "content_block_start",   # synthesized empty text block
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_no_finish_reason_defaults_to_end_turn() -> None:
    """Some adapter paths may close the stream without ever sending
    finish_reason. Default to end_turn rather than null."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="hi"))  # no finish_reason chunk
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.delta.stop_reason == "end_turn"


def test_usage_only_final_chunk_with_no_choices() -> None:
    """OpenAI sometimes emits a final chunk with usage but choices=[] when
    stream_options.include_usage is set. We must extract the usage cleanly."""
    chunk_no_choices = ChatCompletionChunk(
        id="chatcmpl-x",
        object="chat.completion.chunk",
        created=1,
        model="m",
        choices=[],
        usage=Usage(prompt_tokens=10, completion_tokens=42, total_tokens=52),
    )
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(
        t,
        _chunk(role="assistant", content="x"),
        _chunk(finish_reason="stop"),
        chunk_no_choices,
    )
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.usage.output_tokens == 42


# ---------------------------------------------------------------------------
# Realistic Claude-Code-shaped stream (text + multi-tool, end-to-end)
# ---------------------------------------------------------------------------

def test_realistic_text_plus_two_tools_stream() -> None:
    """Mimics what Claude Code receives when the model says 'let me look',
    then calls read_file, then calls grep — full event sequence verified
    block-by-block."""
    t = OpenAIToAnthropicStreamTranslator(
        request_model="claude-haiku-4-5-20251001", input_tokens=87,
    )
    events = _run(
        t,
        _chunk(role="assistant", content="Let me check the file then search.",
               id="chatcmpl-realistic"),
        # First tool: read_file
        _chunk(tool_calls=[ToolCallDelta(
            index=0, id="tu_read", type="function",
            function=FunctionCallDelta(name="read_file", arguments=""),
        )]),
        _chunk(tool_calls=[ToolCallDelta(
            index=0, function=FunctionCallDelta(arguments='{"path":'),
        )]),
        _chunk(tool_calls=[ToolCallDelta(
            index=0, function=FunctionCallDelta(arguments='"/etc/hosts"}'),
        )]),
        # Second tool: grep (parallel)
        _chunk(tool_calls=[ToolCallDelta(
            index=1, id="tu_grep", type="function",
            function=FunctionCallDelta(name="grep", arguments='{"pattern":"127"}'),
        )]),
        _chunk(finish_reason="tool_calls",
               usage=Usage(prompt_tokens=87, completion_tokens=33, total_tokens=120)),
    )

    # Verify the structural sequence
    assert _types(events) == [
        "message_start",
        "content_block_start",    # text block 0
        "content_block_delta",    # "Let me check the file then search."
        "content_block_stop",     # text block 0 closes
        "content_block_start",    # tool_use block 1: read_file
        "content_block_delta",    # arguments piece 1: '{"path":'
        "content_block_delta",    # arguments piece 2: '"/etc/hosts"}'
        "content_block_stop",     # tool_use block 1 closes
        "content_block_start",    # tool_use block 2: grep
        "content_block_delta",    # arguments full
        "content_block_stop",     # tool_use block 2 closes (finalize)
        "message_delta",
        "message_stop",
    ]

    # Verify the final usage + stop_reason
    msg_delta = next(e for e in events if isinstance(e, AnthropicMessageDeltaEvent))
    assert msg_delta.delta.stop_reason == "tool_use"
    assert msg_delta.usage.output_tokens == 33

    # Verify the tool ids/names landed on the right blocks
    block_starts = [e for e in events if isinstance(e, AnthropicContentBlockStartEvent)]
    assert block_starts[0].content_block.type == "text"
    assert block_starts[1].content_block.type == "tool_use"
    assert block_starts[1].content_block.id == "tu_read"   # type: ignore[union-attr]
    assert block_starts[1].content_block.name == "read_file"  # type: ignore[union-attr]
    assert block_starts[2].content_block.type == "tool_use"
    assert block_starts[2].content_block.id == "tu_grep"   # type: ignore[union-attr]
    assert block_starts[2].content_block.name == "grep"  # type: ignore[union-attr]

    # Verify reassembled tool arguments parse as valid JSON
    deltas_block_1 = [
        e for e in events
        if isinstance(e, AnthropicContentBlockDeltaEvent) and e.index == 1
    ]
    json_pieces = [d.delta.partial_json for d in deltas_block_1]  # type: ignore[union-attr]
    reassembled = "".join(json_pieces)
    assert json.loads(reassembled) == {"path": "/etc/hosts"}


# ---------------------------------------------------------------------------
# SSE wire-format
# ---------------------------------------------------------------------------

def test_sse_format_includes_event_line_and_data_line_separated_by_blank() -> None:
    """Anthropic SSE: `event: <type>\\n` then `data: <json>\\n` then blank `\\n`."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="hi"), _chunk(finish_reason="stop"))
    msg_start_sse = format_anthropic_sse(events[0])

    lines = msg_start_sse.split("\n")
    assert lines[0] == "event: message_start"
    assert lines[1].startswith("data: ")
    assert lines[2] == ""  # blank line terminator
    # Verify the JSON in data: line is parseable
    json_str = lines[1][len("data: "):]
    payload = json.loads(json_str)
    assert payload["type"] == "message_start"
    assert payload["message"]["model"] == "m"


def test_sse_format_omits_null_fields() -> None:
    """Anthropic spec omits null/absent optional fields rather than emitting
    `field: null`. exclude_none in model_dump_json handles this."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="hi"), _chunk(finish_reason="stop"))
    msg_start_sse = format_anthropic_sse(events[0])
    # stop_reason is None at message_start; should not appear in the JSON
    assert '"stop_reason":null' not in msg_start_sse
    assert '"stop_reason"' not in msg_start_sse


def test_sse_format_for_text_delta() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="hello"), _chunk(finish_reason="stop"))
    # Find the text delta event
    text_delta = next(
        e for e in events
        if isinstance(e, AnthropicContentBlockDeltaEvent)
    )
    sse = format_anthropic_sse(text_delta)
    lines = sse.split("\n")
    assert lines[0] == "event: content_block_delta"
    payload = json.loads(lines[1][len("data: "):])
    assert payload == {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    }


def test_sse_format_for_message_stop() -> None:
    """message_stop has no body fields beyond `type`. SSE data is just the type."""
    t = OpenAIToAnthropicStreamTranslator(request_model="m")
    events = _run(t, _chunk(role="assistant", content="x"), _chunk(finish_reason="stop"))
    msg_stop = next(e for e in events if isinstance(e, AnthropicMessageStopEvent))
    sse = format_anthropic_sse(msg_stop)
    lines = sse.split("\n")
    assert lines[0] == "event: message_stop"
    payload = json.loads(lines[1][len("data: "):])
    assert payload == {"type": "message_stop"}

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""ResponsesStreamTranslator: ChatCompletion chunks → Responses SSE events.

The load-bearing test validates every emitted event against the OpenAI SDK's
own Pydantic event models — the same library Codex parses with — so a passing
suite means the wire shapes match what Codex expects.
"""

from __future__ import annotations

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from modelmeld.api.schemas import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    FunctionCallDelta,
    ToolCallDelta,
    Usage,
)
from modelmeld.translation.responses_stream import (
    ResponsesStreamTranslator,
    format_responses_sse,
)

_SDK_EVENT = {
    "response.created": ResponseCreatedEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.output_text.done": ResponseTextDoneEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
    "response.completed": ResponseCompletedEvent,
}


def _tool_open(idx: int, call_id: str, name: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c", created=0, model="qwen",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
            ToolCallDelta(index=idx, id=call_id, type="function",
                          function=FunctionCallDelta(name=name, arguments="")),
        ]))],
    )


def _tool_args(idx: int, frag: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="c", created=0, model="qwen",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
            ToolCallDelta(index=idx, function=FunctionCallDelta(arguments=frag)),
        ]))],
    )


def _chunk(content: str | None = None, *, usage: Usage | None = None) -> ChatCompletionChunk:
    delta = ChoiceDelta(role="assistant", content=content) if content is not None else ChoiceDelta()
    return ChatCompletionChunk(
        id="c", created=0, model="qwen",
        choices=[ChunkChoice(index=0, delta=delta)], usage=usage,
    )


def _run() -> list[dict]:
    t = ResponsesStreamTranslator(model="qwen3-coder-next")
    events: list[dict] = []
    events += t.translate_chunk(_chunk("Hello "))
    events += t.translate_chunk(_chunk("world"))
    events += t.translate_chunk(_chunk(usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12)))
    events += t.finalize()
    return events


def test_event_sequence() -> None:
    types = [e["type"] for e in _run()]
    assert types == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]


def test_every_event_validates_against_openai_sdk() -> None:
    # If a dict doesn't match the SDK's shape, model_validate raises — meaning
    # Codex (which uses the same SDK) wouldn't parse it either.
    for event in _run():
        cls = _SDK_EVENT[event["type"]]
        cls.model_validate(event)


def test_text_accumulates_and_usage_propagates() -> None:
    events = _run()
    done = next(e for e in events if e["type"] == "response.output_text.done")
    assert done["text"] == "Hello world"
    completed = next(e for e in events if e["type"] == "response.completed")
    assert completed["response"]["status"] == "completed"
    assert completed["response"]["usage"]["output_tokens"] == 2
    assert completed["response"]["usage"]["input_tokens"] == 10


def test_sequence_numbers_are_monotonic_from_zero() -> None:
    seqs = [e["sequence_number"] for e in _run()]
    assert seqs == list(range(len(seqs)))


def test_finalize_is_idempotent() -> None:
    t = ResponsesStreamTranslator(model="m")
    t.translate_chunk(_chunk("hi"))
    first = t.finalize()
    assert first
    assert t.finalize() == []


def test_sse_frame_format() -> None:
    frame = format_responses_sse({"type": "response.completed", "sequence_number": 0})
    assert frame.startswith("event: response.completed\ndata: ")
    assert frame.endswith("\n\n")


# ---------------------------------------------------------------------------
# Tool-call streaming (2b)
# ---------------------------------------------------------------------------

def _validate_all(events: list[dict]) -> None:
    for event in events:
        _SDK_EVENT[event["type"]].model_validate(event)


def test_tool_call_streaming_lifecycle_and_sdk_shapes() -> None:
    t = ResponsesStreamTranslator(model="qwen3-coder-next")
    events: list[dict] = []
    events += t.translate_chunk(_tool_open(0, "call_1", "get_weather"))
    events += t.translate_chunk(_tool_args(0, '{"city":'))
    events += t.translate_chunk(_tool_args(0, '"SF"}'))
    events += t.finalize()

    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    assert "response.output_item.added" in types          # function_call item opened
    assert types.count("response.function_call_arguments.delta") == 2
    assert "response.function_call_arguments.done" in types
    assert types[-1] == "response.completed"
    _validate_all(events)

    fc = next(o for o in events[-1]["response"]["output"] if o["type"] == "function_call")
    assert fc["name"] == "get_weather"
    assert fc["arguments"] == '{"city":"SF"}'
    assert fc["call_id"] == "call_1"


def test_mixed_text_then_tool_call_indices() -> None:
    t = ResponsesStreamTranslator(model="m")
    events: list[dict] = []
    events += t.translate_chunk(_chunk("Let me check. "))
    events += t.translate_chunk(_tool_open(0, "call_9", "search"))
    events += t.translate_chunk(_tool_args(0, '{"q":"x"}'))
    events += t.finalize()

    _validate_all(events)
    output = events[-1]["response"]["output"]
    # Message item at index 0, function_call at index 1 (order of appearance).
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "Let me check. "
    assert output[1]["type"] == "function_call"
    assert output[1]["name"] == "search"
    assert output[1]["arguments"] == '{"q":"x"}'


def test_two_parallel_tool_calls_get_distinct_items() -> None:
    t = ResponsesStreamTranslator(model="m")
    events: list[dict] = []
    events += t.translate_chunk(_tool_open(0, "call_a", "alpha"))
    events += t.translate_chunk(_tool_open(1, "call_b", "beta"))
    events += t.translate_chunk(_tool_args(0, "{}"))
    events += t.translate_chunk(_tool_args(1, "{}"))
    events += t.finalize()

    _validate_all(events)
    fcs = [o for o in events[-1]["response"]["output"] if o["type"] == "function_call"]
    assert [f["name"] for f in fcs] == ["alpha", "beta"]
    assert len({f["call_id"] for f in fcs}) == 2

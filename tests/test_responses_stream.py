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
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from modelmeld.api.schemas import ChatCompletionChunk, ChoiceDelta, ChunkChoice, Usage
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
    "response.completed": ResponseCompletedEvent,
}


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

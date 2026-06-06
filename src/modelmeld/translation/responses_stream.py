# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Internal Chat chunk stream → OpenAI Responses API SSE events.

The Responses streaming protocol is a lifecycle of typed events, each with a
monotonic `sequence_number`:

    response.created
    response.output_item.added        (the assistant message item)
    response.content_part.added       (an output_text part)
    response.output_text.delta × N    (the text, chunk by chunk)
    response.output_text.done
    response.content_part.done
    response.output_item.done
    response.completed                (terminal; carries usage)

Phase 2a handles text. Tool-call streaming (function_call items +
response.function_call_arguments.delta/done) is a 2b follow-up; the
non-streaming path already covers tool calls.

Events are emitted as plain dicts shaped to the wire format. They are validated
against the OpenAI SDK's own event models in the tests, so "matches the SDK" ==
"matches what Codex parses."
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from modelmeld.api.schemas import ChatCompletionChunk

_OUTPUT_INDEX = 0
_CONTENT_INDEX = 0


def format_responses_sse(event: dict[str, Any]) -> str:
    """Serialize one event dict to a Responses SSE frame."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


class ResponsesStreamTranslator:
    """State machine: ChatCompletionChunk stream → Responses SSE event dicts."""

    def __init__(self, model: str, *, response_id: str | None = None) -> None:
        self.model = model
        self.response_id = response_id or f"resp_{uuid4().hex[:24]}"
        self.item_id = f"msg_{uuid4().hex[:24]}"
        self._created_at = int(time.time())
        self._seq = 0
        self._started = False
        self._finished = False
        self._text_parts: list[str] = []
        self._input_tokens = 0
        self._output_tokens = 0

    # -- public API --------------------------------------------------------

    def translate_chunk(self, chunk: ChatCompletionChunk) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self._started:
            events.extend(self._start_events())
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                self._text_parts.append(delta.content)
                events.append({
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_seq(),
                    "item_id": self.item_id,
                    "output_index": _OUTPUT_INDEX,
                    "content_index": _CONTENT_INDEX,
                    "delta": delta.content,
                    "logprobs": [],
                })
        if chunk.usage is not None:
            self._output_tokens = chunk.usage.completion_tokens
            self._input_tokens = chunk.usage.prompt_tokens
        return events

    def finalize(self) -> list[dict[str, Any]]:
        """Closing events. Idempotent — second call returns []."""
        if self._finished:
            return []
        self._finished = True
        events: list[dict[str, Any]] = []
        if not self._started:
            events.extend(self._start_events())

        full_text = "".join(self._text_parts)
        part = {"type": "output_text", "text": full_text, "annotations": []}
        events.append({
            "type": "response.output_text.done",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id, "output_index": _OUTPUT_INDEX,
            "content_index": _CONTENT_INDEX, "text": full_text, "logprobs": [],
        })
        events.append({
            "type": "response.content_part.done",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id, "output_index": _OUTPUT_INDEX,
            "content_index": _CONTENT_INDEX, "part": part,
        })
        done_item = self._message_item("completed", [part])
        events.append({
            "type": "response.output_item.done",
            "sequence_number": self._next_seq(),
            "output_index": _OUTPUT_INDEX, "item": done_item,
        })
        events.append({
            "type": "response.completed",
            "sequence_number": self._next_seq(),
            "response": self._response_obj("completed", [done_item], self._usage()),
        })
        return events

    # -- internal ----------------------------------------------------------

    def _next_seq(self) -> int:
        n = self._seq
        self._seq += 1
        return n

    def _start_events(self) -> Iterable[dict[str, Any]]:
        self._started = True
        yield {
            "type": "response.created",
            "sequence_number": self._next_seq(),
            "response": self._response_obj("in_progress", []),
        }
        yield {
            "type": "response.output_item.added",
            "sequence_number": self._next_seq(),
            "output_index": _OUTPUT_INDEX,
            "item": self._message_item("in_progress", []),
        }
        yield {
            "type": "response.content_part.added",
            "sequence_number": self._next_seq(),
            "item_id": self.item_id, "output_index": _OUTPUT_INDEX,
            "content_index": _CONTENT_INDEX,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }

    def _message_item(self, status: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "type": "message", "id": self.item_id, "status": status,
            "role": "assistant", "content": content,
        }

    def _usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def _response_obj(
        self, status: str, output: list[dict[str, Any]], usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": self.response_id, "object": "response",
            "created_at": self._created_at, "model": self.model,
            "status": status, "output": output,
            "error": None, "incomplete_details": None, "instructions": None,
            "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "top_p": None,
            "tool_choice": "auto", "tools": [],
        }
        obj["usage"] = usage
        return obj

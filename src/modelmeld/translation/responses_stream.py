# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Internal Chat chunk stream → OpenAI Responses API SSE events.

The Responses stream is a lifecycle of typed events with a monotonic
`sequence_number`. Output is an ordered set of items — a `message` item for
assistant text, and one `function_call` item per tool call — each opened lazily
at its own `output_index` as it first appears in the chunk stream:

    response.created
    (text)  output_item.added(message) → content_part.added
            → output_text.delta × N
    (tool)  output_item.added(function_call)
            → function_call_arguments.delta × N
    (close, in output_index order)
            text:  output_text.done → content_part.done → output_item.done
            tool:  function_call_arguments.done → output_item.done
    response.completed   (terminal; carries the full output + usage)

Events are emitted as dicts shaped to the wire format and validated against the
OpenAI SDK's own event models in the tests, so "matches the SDK" == "matches
what Codex parses."
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from modelmeld.api.schemas import ChatCompletionChunk

_CONTENT_INDEX = 0


def format_responses_sse(event: dict[str, Any]) -> str:
    """Serialize one event dict to a Responses SSE frame."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


class ResponsesStreamTranslator:
    """State machine: ChatCompletionChunk stream → Responses SSE event dicts."""

    def __init__(self, model: str, *, response_id: str | None = None) -> None:
        self.model = model
        self.response_id = response_id or f"resp_{uuid4().hex[:24]}"
        self._created_at = int(time.time())
        self._seq = 0
        self._created_emitted = False
        self._finished = False
        self._next_output_index = 0

        # Text (message) item — opened lazily on the first text delta.
        self._text_open = False
        self._text_index = 0
        self._text_item_id = ""
        self._text_parts: list[str] = []

        # Tool (function_call) items, keyed by the OpenAI tool_call index.
        # value: {output_index, item_id, call_id, name, args: list[str]}
        self._tools: dict[int, dict[str, Any]] = {}

        # Usage. OSS providers usually omit usage from stream chunks, so we
        # track whether any real usage arrived; if none did, finalize() falls
        # back to a char-based estimate (output accumulated here across both
        # text and tool-call argument deltas; input supplied by the caller).
        self._input_tokens = 0
        self._output_tokens = 0
        self._saw_usage = False
        self._output_chars = 0

    # -- public API --------------------------------------------------------

    def translate_chunk(self, chunk: ChatCompletionChunk) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self._created_emitted:
            self._created_emitted = True
            events.append({
                "type": "response.created",
                "sequence_number": self._next_seq(),
                "response": self._response_obj("in_progress", []),
            })

        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                if not self._text_open:
                    events.extend(self._open_text())
                self._text_parts.append(delta.content)
                self._output_chars += len(delta.content)
                events.append({
                    "type": "response.output_text.delta",
                    "sequence_number": self._next_seq(),
                    "item_id": self._text_item_id,
                    "output_index": self._text_index,
                    "content_index": _CONTENT_INDEX,
                    "delta": delta.content,
                    "logprobs": [],
                })
            if delta.tool_calls:
                for tcd in delta.tool_calls:
                    events.extend(self._handle_tool_delta(tcd))

        if chunk.usage is not None:
            self._saw_usage = True
            self._output_tokens = chunk.usage.completion_tokens
            self._input_tokens = chunk.usage.prompt_tokens
        return events

    def finalize(self, *, input_tokens: int = 0) -> list[dict[str, Any]]:
        """Closing events. Idempotent — second call returns [].

        `input_tokens` is a caller-supplied estimate used only when the upstream
        stream never reported real usage; output tokens are then estimated from
        the accumulated text + tool-argument characters (≈4 chars/token).
        """
        if self._finished:
            return []
        self._finished = True
        if not self._saw_usage:
            self._input_tokens = input_tokens
            self._output_tokens = (
                max(1, self._output_chars // 4) if self._output_chars else 0
            )
        events: list[dict[str, Any]] = []
        if not self._created_emitted:
            self._created_emitted = True
            events.append({
                "type": "response.created",
                "sequence_number": self._next_seq(),
                "response": self._response_obj("in_progress", []),
            })

        # Responses guarantees ≥1 output item; if the model produced neither
        # text nor tool calls, emit an empty message.
        if not self._text_open and not self._tools:
            events.extend(self._open_text())

        # Indexed list of completed items for the terminal response object.
        completed_items: list[tuple[int, dict[str, Any]]] = []

        if self._text_open:
            full_text = "".join(self._text_parts)
            part = {"type": "output_text", "text": full_text, "annotations": []}
            events.append({
                "type": "response.output_text.done",
                "sequence_number": self._next_seq(),
                "item_id": self._text_item_id, "output_index": self._text_index,
                "content_index": _CONTENT_INDEX, "text": full_text, "logprobs": [],
            })
            events.append({
                "type": "response.content_part.done",
                "sequence_number": self._next_seq(),
                "item_id": self._text_item_id, "output_index": self._text_index,
                "content_index": _CONTENT_INDEX, "part": part,
            })
            done_item = self._message_item("completed", self._text_item_id, [part])
            events.append({
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": self._text_index, "item": done_item,
            })
            completed_items.append((self._text_index, done_item))

        for tool in sorted(self._tools.values(), key=lambda t: t["index"]):
            full_args = "".join(tool["args"])
            events.append({
                "type": "response.function_call_arguments.done",
                "sequence_number": self._next_seq(),
                "item_id": tool["item_id"], "output_index": tool["index"],
                "name": tool["name"], "arguments": full_args,
            })
            fc_item = self._function_call_item(tool, full_args)
            events.append({
                "type": "response.output_item.done",
                "sequence_number": self._next_seq(),
                "output_index": tool["index"], "item": fc_item,
            })
            completed_items.append((tool["index"], fc_item))

        output = [item for _, item in sorted(completed_items, key=lambda x: x[0])]
        events.append({
            "type": "response.completed",
            "sequence_number": self._next_seq(),
            "response": self._response_obj("completed", output, self._usage()),
        })
        return events

    # -- internal ----------------------------------------------------------

    def _next_seq(self) -> int:
        n = self._seq
        self._seq += 1
        return n

    def _open_text(self) -> list[dict[str, Any]]:
        self._text_open = True
        self._text_index = self._next_output_index
        self._next_output_index += 1
        self._text_item_id = f"msg_{uuid4().hex[:24]}"
        return [
            {
                "type": "response.output_item.added",
                "sequence_number": self._next_seq(),
                "output_index": self._text_index,
                "item": self._message_item("in_progress", self._text_item_id, []),
            },
            {
                "type": "response.content_part.added",
                "sequence_number": self._next_seq(),
                "item_id": self._text_item_id, "output_index": self._text_index,
                "content_index": _CONTENT_INDEX,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]

    def _handle_tool_delta(self, tcd: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        idx = tcd.index
        fn = getattr(tcd, "function", None)
        if idx not in self._tools:
            output_index = self._next_output_index
            self._next_output_index += 1
            item_id = f"fc_{uuid4().hex[:24]}"
            tool = {
                "output_index": output_index, "index": output_index,
                "item_id": item_id,
                "call_id": tcd.id or f"call_{uuid4().hex[:24]}",
                "name": (fn.name if fn and fn.name else ""),
                "args": [],
            }
            self._tools[idx] = tool
            events.append({
                "type": "response.output_item.added",
                "sequence_number": self._next_seq(),
                "output_index": output_index,
                "item": self._function_call_item(tool, ""),
            })
        tool = self._tools[idx]
        if fn and fn.name and not tool["name"]:
            tool["name"] = fn.name
        if fn and fn.arguments:
            tool["args"].append(fn.arguments)
            self._output_chars += len(fn.arguments)
            events.append({
                "type": "response.function_call_arguments.delta",
                "sequence_number": self._next_seq(),
                "item_id": tool["item_id"], "output_index": tool["index"],
                "delta": fn.arguments,
            })
        return events

    def _message_item(self, status: str, item_id: str, content: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "type": "message", "id": item_id, "status": status,
            "role": "assistant", "content": content,
        }

    def _function_call_item(self, tool: dict[str, Any], arguments: str) -> dict[str, Any]:
        return {
            "type": "function_call", "id": tool["item_id"],
            "call_id": tool["call_id"], "name": tool["name"],
            "arguments": arguments, "status": "completed",
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
        return {
            "id": self.response_id, "object": "response",
            "created_at": self._created_at, "model": self.model,
            "status": status, "output": output,
            "error": None, "incomplete_details": None, "instructions": None,
            "metadata": {}, "parallel_tool_calls": True,
            "temperature": None, "top_p": None,
            "tool_choice": "auto", "tools": [], "usage": usage,
        }

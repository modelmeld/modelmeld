# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Pydantic models for the OpenAI Responses API surface (`POST /v1/responses`).

This is the third wire format the gateway speaks (alongside Chat Completions and
Anthropic Messages). Codex CLI talks the Responses API natively, so exposing it
at the standard `/v1/responses` path lets Codex point at the gateway with no
plugin shim.

Request shape: an `input` (string or list of role/content items) plus a separate
`instructions` string. Response shape: an `output` array of typed items
(`message`, `function_call`) rather than `choices`. Models are permissive
(`extra="allow"`) where Codex sends fields we don't act on, so unknown fields
round-trip instead of 422-ing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ResponsesContentPart(BaseModel):
    """One content part of an input item. Responses uses typed parts
    (`input_text`, `output_text`, `input_image`, …); Phase 1 reads text parts."""

    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None


class ResponsesInputItem(BaseModel):
    """One entry in the `input` array.

    The array is heterogeneous: a multi-turn client (e.g. Codex) replays the
    whole conversation, so beyond role/content `message` items it also sends
    `function_call` (a prior tool call), `function_call_output` (a tool result),
    and `reasoning` items — none of which carry a `role`. Every field is
    therefore optional and the item is keyed by `type`; the translator decides
    what each item becomes (or skips it)."""

    model_config = ConfigDict(extra="allow")
    type: str | None = None
    # message items
    role: str | None = None  # user | assistant | system | developer
    content: str | list[ResponsesContentPart] | None = None
    # function_call items (assistant's prior tool invocation)
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # function_call_output items (tool result fed back in)
    output: Any | None = None


class ResponsesRequest(BaseModel):
    """OpenAI Responses API request body."""

    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[ResponsesInputItem]
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    store: bool | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class ResponsesOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponsesMessageItem(BaseModel):
    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["completed"] = "completed"
    content: list[ResponsesOutputText]


class ResponsesFunctionCallItem(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: Literal["completed"] = "completed"


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed"] = "completed"
    output: list[ResponsesMessageItem | ResponsesFunctionCallItem]
    usage: ResponsesUsage

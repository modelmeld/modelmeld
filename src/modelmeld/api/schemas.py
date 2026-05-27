# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""OpenAI-compatible Pydantic schemas. Full request/response/chunk fidelity.

Modeled from the published OpenAI Chat Completions API surface. Not a source-level
copy of the openai-python SDK — written independently against the public spec.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
]


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex[:24]}"


def _now() -> int:
    return int(time.time())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Content parts (multimodal user/system messages)
# ---------------------------------------------------------------------------

class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class InputAudio(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class AudioPart(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio


ContentPart = Annotated[
    TextPart | ImagePart | AudioPart,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Tool definitions and tool calls
# ---------------------------------------------------------------------------

class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDef


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


# Streaming tool-call deltas may have partial fields and require an index.
class FunctionCallDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


# ---------------------------------------------------------------------------
# Message types (discriminated union on `role`)
# ---------------------------------------------------------------------------

class SystemMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system"]
    content: str | list[TextPart]
    name: str | None = None


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user"]
    content: str | list[ContentPart]
    name: str | None = None


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["assistant"]
    content: str | list[TextPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["tool"]
    content: str | list[TextPart]
    tool_call_id: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Logprobs
# ---------------------------------------------------------------------------

class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    content: list[TokenLogprob] | None = None
    refusal: list[TokenLogprob] | None = None


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class PromptTokensDetails(BaseModel):
    cached_tokens: int | None = None
    audio_tokens: int | None = None


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    audio_tokens: int | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class StreamOptions(BaseModel):
    include_usage: bool | None = False


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    # Forward-compat: unknown fields pass through rather than raising.
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model: str
    messages: list[Message]

    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None


# ---------------------------------------------------------------------------
# Response (non-streaming)
# ---------------------------------------------------------------------------

class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletion(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=_new_completion_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


# ---------------------------------------------------------------------------
# Streaming chunk
# ---------------------------------------------------------------------------

class ChoiceDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    refusal: str | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------

class Model(BaseModel):
    id: str
    # OpenAI uses `object: "model"`; Anthropic-native uses `type: "model"`.
    # Both clients ignore the field they don't care about, so we emit both.
    object: Literal["model"] = "model"
    type: Literal["model"] = "model"
    # OpenAI uses `created` (unix int); Anthropic-native uses `created_at`
    # (RFC 3339 / ISO 8601 string). Claude Code's gateway model discovery
    # (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1) only parses the
    # Anthropic-native shape, so we emit both.
    created: int = Field(default_factory=_now)
    created_at: str = Field(default_factory=_now_iso)
    owned_by: str
    # Anthropic-style human-readable name. Optional for backwards-compat
    # with OpenAI's /v1/models shape (which doesn't include it). Claude
    # Code's /model picker populates from this — required for the picker
    # to render the entry.
    display_name: str | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]
    # Anthropic-native pagination fields. Our endpoint doesn't paginate
    # (everything fits in one page), but Claude Code's gateway model
    # discovery requires these top-level fields to parse the response.
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None

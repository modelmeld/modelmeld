# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Anthropic Messages API Pydantic schemas.

Wire-format models for the `/v1/messages` route. Modeled from the published
Anthropic Messages API spec — not source-copied from the `anthropic` SDK
(see [[ip_no_direct_copy]]).

Scope (v1):
    Request:  text + tool_use + tool_result blocks; tools; tool_choice;
              system as str or list of text blocks; max_tokens required.
    Response: text + tool_use blocks; stop_reason; usage.
    Streaming: message_start / content_block_start / content_block_delta /
               content_block_stop / message_delta / message_stop event shapes.
    Deferred to v2: image content blocks; cache_control honoring (we accept
                    the field on input but ignore it); ping events.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Cache control (accepted but ignored in v1)
# ---------------------------------------------------------------------------

class AnthropicCacheControl(BaseModel):
    """Prompt-caching marker on a content block. Accepted but ignored in v1."""
    type: Literal["ephemeral"]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

class AnthropicTextBlock(BaseModel):
    """`{"type": "text", "text": "..."}`."""
    model_config = ConfigDict(extra="allow")
    type: Literal["text"]
    text: str
    cache_control: AnthropicCacheControl | None = None


class AnthropicImageSourceBase64(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicImageSourceUrl(BaseModel):
    type: Literal["url"]
    url: str


AnthropicImageSource = Annotated[
    AnthropicImageSourceBase64 | AnthropicImageSourceUrl,
    Field(discriminator="type"),
]


class AnthropicImageBlock(BaseModel):
    """`{"type": "image", "source": {...}}`. Translation raises in v1 — deferred."""
    model_config = ConfigDict(extra="allow")
    type: Literal["image"]
    source: AnthropicImageSource
    cache_control: AnthropicCacheControl | None = None


class AnthropicToolUseBlock(BaseModel):
    """`{"type": "tool_use", "id": "...", "name": "...", "input": {...}}`.

    Appears on assistant turns in multi-turn conversations: the model
    previously called a tool and the client is replaying the call back.
    """
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    cache_control: AnthropicCacheControl | None = None


class AnthropicToolResultBlock(BaseModel):
    """`{"type": "tool_result", "tool_use_id": "...", "content": ..., "is_error": ...}`.

    Appears on user turns: the client returning the result of a tool call
    that the model previously requested. `content` may be a string OR a list
    of text blocks (per D-3 we accept both; image-bearing tool results raise).
    """
    model_config = ConfigDict(extra="allow")
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[AnthropicTextBlock]
    is_error: bool | None = None
    cache_control: AnthropicCacheControl | None = None


# Discriminated union of all content block types a CLIENT may send in
# request messages. (Response blocks are a narrower set — see below.)
AnthropicRequestContentBlock = Annotated[
    AnthropicTextBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock,
    Field(discriminator="type"),
]


# Narrower union for response content: text + tool_use only.
AnthropicResponseContentBlock = Annotated[
    AnthropicTextBlock | AnthropicToolUseBlock,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class AnthropicMessage(BaseModel):
    """A single message in the conversation history."""
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"]
    content: str | list[AnthropicRequestContentBlock]


# ---------------------------------------------------------------------------
# Tool definitions and tool choice
# ---------------------------------------------------------------------------

class AnthropicToolDef(BaseModel):
    """Tool the model may call. `input_schema` is the JSON Schema for input."""
    model_config = ConfigDict(extra="allow")
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    cache_control: AnthropicCacheControl | None = None


class AnthropicToolChoiceAuto(BaseModel):
    type: Literal["auto"]
    disable_parallel_tool_use: bool | None = None


class AnthropicToolChoiceAny(BaseModel):
    """`any` ≡ OpenAI's `required` — model MUST call some tool."""
    type: Literal["any"]
    disable_parallel_tool_use: bool | None = None


class AnthropicToolChoiceSpecific(BaseModel):
    """`{"type":"tool", "name":"..."}` — model MUST call this specific tool."""
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: bool | None = None


class AnthropicToolChoiceNone(BaseModel):
    """`{"type":"none"}` — model MUST NOT call any tool."""
    type: Literal["none"]


AnthropicToolChoice = Annotated[
    AnthropicToolChoiceAuto | AnthropicToolChoiceAny | AnthropicToolChoiceSpecific | AnthropicToolChoiceNone,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# System prompt (string OR list of text blocks)
# ---------------------------------------------------------------------------

# System is either a plain string or a list of text blocks (the list form
# exists to support `cache_control` markers on segments). We accept both
# shapes; D-2: list shape is joined with "\n\n" into a single string at
# translation time.
AnthropicSystemPrompt = Union[str, list[AnthropicTextBlock]]


# ---------------------------------------------------------------------------
# Metadata (e.g., end-user ID)
# ---------------------------------------------------------------------------

class AnthropicMetadata(BaseModel):
    """Optional metadata Anthropic uses for safety/abuse tracking."""
    model_config = ConfigDict(extra="allow")
    user_id: str | None = None


# ---------------------------------------------------------------------------
# Top-level request
# ---------------------------------------------------------------------------

class AnthropicMessagesRequest(BaseModel):
    """`POST /v1/messages` request body."""
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]

    # D-1: max_tokens is required by Anthropic. We enforce the same.
    max_tokens: int = Field(..., ge=1)

    # D-2: system may be str or list of text blocks.
    system: AnthropicSystemPrompt | None = None

    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: list[str] | None = None
    stream: bool | None = False

    tools: list[AnthropicToolDef] | None = None
    tool_choice: AnthropicToolChoice | None = None

    metadata: AnthropicMetadata | None = None


class AnthropicCountTokensRequest(AnthropicMessagesRequest):
    """`POST /v1/messages/count_tokens` request body.

    Same shape as `AnthropicMessagesRequest` except `max_tokens` is
    optional — `count_tokens` measures the input prompt; it does not
    generate any output, so `max_tokens` has no semantic role on this
    endpoint. Anthropic's API accepts count_tokens requests without
    `max_tokens`; we follow suit.

    Inherits all other fields (`temperature`, `top_p`, etc.) so
    `from_anthropic_request` can translate either shape uniformly —
    the generation-only fields are simply ignored by the
    token-counting path.
    """
    # Override the required-int constraint to optional. Pyright/mypy
    # would flag this as a Liskov-style override (subclass loosening a
    # parent constraint), but Pydantic supports it and the semantics
    # are intentional here. The `type: ignore` is on the field
    # redefinition.
    max_tokens: int | None = Field(default=None, ge=1)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Response (non-streaming)
# ---------------------------------------------------------------------------

AnthropicStopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "refusal",
]


class AnthropicUsage(BaseModel):
    """Token counts surfaced to the caller in `/v1/messages` responses.

    `cache_creation_input_tokens` / `cache_read_input_tokens` are
    populated when the upstream (Anthropic) reports them — they're the
    visible signal that the customer's `cache_control` markers are
    working through our gateway. Populated end-to-end via
    `from_anthropic_response` (upstream → ChatCompletion) +
    `from_openai_anthropic` (ChatCompletion → AnthropicMessagesResponse)
    so non-streaming responses preserve the cache stats verbatim.

    Streaming responses today propagate input/output tokens only; cache
    stats in the upstream `message_start` event are not yet plumbed
    through the stream-translation pipeline (tracked as a follow-up).
    """
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicMessagesResponse(BaseModel):
    """`POST /v1/messages` non-streaming response."""
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicResponseContentBlock]
    model: str
    stop_reason: AnthropicStopReason | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

class _MessageStartShell(BaseModel):
    """The `message` field inside a `message_start` event: a response shell
    with empty content + input_tokens populated, output_tokens=0."""
    model_config = ConfigDict(extra="allow")
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicResponseContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: AnthropicStopReason | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicMessageStartEvent(BaseModel):
    type: Literal["message_start"]
    message: _MessageStartShell


class _ContentBlockStartTextShell(BaseModel):
    type: Literal["text"]
    text: str = ""


class _ContentBlockStartToolUseShell(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


_ContentBlockStartShell = Annotated[
    _ContentBlockStartTextShell | _ContentBlockStartToolUseShell,
    Field(discriminator="type"),
]


class AnthropicContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"]
    index: int
    content_block: _ContentBlockStartShell


class _TextDelta(BaseModel):
    type: Literal["text_delta"]
    text: str


class _InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"]
    partial_json: str


_ContentBlockDelta = Annotated[
    _TextDelta | _InputJsonDelta,
    Field(discriminator="type"),
]


class AnthropicContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"]
    index: int
    delta: _ContentBlockDelta


class AnthropicContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"]
    index: int


class _MessageDelta(BaseModel):
    stop_reason: AnthropicStopReason | None = None
    stop_sequence: str | None = None


class _MessageDeltaUsage(BaseModel):
    """Usage block on `message_delta`. Anthropic emits output_tokens only here."""
    output_tokens: int


class AnthropicMessageDeltaEvent(BaseModel):
    type: Literal["message_delta"]
    delta: _MessageDelta
    usage: _MessageDeltaUsage


class AnthropicMessageStopEvent(BaseModel):
    type: Literal["message_stop"]


class AnthropicPingEvent(BaseModel):
    """Anthropic occasionally emits ping events on long streams. v1 doesn't
    generate them but we define the shape for completeness."""
    type: Literal["ping"]


AnthropicStreamEvent = Annotated[
    AnthropicMessageStartEvent | AnthropicContentBlockStartEvent | AnthropicContentBlockDeltaEvent | AnthropicContentBlockStopEvent | AnthropicMessageDeltaEvent | AnthropicMessageStopEvent | AnthropicPingEvent,
    Field(discriminator="type"),
]


__all__ = [
    # Misc
    "AnthropicCacheControl",
    "AnthropicContentBlockDeltaEvent",
    "AnthropicContentBlockStartEvent",
    "AnthropicContentBlockStopEvent",
    "AnthropicImageBlock",
    "AnthropicImageSource",
    "AnthropicImageSourceBase64",
    "AnthropicImageSourceUrl",
    "AnthropicMessage",
    "AnthropicMessageDeltaEvent",
    "AnthropicMessageStartEvent",
    "AnthropicMessageStopEvent",
    # Request
    "AnthropicMessagesRequest",
    # Response
    "AnthropicMessagesResponse",
    "AnthropicMetadata",
    "AnthropicPingEvent",
    "AnthropicRequestContentBlock",
    "AnthropicResponseContentBlock",
    "AnthropicStopReason",
    # Stream events
    "AnthropicStreamEvent",
    "AnthropicSystemPrompt",
    # Content blocks
    "AnthropicTextBlock",
    "AnthropicToolChoice",
    "AnthropicToolChoiceAny",
    "AnthropicToolChoiceAuto",
    "AnthropicToolChoiceNone",
    "AnthropicToolChoiceSpecific",
    # Tools
    "AnthropicToolDef",
    "AnthropicToolResultBlock",
    "AnthropicToolUseBlock",
    "AnthropicUsage",
]

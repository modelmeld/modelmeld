# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""OpenAI Chat Completions ⇄ Anthropic Messages API translation.

Translates in both directions on pure dicts so tests can run without the
`anthropic` SDK installed. The AnthropicAdapter calls `model_dump()` on SDK
objects before handing them to `from_anthropic_response` / event translators.

Scope:
    ✓ Text messages, multi-turn, system prompts
    ✓ Tool definitions, tool calls (assistant), tool results (user)
    ✓ Images via image_url (URL and base64 data URLs)
    ✓ Streaming events (message_start / content_block_* / message_delta / message_stop)
    ✗ input_audio content parts (Anthropic doesn't accept audio — raises)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from uuid import uuid4

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    FinishReason,
    FunctionCall,
    FunctionDef,
    ImagePart,
    Message,
    PromptTokensDetails,
    ResponseMessage,
    SystemMessage,
    TextPart,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicMessageDeltaEvent,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicResponseContentBlock,
    AnthropicStopReason,
    AnthropicStreamEvent,
    AnthropicTextBlock,
    AnthropicToolChoice,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceAuto,
    AnthropicToolChoiceNone,
    AnthropicToolChoiceSpecific,
    AnthropicToolDef,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    _ContentBlockStartTextShell,
    _ContentBlockStartToolUseShell,
    _InputJsonDelta,
    _MessageDelta,
    _MessageDeltaUsage,
    _MessageStartShell,
    _TextDelta,
)


class TranslationError(Exception):
    """Raised when a request/response cannot be losslessly translated."""


_DEFAULT_MAX_TOKENS = 4096
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)

# Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


# ---------------------------------------------------------------------------
# Request: OpenAI → Anthropic
# ---------------------------------------------------------------------------

def to_anthropic_params(request: ChatCompletionRequest) -> dict[str, Any]:
    system_text, messages = _split_system_and_messages(request.messages)

    params: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        # Anthropic requires max_tokens. Default if neither OpenAI field is set.
        "max_tokens": request.max_completion_tokens
        or request.max_tokens
        or _DEFAULT_MAX_TOKENS,
    }
    if system_text:
        params["system"] = system_text
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.stop:
        params["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else list(request.stop)
        )
    if request.tools:
        params["tools"] = [_translate_tool_def(t) for t in request.tools]
    if request.tool_choice is not None:
        tc = _translate_tool_choice(request.tool_choice)
        if tc is not None:
            params["tool_choice"] = tc
    return params


def _split_system_and_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(_extract_text(msg.content))
            continue
        converted.append(_translate_message(msg))
    return ("\n\n".join(system_parts) if system_parts else None, converted)


def _extract_text(content: str | list[TextPart]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content)


def _translate_message(msg: Message) -> dict[str, Any]:
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": _translate_user_content(msg.content)}
    if isinstance(msg, AssistantMessage):
        return {"role": "assistant", "content": _translate_assistant_content(msg)}
    if isinstance(msg, ToolMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": _extract_text(msg.content),
                }
            ],
        }
    raise TranslationError(f"Unsupported message type: {type(msg).__name__}")


def _translate_user_content(content: str | list[Any]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image", "source": _translate_image_source(part.image_url.url)})
        else:
            # AudioPart and any future part types
            raise TranslationError(
                f"Anthropic does not support content part type: {getattr(part, 'type', type(part).__name__)}"
            )
    return blocks


def _translate_image_source(url: str) -> dict[str, Any]:
    match = _DATA_URL_RE.match(url)
    if match:
        return {
            "type": "base64",
            "media_type": match.group("mime"),
            "data": match.group("data"),
        }
    return {"type": "url", "url": url}


def _translate_assistant_content(msg: AssistantMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if msg.content:
        text = _extract_text(msg.content) if isinstance(msg.content, list) else msg.content
        blocks.append({"type": "text", "text": text})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                input_dict = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                input_dict = {"_raw_arguments": tc.function.arguments}
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": input_dict,
            })
    return blocks or [{"type": "text", "text": ""}]


def _translate_tool_def(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.function.name,
        "description": tool.function.description or "",
        "input_schema": tool.function.parameters or {"type": "object"},
    }


def _translate_tool_choice(choice: str | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return None  # Anthropic's equivalent is to omit tools; we keep tools but signal preference
    elif isinstance(choice, dict):
        if choice.get("type") == "function":
            fn = choice.get("function", {})
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": name}
    return {"type": "auto"}


# ---------------------------------------------------------------------------
# Response: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def from_anthropic_response(response: dict[str, Any]) -> ChatCompletion:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    type="function",
                    function=FunctionCall(
                        name=block["name"],
                        arguments=json.dumps(block.get("input", {})),
                    ),
                )
            )

    content = "\n".join(text_parts) if text_parts else None
    stop_reason = response.get("stop_reason") or "end_turn"
    finish_reason: FinishReason = _STOP_REASON_MAP.get(stop_reason, "stop")

    usage = response.get("usage", {})
    in_tok = int(usage.get("input_tokens", 0))
    out_tok = int(usage.get("output_tokens", 0))

    # Preserve Anthropic's cache stats — `cache_creation_input_tokens`
    # (tokens billed at full rate on cache write) and
    # `cache_read_input_tokens` (tokens billed at 10% rate on cache
    # hit). Without this, the customer can't verify their cache_control
    # markers are working through our gateway, even though they're
    # getting the discount on Anthropic's bill.
    cache_write = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    prompt_details: PromptTokensDetails | None = None
    if cache_write is not None or cache_read is not None:
        prompt_details = PromptTokensDetails(
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            # Mirror cache_read into the cross-vendor `cached_tokens`
            # slot so consumers asking "how many tokens hit cache?"
            # get a consistent answer regardless of upstream vendor.
            cached_tokens=cache_read,
        )

    return ChatCompletion(
        id=response.get("id") or f"chatcmpl-{uuid4().hex[:24]}",
        created=int(time.time()),
        model=response.get("model", ""),
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            prompt_tokens_details=prompt_details,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming: Anthropic events → OpenAI chunks
# ---------------------------------------------------------------------------

def _usage_from_anthropic_message_start(usage_dict: dict[str, Any]) -> Usage | None:
    """Carry an Anthropic `message_start` usage block — input tokens + cache
    stats — onto the first OpenAI chunk, so downstream consumers (and the
    OpenAI→Anthropic re-translation) can surface them. Anthropic reports cache
    counts ONLY in message_start, so this is the one chance to capture them in a
    stream. Returns None when there's nothing to carry. Mirrors
    `from_anthropic_response`.
    """
    if not usage_dict:
        return None
    cache_write = usage_dict.get("cache_creation_input_tokens")
    cache_read = usage_dict.get("cache_read_input_tokens")
    prompt_details: PromptTokensDetails | None = None
    if cache_write is not None or cache_read is not None:
        prompt_details = PromptTokensDetails(
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            cached_tokens=cache_read,
        )
    in_tok = int(usage_dict.get("input_tokens", 0))
    if prompt_details is None and in_tok == 0:
        return None
    return Usage(
        prompt_tokens=in_tok,
        completion_tokens=0,
        total_tokens=in_tok,
        prompt_tokens_details=prompt_details,
    )


class AnthropicStreamTranslator:
    """Accumulates state across an Anthropic stream and emits OpenAI chunks.

    `translate_event` consumes one event dict, returns 0 or 1 chunk(s).
    """

    def __init__(self) -> None:
        self.id: str = ""
        self.model: str = ""
        self.created: int = int(time.time())
        self.role_emitted = False
        # block_index → tool_call_index (OpenAI tool_calls are per-message-indexed)
        self._block_to_tool_index: dict[int, int] = {}
        self._next_tool_index = 0

    def _chunk(
        self,
        delta: ChoiceDelta,
        finish_reason: FinishReason | None = None,
        usage: Usage | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self.id or f"chatcmpl-{uuid4().hex[:24]}",
            created=self.created,
            model=self.model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
            usage=usage,
        )

    def translate_event(self, event: dict[str, Any]) -> ChatCompletionChunk | None:
        etype = event.get("type")

        if etype == "message_start":
            message = event.get("message", {})
            self.id = message.get("id", self.id)
            self.model = message.get("model", self.model)
            self.created = int(time.time())
            # Emit the role chunk now, carrying any cache stats from the
            # message_start usage (the only place Anthropic reports them).
            self.role_emitted = True
            return self._chunk(
                ChoiceDelta(role="assistant", content=""),
                usage=_usage_from_anthropic_message_start(message.get("usage", {})),
            )

        if etype == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            btype = block.get("type")
            if btype == "tool_use":
                tool_idx = self._next_tool_index
                self._next_tool_index += 1
                self._block_to_tool_index[idx] = tool_idx
                from modelmeld.api.schemas import FunctionCallDelta, ToolCallDelta
                return self._chunk(
                    ChoiceDelta(
                        tool_calls=[
                            ToolCallDelta(
                                index=tool_idx,
                                id=block.get("id"),
                                type="function",
                                function=FunctionCallDelta(
                                    name=block.get("name"),
                                    arguments="",
                                ),
                            )
                        ]
                    )
                )
            return None  # text block start — wait for the deltas

        if etype == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                return self._chunk(ChoiceDelta(content=delta.get("text", "")))
            if dtype == "input_json_delta":
                tool_idx = self._block_to_tool_index.get(idx)
                if tool_idx is None:
                    return None
                from modelmeld.api.schemas import FunctionCallDelta, ToolCallDelta
                return self._chunk(
                    ChoiceDelta(
                        tool_calls=[
                            ToolCallDelta(
                                index=tool_idx,
                                function=FunctionCallDelta(
                                    arguments=delta.get("partial_json", "")
                                ),
                            )
                        ]
                    )
                )
            return None

        if etype == "content_block_stop":
            return None  # nothing to emit; the deltas covered the content

        if etype == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")
            finish: FinishReason | None = (
                _STOP_REASON_MAP.get(stop_reason, "stop") if stop_reason else None
            )
            usage_dict = event.get("usage", {})
            usage = None
            if usage_dict:
                # Anthropic only sends output_tokens here.
                usage = Usage(
                    prompt_tokens=0,
                    completion_tokens=int(usage_dict.get("output_tokens", 0)),
                    total_tokens=int(usage_dict.get("output_tokens", 0)),
                )
            return self._chunk(ChoiceDelta(), finish_reason=finish, usage=usage)

        if etype == "message_stop":
            return None

        # Unknown event type — ignore quietly.
        return None


# ---------------------------------------------------------------------------
# Request: Anthropic → OpenAI  (feeds the /v1/messages route)
# ---------------------------------------------------------------------------

def from_anthropic_request(req: AnthropicMessagesRequest) -> ChatCompletionRequest:
    """Translate an Anthropic Messages request into the internal OpenAI shape.

    Mirror of `to_anthropic_params`. The translation tables (request fields,
    content blocks, tool definitions, tool_choice variants) are documented in
    docs/design-anthropic-messages-api.md.

    A single Anthropic message MAY produce multiple OpenAI messages — a user
    turn with mixed tool_result + text blocks splits into ToolMessage(s) +
    UserMessage in the order they appeared.

    Raises TranslationError on image content blocks (deferred to v2),
    on tool_result blocks bearing image content, on tool_use blocks on user
    role (malformed input), and on tool_result blocks on assistant role
    (also malformed — tool_results must come back to the assistant via user).
    """
    messages: list[Message] = []

    # 1. System (D-2): str or list[AnthropicTextBlock] → leading SystemMessage
    if req.system is not None:
        system_text = _anthropic_system_to_text(req.system)
        if system_text:
            messages.append(SystemMessage(role="system", content=system_text))

    # 2. Per-message conversion. One Anthropic message may yield multiple
    #    OpenAI messages (mixed tool_result + text in a single user turn).
    for msg in req.messages:
        messages.extend(_anthropic_message_to_openai(msg))

    # 3. Tools
    tools: list[Tool] | None = None
    if req.tools:
        tools = [_anthropic_tool_def_to_openai(t) for t in req.tools]

    # 4. tool_choice
    tool_choice: str | dict[str, Any] | None = None
    if req.tool_choice is not None:
        tool_choice = _anthropic_tool_choice_to_openai(req.tool_choice)

    # 5. stop_sequences: Anthropic uses list[str]; OpenAI accepts str | list[str].
    #    Pass list straight through; a single-element list is still semantically
    #    correct on the OpenAI side.
    stop: str | list[str] | None = None
    if req.stop_sequences:
        stop = list(req.stop_sequences)

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=stop,
        stream=req.stream or False,
        tools=tools,
        tool_choice=tool_choice,
        # metadata.user_id maps onto the OpenAI `user` field (similar purpose:
        # abuse/safety tracking by end-user identity).
        user=(req.metadata.user_id if req.metadata else None),
    )


def _anthropic_system_to_text(system: str | list[AnthropicTextBlock]) -> str:
    """D-2: list-form joins text blocks with '\\n\\n'. cache_control ignored."""
    if isinstance(system, str):
        return system
    return "\n\n".join(block.text for block in system)


def _anthropic_message_to_openai(msg: AnthropicMessage) -> list[Message]:
    """Convert one Anthropic message → one or more OpenAI messages."""
    if msg.role == "user":
        return _anthropic_user_to_openai(msg)
    if msg.role == "assistant":
        return [_anthropic_assistant_to_openai(msg)]
    if msg.role == "system":
        # Non-standard for Anthropic's own API, but real clients (Claude Code
        # headless) send it. Hoist to a SystemMessage; the egress translator
        # folds all SystemMessages back into the top-level system prompt.
        if isinstance(msg.content, str):
            text = msg.content
        else:
            text = "\n".join(
                t for b in msg.content if (t := getattr(b, "text", ""))
            )
        return [SystemMessage(role="system", content=text)]
    # Pydantic literal validation should prevent reaching here.
    raise TranslationError(f"Unsupported message role: {msg.role!r}")


def _anthropic_user_to_openai(msg: AnthropicMessage) -> list[Message]:
    """User message → potentially multiple OpenAI messages.

    Pure-string content → single UserMessage.
    List of blocks: tool_result blocks become their own ToolMessages
    (one per block); contiguous text/image blocks accumulate into a
    UserMessage. The original block order is preserved.
    """
    if isinstance(msg.content, str):
        return [UserMessage(role="user", content=msg.content)]

    out: list[Message] = []
    pending_user_parts: list[TextPart | ImagePart] = []

    def _flush_pending() -> None:
        if not pending_user_parts:
            return
        # Simplify the common case of a single text block → plain string.
        if len(pending_user_parts) == 1 and isinstance(pending_user_parts[0], TextPart):
            out.append(UserMessage(role="user", content=pending_user_parts[0].text))
        else:
            out.append(UserMessage(role="user", content=list(pending_user_parts)))
        pending_user_parts.clear()

    for block in msg.content:
        if isinstance(block, AnthropicTextBlock):
            pending_user_parts.append(TextPart(type="text", text=block.text))
        elif isinstance(block, AnthropicImageBlock):
            raise TranslationError(
                "Anthropic image content blocks are not supported in v1 "
                "of the /v1/messages route. Deferred per the design doc."
            )
        elif isinstance(block, AnthropicToolResultBlock):
            _flush_pending()
            content_text = _anthropic_tool_result_content_to_text(block.content)
            out.append(ToolMessage(
                role="tool",
                tool_call_id=block.tool_use_id,
                content=content_text,
            ))
        elif isinstance(block, AnthropicToolUseBlock):
            # tool_use blocks belong on assistant turns. Seeing one on a
            # user turn means malformed input from the client.
            raise TranslationError(
                f"tool_use block found on user message "
                f"(id={block.id!r}); tool_use belongs on assistant role"
            )
        else:  # pragma: no cover — discriminator should make this unreachable
            raise TranslationError(
                f"Unknown user content block type: {type(block).__name__}"
            )

    _flush_pending()
    return out


def _anthropic_tool_result_content_to_text(
    content: str | list[AnthropicTextBlock],
) -> str:
    """D-3: text-only tool_result. Schema restricts to str or list[TextBlock],
    so anything else (image-bearing) wouldn't have parsed in the first place
    and any new block type would raise here defensively."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, AnthropicTextBlock):
                raise TranslationError(
                    f"tool_result.content list may only contain text blocks "
                    f"in v1; got {type(block).__name__}"
                )
            parts.append(block.text)
        return "\n".join(parts)
    raise TranslationError(
        f"Unexpected tool_result.content type: {type(content).__name__}"
    )


def _anthropic_assistant_to_openai(msg: AnthropicMessage) -> AssistantMessage:
    """Assistant message → single AssistantMessage with optional tool_calls.

    Pure-string content → AssistantMessage(content=string).
    List of blocks: text blocks concatenate with '\\n' into content;
    tool_use blocks accumulate into tool_calls. If no text blocks present,
    content is None (OpenAI's idiom for tool-call-only assistant turns).
    """
    if isinstance(msg.content, str):
        return AssistantMessage(role="assistant", content=msg.content)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in msg.content:
        if isinstance(block, AnthropicTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, AnthropicToolUseBlock):
            tool_calls.append(ToolCall(
                id=block.id,
                type="function",
                function=FunctionCall(
                    name=block.name,
                    arguments=json.dumps(block.input),
                ),
            ))
        elif isinstance(block, AnthropicImageBlock):
            raise TranslationError(
                "Assistant messages must not contain image blocks"
            )
        elif isinstance(block, AnthropicToolResultBlock):
            raise TranslationError(
                f"tool_result block found on assistant message "
                f"(tool_use_id={block.tool_use_id!r}); tool_results belong on user role"
            )
        else:  # pragma: no cover
            raise TranslationError(
                f"Unknown assistant content block type: {type(block).__name__}"
            )

    content: str | None = "\n".join(text_parts) if text_parts else None
    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls if tool_calls else None,
    )


def _anthropic_tool_def_to_openai(tool: AnthropicToolDef) -> Tool:
    """Anthropic tool def → OpenAI Tool(function=FunctionDef)."""
    return Tool(
        type="function",
        function=FunctionDef(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        ),
    )


def _anthropic_tool_choice_to_openai(
    choice: AnthropicToolChoice,
) -> str | dict[str, Any]:
    """Anthropic tool_choice variants → OpenAI tool_choice shape.

      {type:"auto"}            → "auto"
      {type:"any"}             → "required"
      {type:"none"}            → "none"
      {type:"tool", name:"X"}  → {"type":"function","function":{"name":"X"}}
    """
    if isinstance(choice, AnthropicToolChoiceAuto):
        return "auto"
    if isinstance(choice, AnthropicToolChoiceAny):
        return "required"
    if isinstance(choice, AnthropicToolChoiceNone):
        return "none"
    if isinstance(choice, AnthropicToolChoiceSpecific):
        return {"type": "function", "function": {"name": choice.name}}
    # pragma: no cover — discriminator covers all variants
    raise TranslationError(
        f"Unknown tool_choice variant: {type(choice).__name__}"
    )


# ---------------------------------------------------------------------------
# Response: OpenAI → Anthropic  (feeds the /v1/messages route)
# ---------------------------------------------------------------------------

# Reverse of _STOP_REASON_MAP. Defined explicitly rather than computed by
# inversion because the original map is many-to-one (both end_turn and
# stop_sequence map to "stop"); the forward map resolves the ambiguity.
_OPENAI_FINISH_TO_ANTHROPIC_STOP: dict[FinishReason, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    # OpenAI's legacy "function_call" finish reason predates the tool_calls
    # API; treat it like tool_use semantically.
    "function_call": "tool_use",
}


def to_anthropic_response(
    completion: ChatCompletion,
    request_model: str,
) -> AnthropicMessagesResponse:
    """Translate an internal ChatCompletion → Anthropic Messages response.

    Reverse of `from_anthropic_response`. See translation table in
    docs/design-anthropic-messages-api.md.

    `request_model` is the model string from the original Anthropic request.
    We echo it back rather than `completion.model` because the internal
    pipeline may have rewritten the model via `_apply_model_override`
    (capability routing) — the client should see the model they asked for,
    not the routing target.

    Anthropic responses model a single message. If the underlying
    completion has multiple choices (OpenAI n>1), we use choice[0] only.
    Coding-tool traffic never sets n>1 in practice.
    """
    if not completion.choices:
        raise TranslationError("ChatCompletion has no choices to translate")
    choice = completion.choices[0]
    msg = choice.message

    content_blocks: list[AnthropicResponseContentBlock] = []

    # Text block — only emitted if the assistant produced text content.
    # An empty string is still "produced text" and we honor it; None means
    # the assistant produced only tool calls.
    if msg.content is not None and msg.content != "":
        content_blocks.append(AnthropicTextBlock(type="text", text=msg.content))

    # Tool-use blocks — one per OpenAI tool_call, preserving order.
    if msg.tool_calls:
        for tc in msg.tool_calls:
            content_blocks.append(_tool_call_to_anthropic_block(tc))

    # Anthropic responses must have at least one content block. If the
    # assistant produced neither text nor tool calls (edge case: empty
    # completion), emit a single empty text block so the shape is valid.
    if not content_blocks:
        content_blocks.append(AnthropicTextBlock(type="text", text=""))

    # finish_reason → stop_reason. None → "end_turn" (the most common
    # natural-completion signal); anything unrecognized → "end_turn" too.
    stop_reason: AnthropicStopReason = "end_turn"
    if choice.finish_reason is not None:
        stop_reason = _OPENAI_FINISH_TO_ANTHROPIC_STOP.get(
            choice.finish_reason, "end_turn",
        )

    # Usage. ChatCompletion.usage may be None on some adapter paths;
    # fall back to zeros so the response shape stays valid.
    usage = completion.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    # Anthropic cache stats — surfaced from prompt_tokens_details if the
    # upstream populated them (only Anthropic-routed requests, today).
    # Without this propagation, customers can't verify their
    # `cache_control` markers are taking effect even though they ARE
    # getting the discount on Anthropic's bill.
    cache_write: int | None = None
    cache_read: int | None = None
    if usage and usage.prompt_tokens_details is not None:
        cache_write = usage.prompt_tokens_details.cache_creation_input_tokens
        cache_read = usage.prompt_tokens_details.cache_read_input_tokens

    # ID: pass through if it already looks Anthropic-shaped (msg_*);
    # otherwise generate a fresh msg_* id. Preserves upstream id when the
    # request was actually routed to Anthropic (so observability tools can
    # correlate); rewrites for local adapters that emit chatcmpl-*.
    msg_id = completion.id
    if not msg_id.startswith("msg_"):
        msg_id = f"msg_{uuid4().hex[:24]}"

    return AnthropicMessagesResponse(
        id=msg_id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=request_model,
        stop_reason=stop_reason,
        # We can't determine which stop sequence matched from the
        # OpenAI-shape finish_reason alone (it's just "stop"). Leave null;
        # documented as known minor limitation in the design doc.
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
    )


def _tool_call_to_anthropic_block(tc: ToolCall) -> AnthropicToolUseBlock:
    """Translate one OpenAI ToolCall → Anthropic tool_use block.

    `tool_call.function.arguments` is a JSON-encoded string in OpenAI's
    schema; Anthropic's `input` is a parsed object. On malformed JSON we
    fall back to wrapping the raw string in `_raw_arguments` — symmetric
    with `_translate_assistant_content` going the other direction. The
    client (Claude Code) sees structured input either way.
    """
    try:
        input_dict = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except json.JSONDecodeError:
        input_dict = {"_raw_arguments": tc.function.arguments}
    return AnthropicToolUseBlock(
        type="tool_use",
        id=tc.id,
        name=tc.function.name,
        input=input_dict,
    )


# ---------------------------------------------------------------------------
# Streaming: OpenAI chunks → Anthropic SSE events
# ---------------------------------------------------------------------------

class OpenAIToAnthropicStreamTranslator:
    """State machine: OpenAI ChatCompletionChunk stream → Anthropic SSE events.

    Reverse direction of `AnthropicStreamTranslator`. Used by the
    /v1/messages streaming path to translate the gateway's internal
    streaming format into Anthropic's wire-format event sequence.

    Usage:
        translator = OpenAIToAnthropicStreamTranslator(
            request_model="claude-haiku-4-5-20251001",
            input_tokens=42,  # pre-counted by the route from the request body
        )
        async for chunk in upstream_aiter:
            for event in translator.translate_chunk(chunk):
                yield format_anthropic_sse(event)
        for event in translator.finalize():
            yield format_anthropic_sse(event)

    Event ordering invariants (must match Anthropic spec):
        message_start
        (for each content block, in order it appeared:
            content_block_start
            content_block_delta+
            content_block_stop)
        message_delta   (carries stop_reason + final output_tokens)
        message_stop

    Block-switching rules:
        - Text deltas all accumulate in a single open text block UNTIL
          interrupted by a tool_use delta with an unseen OpenAI tool_index.
        - The first tool_use delta with a given OpenAI index closes the
          previous block and opens a new tool_use block.
        - Continuation deltas (same OpenAI index, no `id`/`name`) emit
          `input_json_delta` events on the existing tool_use block.
        - We do NOT support reopening a closed block — Anthropic spec
          forbids it. OpenAI doesn't interleave parallel tool-call argument
          streams in practice, so this constraint is satisfiable.

    `input_tokens` is reported in `message_start` and is fixed for the
    response. Pass a pre-computed count from the route; if unknown, 0 is
    acceptable (real Anthropic clients tolerate it and rely on
    message_delta usage at the end for final totals).
    """

    def __init__(self, request_model: str, input_tokens: int = 0) -> None:
        self.request_model = request_model
        self.input_tokens = input_tokens

        self._message_started = False
        self._message_id: str = ""

        # Anthropic block index counter (independent of OpenAI tool_call.index).
        self._next_block_index = 0
        # The currently open block, if any. None means no block is open.
        self._current_block_index: int | None = None
        self._current_block_kind: str | None = None  # "text" | "tool_use" | None

        # Map OpenAI tool_call.index → assigned Anthropic block_index.
        # Lets continuation deltas (same OpenAI index, no id/name) find their
        # already-opened tool_use block.
        self._openai_tool_idx_to_block: dict[int, int] = {}

        # Accumulated state, emitted at finalize().
        self._finish_reason: FinishReason | None = None
        self._final_output_tokens: int = 0
        # Anthropic cache stats, captured from the first chunk that carries them
        # (they ride in prompt_tokens_details) and emitted in message_start.
        self._cache_creation: int | None = None
        self._cache_read: int | None = None
        # Whether any content_block was ever opened. If still False at finalize,
        # emit a single empty text block so Anthropic's "≥1 block per response"
        # invariant holds.
        self._any_block_opened = False

    # -- public API --------------------------------------------------------

    def translate_chunk(self, chunk: ChatCompletionChunk) -> list[AnthropicStreamEvent]:
        """Process one OpenAI chunk; return 0+ Anthropic events to emit."""
        events: list[AnthropicStreamEvent] = []

        # Capture usage (output tokens + Anthropic cache stats) BEFORE emitting
        # message_start: the cache counts ride on the first chunk, and
        # message_start is emitted on that same chunk, so they must be read
        # first to make it into the event.
        if chunk.usage is not None:
            self._final_output_tokens = chunk.usage.completion_tokens
            details = chunk.usage.prompt_tokens_details
            if details is not None:
                if details.cache_creation_input_tokens is not None:
                    self._cache_creation = details.cache_creation_input_tokens
                if details.cache_read_input_tokens is not None:
                    self._cache_read = details.cache_read_input_tokens

        # message_start (exactly once, on the first chunk).
        if not self._message_started:
            self._message_started = True
            self._message_id = self._normalize_id(chunk.id)
            events.append(self._make_message_start())

        # Pull finish_reason from the chunk regardless of choices shape.
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta

            # Text content delta.
            if delta.content:
                events.extend(self._emit_text_delta(delta.content))

            # Tool-call deltas (may be multiple, one per parallel tool index).
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    events.extend(self._emit_tool_call_delta(tc_delta))

            if choice.finish_reason is not None:
                self._finish_reason = choice.finish_reason

        return events

    def finalize(self) -> list[AnthropicStreamEvent]:
        """Emit closing events: content_block_stop (if a block is open),
        message_delta, message_stop. Idempotent — calling twice is a bug,
        but safe (returns empty list on the second call)."""
        if not self._message_started:
            # Stream was empty (no chunks ever processed). Emit a synthetic
            # message_start so the SSE sequence is still well-formed.
            self._message_started = True
            self._message_id = f"msg_{uuid4().hex[:24]}"
            yielded_message_start: list[AnthropicStreamEvent] = [self._make_message_start()]
        else:
            yielded_message_start = []

        events: list[AnthropicStreamEvent] = list(yielded_message_start)

        # Close any currently open block.
        if self._current_block_index is not None:
            events.append(self._content_block_stop(self._current_block_index))
            self._current_block_index = None
            self._current_block_kind = None

        # If no block was ever opened (truly empty response: no text, no
        # tool calls), open + close an empty text block. Anthropic spec
        # requires ≥1 content block per response.
        if not self._any_block_opened:
            events.append(self._content_block_start_text(0))
            events.append(self._content_block_stop(0))
            self._next_block_index = 1
            self._any_block_opened = True

        # message_delta: stop_reason + final output_tokens.
        stop_reason: AnthropicStopReason = "end_turn"
        if self._finish_reason is not None:
            stop_reason = _OPENAI_FINISH_TO_ANTHROPIC_STOP.get(
                self._finish_reason, "end_turn",
            )
        events.append(AnthropicMessageDeltaEvent(
            type="message_delta",
            delta=_MessageDelta(stop_reason=stop_reason, stop_sequence=None),
            usage=_MessageDeltaUsage(output_tokens=self._final_output_tokens),
        ))
        events.append(AnthropicMessageStopEvent(type="message_stop"))

        return events

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _normalize_id(openai_id: str) -> str:
        """Pass through msg_* ids; rewrite anything else to msg_<uuid>.
        Same rule as `to_anthropic_response`."""
        if openai_id.startswith("msg_"):
            return openai_id
        return f"msg_{uuid4().hex[:24]}"

    def _make_message_start(self) -> AnthropicMessageStartEvent:
        return AnthropicMessageStartEvent(
            type="message_start",
            message=_MessageStartShell(
                id=self._message_id,
                type="message",
                role="assistant",
                content=[],
                model=self.request_model,
                stop_reason=None,
                stop_sequence=None,
                usage=AnthropicUsage(
                    input_tokens=self.input_tokens,
                    output_tokens=0,
                    cache_creation_input_tokens=self._cache_creation,
                    cache_read_input_tokens=self._cache_read,
                ),
            ),
        )

    def _emit_text_delta(self, text: str) -> list[AnthropicStreamEvent]:
        """Emit a text delta. If no text block is currently open, close any
        open block first and open a new text block."""
        events: list[AnthropicStreamEvent] = []

        if self._current_block_kind != "text":
            # Close current block (if any).
            if self._current_block_index is not None:
                events.append(self._content_block_stop(self._current_block_index))
            # Open new text block.
            block_idx = self._next_block_index
            self._next_block_index += 1
            self._current_block_index = block_idx
            self._current_block_kind = "text"
            self._any_block_opened = True
            events.append(self._content_block_start_text(block_idx))

        # Emit the delta on the (now-open) text block.
        assert self._current_block_index is not None
        events.append(AnthropicContentBlockDeltaEvent(
            type="content_block_delta",
            index=self._current_block_index,
            delta=_TextDelta(type="text_delta", text=text),
        ))
        return events

    def _emit_tool_call_delta(self, tc_delta: Any) -> list[AnthropicStreamEvent]:
        """Emit events for one OpenAI ToolCallDelta. Handles three sub-cases:

          1. First-time delta for a new OpenAI tool_index: close current
             block, open a new tool_use block (id + name come on this delta),
             optionally emit any partial arguments.
          2. Continuation delta for a tool_index we've already opened:
             emit input_json_delta on the existing block.
          3. Delta carrying only the tool_call header (no arguments yet):
             open the block but emit no input_json_delta yet.
        """
        events: list[AnthropicStreamEvent] = []
        openai_idx = tc_delta.index

        if openai_idx not in self._openai_tool_idx_to_block:
            # First time we see this tool_index — open a new tool_use block.
            if self._current_block_index is not None:
                events.append(self._content_block_stop(self._current_block_index))

            block_idx = self._next_block_index
            self._next_block_index += 1
            self._openai_tool_idx_to_block[openai_idx] = block_idx
            self._current_block_index = block_idx
            self._current_block_kind = "tool_use"
            self._any_block_opened = True

            # `id` and `name` arrive on the first delta for this tool_index.
            # If a local adapter omits the id, synthesize one (Claude Code
            # requires non-empty id on tool_use blocks).
            tc_id = tc_delta.id or f"toolu_{uuid4().hex[:16]}"
            tc_name = ""
            if tc_delta.function is not None and tc_delta.function.name:
                tc_name = tc_delta.function.name

            events.append(self._content_block_start_tool_use(
                block_idx, tc_id, tc_name,
            ))
        else:
            # Continuation delta — block is already open.
            block_idx = self._openai_tool_idx_to_block[openai_idx]
            # If the current open block is something else (shouldn't happen in
            # practice — OpenAI doesn't interleave parallel tool argument
            # streams), silently treat as a continuation. The Anthropic client
            # will receive input_json_delta on a closed block, which would be
            # malformed — but real OpenAI streams don't trigger this branch.

        # Emit arguments delta if there's any payload.
        if tc_delta.function is not None and tc_delta.function.arguments:
            events.append(AnthropicContentBlockDeltaEvent(
                type="content_block_delta",
                index=self._openai_tool_idx_to_block[openai_idx],
                delta=_InputJsonDelta(
                    type="input_json_delta",
                    partial_json=tc_delta.function.arguments,
                ),
            ))

        return events

    @staticmethod
    def _content_block_start_text(block_idx: int) -> AnthropicContentBlockStartEvent:
        return AnthropicContentBlockStartEvent(
            type="content_block_start",
            index=block_idx,
            content_block=_ContentBlockStartTextShell(type="text", text=""),
        )

    @staticmethod
    def _content_block_start_tool_use(
        block_idx: int, tc_id: str, tc_name: str,
    ) -> AnthropicContentBlockStartEvent:
        return AnthropicContentBlockStartEvent(
            type="content_block_start",
            index=block_idx,
            content_block=_ContentBlockStartToolUseShell(
                type="tool_use", id=tc_id, name=tc_name, input={},
            ),
        )

    @staticmethod
    def _content_block_stop(block_idx: int) -> AnthropicContentBlockStopEvent:
        return AnthropicContentBlockStopEvent(
            type="content_block_stop",
            index=block_idx,
        )


def format_anthropic_sse(event: AnthropicStreamEvent) -> str:
    """Format one Anthropic stream event as an SSE message.

    Anthropic SSE uses `event: <type>` + `data: <json>` lines, separated
    by a blank line. This differs from OpenAI's SSE format (which only
    uses `data: <json>` with no `event:` line + a final `data: [DONE]`).

    Example output:
        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}
        <blank line>
    """
    # The pydantic discriminator-union machinery requires a model_dump()
    # path that picks the right concrete shape per event type. Use
    # model_dump_json with exclude_none so we don't emit `null`s where
    # the Anthropic spec expects fields to be absent.
    payload = event.model_dump_json(exclude_none=True)
    return f"event: {event.type}\ndata: {payload}\n\n"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Translation between the OpenAI Responses API and our internal Chat shape.

Inbound  (`from_responses_request`): Responses request → ChatCompletionRequest.
    `instructions` becomes a system message; `input` items become role/content
    messages; Responses function tools are rewritten to Chat tool shape.

Outbound (`to_responses_response`): ChatCompletion → Responses result. The
    assistant's text becomes a `message` output item; each tool call becomes a
    `function_call` output item (Responses surfaces tool calls in the output
    array, not as a block on an assistant message).

Phase 1 is non-streaming. The Responses SSE event stream is a separate follow-up.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionRequest,
    FunctionCall,
    FunctionDef,
    Message,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from modelmeld.api.schemas_responses import (
    ResponsesContentPart,
    ResponsesFunctionCallItem,
    ResponsesInputItem,
    ResponsesMessageItem,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)

# ---------------------------------------------------------------------------
# Inbound: Responses request → ChatCompletionRequest
# ---------------------------------------------------------------------------

def _item_text(content: str | list[ResponsesContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if part.text)


def _output_text(output: Any) -> str:
    """A `function_call_output` carries its result in `output`, which may be a
    plain string, a list of typed parts, or an object — normalize to text."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(
            (p.get("text") or "") for p in output if isinstance(p, dict)
        )
    if isinstance(output, dict):
        return output.get("text") or output.get("output") or ""
    return str(output)


def _to_message(role: str, text: str) -> Message:
    # `developer` is the Responses-era rename of the system role.
    if role in ("system", "developer"):
        return SystemMessage(role="system", content=text)
    if role == "assistant":
        return AssistantMessage(role="assistant", content=text)
    return UserMessage(role="user", content=text)


def _input_item_to_message(item: ResponsesInputItem) -> Message | None:
    """Map one heterogeneous `input` item to a chat message, or None to skip.

    Multi-turn Responses clients replay the full transcript, so the array mixes
    role/content `message` items with `function_call` (assistant tool call),
    `function_call_output` (tool result), and `reasoning` items."""
    if item.type == "function_call":
        # Assistant's prior tool invocation → assistant msg carrying the call.
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(
                id=item.call_id or "",
                type="function",
                function=FunctionCall(
                    name=item.name or "", arguments=item.arguments or "",
                ),
            )],
        )
    if item.type == "function_call_output":
        # Tool result fed back in → tool-role message keyed by call_id.
        return ToolMessage(
            role="tool",
            tool_call_id=item.call_id or "",
            content=_output_text(item.output),
        )
    # `reasoning` (and any other non-message type without a role) has no chat
    # equivalent — drop it rather than 422.
    if item.role is None:
        return None
    return _to_message(item.role, _item_text(item.content))


def _to_chat_tools(tools: list[dict[str, Any]] | None) -> list[Tool] | None:
    """Responses function tools are flat (`{type, name, parameters, …}`); Chat
    nests them under `function`. Translate function tools; drop other tool types
    (e.g. built-in `web_search`) which have no Chat-Completions equivalent."""
    if not tools:
        return None
    out: list[Tool] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        # Flat Responses shape or already-nested Chat shape.
        fn = tool.get("function") or tool
        name = fn.get("name")
        if not name:
            continue
        out.append(Tool(
            type="function",
            function=FunctionDef(
                name=name,
                description=fn.get("description"),
                parameters=fn.get("parameters"),
                strict=fn.get("strict"),
            ),
        ))
    return out or None


def from_responses_request(req: ResponsesRequest) -> ChatCompletionRequest:
    messages: list[Message] = []
    if req.instructions:
        messages.append(SystemMessage(role="system", content=req.instructions))
    if isinstance(req.input, str):
        messages.append(UserMessage(role="user", content=req.input))
    else:
        for item in req.input:
            msg = _input_item_to_message(item)
            if msg is not None:
                messages.append(msg)
    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_output_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tools=_to_chat_tools(req.tools),
        tool_choice=req.tool_choice,
        stream=req.stream,
    )


# ---------------------------------------------------------------------------
# Outbound: ChatCompletion → Responses result
# ---------------------------------------------------------------------------

def to_responses_response(completion: ChatCompletion, model: str) -> ResponsesResponse:
    output: list[ResponsesMessageItem | ResponsesFunctionCallItem] = []
    if completion.choices:
        msg = completion.choices[0].message
        if msg.content:
            output.append(ResponsesMessageItem(
                id=f"msg_{uuid4().hex[:24]}",
                content=[ResponsesOutputText(text=msg.content)],
            ))
        for tc in msg.tool_calls or []:
            output.append(ResponsesFunctionCallItem(
                id=f"fc_{uuid4().hex[:24]}",
                call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            ))
    # Responses guarantees ≥1 output item; emit an empty message if the
    # completion produced neither text nor tool calls.
    if not output:
        output.append(ResponsesMessageItem(
            id=f"msg_{uuid4().hex[:24]}",
            content=[ResponsesOutputText(text="")],
        ))

    usage = completion.usage
    return ResponsesResponse(
        id=completion.id or f"resp_{uuid4().hex[:24]}",
        created_at=int(time.time()),
        model=model,
        output=output,
        usage=ResponsesUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
    )

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
    Message,
    SystemMessage,
    UserMessage,
)
from modelmeld.api.schemas_responses import (
    ResponsesContentPart,
    ResponsesFunctionCallItem,
    ResponsesMessageItem,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)

# ---------------------------------------------------------------------------
# Inbound: Responses request → ChatCompletionRequest
# ---------------------------------------------------------------------------

def _item_text(content: str | list[ResponsesContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if part.text)


def _to_message(role: str, text: str) -> Message:
    # `developer` is the Responses-era rename of the system role.
    if role in ("system", "developer"):
        return SystemMessage(role="system", content=text)
    if role == "assistant":
        return AssistantMessage(role="assistant", content=text)
    return UserMessage(role="user", content=text)


def _to_chat_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Responses function tools are flat (`{type, name, parameters, …}`); Chat
    nests them under `function`. Translate function tools; drop other tool types
    (e.g. built-in `web_search`) which have no Chat-Completions equivalent."""
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") or {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
        }
        out.append({
            "type": "function",
            "function": {k: v for k, v in fn.items() if v is not None},
        })
    return out or None


def from_responses_request(req: ResponsesRequest) -> ChatCompletionRequest:
    messages: list[Message] = []
    if req.instructions:
        messages.append(SystemMessage(role="system", content=req.instructions))
    if isinstance(req.input, str):
        messages.append(UserMessage(role="user", content=req.input))
    else:
        for item in req.input:
            messages.append(_to_message(item.role, _item_text(item.content)))
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

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Responses API ↔ internal Chat translation (Phase 1, non-streaming)."""

from __future__ import annotations

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    Choice,
    FunctionCall,
    ResponseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from modelmeld.api.schemas_responses import ResponsesRequest
from modelmeld.translation.openai_responses import (
    from_responses_request,
    to_responses_response,
)

# ---------------------------------------------------------------------------
# Inbound: Responses request → ChatCompletionRequest
# ---------------------------------------------------------------------------

def test_string_input_becomes_single_user_message() -> None:
    req = ResponsesRequest(model="anthropic/modelmeld-auto", input="hello there")
    out = from_responses_request(req)
    assert out.model == "anthropic/modelmeld-auto"
    assert len(out.messages) == 1
    assert isinstance(out.messages[0], UserMessage)
    assert out.messages[0].content == "hello there"


def test_instructions_become_system_message_and_roles_map() -> None:
    req = ResponsesRequest(
        model="m",
        instructions="You are terse.",
        input=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "developer", "content": "be careful"},  # developer → system
        ],
    )
    out = from_responses_request(req)
    assert isinstance(out.messages[0], SystemMessage)
    assert out.messages[0].content == "You are terse."
    assert isinstance(out.messages[1], UserMessage)
    assert isinstance(out.messages[2], AssistantMessage)
    assert isinstance(out.messages[3], SystemMessage)  # developer mapped to system


def test_content_parts_text_is_extracted() -> None:
    req = ResponsesRequest(
        model="m",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "part one "},
            {"type": "input_text", "text": "part two"},
        ]}],
    )
    out = from_responses_request(req)
    assert out.messages[0].content == "part one part two"


def test_responses_function_tool_becomes_chat_tool_shape() -> None:
    req = ResponsesRequest(
        model="m",
        input="x",
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {}},
        }],
    )
    out = from_responses_request(req)
    assert out.tools is not None and len(out.tools) == 1
    tool = out.tools[0]
    # Chat shape nests under `function`.
    assert tool.type == "function"
    assert tool.function.name == "get_weather"


def test_non_function_tools_are_dropped() -> None:
    req = ResponsesRequest(
        model="m", input="x",
        tools=[{"type": "web_search"}],  # no Chat-Completions equivalent
    )
    out = from_responses_request(req)
    assert out.tools is None


# -- Multi-turn replay: the heterogeneous `input` array a real agentic client
#    (Codex) sends back after a tool round-trip. Regression for the 422 where
#    function_call / function_call_output / reasoning items were rejected.

def test_function_call_item_becomes_assistant_tool_call() -> None:
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "weather in SF?"}]},
            {"type": "function_call", "call_id": "call_9",
             "name": "get_weather", "arguments": '{"city":"SF"}'},
        ],
    )
    out = from_responses_request(req)
    assert isinstance(out.messages[0], UserMessage)
    assert isinstance(out.messages[1], AssistantMessage)
    tcs = out.messages[1].tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].id == "call_9"
    assert tcs[0].function.name == "get_weather"
    assert tcs[0].function.arguments == '{"city":"SF"}'


def test_function_call_output_becomes_tool_message() -> None:
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "function_call", "call_id": "call_9",
             "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_9",
             "output": "72F and sunny"},
        ],
    )
    out = from_responses_request(req)
    assert isinstance(out.messages[1], ToolMessage)
    assert out.messages[1].tool_call_id == "call_9"
    assert out.messages[1].content == "72F and sunny"


def test_function_call_output_list_form_is_flattened() -> None:
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "function_call_output", "call_id": "c1",
             "output": [{"type": "output_text", "text": "ab"},
                        {"type": "output_text", "text": "cd"}]},
        ],
    )
    out = from_responses_request(req)
    assert isinstance(out.messages[0], ToolMessage)
    assert out.messages[0].content == "abcd"


def test_reasoning_item_is_skipped_not_rejected() -> None:
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "reasoning", "summary": [], "id": "rs_1"},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "continue"}]},
        ],
    )
    out = from_responses_request(req)
    # Reasoning dropped; only the user message survives.
    assert len(out.messages) == 1
    assert isinstance(out.messages[0], UserMessage)
    assert out.messages[0].content == "continue"


def test_full_codex_replay_round_trips() -> None:
    # The exact shape that 422'd against the hosted gateway: a developer
    # message with input_text parts, a prior tool call, its output, and the
    # next user turn — all in one `input` array.
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "<permissions...>"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "list files"}]},
            {"type": "function_call", "call_id": "c1",
             "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "a.py\nb.py"},
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "now read a.py"}]},
        ],
    )
    out = from_responses_request(req)
    roles = [type(m).__name__ for m in out.messages]
    assert roles == [
        "SystemMessage",     # developer
        "UserMessage",
        "AssistantMessage",  # function_call
        "ToolMessage",       # function_call_output
        "UserMessage",       # reasoning dropped between them
    ]


# ---------------------------------------------------------------------------
# Outbound: ChatCompletion → Responses result
# ---------------------------------------------------------------------------

def _completion(message: ResponseMessage, *, model: str = "qwen3-coder-next") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc", model=model,
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


def test_text_completion_becomes_message_output_item() -> None:
    resp = to_responses_response(
        _completion(ResponseMessage(role="assistant", content="the answer")),
        model="qwen3-coder-next",
    )
    assert resp.object == "response"
    assert resp.status == "completed"
    assert resp.model == "qwen3-coder-next"
    assert len(resp.output) == 1
    item = resp.output[0]
    assert item.type == "message"
    assert item.content[0].type == "output_text"
    assert item.content[0].text == "the answer"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 7
    assert resp.usage.total_tokens == 18


def test_tool_calls_become_function_call_items() -> None:
    msg = ResponseMessage(
        role="assistant", content=None,
        tool_calls=[ToolCall(
            id="call_1", type="function",
            function=FunctionCall(name="get_weather", arguments='{"city":"SF"}'),
        )],
    )
    resp = to_responses_response(_completion(msg), model="m")
    fcs = [o for o in resp.output if o.type == "function_call"]
    assert len(fcs) == 1
    assert fcs[0].name == "get_weather"
    assert fcs[0].arguments == '{"city":"SF"}'
    assert fcs[0].call_id == "call_1"


def test_empty_completion_emits_one_empty_message_item() -> None:
    resp = to_responses_response(
        _completion(ResponseMessage(role="assistant", content=None)),
        model="m",
    )
    assert len(resp.output) == 1
    assert resp.output[0].type == "message"
    assert resp.output[0].content[0].text == ""

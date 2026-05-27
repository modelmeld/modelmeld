# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for `to_anthropic_response`.

Validates internal OpenAI ChatCompletion → Anthropic Messages response
translation. Pairs with `from_anthropic_request` (chunk 2); together
they form the non-streaming HTTP boundary for the upcoming /v1/messages
route.
"""

from __future__ import annotations

import json

import pytest

from modelmeld.api.schemas import (
    ChatCompletion,
    Choice,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    Usage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicToolUseBlock,
)
from modelmeld.translation import TranslationError, to_anthropic_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completion(
    *,
    id: str = "chatcmpl-abc123",
    model: str = "internal-model",
    content: str | None = "Hi.",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str | None = "stop",
    usage: Usage | None = None,
) -> ChatCompletion:
    """Build a minimal ChatCompletion for testing."""
    return ChatCompletion(
        id=id,
        model=model,
        choices=[Choice(
            index=0,
            message=ResponseMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,  # type: ignore[arg-type]
        )],
        usage=usage or Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ---------------------------------------------------------------------------
# Top-level fields
# ---------------------------------------------------------------------------

def test_basic_text_response_translates_to_single_text_block() -> None:
    c = _completion(content="Hello world.")
    r = to_anthropic_response(c, request_model="claude-haiku-4-5-20251001")
    assert isinstance(r, AnthropicMessagesResponse)
    assert r.type == "message"
    assert r.role == "assistant"
    assert len(r.content) == 1
    block = r.content[0]
    assert isinstance(block, AnthropicTextBlock)
    assert block.type == "text"
    assert block.text == "Hello world."


def test_response_echoes_request_model_not_internal_model() -> None:
    """Capability routing may rewrite internal completion.model; the client
    should see the model they requested."""
    c = _completion(model="vllm/qwen-7b")
    r = to_anthropic_response(c, request_model="claude-haiku-4-5-20251001")
    assert r.model == "claude-haiku-4-5-20251001"


def test_usage_translation() -> None:
    c = _completion(usage=Usage(prompt_tokens=123, completion_tokens=45, total_tokens=168))
    r = to_anthropic_response(c, request_model="m")
    assert r.usage.input_tokens == 123
    assert r.usage.output_tokens == 45


def test_missing_usage_yields_zero_tokens() -> None:
    """Some adapter paths may emit ChatCompletion with usage=None.
    Translation falls back to zeros — shape stays valid."""
    c = ChatCompletion(
        id="chatcmpl-abc",
        model="m",
        choices=[Choice(
            index=0,
            message=ResponseMessage(content="x"),
            finish_reason="stop",
        )],
        usage=None,
    )
    r = to_anthropic_response(c, request_model="m")
    assert r.usage.input_tokens == 0
    assert r.usage.output_tokens == 0


def test_stop_sequence_field_is_null_per_design() -> None:
    """OpenAI finish_reason='stop' doesn't say which stop sequence matched.
    Documented limitation: we leave stop_sequence=None."""
    c = _completion(finish_reason="stop")
    r = to_anthropic_response(c, request_model="m")
    assert r.stop_sequence is None


# ---------------------------------------------------------------------------
# ID handling
# ---------------------------------------------------------------------------

def test_anthropic_shaped_id_passes_through() -> None:
    """When the upstream id already looks Anthropic-shaped (msg_*), preserve
    it — useful for observability correlation when routing TO Anthropic."""
    c = _completion(id="msg_anthropic_upstream_id_xyz")
    r = to_anthropic_response(c, request_model="m")
    assert r.id == "msg_anthropic_upstream_id_xyz"


def test_openai_shaped_id_gets_rewritten_to_anthropic_shape() -> None:
    """When the id is OpenAI-shaped (chatcmpl-*), generate a fresh msg_*.
    Local adapters (vLLM openai-compat) emit chatcmpl-*; Claude Code expects
    msg_* idiomatically."""
    c = _completion(id="chatcmpl-vllm-local-9876")
    r = to_anthropic_response(c, request_model="m")
    assert r.id.startswith("msg_")
    assert r.id != "chatcmpl-vllm-local-9876"


def test_arbitrary_id_format_gets_normalized() -> None:
    """Any non-msg_* prefix triggers regeneration."""
    c = _completion(id="custom-id-7")
    r = to_anthropic_response(c, request_model="m")
    assert r.id.startswith("msg_")


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("openai_finish", "expected_stop"), [
    ("stop", "end_turn"),
    ("length", "max_tokens"),
    ("tool_calls", "tool_use"),
    ("content_filter", "refusal"),
    ("function_call", "tool_use"),  # legacy openai field, semantically equivalent
])
def test_finish_reason_maps_correctly(openai_finish: str, expected_stop: str) -> None:
    c = _completion(finish_reason=openai_finish)
    r = to_anthropic_response(c, request_model="m")
    assert r.stop_reason == expected_stop


def test_none_finish_reason_defaults_to_end_turn() -> None:
    """Some adapter paths leave finish_reason=None; default to end_turn
    rather than producing a null stop_reason (which would confuse clients)."""
    c = _completion(finish_reason=None)
    r = to_anthropic_response(c, request_model="m")
    assert r.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Tool call translation
# ---------------------------------------------------------------------------

def test_assistant_with_text_and_one_tool_call_emits_two_blocks() -> None:
    c = _completion(
        content="Let me check that.",
        tool_calls=[ToolCall(
            id="tu_1",
            type="function",
            function=FunctionCall(
                name="read_file",
                arguments='{"path": "/tmp/x.txt"}',
            ),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    assert len(r.content) == 2
    assert isinstance(r.content[0], AnthropicTextBlock)
    assert r.content[0].text == "Let me check that."
    assert isinstance(r.content[1], AnthropicToolUseBlock)
    assert r.content[1].id == "tu_1"
    assert r.content[1].name == "read_file"
    assert r.content[1].input == {"path": "/tmp/x.txt"}
    assert r.stop_reason == "tool_use"


def test_tool_call_only_response_has_no_text_block() -> None:
    """When the LLM produces only a tool call (content=None), emit a
    tool_use block alone — no leading empty text block."""
    c = _completion(
        content=None,
        tool_calls=[ToolCall(
            id="tu_1",
            type="function",
            function=FunctionCall(name="fn", arguments="{}"),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    assert len(r.content) == 1
    assert isinstance(r.content[0], AnthropicToolUseBlock)


def test_multiple_tool_calls_become_multiple_tool_use_blocks_preserving_order() -> None:
    c = _completion(
        content=None,
        tool_calls=[
            ToolCall(id="tu_a", type="function",
                     function=FunctionCall(name="read", arguments='{"p":"a"}')),
            ToolCall(id="tu_b", type="function",
                     function=FunctionCall(name="read", arguments='{"p":"b"}')),
        ],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    assert len(r.content) == 2
    assert all(isinstance(b, AnthropicToolUseBlock) for b in r.content)
    assert [b.id for b in r.content] == ["tu_a", "tu_b"]  # type: ignore[union-attr]
    assert r.content[0].input == {"p": "a"}  # type: ignore[union-attr]
    assert r.content[1].input == {"p": "b"}  # type: ignore[union-attr]


def test_tool_call_with_empty_arguments_string_becomes_empty_object() -> None:
    """OpenAI tool_calls with arguments='' (or missing) → empty dict in
    Anthropic input. Some local models emit empty-string arguments
    rather than '{}'."""
    c = _completion(
        content=None,
        tool_calls=[ToolCall(
            id="tu_x",
            type="function",
            function=FunctionCall(name="ping", arguments=""),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    block = r.content[0]
    assert isinstance(block, AnthropicToolUseBlock)
    assert block.input == {}


def test_tool_call_with_malformed_json_arguments_falls_back_gracefully() -> None:
    """If a local adapter emits invalid JSON in arguments (real bug we want
    to survive in production), preserve the raw string under _raw_arguments
    rather than crashing. Symmetric with the OpenAI→Anthropic direction."""
    c = _completion(
        content=None,
        tool_calls=[ToolCall(
            id="tu_bad",
            type="function",
            function=FunctionCall(name="fn", arguments='{"x": NOT_VALID_JSON}'),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    block = r.content[0]
    assert isinstance(block, AnthropicToolUseBlock)
    assert block.input == {"_raw_arguments": '{"x": NOT_VALID_JSON}'}


def test_tool_call_arguments_with_nested_structures_preserved() -> None:
    nested = {
        "path": "/tmp/x.txt",
        "options": {"recursive": True, "depth": 3},
        "tags": ["a", "b", "c"],
    }
    c = _completion(
        content=None,
        tool_calls=[ToolCall(
            id="tu_1",
            type="function",
            function=FunctionCall(name="search", arguments=json.dumps(nested)),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    block = r.content[0]
    assert isinstance(block, AnthropicToolUseBlock)
    assert block.input == nested


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_content_and_no_tool_calls_emits_single_empty_text_block() -> None:
    """Anthropic responses must have ≥1 content block. If the assistant
    produced literally nothing (content=None, tool_calls=None — degenerate
    case some adapters can return on a max_tokens=0 edge), emit a single
    empty text block so the response shape stays valid."""
    c = _completion(content=None, tool_calls=None, finish_reason="length")
    r = to_anthropic_response(c, request_model="m")
    assert len(r.content) == 1
    block = r.content[0]
    assert isinstance(block, AnthropicTextBlock)
    assert block.text == ""


def test_empty_string_content_emits_no_text_block_when_tool_calls_present() -> None:
    """content='' (empty string) is semantically 'no text' when paired
    with tool calls. Don't emit a confusing empty text block alongside
    real tool_use blocks."""
    c = _completion(
        content="",
        tool_calls=[ToolCall(
            id="tu_1",
            type="function",
            function=FunctionCall(name="fn", arguments="{}"),
        )],
        finish_reason="tool_calls",
    )
    r = to_anthropic_response(c, request_model="m")
    # Only the tool_use block; no empty text block
    assert len(r.content) == 1
    assert isinstance(r.content[0], AnthropicToolUseBlock)


def test_no_choices_raises_translation_error() -> None:
    """Degenerate input: ChatCompletion with empty choices list. Real
    adapters shouldn't produce this, but we raise rather than emit a
    malformed Anthropic response."""
    c = ChatCompletion(id="chatcmpl-x", model="m", choices=[], usage=None)
    with pytest.raises(TranslationError, match="no choices"):
        to_anthropic_response(c, request_model="m")


def test_multiple_choices_uses_only_first() -> None:
    """OpenAI n>1 produces multiple choices; Anthropic supports only one
    response. We use choice[0] silently. Coding-tool traffic doesn't
    set n>1 in practice."""
    c = ChatCompletion(
        id="chatcmpl-x",
        model="m",
        choices=[
            Choice(index=0, message=ResponseMessage(content="first"), finish_reason="stop"),
            Choice(index=1, message=ResponseMessage(content="second"), finish_reason="stop"),
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    r = to_anthropic_response(c, request_model="m")
    assert len(r.content) == 1
    assert isinstance(r.content[0], AnthropicTextBlock)
    assert r.content[0].text == "first"


def test_full_roundtrip_response_shape() -> None:
    """Build a realistic completion and verify the full Anthropic-shape
    dict the client would receive after FastAPI serializes the model."""
    c = _completion(
        id="msg_real_anthropic_id",
        content="Done.",
        tool_calls=None,
        finish_reason="stop",
        usage=Usage(prompt_tokens=100, completion_tokens=4, total_tokens=104),
    )
    r = to_anthropic_response(c, request_model="claude-haiku-4-5-20251001")
    serialized = r.model_dump(exclude_none=True)

    assert serialized["id"] == "msg_real_anthropic_id"
    assert serialized["type"] == "message"
    assert serialized["role"] == "assistant"
    assert serialized["model"] == "claude-haiku-4-5-20251001"
    assert serialized["stop_reason"] == "end_turn"
    assert "stop_sequence" not in serialized  # excluded because None
    assert serialized["content"] == [{"type": "text", "text": "Done."}]
    assert serialized["usage"] == {"input_tokens": 100, "output_tokens": 4}

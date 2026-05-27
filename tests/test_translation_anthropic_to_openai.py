# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for `from_anthropic_request`.

Validates the Anthropic Messages → internal OpenAI ChatCompletionRequest
translation. Hand-coded inputs cover the translation table cases from
docs/design-anthropic-messages-api.md plus malformed-input edge cases.

The companion direction (OpenAI → Anthropic, used by AnthropicAdapter to
call upstream Anthropic) is tested elsewhere by the existing adapter
suite.
"""

from __future__ import annotations

import json

import pytest

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMetadata,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceAuto,
    AnthropicToolChoiceNone,
    AnthropicToolChoiceSpecific,
    AnthropicToolDef,
)
from modelmeld.translation import TranslationError, from_anthropic_request

# ---------------------------------------------------------------------------
# Top-level fields
# ---------------------------------------------------------------------------

def test_minimal_request_round_trips_through_translation() -> None:
    req = AnthropicMessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[AnthropicMessage(role="user", content="Hello")],
    )
    out = from_anthropic_request(req)

    assert isinstance(out, ChatCompletionRequest)
    assert out.model == "claude-haiku-4-5-20251001"
    assert out.max_tokens == 512
    assert out.stream is False
    assert out.tools is None
    assert out.tool_choice is None
    assert len(out.messages) == 1
    user = out.messages[0]
    assert isinstance(user, UserMessage)
    assert user.content == "Hello"


def test_top_level_scalars_pass_through() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        temperature=0.3, top_p=0.9,
        stop_sequences=["END", "DONE"],
        stream=True,
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.temperature == 0.3
    assert out.top_p == 0.9
    assert out.stop == ["END", "DONE"]
    assert out.stream is True


def test_no_stop_sequences_yields_none_not_empty_list() -> None:
    """Empty stop_sequences should map to None, not [] — the latter could
    confuse downstream adapters that treat empty list as 'never stop'."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.stop is None


def test_metadata_user_id_maps_to_openai_user_field() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        metadata=AnthropicMetadata(user_id="user-42"),
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.user == "user-42"


def test_no_metadata_means_user_is_none() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.user is None


# ---------------------------------------------------------------------------
# System prompt (D-2)
# ---------------------------------------------------------------------------

def test_system_string_becomes_leading_system_message() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        system="You are helpful.",
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 2
    sys_msg = out.messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "You are helpful."


def test_system_list_joins_with_double_newline() -> None:
    """D-2: list-of-text-blocks system → joined with '\\n\\n'."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        system=[
            {"type": "text", "text": "Be helpful."},  # type: ignore[list-item]
            {"type": "text", "text": "Be concise."},  # type: ignore[list-item]
        ],
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    sys_msg = out.messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "Be helpful.\n\nBe concise."


def test_empty_system_string_does_not_inject_system_message() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10, system="",
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    # Empty string is falsy → no SystemMessage prepended
    assert len(out.messages) == 1
    assert isinstance(out.messages[0], UserMessage)


def test_no_system_field_does_not_inject() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 1
    assert isinstance(out.messages[0], UserMessage)


def test_system_cache_control_is_ignored_but_accepted() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        system=[
            {"type": "text", "text": "Be terse.",
             "cache_control": {"type": "ephemeral"}},  # type: ignore[list-item]
        ],
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    sys_msg = out.messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "Be terse."


# ---------------------------------------------------------------------------
# User message variants
# ---------------------------------------------------------------------------

def test_user_message_pure_string_passes_through() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content="Hello world")],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 1
    msg = out.messages[0]
    assert isinstance(msg, UserMessage)
    assert msg.content == "Hello world"


def test_user_message_single_text_block_simplifies_to_string() -> None:
    """A user message that's just one text block becomes string content,
    not a list-of-one-part. Matches what real OpenAI clients expect."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "text", "text": "Hello"},  # type: ignore[list-item]
        ])],
    )
    out = from_anthropic_request(req)
    msg = out.messages[0]
    assert isinstance(msg, UserMessage)
    assert msg.content == "Hello"


def test_user_message_multiple_text_blocks_keeps_list() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "text", "text": "First"},  # type: ignore[list-item]
            {"type": "text", "text": "Second"},  # type: ignore[list-item]
        ])],
    )
    out = from_anthropic_request(req)
    msg = out.messages[0]
    assert isinstance(msg, UserMessage)
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert all(isinstance(p, TextPart) for p in msg.content)
    assert msg.content[0].text == "First"
    assert msg.content[1].text == "Second"


def test_user_message_with_image_block_raises_translation_error() -> None:
    """v1 scope: image content blocks deferred. Translation should error."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "image",  # type: ignore[list-item]
             "source": {"type": "url", "url": "https://example.com/cat.png"}},
        ])],
    )
    with pytest.raises(TranslationError, match="image"):
        from_anthropic_request(req)


# ---------------------------------------------------------------------------
# Tool use round-trip on multi-turn (Claude Code's critical path)
# ---------------------------------------------------------------------------

def test_assistant_message_with_text_and_tool_use_combines_correctly() -> None:
    """An Anthropic assistant message with both text and tool_use blocks
    becomes ONE OpenAI AssistantMessage with content + tool_calls."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "text", "text": "Let me check the file."},  # type: ignore[list-item]
            {"type": "tool_use", "id": "tu_1",  # type: ignore[list-item]
             "name": "read_file", "input": {"path": "/tmp/x.txt"}},
        ])],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 1
    asst = out.messages[0]
    assert isinstance(asst, AssistantMessage)
    assert asst.content == "Let me check the file."
    assert asst.tool_calls is not None and len(asst.tool_calls) == 1
    tc = asst.tool_calls[0]
    assert tc.id == "tu_1"
    assert tc.type == "function"
    assert tc.function.name == "read_file"
    assert json.loads(tc.function.arguments) == {"path": "/tmp/x.txt"}


def test_assistant_message_with_only_tool_use_has_no_content() -> None:
    """OpenAI idiom: assistant message that's purely a tool call has
    content=None and tool_calls populated."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "tool_use", "id": "tu_x",  # type: ignore[list-item]
             "name": "fn", "input": {}},
        ])],
    )
    out = from_anthropic_request(req)
    asst = out.messages[0]
    assert isinstance(asst, AssistantMessage)
    assert asst.content is None
    assert asst.tool_calls is not None and len(asst.tool_calls) == 1


def test_assistant_message_with_multiple_tool_uses_makes_multiple_tool_calls() -> None:
    """Parallel tool-call behavior: multiple tool_use blocks → multiple tool_calls."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "tool_use", "id": "tu_a",  # type: ignore[list-item]
             "name": "read_file", "input": {"path": "a"}},
            {"type": "tool_use", "id": "tu_b",  # type: ignore[list-item]
             "name": "read_file", "input": {"path": "b"}},
        ])],
    )
    out = from_anthropic_request(req)
    asst = out.messages[0]
    assert isinstance(asst, AssistantMessage)
    assert asst.tool_calls is not None and len(asst.tool_calls) == 2
    assert [tc.id for tc in asst.tool_calls] == ["tu_a", "tu_b"]


def test_assistant_message_tool_use_with_empty_input_serializes_as_empty_object() -> None:
    """tool_use.input={} → tool_calls[].function.arguments='{}' (valid JSON)."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "tool_use", "id": "tu_1",  # type: ignore[list-item]
             "name": "ping", "input": {}},
        ])],
    )
    out = from_anthropic_request(req)
    asst = out.messages[0]
    assert isinstance(asst, AssistantMessage)
    assert asst.tool_calls is not None
    assert asst.tool_calls[0].function.arguments == "{}"


def test_assistant_message_pure_string_passes_through() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content="Just text.")],
    )
    out = from_anthropic_request(req)
    asst = out.messages[0]
    assert isinstance(asst, AssistantMessage)
    assert asst.content == "Just text."
    assert asst.tool_calls is None


def test_assistant_message_with_image_block_raises() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "image",  # type: ignore[list-item]
             "source": {"type": "url", "url": "https://example.com/cat.png"}},
        ])],
    )
    with pytest.raises(TranslationError, match="image"):
        from_anthropic_request(req)


def test_assistant_message_with_tool_result_block_raises() -> None:
    """tool_results belong on user role; finding one on assistant is malformed."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="assistant", content=[
            {"type": "tool_result",  # type: ignore[list-item]
             "tool_use_id": "tu_x", "content": "result"},
        ])],
    )
    with pytest.raises(TranslationError, match="tool_result"):
        from_anthropic_request(req)


# ---------------------------------------------------------------------------
# Tool results on user role (D-3)
# ---------------------------------------------------------------------------

def test_user_message_with_tool_result_becomes_tool_message() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "tool_result",  # type: ignore[list-item]
             "tool_use_id": "tu_1", "content": "file contents here"},
        ])],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 1
    tool_msg = out.messages[0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "tu_1"
    assert tool_msg.content == "file contents here"


def test_user_message_with_tool_result_list_content_joins() -> None:
    """D-3: tool_result.content as list of text blocks joins with \\n."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "tool_result", "tool_use_id": "tu_1",  # type: ignore[list-item]
             "content": [
                 {"type": "text", "text": "line 1"},
                 {"type": "text", "text": "line 2"},
             ]},
        ])],
    )
    out = from_anthropic_request(req)
    tool_msg = out.messages[0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.content == "line 1\nline 2"


def test_user_message_mixed_tool_result_and_text_splits_into_multiple_messages() -> None:
    """One Anthropic user message → potentially multiple OpenAI messages
    when there's a mix of tool_result and text blocks. Order preserved."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "tool_result",  # type: ignore[list-item]
             "tool_use_id": "tu_a", "content": "result a"},
            {"type": "tool_result",  # type: ignore[list-item]
             "tool_use_id": "tu_b", "content": "result b"},
            {"type": "text", "text": "Now continue."},  # type: ignore[list-item]
        ])],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 3
    assert isinstance(out.messages[0], ToolMessage)
    assert out.messages[0].tool_call_id == "tu_a"
    assert isinstance(out.messages[1], ToolMessage)
    assert out.messages[1].tool_call_id == "tu_b"
    assert isinstance(out.messages[2], UserMessage)
    assert out.messages[2].content == "Now continue."


def test_user_message_text_before_tool_result_preserves_order() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "text", "text": "About to send tool result."},  # type: ignore[list-item]
            {"type": "tool_result",  # type: ignore[list-item]
             "tool_use_id": "tu_x", "content": "data"},
        ])],
    )
    out = from_anthropic_request(req)
    assert len(out.messages) == 2
    assert isinstance(out.messages[0], UserMessage)
    assert out.messages[0].content == "About to send tool result."
    assert isinstance(out.messages[1], ToolMessage)


def test_user_message_with_tool_use_block_raises() -> None:
    """tool_use belongs on assistant turns. Finding one on user is malformed."""
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        messages=[AnthropicMessage(role="user", content=[
            {"type": "tool_use", "id": "tu_x",  # type: ignore[list-item]
             "name": "fn", "input": {}},
        ])],
    )
    with pytest.raises(TranslationError, match="tool_use"):
        from_anthropic_request(req)


# ---------------------------------------------------------------------------
# Tools + tool_choice
# ---------------------------------------------------------------------------

def test_tool_definitions_translate() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tools=[
            AnthropicToolDef(
                name="read_file",
                description="Read a file from disk",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ],
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tools is not None and len(out.tools) == 1
    tool = out.tools[0]
    assert tool.type == "function"
    assert tool.function.name == "read_file"
    assert tool.function.description == "Read a file from disk"
    assert tool.function.parameters == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def test_tool_def_without_description_passes_through() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tools=[AnthropicToolDef(name="ping")],
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tools is not None
    assert out.tools[0].function.description is None
    # Schema's default input_schema is {"type":"object"}
    assert out.tools[0].function.parameters == {"type": "object"}


def test_tool_choice_auto() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tool_choice=AnthropicToolChoiceAuto(type="auto"),
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tool_choice == "auto"


def test_tool_choice_any_maps_to_required() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tool_choice=AnthropicToolChoiceAny(type="any"),
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tool_choice == "required"


def test_tool_choice_none() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tool_choice=AnthropicToolChoiceNone(type="none"),
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tool_choice == "none"


def test_tool_choice_specific_tool() -> None:
    req = AnthropicMessagesRequest(
        model="m", max_tokens=10,
        tool_choice=AnthropicToolChoiceSpecific(type="tool", name="read_file"),
        messages=[AnthropicMessage(role="user", content="x")],
    )
    out = from_anthropic_request(req)
    assert out.tool_choice == {"type": "function", "function": {"name": "read_file"}}


# ---------------------------------------------------------------------------
# Full Claude-Code-shaped roundtrip
# ---------------------------------------------------------------------------

def test_realistic_claude_code_multi_turn_with_tool_use() -> None:
    """Shape that mirrors what Claude Code actually sends after a few turns:
    system prompt + user question + assistant text+tool_use + user tool_result
    + user follow-up question."""
    req = AnthropicMessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system="You are a coding assistant in Claude Code.",
        tools=[
            AnthropicToolDef(
                name="read_file",
                description="Read a file's contents",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ],
        messages=[
            AnthropicMessage(role="user", content="What's in /etc/hostname?"),
            AnthropicMessage(role="assistant", content=[
                {"type": "text", "text": "Let me read it."},  # type: ignore[list-item]
                {"type": "tool_use", "id": "tu_1",  # type: ignore[list-item]
                 "name": "read_file", "input": {"path": "/etc/hostname"}},
            ]),
            AnthropicMessage(role="user", content=[
                {"type": "tool_result", "tool_use_id": "tu_1",  # type: ignore[list-item]
                 "content": "my-laptop.local\n"},
            ]),
            AnthropicMessage(role="user", content="And what's its IP?"),
        ],
    )
    out = from_anthropic_request(req)

    # SystemMessage, UserMessage (q1), AssistantMessage (text+tool_call),
    # ToolMessage (result), UserMessage (q2) = 5 messages
    assert len(out.messages) == 5

    assert isinstance(out.messages[0], SystemMessage)
    assert out.messages[0].content == "You are a coding assistant in Claude Code."

    assert isinstance(out.messages[1], UserMessage)
    assert out.messages[1].content == "What's in /etc/hostname?"

    asst = out.messages[2]
    assert isinstance(asst, AssistantMessage)
    assert asst.content == "Let me read it."
    assert asst.tool_calls is not None
    assert asst.tool_calls[0].id == "tu_1"
    assert asst.tool_calls[0].function.name == "read_file"

    tool_msg = out.messages[3]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "tu_1"
    assert tool_msg.content == "my-laptop.local\n"

    assert isinstance(out.messages[4], UserMessage)
    assert out.messages[4].content == "And what's its IP?"

    # Tools propagated correctly
    assert out.tools is not None and len(out.tools) == 1
    assert out.tools[0].function.name == "read_file"

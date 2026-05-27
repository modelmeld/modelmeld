"""OpenAI ⇄ Anthropic translation tests. No anthropic SDK required."""

from __future__ import annotations

import json

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.translation.openai_anthropic import (
    AnthropicStreamTranslator,
    TranslationError,
    from_anthropic_response,
    to_anthropic_params,
)

# ---------------------------------------------------------------------------
# Request translation: OpenAI → Anthropic
# ---------------------------------------------------------------------------

def test_simple_text_request() -> None:
    req = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hi"}],
    )
    params = to_anthropic_params(req)
    assert params["model"] == "claude-sonnet-4-6"
    assert params["messages"] == [{"role": "user", "content": "Hi"}]
    assert params["max_tokens"] == 4096  # default
    assert "system" not in params


def test_system_message_extracted_to_top_level() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Capital of France?"},
        ],
    )
    params = to_anthropic_params(req)
    assert params["system"] == "You are concise."
    # System message removed from messages array
    assert all(m["role"] != "system" for m in params["messages"])
    assert params["messages"] == [{"role": "user", "content": "Capital of France?"}]


def test_multiple_system_messages_concatenated() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ],
    )
    params = to_anthropic_params(req)
    assert params["system"] == "Rule 1.\n\nRule 2."


def test_max_completion_tokens_takes_precedence_over_max_tokens() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        max_completion_tokens=200,
    )
    params = to_anthropic_params(req)
    assert params["max_tokens"] == 200


def test_temperature_top_p_stop_passed_through() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        top_p=0.9,
        stop=["END", "STOP"],
    )
    params = to_anthropic_params(req)
    assert params["temperature"] == 0.5
    assert params["top_p"] == 0.9
    assert params["stop_sequences"] == ["END", "STOP"]


def test_stop_string_normalized_to_list() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stop="END",
    )
    params = to_anthropic_params(req)
    assert params["stop_sequences"] == ["END"]


def test_tools_translated_to_anthropic_shape() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
    )
    params = to_anthropic_params(req)
    assert len(params["tools"]) == 1
    tool = params["tools"][0]
    assert tool["name"] == "get_weather"
    assert tool["description"] == "Look up weather"
    assert tool["input_schema"]["type"] == "object"


def test_tool_choice_auto_translated() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="auto",
    )
    assert to_anthropic_params(req)["tool_choice"] == {"type": "auto"}


def test_tool_choice_required_translated_to_any() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="required",
    )
    assert to_anthropic_params(req)["tool_choice"] == {"type": "any"}


def test_tool_choice_specific_function_translated_to_tool() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice={"type": "function", "function": {"name": "do_thing"}},
    )
    assert to_anthropic_params(req)["tool_choice"] == {"type": "tool", "name": "do_thing"}


def test_assistant_tool_call_becomes_tool_use_block() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
            ],
        }
    )
    params = to_anthropic_params(req)
    assistant = params["messages"][1]
    assert assistant["role"] == "assistant"
    use_block = next(b for b in assistant["content"] if b["type"] == "tool_use")
    assert use_block["id"] == "call_1"
    assert use_block["name"] == "get_weather"
    assert use_block["input"] == {"city": "Paris"}


def test_tool_message_becomes_user_tool_result() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        }
    )
    params = to_anthropic_params(req)
    tool_result_msg = params["messages"][-1]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "sunny"


def test_image_url_translated_to_image_block_url_source() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "https://e.com/cat.jpg"}},
                    ],
                }
            ],
        }
    )
    params = to_anthropic_params(req)
    user_msg = params["messages"][0]
    img_block = user_msg["content"][1]
    assert img_block["type"] == "image"
    assert img_block["source"] == {"type": "url", "url": "https://e.com/cat.jpg"}


def test_image_data_url_translated_to_base64_source() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        }
                    ],
                }
            ],
        }
    )
    params = to_anthropic_params(req)
    img_block = params["messages"][0]["content"][0]
    assert img_block["source"]["type"] == "base64"
    assert img_block["source"]["media_type"] == "image/png"
    assert img_block["source"]["data"] == "iVBORw0KGgo="


def test_audio_part_raises_translation_error() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "abc", "format": "wav"},
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(TranslationError, match="audio"):
        to_anthropic_params(req)


# ---------------------------------------------------------------------------
# Response translation: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _anthropic_text_response(text: str = "Hello!") -> dict:
    return {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def test_simple_text_response_translated() -> None:
    completion = from_anthropic_response(_anthropic_text_response("Paris."))
    assert completion.id == "msg_abc123"
    assert completion.model == "claude-sonnet-4-6"
    assert completion.choices[0].message.content == "Paris."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 20
    assert completion.usage.total_tokens == 30


def test_tool_use_block_becomes_tool_call() -> None:
    resp = _anthropic_text_response()
    resp["content"] = [
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_weather",
            "input": {"city": "Paris", "units": "c"},
        }
    ]
    resp["stop_reason"] = "tool_use"
    completion = from_anthropic_response(resp)
    msg = completion.choices[0].message
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].id == "toolu_01"
    assert msg.tool_calls[0].function.name == "get_weather"
    args = json.loads(msg.tool_calls[0].function.arguments)
    assert args == {"city": "Paris", "units": "c"}
    assert completion.choices[0].finish_reason == "tool_calls"


def test_mixed_text_and_tool_use() -> None:
    resp = _anthropic_text_response()
    resp["content"] = [
        {"type": "text", "text": "Let me check that."},
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
    ]
    completion = from_anthropic_response(resp)
    msg = completion.choices[0].message
    assert msg.content == "Let me check that."
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "lookup"


@pytest.mark.parametrize(
    ("anthropic_reason", "openai_reason"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
    ],
)
def test_stop_reason_mapping(anthropic_reason: str, openai_reason: str) -> None:
    resp = _anthropic_text_response()
    resp["stop_reason"] = anthropic_reason
    completion = from_anthropic_response(resp)
    assert completion.choices[0].finish_reason == openai_reason


# ---------------------------------------------------------------------------
# Streaming translation: Anthropic events → OpenAI chunks
# ---------------------------------------------------------------------------

def test_message_start_emits_role_chunk() -> None:
    t = AnthropicStreamTranslator()
    chunk = t.translate_event({
        "type": "message_start",
        "message": {
            "id": "msg_stream_1",
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 5},
        },
    })
    assert chunk is not None
    assert chunk.id == "msg_stream_1"
    assert chunk.model == "claude-sonnet-4-6"
    assert chunk.choices[0].delta.role == "assistant"


def test_text_delta_emits_content_chunk() -> None:
    t = AnthropicStreamTranslator()
    t.translate_event({"type": "message_start", "message": {"id": "x", "model": "m"}})
    chunk = t.translate_event({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    })
    assert chunk is not None
    assert chunk.choices[0].delta.content == "Hello"


def test_tool_use_block_start_emits_tool_call_chunk() -> None:
    t = AnthropicStreamTranslator()
    t.translate_event({"type": "message_start", "message": {"id": "x", "model": "m"}})
    chunk = t.translate_event({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
    })
    assert chunk is not None
    assert chunk.choices[0].delta.tool_calls is not None
    tc = chunk.choices[0].delta.tool_calls[0]
    assert tc.index == 0
    assert tc.id == "toolu_1"
    assert tc.function is not None
    assert tc.function.name == "get_weather"


def test_input_json_delta_emits_argument_fragment() -> None:
    t = AnthropicStreamTranslator()
    t.translate_event({"type": "message_start", "message": {"id": "x", "model": "m"}})
    t.translate_event({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f"},
    })
    chunk = t.translate_event({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"city":"Pari'},
    })
    assert chunk is not None
    tc = chunk.choices[0].delta.tool_calls[0]  # type: ignore[index]
    assert tc.function is not None
    assert tc.function.arguments == '{"city":"Pari'


def test_message_delta_emits_finish_reason() -> None:
    t = AnthropicStreamTranslator()
    t.translate_event({"type": "message_start", "message": {"id": "x", "model": "m"}})
    chunk = t.translate_event({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 25},
    })
    assert chunk is not None
    assert chunk.choices[0].finish_reason == "stop"
    assert chunk.usage is not None
    assert chunk.usage.completion_tokens == 25


def test_unknown_event_returns_none() -> None:
    t = AnthropicStreamTranslator()
    assert t.translate_event({"type": "ping"}) is None
    assert t.translate_event({}) is None


def test_full_stream_roundtrip() -> None:
    """Replay a realistic Anthropic event sequence and verify the OpenAI chunk stream."""
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_full",
                "model": "claude-sonnet-4-6",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 8},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 12},
        },
        {"type": "message_stop"},
    ]
    t = AnthropicStreamTranslator()
    chunks = [c for e in events if (c := t.translate_event(e)) is not None]

    assert chunks[0].choices[0].delta.role == "assistant"
    content_chunks = [c for c in chunks if c.choices[0].delta.content]
    reconstructed = "".join(c.choices[0].delta.content or "" for c in content_chunks)
    assert reconstructed == "Hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 12

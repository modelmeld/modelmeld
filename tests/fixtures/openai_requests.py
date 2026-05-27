"""Representative OpenAI Chat Completions request payloads.

Synthesized to exercise the breadth of the documented API surface — single-turn,
multi-turn, multimodal, tool definitions, tool responses, structured outputs,
streaming, logprobs, reasoning. Each one must parse with modelmeld's
ChatCompletionRequest schema.
"""

from __future__ import annotations

from typing import Any

SIMPLE_TEXT: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
    ],
}

MULTI_TURN: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "Of Spain?"},
    ],
    "temperature": 0.2,
    "max_completion_tokens": 50,
}

MULTIMODAL_IMAGE: dict[str, Any] = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/cat.jpg",
                        "detail": "high",
                    },
                },
            ],
        },
    ],
}

MULTIMODAL_MIXED_PARTS: dict[str, Any] = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You describe images concisely."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}},
                {"type": "text", "text": "Caption?"},
            ],
        },
    ],
}

TOOLS_DEFINED: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "What's the weather in Paris?"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "units": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city"],
                },
                "strict": True,
            },
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
}

TOOL_CHOICE_SPECIFIC: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Force the weather tool."}],
    "tools": [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
}

CONVERSATION_WITH_TOOL_RESULTS: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris", "units": "c"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temp": 18, "conditions": "cloudy"}',
            "tool_call_id": "call_abc123",
        },
    ],
}

RESPONSE_FORMAT_JSON: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Return JSON with name and age."},
    ],
    "response_format": {"type": "json_object"},
    "seed": 42,
}

STREAM_WITH_USAGE: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Stream me a poem."}],
    "stream": True,
    "stream_options": {"include_usage": True},
    "temperature": 0.7,
    "top_p": 0.95,
}

LOGPROBS_AND_REASONING: dict[str, Any] = {
    "model": "o3-mini",
    "messages": [
        {"role": "user", "content": "Solve: 23 * 47"},
    ],
    "logprobs": True,
    "top_logprobs": 3,
    "reasoning_effort": "medium",
    "stop": ["\n\n"],
    "frequency_penalty": 0.1,
    "presence_penalty": 0.0,
    "logit_bias": {"50256": -100},
    "user": "user-abc-123",
}

ALL_REQUESTS: list[tuple[str, dict[str, Any]]] = [
    ("simple_text", SIMPLE_TEXT),
    ("multi_turn", MULTI_TURN),
    ("multimodal_image", MULTIMODAL_IMAGE),
    ("multimodal_mixed_parts", MULTIMODAL_MIXED_PARTS),
    ("tools_defined", TOOLS_DEFINED),
    ("tool_choice_specific", TOOL_CHOICE_SPECIFIC),
    ("conversation_with_tool_results", CONVERSATION_WITH_TOOL_RESULTS),
    ("response_format_json", RESPONSE_FORMAT_JSON),
    ("stream_with_usage", STREAM_WITH_USAGE),
    ("logprobs_and_reasoning", LOGPROBS_AND_REASONING),
]

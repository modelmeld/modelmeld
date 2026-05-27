"""Representative OpenAI Chat Completions response payloads.

Synthesized to exercise the breadth of the documented response surface.
Each one must parse losslessly with both modelmeld's `ChatCompletion`
schema and the official `openai.types.chat.ChatCompletion` model.
"""

from __future__ import annotations

from typing import Any

SIMPLE_TEXT: dict[str, Any] = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1715000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris."},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
}

WITH_TOOL_CALLS: dict[str, Any] = {
    "id": "chatcmpl-tool1",
    "object": "chat.completion",
    "created": 1715000001,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris","units":"c"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 45, "completion_tokens": 20, "total_tokens": 65},
}

PARALLEL_TOOL_CALLS: dict[str, Any] = {
    "id": "chatcmpl-tool2",
    "object": "chat.completion",
    "created": 1715000002,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 60, "completion_tokens": 38, "total_tokens": 98},
}

WITH_LOGPROBS: dict[str, Any] = {
    "id": "chatcmpl-lp1",
    "object": "chat.completion",
    "created": 1715000003,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "OK",
                        "logprob": -0.05,
                        "bytes": [79, 75],
                        "top_logprobs": [
                            {"token": "OK", "logprob": -0.05, "bytes": [79, 75]},
                            {"token": "Yes", "logprob": -2.3, "bytes": [89, 101, 115]},
                        ],
                    }
                ],
                "refusal": None,
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

N_TWO_CHOICES: dict[str, Any] = {
    "id": "chatcmpl-n2",
    "object": "chat.completion",
    "created": 1715000004,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Choice A"},
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {"role": "assistant", "content": "Choice B"},
            "finish_reason": "stop",
        },
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
}

WITH_USAGE_DETAILS: dict[str, Any] = {
    "id": "chatcmpl-ud",
    "object": "chat.completion",
    "created": 1715000005,
    "model": "o3-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "1081"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
        "prompt_tokens_details": {"cached_tokens": 16},
        "completion_tokens_details": {
            "reasoning_tokens": 320,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
}

FINISH_LENGTH: dict[str, Any] = {
    "id": "chatcmpl-len",
    "object": "chat.completion",
    "created": 1715000006,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "It was the best of"},
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
}

FINISH_CONTENT_FILTER: dict[str, Any] = {
    "id": "chatcmpl-cf",
    "object": "chat.completion",
    "created": 1715000007,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "content_filter",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 0, "total_tokens": 15},
}

WITH_REFUSAL: dict[str, Any] = {
    "id": "chatcmpl-ref",
    "object": "chat.completion",
    "created": 1715000008,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "refusal": "I cannot help with that.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
}

WITH_SYSTEM_FINGERPRINT: dict[str, Any] = {
    "id": "chatcmpl-sf",
    "object": "chat.completion",
    "created": 1715000009,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Sure."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    "system_fingerprint": "fp_4abc123",
    "service_tier": "default",
}

ALL_RESPONSES: list[tuple[str, dict[str, Any]]] = [
    ("simple_text", SIMPLE_TEXT),
    ("with_tool_calls", WITH_TOOL_CALLS),
    ("parallel_tool_calls", PARALLEL_TOOL_CALLS),
    ("with_logprobs", WITH_LOGPROBS),
    ("n_two_choices", N_TWO_CHOICES),
    ("with_usage_details", WITH_USAGE_DETAILS),
    ("finish_length", FINISH_LENGTH),
    ("finish_content_filter", FINISH_CONTENT_FILTER),
    ("with_refusal", WITH_REFUSAL),
    ("with_system_fingerprint", WITH_SYSTEM_FINGERPRINT),
]

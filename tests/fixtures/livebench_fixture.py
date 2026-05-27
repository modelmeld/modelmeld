"""Synthetic LiveBench JSON fixture."""

from __future__ import annotations

LIVEBENCH_BASELINE = {
    "models": [
        {
            "model": "claude-opus-4-7",
            "organization": "Anthropic",
            "context_window": 200000,
            "coding": 73.5,
            "math": 78.1,
            "reasoning": 82.0,
            "data_analysis": 70.4,
            "instruction_following": 91.0,
            "language": 84.3,
            "evaluation_date": "2026-05-01T00:00:00Z",
        },
        {
            "model": "gpt-5-mini",
            "organization": "OpenAI",
            "context_window": 128000,
            "coding": 60.0,
            "math": 64.2,
            "reasoning": 68.5,
            "instruction_following": 88.0,
            "language": 80.0,
            "evaluation_date": "2026-05-01T00:00:00Z",
        },
        {
            "model": "qwen3-coder-next",
            "organization": "Alibaba",
            "coding": 68.0,
            "math": 50.0,
            "reasoning": 60.0,
            "evaluation_date": "2026-05-01T00:00:00Z",
        },
    ]
}

LIVEBENCH_BARE_LIST = LIVEBENCH_BASELINE["models"]

"""Synthetic Artificial Analysis API response fixtures.

Approximates the AA `/v2/models` payload shape based on their public docs.
If AA's actual schema differs in details when we wire to the real API, only
`artificial_analysis.normalize_aa_model` needs adjustment.
"""

from __future__ import annotations

from typing import Any

# A "well-formed" response — clean canonical shape
AA_RESPONSE_BASELINE: dict[str, Any] = {
    "data": [
        {
            "model_id": "claude-opus-4-7",
            "provider": "Anthropic",
            "context_window": 200000,
            "price_input_per_m": 5.00,
            "price_output_per_m": 25.00,
            "evaluations": {
                "GPQA Diamond": 84.5,
                "AA-LCR": 71.2,
                "HLE": 35.1,
                "CritPt": 62.0,
                "Terminal-Bench Hard": 58.3,
                "SciCode": 67.5,
                "LiveCodeBench": 72.0,
                "IFBench": 92.0,
                "GDPval-AA": 88.0,
                "AA-Omniscience": 76.0,
                "τ²-Bench Telecom": 70.0,
            },
            "last_updated": "2026-05-15T00:00:00Z",
        },
        {
            "model_id": "gpt-5-mini",
            "provider": "OpenAI",
            "context_window": 128000,
            "price_input_per_m": 0.25,
            "price_output_per_m": 2.00,
            "evaluations": {
                "GPQA Diamond": 65.2,
                "AA-LCR": 58.4,
                "HLE": 22.5,
                "CritPt": 50.0,
                "Terminal-Bench Hard": 48.0,
                "SciCode": 55.0,
                "LiveCodeBench": 60.0,
                "IFBench": 88.0,
                "GDPval-AA": 80.0,
                "AA-Omniscience": 70.0,
                "τ²-Bench Telecom": 60.0,
            },
            "last_updated": "2026-05-14T00:00:00Z",
        },
        {
            "model_id": "qwen3-coder-next",
            "provider": "Alibaba",
            "context_window": 262144,
            "price_input_per_m": 0.30,
            "price_output_per_m": 0.30,
            "evaluations": {
                "Terminal-Bench Hard": 55.0,
                "SciCode": 60.0,
                "LiveCodeBench": 68.0,
                "GPQA Diamond": 60.0,
                "AA-LCR": 55.0,
                "IFBench": 82.0,
            },
            "last_updated": "2026-05-12T00:00:00Z",
        },
    ]
}


# "Dirty" entries — partial / malformed records the normalizer should handle.
AA_RESPONSE_PARTIAL: dict[str, Any] = {
    "data": [
        {
            # Missing benchmarks entirely — should produce ModelEntry with empty task_scores
            "model_id": "no-benchmarks",
            "provider": "x",
            "context_window": 4096,
            "price_input_per_m": 0.0,
            "price_output_per_m": 0.0,
        },
        {
            # Uses alternative field names: `name` instead of `model_id`, `creator` instead of `provider`
            "name": "Alt-Field-Names",
            "creator": "Some Lab",
            "context_window": 8192,
            "price_input_per_m": 1.0,
            "price_output_per_m": 2.0,
            "evaluations": {"GPQA Diamond": 50.0},
        },
        {
            # No identifiable id — normalizer should reject this one
            "provider": "Anonymous",
            "context_window": 1000,
        },
    ]
}

# A bare-list response (no "data" wrapper) — AA might return either shape
AA_RESPONSE_BARE_LIST: list[dict[str, Any]] = AA_RESPONSE_BASELINE["data"]

"""Synthetic LMArena daily JSON snapshot fixtures."""

from __future__ import annotations

# text_arena.json — general-chat preference
LMARENA_TEXT = [
    {
        "model": "claude-opus-4-7",
        "organization": "Anthropic",
        "rating": 1285,
        "rank": 1,
        "date": "2026-05-15",
    },
    {
        "model": "gpt-5-4",
        "organization": "OpenAI",
        "rating": 1278,
        "rank": 2,
        "date": "2026-05-15",
    },
    {
        "model": "gpt-5-mini",
        "organization": "OpenAI",
        "rating": 1195,
        "rank": 12,
        "date": "2026-05-15",
    },
]

# code_arena.json — coding-only preference
LMARENA_CODE = [
    {
        "model": "claude-opus-4-7",
        "organization": "Anthropic",
        "rating": 1310,
        "rank": 1,
        "date": "2026-05-15",
    },
    {
        "model": "qwen3-coder-next",
        "organization": "Alibaba",
        "rating": 1235,
        "rank": 5,
        "date": "2026-05-15",
    },
]

# Malformed entries should be skipped, not crash the fetcher
LMARENA_TEXT_MALFORMED = [
    {"model": "no-rating-here"},
    {"rating": 1200},                # no model
    {"model": "bad-rating", "rating": "not-a-number"},
]

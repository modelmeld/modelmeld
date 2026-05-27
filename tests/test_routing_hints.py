"""routing_hints — header parsing + validation + RoutingHints semantics."""

from __future__ import annotations

import pytest

from modelmeld.api.routing_hints import (
    AGENT_ROLE_TO_CATEGORY,
    HEADER_AGENT_ROLE,
    HEADER_EXCLUDE_PROVIDERS,
    HEADER_QUALITY_THRESHOLD,
    HEADER_TASK_CATEGORY,
    RoutingHintError,
    RoutingHints,
    extract_hints_from_headers,
)
from modelmeld.scout.task_category import TASK_CATEGORIES


def test_empty_headers_returns_empty_hints() -> None:
    hints = extract_hints_from_headers({})
    assert hints == RoutingHints()
    assert hints.has_category_hint is False
    assert hints.effective_category() is None


# ---------------------------------------------------------------------------
# task_category header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", list(TASK_CATEGORIES))
def test_each_task_category_accepted(category: str) -> None:
    hints = extract_hints_from_headers({HEADER_TASK_CATEGORY: category})
    assert hints.task_category == category
    assert hints.effective_category() == category
    assert hints.has_category_hint is True


def test_task_category_is_case_insensitive() -> None:
    hints = extract_hints_from_headers({HEADER_TASK_CATEGORY: "CODING"})
    assert hints.task_category == "coding"


def test_invalid_task_category_rejected() -> None:
    with pytest.raises(RoutingHintError, match="not in"):
        extract_hints_from_headers({HEADER_TASK_CATEGORY: "code-review"})


def test_empty_string_task_category_treated_as_unset() -> None:
    hints = extract_hints_from_headers({HEADER_TASK_CATEGORY: "   "})
    assert hints.task_category is None


# ---------------------------------------------------------------------------
# agent_role header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("coder", "coding"),
        ("Coder", "coding"),
        ("software-engineer", "coding"),  # hyphens normalized
        ("researcher", "reasoning"),
        ("summarizer", "summarization"),
        ("executor", "tool_use"),
        ("assistant", "simple_qa"),
    ],
)
def test_known_roles_map_to_categories(role: str, expected: str) -> None:
    hints = extract_hints_from_headers({HEADER_AGENT_ROLE: role})
    assert hints.derived_category == expected
    assert hints.effective_category() == expected


def test_unknown_role_rejected() -> None:
    with pytest.raises(RoutingHintError, match="unknown"):
        extract_hints_from_headers({HEADER_AGENT_ROLE: "vibe_master_general"})


def test_task_category_wins_over_agent_role() -> None:
    """Explicit task_category overrides the role-derived category."""
    hints = extract_hints_from_headers({
        HEADER_TASK_CATEGORY: "reasoning",
        HEADER_AGENT_ROLE: "coder",  # would normally → coding
    })
    assert hints.task_category == "reasoning"
    assert hints.derived_category == "coding"
    assert hints.effective_category() == "reasoning"


def test_agent_role_to_category_map_only_uses_known_categories() -> None:
    """Sanity: every role maps to a valid task category."""
    for role, category in AGENT_ROLE_TO_CATEGORY.items():
        assert category in TASK_CATEGORIES, f"role {role!r} maps to bogus category {category!r}"


# ---------------------------------------------------------------------------
# quality_threshold header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["0.0", "0.5", "0.75", "1.0"])
def test_valid_quality_threshold_accepted(value: str) -> None:
    hints = extract_hints_from_headers({HEADER_QUALITY_THRESHOLD: value})
    assert hints.quality_threshold == float(value)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "100", "abc"])
def test_invalid_quality_threshold_rejected(value: str) -> None:
    with pytest.raises(RoutingHintError):
        extract_hints_from_headers({HEADER_QUALITY_THRESHOLD: value})


def test_empty_threshold_is_unset() -> None:
    hints = extract_hints_from_headers({HEADER_QUALITY_THRESHOLD: ""})
    assert hints.quality_threshold is None


# ---------------------------------------------------------------------------
# exclude_providers header
# ---------------------------------------------------------------------------

def test_exclude_providers_parsed_as_lowercase_frozenset() -> None:
    hints = extract_hints_from_headers({
        HEADER_EXCLUDE_PROVIDERS: "OpenAI, anthropic, ,VLLM"
    })
    assert hints.excluded_providers == frozenset({"openai", "anthropic", "vllm"})


def test_exclude_providers_empty_is_unset() -> None:
    hints = extract_hints_from_headers({HEADER_EXCLUDE_PROVIDERS: " , ,"})
    assert hints.excluded_providers is None


def test_single_provider_exclude() -> None:
    hints = extract_hints_from_headers({HEADER_EXCLUDE_PROVIDERS: "openai"})
    assert hints.excluded_providers == frozenset({"openai"})


# ---------------------------------------------------------------------------
# Header casing
# ---------------------------------------------------------------------------

def test_headers_are_case_insensitive() -> None:
    """ASGI / Starlette lowercase headers; extractor must too."""
    hints = extract_hints_from_headers({
        "X-ModelMeld-Task-Category": "coding",
        "X-MODELMELD-AGENT-ROLE": "researcher",
        "x-modelmeld-quality-threshold": "0.9",
    })
    assert hints.task_category == "coding"
    assert hints.agent_role == "researcher"
    assert hints.quality_threshold == 0.9


# ---------------------------------------------------------------------------
# All four together
# ---------------------------------------------------------------------------

def test_all_four_headers_combine() -> None:
    hints = extract_hints_from_headers({
        HEADER_TASK_CATEGORY: "coding",
        HEADER_AGENT_ROLE: "engineer",
        HEADER_QUALITY_THRESHOLD: "0.85",
        HEADER_EXCLUDE_PROVIDERS: "openai",
    })
    assert hints.task_category == "coding"
    assert hints.agent_role == "engineer"
    assert hints.quality_threshold == 0.85
    assert hints.excluded_providers == frozenset({"openai"})
    assert hints.effective_category() == "coding"

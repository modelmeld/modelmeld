"""TaskCategoryClassifier — heuristic prompt → category."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    FunctionDef,
    SystemMessage,
    Tool,
    UserMessage,
)
from modelmeld.scout.task_category import (
    TASK_CATEGORIES,
    TaskCategoryClassifier,
)


def _req(text: str, tools: list[Tool] | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
        tools=tools or [],
    )


# ---------------------------------------------------------------------------
# Definitive: tools declared → tool_use
# ---------------------------------------------------------------------------

def test_tools_declared_routes_to_tool_use() -> None:
    tools = [
        Tool(
            type="function",
            function=FunctionDef(
                name="search", description="", parameters={"type": "object"}
            ),
        )
    ]
    req = _req("write a function to sum a list", tools=tools)
    decision = TaskCategoryClassifier().classify(req)
    assert decision.category == "tool_use"
    assert decision.confidence == 1.0


# ---------------------------------------------------------------------------
# Coding signals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "Please refactor this function to remove the nested loops.",
        "Write a Python class for a binary tree with insert and traverse methods.",
        "Fix this bug: TypeError: 'NoneType' object is not iterable",
        "Implement a fibonacci function in TypeScript.",
        "Add a docstring to this method.",
        "Rename the variable `tmp` to `buffer` across this file.",
        "Here's the relevant code:\n```\ndef foo(): pass\n```\nMake it idempotent.",
        "I'm getting a syntax error in my main.py — please debug.",
    ],
)
def test_coding_signals_route_to_coding(prompt: str) -> None:
    decision = TaskCategoryClassifier().classify(_req(prompt))
    assert decision.category == "coding", f"Expected coding for: {prompt!r}; got {decision}"


# ---------------------------------------------------------------------------
# Reasoning signals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "Prove that the sum of two odd numbers is even, step by step.",
        "Derive the closed-form solution for this recurrence.",
        "Analyze the trade-offs between eventual consistency and strong consistency.",
        "Explain why CAP theorem holds; reason about partition tolerance.",
        "Compare and contrast B-trees vs LSM-trees for write-heavy workloads.",
    ],
)
def test_reasoning_signals_route_to_reasoning(prompt: str) -> None:
    decision = TaskCategoryClassifier().classify(_req(prompt))
    assert decision.category == "reasoning"


# ---------------------------------------------------------------------------
# Summarization signals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "Summarize the main findings of the paper.",
        "TL;DR of this Slack thread please.",
        "Give me the key points from this transcript.",
        "Condense this into three bullet points.",
    ],
)
def test_summarization_signals_route_to_summarization(prompt: str) -> None:
    decision = TaskCategoryClassifier().classify(_req(prompt))
    assert decision.category == "summarization"


# ---------------------------------------------------------------------------
# Simple QA signals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of France?",
        "Who is the CEO of Anthropic?",
        "When was the transformer paper published?",
        "Define entropy in the information-theory sense.",
    ],
)
def test_simple_qa_signals_route_to_simple_qa(prompt: str) -> None:
    decision = TaskCategoryClassifier().classify(_req(prompt))
    assert decision.category == "simple_qa"


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def test_empty_prompt_falls_back_to_simple_qa() -> None:
    decision = TaskCategoryClassifier().classify(_req("just sayin"))
    assert decision.category == "simple_qa"
    assert decision.confidence == 0.0
    assert "no_signals" in decision.rationale


def test_long_prompt_with_no_signals_goes_to_summarization() -> None:
    # 1500 tokens ≈ 6000 chars in our 4-char-per-token model
    blob = "Quarterly revenue overview. " * 500
    decision = TaskCategoryClassifier().classify(_req(blob))
    assert decision.category == "summarization"
    assert "long_prompt" in decision.rationale


def test_long_coding_prompt_stays_coding_not_summarization() -> None:
    """Code-heavy long inputs should land in coding, not summarization."""
    blob = "Refactor this function:\n```\ndef foo(x): return x + 1\n```\n" * 100
    decision = TaskCategoryClassifier().classify(_req(blob))
    assert decision.category == "coding"


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

def test_coding_wins_ties_against_simple_qa() -> None:
    """Per _CATEGORY_PRIORITY, coding beats simple_qa on equal scores."""
    # "what is" → simple_qa; "function" → coding. Each gets one hit.
    decision = TaskCategoryClassifier().classify(
        _req("what is a function in Python?")
    )
    # With one hit each, coding's higher priority should win the tie.
    assert decision.category == "coding"


def test_category_decision_per_category_scores_are_complete() -> None:
    decision = TaskCategoryClassifier().classify(_req("refactor this code"))
    assert set(decision.per_category_scores.keys()) >= set(TASK_CATEGORIES) - {"tool_use"}


# ---------------------------------------------------------------------------
# System messages contribute too
# ---------------------------------------------------------------------------

def test_classifier_reads_system_messages() -> None:
    req = ChatCompletionRequest(
        model="x",
        messages=[
            SystemMessage(role="system", content="You refactor Python code."),
            UserMessage(role="user", content="here"),
        ],
        tools=[],
    )
    decision = TaskCategoryClassifier().classify(req)
    assert decision.category == "coding"

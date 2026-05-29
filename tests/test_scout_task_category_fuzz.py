"""Deterministic paraphrase coverage for TaskCategoryClassifier.

The goal is to make classifier blind spots visible without changing the
classifier in the same PR. Each category gets a small hand-written paraphrase
set derived from the canonical examples in test_scout_task_category.py.
"""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest, UserMessage
from modelmeld.scout.task_category import TaskCategoryClassifier


def _req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
        tools=[],
    )


PARAPHRASES_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "coding": (
        "Clean up this function so it no longer uses nested loops.",
        "Write a Python class that stores tree nodes and supports traversal.",
        "Fix this bug where None is being treated like an iterable.",
        "Implement fibonacci in TypeScript with a small test case.",
        "Add a docstring that explains the method inputs and return value.",
        "Rename the tmp variable to buffer in this module.",
        "Here is the code block: ```def foo(): pass``` make it safe to run twice.",
        "My main.py has a syntax error; please debug it.",
    ),
    "reasoning": (
        "Show a step-by-step proof that adding two odd numbers gives an even number.",
        "Derive a closed-form formula for this recurrence relation.",
        "Analyze strong consistency versus eventual consistency trade-offs.",
        "Explain why partition tolerance matters under the CAP theorem.",
        "Compare and contrast B-trees and LSM-trees for write-heavy databases.",
    ),
    "summarization": (
        "Summarize the paper's main findings.",
        "Give me a TLDR for this Slack discussion.",
        "Extract the key points from this transcript.",
        "Condense this document into three bullet points.",
    ),
    "simple_qa": (
        "What is France's capital city?",
        "Who is Anthropic's CEO?",
        "When was the Transformer paper first published?",
        "Define entropy as used in information theory.",
    ),
}


@pytest.mark.parametrize("expected_category", sorted(PARAPHRASES_BY_CATEGORY))
def test_task_category_paraphrases_meet_category_threshold(
    expected_category: str,
) -> None:
    classifier = TaskCategoryClassifier()
    prompts = PARAPHRASES_BY_CATEGORY[expected_category]
    decisions = [(prompt, classifier.classify(_req(prompt))) for prompt in prompts]
    failures = [
        (prompt, decision.category)
        for prompt, decision in decisions
        if decision.category != expected_category
    ]
    pass_ratio = (len(prompts) - len(failures)) / len(prompts)

    assert pass_ratio >= 0.8, (
        f"{expected_category} paraphrase coverage fell to {pass_ratio:.0%}; "
        f"misclassified prompts: {failures}"
    )

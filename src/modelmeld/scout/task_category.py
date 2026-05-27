# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Task-category classification — what kind of work is this prompt asking for?

Distinct from `HeuristicScout` (tier classifier). Output is one of:
    coding, reasoning, simple_qa, summarization, tool_use

The CapabilityScout consults the ModelRegistry with this label
to pick the cheapest model that meets the quality threshold for the category.

The classifier is pure-heuristic for now; an embedding-based or fine-tuned
classifier can plug in via the same interface later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.heuristics import _estimate_tokens, _extract_text

TASK_CATEGORIES = ("coding", "reasoning", "simple_qa", "summarization", "tool_use")


@dataclass(frozen=True)
class TaskCategoryWeights:
    """Tunable scoring constants for `TaskCategoryClassifier`.

    Override for organization-specific tuning (e.g. a data-science-heavy
    deployment that wants summarization to outrank coding in ties). The
    OSS default ships the current production-tuned values.
    """

    # Tie-break preference when match counts are equal — higher wins.
    # Coding > reasoning because routing a reasoning-flavored coding prompt
    # to a pure reasoning model costs more (Opus on a Sonnet-suitable refactor).
    category_priority: dict[str, int] = field(
        default_factory=lambda: {
            "coding": 5,
            "reasoning": 4,
            "summarization": 3,
            "simple_qa": 2,
            "tool_use": 1,
        },
    )
    # Long inputs without strong code/reasoning signals → likely summarization.
    long_prompt_tokens: int = 1500
    # Score nudge for long-prompt→summarization heuristic. Soft enough to still
    # lose to a clear match in another category.
    long_prompt_summarization_score: float = 0.5


DEFAULT_TASK_CATEGORY_WEIGHTS = TaskCategoryWeights()

_CODING_PATTERNS = (
    r"\b(code|function|class|method|module|package)\b",
    r"\b(rename|refactor|lint|format|prettify)\b",
    r"\b(docstring|type ?hint|annotation|return type)\b",
    r"\b(implement|write a|fix this|debug|bug|exception|error)\b",
    r"\b(diff|merge|commit|pull request|pr)\b",
    r"\b(syntax|compile|build|import|export)\b",
    r"\b(unit test|integration test|test case|assert)\b",
    # File-extension mentions strongly imply code work
    r"\.(py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|rb|php|cs|swift|kt|scala|sh|sql)\b",
    # Triple-backtick code blocks
    r"```",
    # Common code-shape signatures
    r"\b(def|func|fn|function)\s+\w+\s*\(",
    r"\bclass\s+\w+(\(|:)",
)

_REASONING_PATTERNS = (
    r"\b(prove|derive|deduce|infer)\b",
    r"\bstep[- ]by[- ]step\b",
    r"\b(explain why|reason about|think through)\b",
    r"\b(theorem|proof|equation|formula)\b",
    r"\b(analy[sz]e|analy[sz]is)\b",
    r"\b(compare and contrast|trade-?offs?)\b",
    r"\b(design (?:a|an|the))\s+(distributed|scalable|fault|micro|architecture|system)",
    r"\barchitect(ure|ural)?\b",
    r"\b(algorithm|complexity|big[- ]o)\b",
)

_SIMPLE_QA_PATTERNS = (
    r"\bwhat (is|are|does|do)\b",
    r"\bwho (is|was)\b",
    r"\b(when|where) (was|is|did)\b",
    r"\b(define|definition of|meaning of)\b",
)

_SUMMARIZATION_PATTERNS = (
    r"\bsummari[sz]e\b",
    r"\bsummary\b",
    r"\btl;?dr\b",
    r"\b(key|main) points\b",
    r"\b(condense|abridge)\b",
    r"\b(in brief|in short)\b",
    r"\babstract( of)?\b",
    r"\b(extract|pull out) (the )?(main|key)\b",
)


def _compile_any(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_PATTERNS_BY_CATEGORY: dict[str, re.Pattern[str]] = {
    "coding": _compile_any(_CODING_PATTERNS),
    "reasoning": _compile_any(_REASONING_PATTERNS),
    "simple_qa": _compile_any(_SIMPLE_QA_PATTERNS),
    "summarization": _compile_any(_SUMMARIZATION_PATTERNS),
}


@dataclass(frozen=True)
class TaskCategoryDecision:
    """Output of TaskCategoryClassifier.classify()."""

    category: str
    confidence: float                  # 0..1, max_score / total_score; 0 when nothing matched
    rationale: str                     # short trace for logs
    per_category_scores: dict[str, float]


class TaskCategoryClassifier:
    """Heuristic prompt → task category classifier.

    Algorithm:
      1. If the request declares tools → `tool_use` (definitive).
      2. Otherwise count regex matches per category; ties broken by
         `_CATEGORY_PRIORITY` (coding > reasoning > summarization > simple_qa).
      3. Long prompts with no strong code/reasoning signal → `summarization`
         (with a soft +0.5 score so it still loses to clear matches).
      4. Empty result (no signals at all) → `simple_qa` (cheapest default).
    """

    name = "heuristic_task_category"

    def __init__(self, weights: TaskCategoryWeights | None = None) -> None:
        self.weights = weights or DEFAULT_TASK_CATEGORY_WEIGHTS

    def classify(self, request: ChatCompletionRequest) -> TaskCategoryDecision:
        if request.tools:
            return TaskCategoryDecision(
                category="tool_use",
                confidence=1.0,
                rationale="tools_declared",
                per_category_scores={"tool_use": 1.0},
            )

        text = _extract_text(request)
        tokens = _estimate_tokens(text)
        w = self.weights
        scores: dict[str, float] = {c: 0.0 for c in TASK_CATEGORIES}
        signals: list[str] = []

        for category, pattern in _PATTERNS_BY_CATEGORY.items():
            matches = pattern.findall(text)
            n = len(matches)
            if n > 0:
                scores[category] = float(n)
                signals.append(f"{category}:{n}")

        # Long-prompt nudge toward summarization
        if tokens >= w.long_prompt_tokens and scores["summarization"] == 0.0 \
                and scores["coding"] == 0.0:
            scores["summarization"] = w.long_prompt_summarization_score
            signals.append(f"long_prompt({tokens}tok)→summarization")

        total = sum(scores.values())
        if total == 0.0:
            return TaskCategoryDecision(
                category="simple_qa",
                confidence=0.0,
                rationale="no_signals",
                per_category_scores=scores,
            )

        # Pick max score, break ties by priority
        best_cat = max(
            scores.items(),
            key=lambda kv: (kv[1], w.category_priority.get(kv[0], 0)),
        )[0]
        confidence = scores[best_cat] / total

        return TaskCategoryDecision(
            category=best_cat,
            confidence=confidence,
            rationale=",".join(signals) if signals else "no_signals",
            per_category_scores=scores,
        )

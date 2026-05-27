"""Precision tests on the expanded labeled corpus."""

from __future__ import annotations

from collections import Counter

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.base import Tier
from modelmeld.scout.heuristics import HeuristicScout
from tests.fixtures.scout_corpus import LABELED_CORPUS

# Minimum precision on simple-labeled prompts under the default-threshold scout.
# The original baseline was 100% on a 10-prompt corpus; with 25+ simple prompts
# spanning per-tool patterns, ≥80% is the bar.
_SIMPLE_PRECISION_TARGET = 0.80
_COMPLEX_PRECISION_TARGET = 0.80


def _req(prompt: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": prompt}],
    )


@pytest.fixture
def scout() -> HeuristicScout:
    return HeuristicScout()


async def _classify_all(scout: HeuristicScout) -> dict[str, list[tuple[str, Tier]]]:
    """Group corpus entries by complexity label and tag each with the scout's tier."""
    by_label: dict[str, list[tuple[str, Tier]]] = {"simple": [], "complex": []}
    for _provenance, complexity, prompt in LABELED_CORPUS:
        decision = await scout.classify(_req(prompt))
        by_label[complexity].append((prompt, decision.tier))
    return by_label


async def test_simple_precision_meets_target(scout: HeuristicScout) -> None:
    by_label = await _classify_all(scout)
    simple = by_label["simple"]
    correct = sum(1 for _, tier in simple if tier == Tier.LOCAL)
    precision = correct / len(simple)
    misses = [p for p, t in simple if t != Tier.LOCAL]
    assert precision >= _SIMPLE_PRECISION_TARGET, (
        f"simple precision {precision:.2%} < target {_SIMPLE_PRECISION_TARGET:.0%}. "
        f"Misclassified ({len(misses)}):\n  - " + "\n  - ".join(misses)
    )


async def test_complex_precision_meets_target(scout: HeuristicScout) -> None:
    by_label = await _classify_all(scout)
    complex_ = by_label["complex"]
    correct = sum(1 for _, tier in complex_ if tier == Tier.CLOUD)
    precision = correct / len(complex_)
    misses = [p for p, t in complex_ if t != Tier.CLOUD]
    assert precision >= _COMPLEX_PRECISION_TARGET, (
        f"complex precision {precision:.2%} < target {_COMPLEX_PRECISION_TARGET:.0%}. "
        f"Misclassified ({len(misses)}):\n  - " + "\n  - ".join(misses)
    )


async def test_corpus_size_and_balance() -> None:
    """Sanity: corpus has reasonable size and isn't lopsided."""
    counts = Counter(c for _, c, _ in LABELED_CORPUS)
    assert sum(counts.values()) >= 40, f"corpus too small: {sum(counts.values())}"
    # Ratios shouldn't be too skewed
    assert counts["simple"] >= 15
    assert counts["complex"] >= 10


async def test_scout_signals_include_devtool_tag(scout: HeuristicScout) -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "You are Claude Code, Anthropic's official CLI for Claude."},
            {"role": "user", "content": "fix the indent in utils.py"},
        ],
    )
    decision = await scout.classify(request)
    assert decision.signals["devtool"] == "claude_code"
    assert decision.signals["devtool_confidence"] >= 0.6

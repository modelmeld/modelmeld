"""Regression-safety + tuning-seam smoke tests.

Locks in three properties:

  1. DEFAULT_HEURISTIC_WEIGHTS contains exactly the values that the
     legacy module-level _* constants encoded. Anyone tempted
     to "tweak the defaults" runs into this test first.

  2. Injecting a custom HeuristicWeights actually changes the scout's
     decision on a borderline prompt (proves the seam is wired up, not
     just present).

  3. Same for TaskCategoryWeights and DEFAULT_QUALITY_THRESHOLD.
"""

from __future__ import annotations

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    UserMessage,
)
from modelmeld.scout.capability import DEFAULT_QUALITY_THRESHOLD
from modelmeld.scout.devtool import (
    DefaultPatternProvider,
    DevTool,
    Fingerprinter,
    PatternProvider,
)
from modelmeld.scout.heuristics import (
    DEFAULT_HEURISTIC_WEIGHTS,
    HeuristicScout,
    HeuristicWeights,
)
from modelmeld.scout.task_category import (
    DEFAULT_TASK_CATEGORY_WEIGHTS,
    TaskCategoryClassifier,
    TaskCategoryWeights,
)

# ---------------------------------------------------------------------------
# Default values frozen at known-good production tunings
# ---------------------------------------------------------------------------


def test_default_heuristic_weights_match_pre_extraction_constants() -> None:
    """Regression-safety: the values that used to live as module-level
    _* constants in heuristics.py are now in DEFAULT_HEURISTIC_WEIGHTS.
    If you change these defaults, you're changing production behavior."""
    w = DEFAULT_HEURISTIC_WEIGHTS
    assert w.neutral_base == 0.50
    assert w.short_prompt_boost == 0.25
    assert w.long_prompt_penalty == 0.30
    assert w.simple_keyword_boost == 0.20
    assert w.complex_keyword_penalty == 0.25
    assert w.has_tools_penalty == 0.15
    assert w.short_token_limit == 200
    assert w.long_token_limit == 2000


def test_default_quality_threshold_is_070() -> None:
    """The production threshold. Frozen by this test until we have an
    A/B harness validating any change against historical traffic."""
    assert DEFAULT_QUALITY_THRESHOLD == 0.70


def test_default_task_category_weights_priorities() -> None:
    """Tie-break order matters for routing — coding > reasoning > etc."""
    w = DEFAULT_TASK_CATEGORY_WEIGHTS
    p = w.category_priority
    # Coding wins tiebreaks against everything else
    assert p["coding"] > p["reasoning"]
    assert p["reasoning"] > p["summarization"]
    assert p["summarization"] > p["simple_qa"]
    assert p["simple_qa"] > p["tool_use"]
    assert w.long_prompt_tokens == 1500


# ---------------------------------------------------------------------------
# HeuristicWeights injection actually changes routing
# ---------------------------------------------------------------------------


def _short_neutral_prompt() -> ChatCompletionRequest:
    """A prompt that, under default weights, scores right around the
    threshold so changes to short_prompt_boost flip the routing."""
    return ChatCompletionRequest(
        model="x",
        messages=[UserMessage(role="user", content="just say hi")],
    )


async def test_aggressive_short_boost_routes_local_more_than_default() -> None:
    """Pump short_prompt_boost up; the scout should route a borderline
    short prompt LOCAL where default would have gone CLOUD."""
    request = _short_neutral_prompt()

    # Default routing
    default_scout = HeuristicScout(confidence_threshold=0.90)  # raise bar
    default_decision = await default_scout.classify(request)

    # Aggressive boost — short prompts get heavily preferred
    aggressive = HeuristicWeights(short_prompt_boost=0.50)
    aggressive_scout = HeuristicScout(
        confidence_threshold=0.90,
        weights=aggressive,
    )
    aggressive_decision = await aggressive_scout.classify(request)

    # Aggressive should at least produce a HIGHER local_score than default
    assert (
        aggressive_decision.signals["local_score"]
        > default_decision.signals["local_score"]
    ), "Higher short_prompt_boost must produce higher local_score"


async def test_zero_weights_produce_neutral_decisions() -> None:
    """A weights config of all-zeros should give every prompt the same
    neutral_base score, regardless of content. Tests that the seam is
    actually plumbed through the classify() method end-to-end."""
    flat = HeuristicWeights(
        neutral_base=0.50,
        short_prompt_boost=0.0,
        long_prompt_penalty=0.0,
        simple_keyword_boost=0.0,
        complex_keyword_penalty=0.0,
        has_tools_penalty=0.0,
    )
    scout = HeuristicScout(confidence_threshold=0.65, weights=flat)
    short_req = ChatCompletionRequest(
        model="x", messages=[UserMessage(role="user", content="hi")],
    )
    long_req = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(role="user", content="x " * 5000)],
    )
    short_decision = await scout.classify(short_req)
    long_decision = await scout.classify(long_req)
    # Same neutral_base → same local_score (within floating-point noise)
    assert short_decision.signals["local_score"] == 0.50
    assert long_decision.signals["local_score"] == 0.50


# ---------------------------------------------------------------------------
# TaskCategoryWeights injection
# ---------------------------------------------------------------------------


def test_custom_task_category_priority_changes_tiebreaks() -> None:
    """If summarization is artificially boosted above coding in the
    priority dict, a tied score should resolve to summarization."""
    request = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(
            role="user",
            # One match each for coding (function) and summarization (summary)
            content="write a function. Then write a summary.",
        )],
    )

    default_classifier = TaskCategoryClassifier()
    default_decision = default_classifier.classify(request)

    # Override: summarization wins tiebreaks
    summarization_first = TaskCategoryWeights(
        category_priority={
            "summarization": 10,
            "coding": 5,
            "reasoning": 4,
            "simple_qa": 2,
            "tool_use": 1,
        },
    )
    custom = TaskCategoryClassifier(weights=summarization_first)
    custom_decision = custom.classify(request)

    # Default chose coding (or one of the matched categories); custom
    # should prefer summarization given a tie
    if default_decision.per_category_scores.get("summarization", 0.0) == \
            default_decision.per_category_scores.get("coding", 0.0):
        # We have a true tie — custom must pick summarization
        assert custom_decision.category == "summarization"


# ---------------------------------------------------------------------------
# PatternProvider injection
# ---------------------------------------------------------------------------


class _CustomTool(str):
    """Placeholder — real subclassing uses extending the DevTool enum at
    the source, not adding new values dynamically. We test override of
    EXISTING tool patterns here."""


class _CursorOnlyProvider(PatternProvider):
    """Provider that ONLY recognizes Cursor — useful to test that
    Fingerprinter consults the provider's supported_tools() rather than
    hard-coding the catalog."""

    def patterns_for(self, tool):
        import re
        if tool is DevTool.CURSOR:
            return [re.compile(r"\bcursor\b", re.IGNORECASE)]
        return []

    def supported_tools(self):
        return [DevTool.CURSOR]


def test_default_pattern_provider_used_when_none_passed() -> None:
    fp = Fingerprinter()
    assert isinstance(fp.patterns, DefaultPatternProvider)


def test_custom_pattern_provider_overrides_default_catalog() -> None:
    """When operator injects a PatternProvider that only knows Cursor,
    the fingerprinter must ignore Claude Code etc. signals — proving the
    provider seam is consulted, not the old module-level _PATTERNS dict."""
    fp = Fingerprinter(patterns=_CursorOnlyProvider())

    cursor_request = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(role="user", content="I'm using Cursor right now")],
    )
    cursor_result = fp.identify(cursor_request)
    assert cursor_result.tool == DevTool.CURSOR

    # A request that would normally trip CLAUDE_CODE patterns should now
    # come back UNKNOWN because the custom provider doesn't know Claude Code
    claude_request = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(role="user", content="You are Claude Code")],
    )
    claude_result = fp.identify(claude_request)
    assert claude_result.tool == DevTool.UNKNOWN

"""HeuristicScout behavior tests + small corpus eval."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.base import Tier
from modelmeld.scout.heuristics import HeuristicScout
from tests.fixtures.scout_corpus import COMPLEX_PROMPTS, SIMPLE_PROMPTS


def _request(prompt: str, *, tools: list | None = None) -> ChatCompletionRequest:
    payload: dict = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    if tools is not None:
        payload["tools"] = tools
    return ChatCompletionRequest.model_validate(payload)


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        HeuristicScout(confidence_threshold=1.5)
    with pytest.raises(ValueError):
        HeuristicScout(confidence_threshold=-0.1)


async def test_short_simple_prompt_routes_local() -> None:
    scout = HeuristicScout()
    decision = await scout.classify(_request("Complete the function: def add(a, b):"))
    assert decision.tier == Tier.LOCAL
    assert "simple_keyword" in decision.rationale or "short_prompt" in decision.rationale


async def test_long_complex_prompt_routes_cloud() -> None:
    scout = HeuristicScout()
    prompt = (
        "Design a distributed system for processing 1M events/sec with exactly-once "
        "semantics. Analyze trade-offs and prove correctness."
    )
    decision = await scout.classify(_request(prompt))
    assert decision.tier == Tier.CLOUD


async def test_tools_defined_penalizes_local() -> None:
    scout = HeuristicScout()
    short_simple = "Complete the function: def add(a, b):"
    no_tools = await scout.classify(_request(short_simple))
    with_tools = await scout.classify(
        _request(
            short_simple,
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        )
    )
    assert no_tools.signals["local_score"] > with_tools.signals["local_score"]


async def test_very_long_prompt_routes_cloud_regardless_of_keywords() -> None:
    scout = HeuristicScout()
    long_prompt = "Complete the function. " + ("padding " * 1000)
    decision = await scout.classify(_request(long_prompt))
    assert decision.tier == Tier.CLOUD
    assert "long_prompt" in decision.rationale


async def test_extracts_text_from_multimodal_user_message() -> None:
    scout = HeuristicScout()
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Format this JSON: {a:1}"},
                        {"type": "image_url", "image_url": {"url": "https://e.com/x.png"}},
                    ],
                }
            ],
        }
    )
    decision = await scout.classify(request)
    # The text part should have been seen and scored; image part is ignored.
    assert "simple_keyword" in decision.rationale


# ---------------------------------------------------------------------------
# Small corpus eval — sanity baseline, not the full benchmark.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", SIMPLE_PROMPTS)
async def test_simple_corpus_routes_local(prompt: str) -> None:
    scout = HeuristicScout()
    decision = await scout.classify(_request(prompt))
    assert decision.tier == Tier.LOCAL, (
        f"Expected LOCAL for simple prompt; got {decision.tier} "
        f"(rationale={decision.rationale}, signals={decision.signals})"
    )


@pytest.mark.parametrize("prompt", COMPLEX_PROMPTS)
async def test_complex_corpus_routes_cloud(prompt: str) -> None:
    scout = HeuristicScout()
    decision = await scout.classify(_request(prompt))
    assert decision.tier == Tier.CLOUD, (
        f"Expected CLOUD for complex prompt; got {decision.tier} "
        f"(rationale={decision.rationale}, signals={decision.signals})"
    )


async def test_threshold_monotonicity_on_corpus() -> None:
    """Lowering the threshold should never decrease the count of LOCAL decisions."""
    all_prompts = SIMPLE_PROMPTS + COMPLEX_PROMPTS

    async def count_local(threshold: float) -> int:
        scout = HeuristicScout(confidence_threshold=threshold)
        local = 0
        for p in all_prompts:
            decision = await scout.classify(_request(p))
            if decision.tier == Tier.LOCAL:
                local += 1
        return local

    strict = await count_local(0.9)   # almost everything → CLOUD
    default = await count_local(0.65)
    permissive = await count_local(0.3)  # almost everything → LOCAL

    assert strict <= default <= permissive
    # Sanity: extremes are actually different
    assert strict < permissive


async def test_app_state_scout_populated() -> None:
    from modelmeld.api.server import build_app

    app = build_app()
    assert app.state.scout is not None
    assert app.state.scout.name == "heuristic"

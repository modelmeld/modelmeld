# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""`-quality` on agentic (tool-bearing) work — capability-first ranking.

Confirmed bug: QUALITY restricts to frontier then ranks cost-ascending, so it
picks the *cheapest* frontier model clearing the threshold — a small/fast
frontier model that no-ops on sustained agentic loops. Fix: for QUALITY +
tool-bearing requests, rank by capability (the request's category score)
descending, cost as the tie-break. Scoped to QUALITY + tools — simple QUALITY
requests keep cost-first, and other policies are unaffected.
"""

from __future__ import annotations

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    FunctionDef,
    Tool,
    UserMessage,
)
from modelmeld.scout.capability import CapabilityScout
from modelmeld.scout.multi_provider_registry import MultiProviderModelRegistry
from modelmeld.scout.registry import ModelEntry

QUALITY = "anthropic/modelmeld-quality"
AUTO = "anthropic/modelmeld-auto"


def _frontier(model_id: str, provider: str, cost_in: float, cost_out: float, score: float) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=provider,                       # anthropic/openai = frontier tier
        context_window=200000,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores={"coding": score, "tool_use": score},
    )


def _scout(*entries: ModelEntry) -> CapabilityScout:
    # eligible_providers left None so QUALITY's frontier filter drives selection.
    return CapabilityScout(
        registry=MultiProviderModelRegistry(list(entries)),
        quality_threshold=0.70,
    )


def _req(model: str, *, with_tools: bool, text: str = "refactor the auth module across these files") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[UserMessage(role="user", content=text)],
        tools=(
            [Tool(type="function", function=FunctionDef(name="edit_file", parameters={"type": "object"}))]
            if with_tools else []
        ),
    )


# Cheapest frontier model clears the bar but is the weakest; a pricier one is stronger.
def _weak_cheap() -> ModelEntry:
    return _frontier("mini-frontier", "openai", 0.25, 2.0, score=0.75)


def _strong_pricey() -> ModelEntry:
    return _frontier("big-frontier", "anthropic", 3.0, 15.0, score=0.90)


# ---------------------------------------------------------------------------

async def test_quality_agentic_picks_strongest_not_cheapest() -> None:
    scout = _scout(_weak_cheap(), _strong_pricey())
    decision = await scout.choose(_req(QUALITY, with_tools=True))
    # The bug would pick mini-frontier ($0.95 blended) over big-frontier ($7.80).
    assert decision.chosen_model_id == "big-frontier"
    assert "quality_agentic=capability_first" in decision.rationale


async def test_quality_agentic_breaks_capability_ties_by_cost() -> None:
    # Two equally-capable frontier models -> prefer the cheaper.
    cheap = _frontier("cheap-strong", "openai", 1.0, 3.0, score=0.90)
    pricey = _frontier("pricey-strong", "anthropic", 3.0, 15.0, score=0.90)
    scout = _scout(pricey, cheap)
    decision = await scout.choose(_req(QUALITY, with_tools=True))
    assert decision.chosen_model_id == "cheap-strong"


async def test_quality_simple_request_stays_cost_first() -> None:
    # No tools + max_tokens None => not autocomplete-shape => QUALITY still goes
    # frontier, but the capability-first re-rank must NOT apply. Cheapest wins.
    scout = _scout(_weak_cheap(), _strong_pricey())
    decision = await scout.choose(_req(QUALITY, with_tools=False))
    assert decision.chosen_model_id == "mini-frontier"
    assert "quality_agentic=capability_first" not in decision.rationale


async def test_auto_with_tools_does_not_get_quality_rerank() -> None:
    scout = _scout(_weak_cheap(), _strong_pricey())
    decision = await scout.choose(_req(AUTO, with_tools=True))
    assert "quality_agentic=capability_first" not in decision.rationale

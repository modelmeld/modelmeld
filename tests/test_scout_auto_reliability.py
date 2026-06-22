# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""`-auto` agentic routing drops MEASURED-unreliable models, then picks cheapest.

A model with a low measured `agentic_coding` (the chronic multi-provider
shortcut-takers) is cheap but ships latent bugs; for agentic (tool-bearing)
coding routes, `-auto` should prefer the cheapest RELIABLE model over the
cheapest-overall. Only rows carrying a measured score are filtered; un-probed
rows pass through; and the filter falls back to the full ranking rather than
503ing when nothing clears the floor.
"""
from __future__ import annotations

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    FunctionDef,
    Tool,
    UserMessage,
)
from modelmeld.scout import CapabilityScout, ModelEntry, ModelRegistry


def _entry(model_id: str, cost_in: float, *, coding: float,
           agentic: float | None, enabled: bool = True) -> ModelEntry:
    # a tool-bearing request classifies as `tool_use`; carry both category scores
    # so the threshold gate passes and the agentic_coding floor is what decides.
    scores: dict[str, float] = {"coding": coding, "tool_use": coding}
    if agentic is not None:
        scores["agentic_coding"] = agentic
    return ModelEntry(
        model_id=model_id, provider="vllm", context_window=100_000,
        cost_per_m_input=cost_in, cost_per_m_output=cost_in * 3,
        task_scores=scores, last_updated="2026-06-17", source="test",
        enabled=enabled,
    )


def _req_auto_tools() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="anthropic/modelmeld-auto",
        messages=[UserMessage(role="user", content="refactor this code")],
        tools=[Tool(type="function", function=FunctionDef(name="edit_file"))],
    )


async def test_floor_excludes_cheap_shortcut_taker(monkeypatch) -> None:
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)  # default 0.40
    registry = ModelRegistry([
        _entry("cheap-shortcut", 0.1, coding=0.85, agentic=0.12),  # cheapest, unreliable
        _entry("reliable", 0.5, coding=0.80, agentic=0.71),         # pricier, reliable
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    assert decision.chosen_model_id == "reliable"


async def test_floor_disabled_picks_cheapest(monkeypatch) -> None:
    monkeypatch.setenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", "0")
    registry = ModelRegistry([
        _entry("cheap-shortcut", 0.1, coding=0.85, agentic=0.12),
        _entry("reliable", 0.5, coding=0.80, agentic=0.71),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    assert decision.chosen_model_id == "cheap-shortcut"


async def test_floor_falls_back_when_all_unreliable(monkeypatch) -> None:
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)
    registry = ModelRegistry([
        _entry("cheap-bad", 0.1, coding=0.85, agentic=0.12),
        _entry("mid-bad", 0.5, coding=0.80, agentic=0.25),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    # no reliable survivor → fall back to cheapest, never 503
    assert decision.chosen_model_id == "cheap-bad"


async def test_unprobed_deprefered_when_reliable_exists(monkeypatch) -> None:
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)
    registry = ModelRegistry([
        _entry("cheap-unscored", 0.1, coding=0.85, agentic=None),  # no measured score
        _entry("reliable", 0.5, coding=0.80, agentic=0.71),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    # un-probed cheapest is DE-PREFERRED: a measured-reliable model exists, so it
    # wins despite being pricier (avoids the cheap-but-unmeasured trap).
    assert decision.chosen_model_id == "reliable"


async def test_unprobed_wins_when_no_measured_reliable(monkeypatch) -> None:
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)
    registry = ModelRegistry([
        _entry("cheap-unscored", 0.1, coding=0.85, agentic=None),   # un-probed, cheapest
        _entry("measured-bad", 0.5, coding=0.80, agentic=0.20),     # measured, below floor
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    # no measured-RELIABLE model → fall back to the full ranking → cheapest wins
    # (a genuinely-new/all-unprobed pool still routes; never 503).
    assert decision.chosen_model_id == "cheap-unscored"


async def test_disabled_unprobed_is_not_chosen_as_fallback(monkeypatch) -> None:
    """A disabled un-probed model must NOT be the -auto agentic fallback.

    Without `enabled`, the de-prefer logic falls back to the full ranking when
    nothing is measured-reliable — which could route to an un-validated model.
    Disabling filters it out of `rank()` BEFORE the fallback, so the enabled
    (measured-reliable) model wins even though the disabled one is cheaper.
    """
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)
    registry = ModelRegistry([
        _entry("cheap-unvalidated", 0.1, coding=0.85, agentic=None, enabled=False),
        _entry("reliable", 0.5, coding=0.80, agentic=0.71, enabled=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    assert decision.chosen_model_id == "reliable"

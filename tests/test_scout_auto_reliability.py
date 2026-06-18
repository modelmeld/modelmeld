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
           agentic: float | None) -> ModelEntry:
    # a tool-bearing request classifies as `tool_use`; carry both category scores
    # so the threshold gate passes and the agentic_coding floor is what decides.
    scores: dict[str, float] = {"coding": coding, "tool_use": coding}
    if agentic is not None:
        scores["agentic_coding"] = agentic
    return ModelEntry(
        model_id=model_id, provider="vllm", context_window=100_000,
        cost_per_m_input=cost_in, cost_per_m_output=cost_in * 3,
        task_scores=scores, last_updated="2026-06-17", source="test",
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


async def test_unprobed_rows_pass_through(monkeypatch) -> None:
    monkeypatch.delenv("MODELMELD_AGENTIC_RELIABILITY_FLOOR", raising=False)
    registry = ModelRegistry([
        _entry("cheap-unscored", 0.1, coding=0.85, agentic=None),  # no measured score
        _entry("reliable", 0.5, coding=0.80, agentic=0.71),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)
    decision = await scout.choose(_req_auto_tools())
    # un-probed cheapest is NOT filtered (can't judge it) → it wins on cost
    assert decision.chosen_model_id == "cheap-unscored"

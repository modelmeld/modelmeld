# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""D1 latency term — the routing-objective redesign's first dimension.

Covers the `ModelEntry` latency fields + `estimated_turn_latency_s`, the
latency-adjusted ranking in `ModelRegistry.rank` /
`MultiProviderModelRegistry.rank`, and the scout wiring that applies the term
ONLY to `-auto` + tool-bearing requests.

The load-bearing claims (see docs/design-routing-objective.md):
  * `-saver` / non-`-auto` ranking is byte-identical to the cost-only ordering
    (latency_weight defaults to 0).
  * the latency term BREAKS NEAR-COST TIES toward the faster option (the
    per-(model x provider) case) but DOES NOT override a genuine cost gap.
  * rows with no latency data are no-ops (never made to look fast).
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
from modelmeld.scout.registry import ModelEntry, ModelRegistry


def _entry(
    model_id: str,
    provider: str,
    cost_in: float,
    cost_out: float,
    *,
    coding: float = 0.8,
    ttft: float | None = None,
    tps: float | None = None,
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=200000,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        # tool-bearing requests classify as `tool_use`; score both so the
        # scout-level tests can pick under either classification.
        task_scores={"coding": coding, "tool_use": coding},
        median_ttft_s=ttft,
        median_output_tps=tps,
        latency_source="test" if tps is not None else "",
    )


# ---------------------------------------------------------------------------
# ModelEntry latency fields + estimate
# ---------------------------------------------------------------------------

def test_latency_fields_default_none() -> None:
    e = _entry("m", "vllm", 1.0, 1.0)
    assert e.median_ttft_s is None
    assert e.median_output_tps is None
    assert e.latency_source == ""


def test_estimated_turn_latency_none_without_throughput() -> None:
    e = _entry("m", "vllm", 1.0, 1.0)  # no tps
    assert e.estimated_turn_latency_s(61000, 128) is None


def test_estimated_turn_latency_uses_ttft_plus_output_only() -> None:
    # v1 intentionally does NOT extrapolate prefill from input_tokens.
    e = _entry("m", "vllm", 1.0, 1.0, ttft=0.5, tps=100.0)
    # 0.5 + 128/100 = 1.78; input size must not change it.
    assert abs(e.estimated_turn_latency_s(1_000, 128) - 1.78) < 1e-9
    assert e.estimated_turn_latency_s(1_000, 128) == e.estimated_turn_latency_s(
        999_999, 128
    )


def test_from_json_round_trips_latency() -> None:
    reg = ModelRegistry.from_json(
        {
            "version": 1,
            "models": [
                {
                    "model_id": "x",
                    "provider": "vllm",
                    "context_window": 1000,
                    "cost_per_m_input": 1.0,
                    "cost_per_m_output": 1.0,
                    "median_ttft_s": 0.5,
                    "median_output_tps": 100.0,
                    "latency_source": "artificial_analysis@medium",
                },
                {  # latency omitted -> None
                    "model_id": "y",
                    "provider": "vllm",
                    "context_window": 1000,
                    "cost_per_m_input": 1.0,
                    "cost_per_m_output": 1.0,
                },
            ],
        }
    )
    x = reg.get("x")
    assert x is not None and x.median_ttft_s == 0.5 and x.median_output_tps == 100.0
    assert x.latency_source == "artificial_analysis@medium"
    y = reg.get("y")
    assert y is not None and y.median_ttft_s is None and y.median_output_tps is None


# ---------------------------------------------------------------------------
# rank() — latency_weight=0 is the unchanged cost-only ordering
# ---------------------------------------------------------------------------

def test_rank_default_is_cost_only_and_stable() -> None:
    reg = ModelRegistry(
        [
            _entry("a", "openai", 10.0, 30.0),
            _entry("b", "openai", 1.0, 3.0),
            _entry("c", "openai", 5.0, 15.0),
        ]
    )
    default_order = [e.model_id for e, _ in reg.rank("coding")]
    explicit_zero = [
        e.model_id for e, _ in reg.rank("coding", latency_weight=0.0)
    ]
    assert default_order == explicit_zero == ["b", "c", "a"]


def test_latency_weight_zero_ignores_latency_data() -> None:
    # Latency data present but weight 0 -> ordering unaffected.
    reg = ModelRegistry(
        [
            _entry("cheap-slow", "vllm", 1.0, 1.0, ttft=2.0, tps=30.0),
            _entry("pricey-fast", "vllm", 2.0, 2.0, ttft=0.2, tps=200.0),
        ]
    )
    order = [e.model_id for e, _ in reg.rank("coding", latency_weight=0.0)]
    assert order == ["cheap-slow", "pricey-fast"]  # pure cost


# ---------------------------------------------------------------------------
# rank() — latency breaks near-cost ties (the per-provider case)
# ---------------------------------------------------------------------------

def test_latency_breaks_equal_cost_tie_toward_fast_provider() -> None:
    # Same model, same cost, two providers, different speed.
    fast = _entry("m", "prov-fast", 0.11, 0.80, ttft=0.4, tps=150.0)
    slow = _entry("m", "prov-slow", 0.11, 0.80, ttft=1.2, tps=60.0)
    reg = MultiProviderModelRegistry([slow, fast])  # insert slow first

    cost_only = [e.provider for e, _ in reg.rank("coding", latency_weight=0.0)]
    assert cost_only == ["prov-slow", "prov-fast"]  # tie -> insertion order

    ranked = reg.rank(
        "coding",
        latency_weight=0.02,
        latency_ref_input_tokens=61000,
        latency_ref_output_tokens=128,
    )
    assert next(e.provider for e, _ in ranked) == "prov-fast"
    # Returned cost must remain the REAL blended cost, not the effective key.
    assert abs(ranked[0][1] - fast.blended_cost_per_m()) < 1e-9


def test_latency_does_not_override_genuine_cost_gap() -> None:
    # cheap+slow vs ~2x-pricier+fast: cost wins, by design.
    cheap_slow = _entry("qwenish", "p1", 0.11, 0.80, ttft=1.2, tps=60.0)
    pricey_fast = _entry("minimaxish", "p2", 0.255, 1.00, ttft=0.4, tps=150.0)
    reg = MultiProviderModelRegistry([cheap_slow, pricey_fast])
    ranked = reg.rank(
        "coding",
        latency_weight=0.02,
        latency_ref_input_tokens=61000,
        latency_ref_output_tokens=128,
    )
    assert ranked[0][0].model_id == "qwenish"


def test_unmeasured_row_is_noop_under_latency() -> None:
    # A row with no latency keeps plain cost; it is not treated as instant.
    measured_fast = _entry("a", "p1", 1.0, 1.0, ttft=0.2, tps=200.0)
    unmeasured_cheaper = _entry("b", "p2", 0.9, 0.9)  # cheaper, no latency
    reg = MultiProviderModelRegistry([measured_fast, unmeasured_cheaper])
    ranked = reg.rank(
        "coding",
        latency_weight=0.02,
        latency_ref_input_tokens=61000,
        latency_ref_output_tokens=128,
    )
    # unmeasured stays at its plain (cheaper) cost -> still first; latency
    # never fabricates a speed advantage for the measured row.
    assert ranked[0][0].model_id == "b"


# ---------------------------------------------------------------------------
# Scout wiring — D1 applies ONLY to -auto + tool-bearing
# ---------------------------------------------------------------------------

def _tool() -> Tool:
    return Tool(
        type="function",
        function=FunctionDef(name="read_file", parameters={"type": "object"}),
    )


def _req(model: str, *, with_tools: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[UserMessage(role="user", content="refactor this module")],
        tools=[_tool()] if with_tools else [],
    )


def _oss_scout() -> CapabilityScout:
    reg = MultiProviderModelRegistry(
        [
            _entry("coderA", "vllm", 0.5, 1.5, ttft=0.5, tps=120.0),
            _entry("coderB", "vllm", 0.6, 1.6, ttft=0.3, tps=180.0),
        ]
    )
    return CapabilityScout(
        registry=reg,
        quality_threshold=0.70,
        eligible_providers=frozenset({"vllm"}),
    )


async def test_auto_with_tools_applies_latency_term() -> None:
    scout = _oss_scout()
    decision = await scout.choose(_req("anthropic/modelmeld-auto", with_tools=True))
    assert "d1=latency" in decision.rationale


async def test_saver_with_tools_does_not_apply_latency() -> None:
    scout = _oss_scout()
    decision = await scout.choose(_req("anthropic/modelmeld-saver", with_tools=True))
    assert "d1=latency" not in decision.rationale


async def test_auto_without_tools_does_not_apply_latency() -> None:
    scout = _oss_scout()
    decision = await scout.choose(_req("anthropic/modelmeld-auto", with_tools=False))
    assert "d1=latency" not in decision.rationale


async def test_non_alias_with_tools_does_not_apply_latency() -> None:
    scout = _oss_scout()
    decision = await scout.choose(_req("qwen3-coder-next", with_tools=True))
    assert "d1=latency" not in decision.rationale

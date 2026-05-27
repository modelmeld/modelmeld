"""ModelRegistry — schema validation, default-registry sanity, and pick() semantics."""

from __future__ import annotations

from modelmeld.scout.registry import ModelEntry, ModelRegistry, default_registry


# ---------------------------------------------------------------------------
# Default registry shipped with the package
# ---------------------------------------------------------------------------

def test_default_registry_loads() -> None:
    reg = default_registry()
    assert len(reg) >= 10, f"default registry should ship ≥10 models, got {len(reg)}"


def test_default_registry_singleton() -> None:
    assert default_registry() is default_registry()


def test_default_registry_covers_each_task_category() -> None:
    reg = default_registry()
    required_tasks = {"coding", "reasoning", "simple_qa", "summarization", "tool_use"}
    for entry in reg.all_entries():
        missing = required_tasks - entry.task_scores.keys()
        assert not missing, (
            f"{entry.model_id} missing task scores for: {missing}"
        )


def test_default_registry_has_known_models() -> None:
    reg = default_registry()
    expected_subset = {
        "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5",
        "gpt-5", "gpt-5-mini",
        "qwen2.5-coder-32b-instruct", "qwen2.5-coder-7b-instruct",
        "deepseek-v3.2",
    }
    actual = {e.model_id for e in reg.all_entries()}
    missing = expected_subset - actual
    assert not missing, f"default registry missing: {missing}"


def test_default_registry_window_and_costs_positive() -> None:
    for entry in default_registry().all_entries():
        assert entry.context_window > 0
        assert entry.cost_per_m_input >= 0  # local models can be near-zero
        assert entry.cost_per_m_output >= 0


def test_default_registry_scores_in_unit_range() -> None:
    for entry in default_registry().all_entries():
        for task, score in entry.task_scores.items():
            assert 0.0 <= score <= 1.0, f"{entry.model_id}.{task} = {score}"


def test_default_registry_includes_sub_haiku_tier() -> None:
    """Three sub-Haiku-tier models for trivial-fast routing."""
    reg = default_registry()
    ids = {e.model_id for e in reg.all_entries()}
    assert "phi-4-mini-instruct" in ids
    assert "granite-4-micro" in ids
    assert "gemma-3-4b" in ids


def test_default_registry_supports_tools_matches_openrouter_audit() -> None:
    """Registry `supports_tools` flags must match
    empirical OpenRouter-side tool-call capability per the audit script
    (`openrouter-tool-capability-audit.ps1`). Sources of false flags:

    1. Small parameter count → unreliable tool-call protocol
       (SLM-for-Agents survey arxiv:2510.03847):
         - phi-4 (14B), phi-4-mini-instruct (3.8B)
         - granite-4-micro (3B), gemma-3-4b (4B)
    2. OpenRouter sub-provider routing landing on non-tool-capable
       endpoint even though the canonical model supports tools elsewhere:
         - deepseek-r1-distill-llama-70b (distilled reasoning;
           Tools fail on OR despite Together/Fireworks supporting)
         - qwen2.5-coder-7b-instruct (404 on OR — possibly deprecated)
         - qwen2.5-coder-32b-instruct (404 on OR for tool-use endpoint)

    These flags must stay accurate or scout's tool-capability filter
    routes tool-bearing requests to models OpenRouter rejects with
    'no endpoints found that support tool use' (HTTP 404).
    """
    reg = default_registry()
    expected_no_tool_support = {
        "phi-4",
        "phi-4-mini-instruct",
        "granite-4-micro",
        "gemma-3-4b",
        "deepseek-r1-distill-llama-70b",
        "qwen2.5-coder-7b-instruct",
        "qwen2.5-coder-32b-instruct",
    }
    for model_id in expected_no_tool_support:
        entry = reg.get(model_id)
        assert entry is not None, f"Expected {model_id} in default registry"
        assert entry.supports_tools is False, (
            f"{model_id} must NOT advertise tool support — "
            f"OpenRouter returns 404 'no tool endpoints' for it"
        )


def test_default_registry_oss_premium_tier_supports_tools() -> None:
    """The 30B+ class models in the OSS premium tier reliably handle
    tool calls per SLM-for-Agents research. Default supports_tools=True."""
    reg = default_registry()
    for premium_id in (
        "deepseek-v4-pro", "kimi-k2.6", "qwen3-coder-480b",
        "gpt-oss-120b", "llama-4-scout", "deepseek-r1",
    ):
        entry = reg.get(premium_id)
        if entry is not None:  # not all may be present in every snapshot
            assert entry.supports_tools is True, (
                f"{premium_id} should support tools (default True)"
            )


# ---------------------------------------------------------------------------
# ModelEntry helpers
# ---------------------------------------------------------------------------

def test_blended_cost_default_weights() -> None:
    entry = ModelEntry(
        model_id="x", provider="p", context_window=100,
        cost_per_m_input=1.0, cost_per_m_output=10.0,
    )
    # 0.6 * 1.0 + 0.4 * 10.0 = 4.6
    assert entry.blended_cost_per_m() == 4.6


def test_blended_cost_custom_weights() -> None:
    entry = ModelEntry(
        model_id="x", provider="p", context_window=100,
        cost_per_m_input=1.0, cost_per_m_output=10.0,
    )
    assert entry.blended_cost_per_m(input_weight=1.0, output_weight=0.0) == 1.0
    assert entry.blended_cost_per_m(input_weight=0.0, output_weight=1.0) == 10.0


def test_meets_threshold() -> None:
    entry = ModelEntry(
        model_id="x", provider="p", context_window=100,
        cost_per_m_input=1.0, cost_per_m_output=2.0,
        task_scores={"coding": 0.8},
    )
    assert entry.meets_threshold("coding", 0.8) is True
    assert entry.meets_threshold("coding", 0.81) is False
    # Missing scores default to 0.0 → fails any positive threshold
    assert entry.meets_threshold("reasoning", 0.5) is False
    assert entry.meets_threshold("reasoning", 0.0) is True  # 0.0 >= 0.0


# ---------------------------------------------------------------------------
# pick() semantics — table-driven
# ---------------------------------------------------------------------------

def _toy_registry() -> ModelRegistry:
    """Three-model registry with deterministic costs for pick() testing."""
    return ModelRegistry([
        ModelEntry(
            model_id="premium",
            provider="cloud-a",
            context_window=200_000,
            cost_per_m_input=5.0, cost_per_m_output=25.0,
            task_scores={"coding": 0.90, "simple_qa": 0.95},
        ),
        ModelEntry(
            model_id="mid",
            provider="cloud-b",
            context_window=128_000,
            cost_per_m_input=0.25, cost_per_m_output=2.0,
            task_scores={"coding": 0.70, "simple_qa": 0.92},
        ),
        ModelEntry(
            model_id="cheap-local",
            provider="vllm",
            context_window=32_000,
            cost_per_m_input=0.07, cost_per_m_output=0.07,
            task_scores={"coding": 0.55, "simple_qa": 0.78},
        ),
    ])


def test_pick_returns_cheapest_meeting_threshold() -> None:
    reg = _toy_registry()
    # All 3 meet 0.55, cheap-local is cheapest
    assert reg.pick("coding", quality_threshold=0.55).model_id == "cheap-local"
    # Only premium + mid meet 0.65, mid is cheaper
    assert reg.pick("coding", quality_threshold=0.65).model_id == "mid"
    # Only premium meets 0.80
    assert reg.pick("coding", quality_threshold=0.80).model_id == "premium"


def test_pick_returns_none_when_no_candidate() -> None:
    reg = _toy_registry()
    assert reg.pick("coding", quality_threshold=0.99) is None


def test_pick_eligible_providers_filter() -> None:
    reg = _toy_registry()
    # Restrict to vllm — only cheap-local qualifies for any threshold
    assert reg.pick("coding", quality_threshold=0.55, eligible_providers=frozenset({"vllm"})).model_id == "cheap-local"
    # Restrict to vllm with threshold > local's score → no candidate
    assert reg.pick("coding", quality_threshold=0.60, eligible_providers=frozenset({"vllm"})) is None
    # Restrict to clouds only with threshold 0.80
    chosen = reg.pick("coding", quality_threshold=0.80, eligible_providers=frozenset({"cloud-a", "cloud-b"}))
    assert chosen.model_id == "premium"


def test_pick_min_context_window_filter() -> None:
    reg = _toy_registry()
    # 32K window excludes cheap-local
    assert reg.pick("coding", quality_threshold=0.50, min_context_window=100_000).model_id == "mid"
    # Higher window still includes premium
    assert reg.pick("coding", quality_threshold=0.50, min_context_window=150_000).model_id == "premium"


def test_pick_missing_task_category_returns_none() -> None:
    reg = _toy_registry()
    # No model has scores for "vision"
    assert reg.pick("vision", quality_threshold=0.5) is None


def test_pick_tiebreak_prefers_larger_window() -> None:
    reg = ModelRegistry([
        ModelEntry(
            model_id="small-window",
            provider="p", context_window=8_000,
            cost_per_m_input=1.0, cost_per_m_output=2.0,
            task_scores={"coding": 0.8},
        ),
        ModelEntry(
            model_id="big-window",
            provider="p", context_window=200_000,
            cost_per_m_input=1.0, cost_per_m_output=2.0,
            task_scores={"coding": 0.8},
        ),
    ])
    # Same cost; tie-breaker prefers larger window
    assert reg.pick("coding", quality_threshold=0.8).model_id == "big-window"


def test_rank_returns_all_candidates_sorted() -> None:
    reg = _toy_registry()
    ranked = reg.rank("simple_qa", quality_threshold=0.0)
    assert [e.model_id for e, _ in ranked] == ["cheap-local", "mid", "premium"]
    # Threshold filters
    ranked = reg.rank("simple_qa", quality_threshold=0.90)
    assert [e.model_id for e, _ in ranked] == ["mid", "premium"]


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------

def test_from_json_round_trip() -> None:
    payload = {
        "version": 1,
        "models": [
            {
                "model_id": "m1",
                "provider": "p",
                "context_window": 1000,
                "cost_per_m_input": 0.1,
                "cost_per_m_output": 0.2,
                "task_scores": {"coding": 0.7},
                "last_updated": "2026-05-17T00:00:00Z",
                "source": "test",
            }
        ],
    }
    reg = ModelRegistry.from_json(payload)
    assert "m1" in reg
    entry = reg.get("m1")
    assert entry.context_window == 1000
    assert entry.task_scores["coding"] == 0.7


def test_from_json_rejects_future_version() -> None:
    import pytest
    with pytest.raises(ValueError, match="version"):
        ModelRegistry.from_json({"version": 99, "models": []})


def test_from_json_empty_models() -> None:
    reg = ModelRegistry.from_json({"version": 1, "models": []})
    assert len(reg) == 0
    assert reg.pick("coding") is None


# ---------------------------------------------------------------------------
# Default registry sanity: realistic routing decisions
# ---------------------------------------------------------------------------

def test_default_registry_picks_haiku_for_simple_qa_at_low_threshold() -> None:
    """Simple Q&A at lenient threshold should land on a cheap model."""
    reg = default_registry()
    chosen = reg.pick("simple_qa", quality_threshold=0.90)
    assert chosen is not None
    # Should be a cheap model (Haiku, mini, or local) — not Opus
    assert chosen.model_id != "claude-opus-4-7"
    assert chosen.blended_cost_per_m() < 5.0


def test_default_registry_picks_competent_coder_at_high_threshold() -> None:
    """High-quality coding picks should beat the 0.80 task-score bar.

    Note: as of the 2026-05-24 lineup refresh, open-weights coders like
    Qwen3-Coder-480B (coding=0.90) now exceed frontier scores. The
    cheapest-at-quality picker correctly picks Qwen3-Coder-480B at
    $0.90/Mtok over Claude Opus at $5/Mtok input. This is the moat
    working: customers get better-than-frontier coding quality at
    a fraction of the cost. The old version of this test asserted
    frontier wins — that's no longer the case for coding specifically.
    """
    reg = default_registry()
    chosen = reg.pick("coding", quality_threshold=0.80)
    assert chosen is not None
    assert chosen.task_scores["coding"] >= 0.80
    # Cost-aware pick: under $5/Mtok blended (excludes Claude Opus)
    assert chosen.blended_cost_per_m() < 5.0


def test_default_registry_local_only_routing() -> None:
    """Sovereignty mode: restrict to vllm provider."""
    reg = default_registry()
    chosen = reg.pick(
        "coding",
        quality_threshold=0.70,
        eligible_providers=frozenset({"vllm"}),
    )
    assert chosen is not None
    assert chosen.provider == "vllm"
    # Should pick a competent local coder
    assert chosen.task_scores["coding"] >= 0.70

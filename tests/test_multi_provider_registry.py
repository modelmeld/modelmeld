# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""MultiProviderModelRegistry — composite-key indexing across providers."""

from __future__ import annotations

import logging

import pytest

from modelmeld.scout.multi_provider_registry import (
    MultiProviderModelRegistry,
    default_multi_provider_registry,
)
from modelmeld.scout.registry import ModelEntry


def _entry(model_id: str, provider: str, cost: float = 0.5) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=131072,
        cost_per_m_input=cost,
        cost_per_m_output=cost,
        task_scores={
            "coding": 0.80,
            "reasoning": 0.75,
            "simple_qa": 0.85,
            "summarization": 0.78,
            "tool_use": 0.82,
        },
    )


def test_multiple_entries_same_model_id_different_provider() -> None:
    """The whole point: same logical model across multiple providers."""
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks", cost=0.30),
            _entry("qwen3-coder-30b", "together", cost=0.35),
            _entry("qwen3-coder-30b", "openrouter", cost=0.40),
        ]
    )
    assert len(reg) == 3
    assert len(reg.entries_for("qwen3-coder-30b")) == 3
    assert reg.providers_for("qwen3-coder-30b") == frozenset(
        {"fireworks", "together", "openrouter"}
    )


def test_get_by_key_returns_specific_provider() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks", cost=0.30),
            _entry("qwen3-coder-30b", "together", cost=0.35),
        ]
    )
    fw = reg.get_by_key("qwen3-coder-30b", "fireworks")
    tg = reg.get_by_key("qwen3-coder-30b", "together")
    assert fw is not None and fw.cost_per_m_input == 0.30
    assert tg is not None and tg.cost_per_m_input == 0.35


def test_get_by_key_returns_none_for_missing() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
        ]
    )
    assert reg.get_by_key("qwen3-coder-30b", "openrouter") is None
    assert reg.get_by_key("unknown-model", "fireworks") is None


def test_entries_for_returns_all_providers_for_model() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
            _entry("qwen3-coder-30b", "together"),
            _entry("deepseek-v3.2", "openrouter"),  # different model
        ]
    )
    entries = reg.entries_for("qwen3-coder-30b")
    assert len(entries) == 2
    assert {e.provider for e in entries} == {"fireworks", "together"}


def test_entries_for_unknown_model_returns_empty() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
        ]
    )
    assert reg.entries_for("unknown-model") == []


def test_all_entries_multi_includes_every_row() -> None:
    """Base class's all_entries() collapses by model_id; all_entries_multi() doesn't."""
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
            _entry("qwen3-coder-30b", "together"),
            _entry("deepseek-v3.2", "openrouter"),
        ]
    )
    all_multi = reg.all_entries_multi()
    assert len(all_multi) == 3
    # Base class's all_entries collapses to one-per-model_id
    base_all = reg.all_entries()
    assert len(base_all) == 2


def test_providers_for_returns_provider_set() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
            _entry("qwen3-coder-30b", "together"),
        ]
    )
    assert reg.providers_for("qwen3-coder-30b") == frozenset({"fireworks", "together"})
    assert reg.providers_for("unknown") == frozenset()


def test_duplicate_key_keeps_last_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Two entries with identical (model_id, provider) — the second wins."""
    first = _entry("qwen3-coder-30b", "fireworks", cost=0.30)
    second = _entry("qwen3-coder-30b", "fireworks", cost=0.99)  # later, wins
    with caplog.at_level(logging.WARNING):
        reg = MultiProviderModelRegistry([first, second])
    assert len(reg) == 1
    kept = reg.get_by_key("qwen3-coder-30b", "fireworks")
    assert kept is not None and kept.cost_per_m_input == 0.99
    assert any("duplicate" in rec.getMessage() for rec in caplog.records)


def test_base_class_get_still_works_for_back_compat() -> None:
    """get(model_id) returns *some* entry — exact provider isn't guaranteed."""
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks"),
            _entry("qwen3-coder-30b", "together"),
        ]
    )
    entry = reg.get("qwen3-coder-30b")
    assert entry is not None
    assert entry.model_id == "qwen3-coder-30b"


def test_empty_registry_handles_cleanly() -> None:
    reg = MultiProviderModelRegistry([])
    assert len(reg) == 0
    assert reg.entries_for("anything") == []
    assert reg.providers_for("anything") == frozenset()
    assert reg.get_by_key("anything", "anywhere") is None


def test_pick_iterates_multi_provider_rows() -> None:
    """Multi-provider pick() sees ALL (model_id, provider) rows, not just
    the base's collapsed _by_id representatives. So eligible_providers
    filtering can actually find the right provider."""
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks", cost=0.30),
            _entry("qwen3-coder-30b", "together", cost=0.35),
            _entry("deepseek-v3.2", "openrouter", cost=0.50),
        ]
    )
    picked = reg.pick(task_category="coding", quality_threshold=0.70)
    assert picked is not None
    # Cheapest across all rows: qwen3-coder-30b@fireworks at 0.30
    assert picked.model_id == "qwen3-coder-30b"
    assert picked.provider == "fireworks"


def test_rank_respects_eligible_providers_across_multi_rows() -> None:
    """Provider filter applies to every (model_id, provider) row, not just
    the collapsed _by_id representative. Customer with only Fireworks
    configured can still pick a model with Fireworks availability even if
    vLLM was the last-inserted entry for that model_id."""
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks", cost=0.30),
            _entry("qwen3-coder-30b", "together", cost=0.35),
            _entry("qwen3-coder-30b", "vllm", cost=0.10),  # inserted last
        ]
    )
    ranked = reg.rank(
        task_category="coding",
        eligible_providers=frozenset({"fireworks"}),
    )
    # Without the override this would return [] (because _by_id would only
    # have vllm, which isn't in eligible). With the override, the
    # fireworks row is found.
    assert len(ranked) == 1
    entry, _cost = ranked[0]
    assert entry.provider == "fireworks"


def test_rank_returns_empty_when_no_provider_matches() -> None:
    reg = MultiProviderModelRegistry(
        [
            _entry("qwen3-coder-30b", "fireworks", cost=0.30),
            _entry("qwen3-coder-30b", "together", cost=0.35),
        ]
    )
    ranked = reg.rank(
        task_category="coding",
        eligible_providers=frozenset({"openrouter"}),
    )
    assert ranked == []


def test_rank_filters_on_tool_support_across_multi_rows() -> None:
    """Tool-support filter applies per-row. A model with one tool-capable
    provider and one tool-incapable should yield only the capable row."""
    fw = _entry("special-model", "fireworks", cost=0.30)
    # Patch the second row to have supports_tools=False
    or_ = ModelEntry(
        model_id="special-model",
        provider="openrouter",
        context_window=131072,
        cost_per_m_input=0.25,
        cost_per_m_output=0.25,
        task_scores={
            "coding": 0.80,
            "reasoning": 0.75,
            "simple_qa": 0.85,
            "summarization": 0.78,
            "tool_use": 0.82,
        },
        supports_tools=False,
    )
    reg = MultiProviderModelRegistry([fw, or_])
    ranked = reg.rank(task_category="coding", require_tool_support=True)
    assert len(ranked) == 1
    assert ranked[0][0].provider == "fireworks"


# ---------------------------------------------------------------------------
# load_default — base + overlay merge
# ---------------------------------------------------------------------------


def test_load_default_returns_multi_provider_registry() -> None:
    reg = MultiProviderModelRegistry.load_default()
    assert isinstance(reg, MultiProviderModelRegistry)
    # base (30 entries) + overlay (~15 rows) — sanity check on size
    assert len(reg) >= 40, f"expected ≥40 rows, got {len(reg)}"


def test_load_default_overlay_models_have_multiple_providers() -> None:
    """The whole point: overlay models route across multiple providers."""
    reg = MultiProviderModelRegistry.load_default()
    overlay_expected_multi = [
        "deepseek-v4-pro",  # fireworks + together + openrouter + vllm
        "gpt-oss-120b",  # same
        "llama-3.3-70b-instruct",  # together + openrouter + vllm
        "kimi-k2.6",  # fireworks + together + vllm
    ]
    for model_id in overlay_expected_multi:
        providers = reg.providers_for(model_id)
        assert len(providers) >= 3, f"{model_id} should have ≥3 providers, got {sorted(providers)}"


def test_load_default_overlay_task_scores_inherit_from_base() -> None:
    """Overlay rows with empty task_scores inherit from base by model_id."""
    reg = MultiProviderModelRegistry.load_default()
    # qwen3-coder-next@openrouter should inherit task_scores from the base
    # qwen3-coder-next@vllm entry (overlay JSON has empty task_scores).
    overlay_entry = reg.get_by_key("qwen3-coder-next", "openrouter")
    base_entry = reg.get_by_key("qwen3-coder-next", "vllm")
    assert overlay_entry is not None
    assert base_entry is not None
    assert overlay_entry.task_scores == base_entry.task_scores


def test_load_default_overlay_supports_tools_inherits_from_base() -> None:
    """Conservative AND: if base says no-tools, overlay row also says no-tools."""
    reg = MultiProviderModelRegistry.load_default()
    # phi-4 has supports_tools=False in the base — overlay rows should inherit.
    overlay_entry = reg.get_by_key("phi-4", "openrouter")
    assert overlay_entry is not None
    assert overlay_entry.supports_tools is False


def test_load_default_overlay_supports_tools_inherits_true() -> None:
    """If base says tools=True, overlay row should also say tools=True."""
    reg = MultiProviderModelRegistry.load_default()
    overlay_entry = reg.get_by_key("qwen3-coder-next", "openrouter")
    assert overlay_entry is not None
    assert overlay_entry.supports_tools is True


def test_load_default_overlay_costs_match_overlay_json() -> None:
    """Overlay row costs come from the overlay JSON, not the base."""
    reg = MultiProviderModelRegistry.load_default()
    overlay = reg.get_by_key("qwen3-coder-next", "openrouter")
    base = reg.get_by_key("qwen3-coder-next", "vllm")
    assert overlay is not None
    assert base is not None
    # The openrouter row's costs come from the overlay JSON and must
    # differ from the base vllm row (different provider, different cost).
    # Specific values validated against OpenRouter's live catalog
    # 2026-05-30 — see scripts/validate_overlay.py.
    assert overlay.cost_per_m_input == 0.11
    assert overlay.cost_per_m_output == 0.80
    assert overlay.cost_per_m_input != base.cost_per_m_input


def test_load_default_overlay_provider_model_id_is_set() -> None:
    """Overlay rows carry the provider-specific model identifier."""
    reg = MultiProviderModelRegistry.load_default()
    fw = reg.get_by_key("deepseek-v4-pro", "fireworks")
    or_ = reg.get_by_key("deepseek-v4-pro", "openrouter")
    assert fw is not None
    assert or_ is not None
    assert fw.provider_model_id == "accounts/fireworks/models/deepseek-v4-pro"
    assert or_.provider_model_id == "deepseek/deepseek-v4-pro"


def test_default_multi_provider_registry_singleton() -> None:
    assert default_multi_provider_registry() is default_multi_provider_registry()


def test_default_multi_provider_registry_includes_base_models() -> None:
    """The merge keeps base entries (anthropic, openai, google rows) intact."""
    reg = default_multi_provider_registry()
    base_models = {"claude-opus-4-7", "gpt-5", "gemini-2-5-pro"}
    actual = {e.model_id for e in reg.all_entries_multi()}
    missing = base_models - actual
    assert not missing, f"base registry models missing after merge: {missing}"

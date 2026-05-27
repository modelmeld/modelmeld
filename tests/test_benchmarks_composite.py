"""Multi-source composition + weighted merging in RegistryRefresher."""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock

import pytest

from modelmeld.scout.benchmarks import (
    DEFAULT_SOURCE_WEIGHTS,
    RegistryRefresher,
)
from modelmeld.scout.benchmarks.base import BenchmarkSource
from modelmeld.scout.benchmarks.refresher import _merge_sources
from modelmeld.scout.registry import ModelEntry, ModelRegistry


class _StaticSource(BenchmarkSource):
    """Source that returns a fixed list of entries; tests-only."""

    def __init__(self, source_name: str, entries: Iterable[ModelEntry]) -> None:
        self.name = source_name
        self._entries = list(entries)

    async def fetch(self) -> list[ModelEntry]:
        return list(self._entries)


def _entry(
    model_id: str,
    source: str,
    *,
    coding: float | None = None,
    reasoning: float | None = None,
    cost_in: float = 0.0,
    cost_out: float = 0.0,
    context_window: int = 0,
    provider: str = "unknown",
) -> ModelEntry:
    scores = {}
    if coding is not None:
        scores["coding"] = coding
    if reasoning is not None:
        scores["reasoning"] = reasoning
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores=scores,
        last_updated="2026-05-17T00:00:00Z",
        source=source,
    )


# ---------------------------------------------------------------------------
# _merge_sources — weighted average across sources
# ---------------------------------------------------------------------------

def test_merge_weights_aider_higher_than_aa_for_coding() -> None:
    """Aider weight 1.5 vs AA weight 1.0; weighted average favors Aider."""
    per_source = {
        "artificial_analysis": [_entry("x", "artificial_analysis", coding=0.70)],
        "aider_polyglot": [_entry("x", "aider_polyglot", coding=0.90)],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    assert len(merged) == 1
    # weighted: (0.70 * 1.0 + 0.90 * 1.5) / (1.0 + 1.5) = 2.05 / 2.5 = 0.82
    assert abs(merged[0].task_scores["coding"] - 0.82) < 1e-6


def test_merge_picks_metadata_from_highest_weighted_source() -> None:
    """Aider has weight 1.5 — its provider hint wins over AA's 1.0."""
    per_source = {
        "artificial_analysis": [
            _entry("x", "artificial_analysis", coding=0.70, provider="anthropic", cost_in=5.0, cost_out=25.0, context_window=200000)
        ],
        "aider_polyglot": [
            _entry("x", "aider_polyglot", coding=0.90, provider="anthropic-aider")
        ],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    # Aider's provider hint wins (higher weight, non-default value)
    assert merged[0].provider == "anthropic-aider"
    # AA's context_window/cost survives because Aider's are zero (default)
    assert merged[0].context_window == 200000
    assert merged[0].cost_per_m_input == 5.0
    assert merged[0].cost_per_m_output == 25.0


def test_merge_task_independence() -> None:
    """Aider contributes only to coding; LiveBench contributes only to reasoning."""
    per_source = {
        "aider_polyglot": [_entry("x", "aider_polyglot", coding=0.80)],
        "livebench": [_entry("x", "livebench", reasoning=0.75)],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    assert set(merged[0].task_scores.keys()) == {"coding", "reasoning"}
    assert merged[0].task_scores["coding"] == pytest.approx(0.80)
    assert merged[0].task_scores["reasoning"] == pytest.approx(0.75)


def test_merge_lmarena_low_weight_doesnt_dominate() -> None:
    """LMArena weight 0.3 means its score is heavily diluted by AA (1.0)."""
    per_source = {
        "artificial_analysis": [_entry("x", "artificial_analysis", coding=0.80)],
        "lmarena": [_entry("x", "lmarena", coding=0.20)],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    # (0.80 * 1.0 + 0.20 * 0.3) / 1.3 = 0.86 / 1.3 ≈ 0.662
    assert abs(merged[0].task_scores["coding"] - (0.86 / 1.3)) < 1e-6


def test_merge_concats_source_attribution() -> None:
    per_source = {
        "artificial_analysis": [_entry("x", "artificial_analysis", coding=0.7)],
        "aider_polyglot": [_entry("x", "aider_polyglot", coding=0.8)],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    assert merged[0].source == "aider_polyglot,artificial_analysis"


def test_merge_single_source_passes_through() -> None:
    per_source = {"artificial_analysis": [_entry("x", "artificial_analysis", coding=0.7)]}
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    assert merged[0].source == "artificial_analysis"
    assert merged[0].task_scores["coding"] == 0.7


def test_merge_disjoint_models_kept_separate() -> None:
    per_source = {
        "artificial_analysis": [_entry("a", "artificial_analysis", coding=0.7)],
        "livebench": [_entry("b", "livebench", coding=0.8)],
    }
    merged = _merge_sources(per_source, DEFAULT_SOURCE_WEIGHTS)
    ids = {e.model_id for e in merged}
    assert ids == {"a", "b"}


def test_merge_zero_weight_excludes_source() -> None:
    weights = dict(DEFAULT_SOURCE_WEIGHTS)
    weights["lmarena"] = 0.0
    per_source = {
        "artificial_analysis": [_entry("x", "artificial_analysis", coding=0.8)],
        "lmarena": [_entry("x", "lmarena", coding=0.3)],
    }
    merged = _merge_sources(per_source, weights)
    # lmarena contributes weight 0 → only AA matters → 0.8
    assert merged[0].task_scores["coding"] == 0.8


# ---------------------------------------------------------------------------
# RegistryRefresher multi-source end-to-end
# ---------------------------------------------------------------------------

async def test_refresher_composes_two_sources() -> None:
    refresher = RegistryRefresher(
        sources=[
            _StaticSource("artificial_analysis", [
                _entry("opus", "artificial_analysis", coding=0.81, reasoning=0.90,
                       provider="anthropic", cost_in=5.0, cost_out=25.0, context_window=200000),
            ]),
            _StaticSource("aider_polyglot", [
                _entry("opus", "aider_polyglot", coding=0.85, provider="anthropic"),
            ]),
        ],
    )
    update = await refresher.refresh(current=ModelRegistry([]))
    assert update.fetched_count == 2
    assert update.per_source_counts == {"artificial_analysis": 1, "aider_polyglot": 1}
    assert len(update.added) == 1
    merged = update.added[0]
    # Coding score weight-averaged: (0.81 + 0.85*1.5) / (1.0 + 1.5)
    expected = (0.81 * 1.0 + 0.85 * 1.5) / (1.0 + 1.5)
    assert abs(merged.task_scores["coding"] - expected) < 1e-6
    # Reasoning only from AA
    assert abs(merged.task_scores["reasoning"] - 0.90) < 1e-6


async def test_refresher_per_source_failure_is_isolated() -> None:
    class _FailingSource(BenchmarkSource):
        name = "broken"
        async def fetch(self) -> list[ModelEntry]:
            raise RuntimeError("upstream is down")

    refresher = RegistryRefresher(
        sources=[
            _FailingSource(),
            _StaticSource("artificial_analysis", [
                _entry("x", "artificial_analysis", coding=0.7),
            ]),
        ],
    )
    update = await refresher.refresh(current=ModelRegistry([]))
    # Failing source isolated to per_source_failures; AA still contributes.
    assert update.per_source_counts == {"artificial_analysis": 1}
    assert "broken" in update.per_source_failures
    assert "upstream is down" in update.per_source_failures["broken"]
    assert update.all_sources_failed() is False
    assert len(update.added) == 1
    assert update.added[0].task_scores["coding"] == 0.7


async def test_refresher_all_sources_fail_raises() -> None:
    """When every configured source raises, the refresh aborts with RefreshError."""
    from modelmeld.scout.benchmarks.refresher import RefreshError

    class _FailingSource(BenchmarkSource):
        def __init__(self, name: str) -> None:
            self.name = name
        async def fetch(self) -> list[ModelEntry]:
            raise RuntimeError(f"{self.name} is down")

    refresher = RegistryRefresher(sources=[_FailingSource("a"), _FailingSource("b")])
    with pytest.raises(RefreshError) as exc_info:
        await refresher.refresh(current=ModelRegistry([]))
    assert "a" in exc_info.value.per_source_failures
    assert "b" in exc_info.value.per_source_failures


async def test_refresher_no_sources_raises() -> None:
    with pytest.raises(ValueError):
        RegistryRefresher(sources=[])


async def test_refresher_legacy_fetcher_still_works() -> None:
    """The pre-2.8.3 `fetcher=` API is preserved for backwards compat."""
    fake_fetcher = AsyncMock()
    fake_fetcher.fetch_models = AsyncMock(return_value=[
        {
            "model_id": "legacy",
            "provider": "Test",
            "context_window": 1000,
            "price_input_per_m": 1.0,
            "price_output_per_m": 2.0,
            "evaluations": {"GPQA Diamond": 80.0},
        }
    ])
    refresher = RegistryRefresher(fetcher=fake_fetcher)
    update = await refresher.refresh(current=ModelRegistry([]))
    assert len(update.added) == 1
    assert update.added[0].model_id == "legacy"


async def test_refresher_requires_either_sources_or_fetcher() -> None:
    with pytest.raises(ValueError):
        RegistryRefresher()


# ---------------------------------------------------------------------------
# Source weights as exposed
# ---------------------------------------------------------------------------

def test_default_source_weights_complete_and_sane() -> None:
    for source in ["artificial_analysis", "aider_polyglot", "livebench", "lmarena"]:
        assert source in DEFAULT_SOURCE_WEIGHTS
        assert DEFAULT_SOURCE_WEIGHTS[source] > 0
    # LMArena gameable → lowest weight
    assert DEFAULT_SOURCE_WEIGHTS["lmarena"] < DEFAULT_SOURCE_WEIGHTS["artificial_analysis"]
    # Aider authoritative for coding → highest weight
    assert DEFAULT_SOURCE_WEIGHTS["aider_polyglot"] >= DEFAULT_SOURCE_WEIGHTS["artificial_analysis"]

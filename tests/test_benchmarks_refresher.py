"""RegistryRefresher diff semantics + end-to-end refresh flow."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from modelmeld.scout.benchmarks import RegistryRefresher
from modelmeld.scout.benchmarks.refresher import (
    ModelDelta,
    _changed_fields,
    _diff,
    render_update_log,
)
from modelmeld.scout.registry import ModelEntry, ModelRegistry
from tests.fixtures.aa_response import AA_RESPONSE_BASELINE


def _entry(
    model_id: str,
    cost_in: float = 1.0,
    cost_out: float = 2.0,
    context_window: int = 100_000,
    task_scores: dict[str, float] | None = None,
    provider: str = "p",
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores=task_scores or {"coding": 0.7},
    )


# ---------------------------------------------------------------------------
# _changed_fields
# ---------------------------------------------------------------------------

def test_changed_fields_none_when_identical() -> None:
    a = _entry("x")
    b = _entry("x")
    assert _changed_fields(a, b) == []


def test_changed_fields_detects_cost_change() -> None:
    a = _entry("x", cost_in=1.0)
    b = _entry("x", cost_in=2.0)
    assert "cost_per_m_input" in _changed_fields(a, b)


def test_changed_fields_detects_context_window_change() -> None:
    a = _entry("x", context_window=8000)
    b = _entry("x", context_window=128000)
    assert "context_window" in _changed_fields(a, b)


def test_changed_fields_detects_provider_change() -> None:
    a = _entry("x", provider="openai")
    b = _entry("x", provider="anthropic")
    assert "provider" in _changed_fields(a, b)


def test_changed_fields_detects_task_score_change() -> None:
    a = _entry("x", task_scores={"coding": 0.70, "reasoning": 0.60})
    b = _entry("x", task_scores={"coding": 0.85, "reasoning": 0.60})
    diffs = _changed_fields(a, b)
    assert "task_scores.coding" in diffs
    assert "task_scores.reasoning" not in diffs


def test_changed_fields_detects_added_task_score() -> None:
    a = _entry("x", task_scores={"coding": 0.70})
    b = _entry("x", task_scores={"coding": 0.70, "reasoning": 0.60})
    diffs = _changed_fields(a, b)
    assert "task_scores.reasoning" in diffs


def test_changed_fields_ignores_tiny_float_noise() -> None:
    a = _entry("x", cost_in=1.0)
    b = _entry("x", cost_in=1.0 + 1e-9)
    assert _changed_fields(a, b) == []


# ---------------------------------------------------------------------------
# _diff
# ---------------------------------------------------------------------------

def test_diff_added_models() -> None:
    cur = ModelRegistry([_entry("a"), _entry("b")])
    new = ModelRegistry([_entry("a"), _entry("b"), _entry("c")])
    added, removed, updated = _diff(cur, new)
    assert [e.model_id for e in added] == ["c"]
    assert removed == []
    assert updated == []


def test_diff_removed_models() -> None:
    cur = ModelRegistry([_entry("a"), _entry("b"), _entry("c")])
    new = ModelRegistry([_entry("a")])
    added, removed, _updated = _diff(cur, new)
    assert added == []
    assert sorted(e.model_id for e in removed) == ["b", "c"]


def test_diff_updated_models() -> None:
    cur = ModelRegistry([_entry("a", cost_in=1.0)])
    new = ModelRegistry([_entry("a", cost_in=2.0)])
    added, removed, updated = _diff(cur, new)
    assert added == []
    assert removed == []
    assert len(updated) == 1
    assert updated[0].model_id == "a"
    assert "cost_per_m_input" in updated[0].fields_changed


def test_diff_mixed_changes() -> None:
    cur = ModelRegistry([_entry("a"), _entry("b"), _entry("c")])
    new = ModelRegistry([_entry("a"), _entry("b", cost_in=99.0), _entry("d")])
    added, removed, updated = _diff(cur, new)
    assert [e.model_id for e in added] == ["d"]
    assert [e.model_id for e in removed] == ["c"]
    assert [d.model_id for d in updated] == ["b"]


# ---------------------------------------------------------------------------
# RegistryRefresher end-to-end with mocked fetcher
# ---------------------------------------------------------------------------

async def test_refresh_with_baseline_fixture() -> None:
    """Refresh against the empty registry → all 3 models are 'added'."""
    fake_fetcher = AsyncMock()
    fake_fetcher.fetch_models = AsyncMock(return_value=AA_RESPONSE_BASELINE["data"])

    refresher = RegistryRefresher(fetcher=fake_fetcher)
    update = await refresher.refresh(current=ModelRegistry([]))

    assert update.fetched_count == 3
    assert len(update.added) == 3
    assert update.removed == []
    assert update.updated == []
    assert update.has_changes() is True


async def test_refresh_no_changes_when_identical() -> None:
    """Refresh against a registry matching the upstream → no changes."""
    fake_fetcher = AsyncMock()
    fake_fetcher.fetch_models = AsyncMock(return_value=AA_RESPONSE_BASELINE["data"])

    refresher = RegistryRefresher(fetcher=fake_fetcher)
    first = await refresher.refresh(current=ModelRegistry([]))
    second = await refresher.refresh(current=first.new_registry)
    assert second.has_changes() is False
    assert second.summary() == {"added": 0, "removed": 0, "updated": 0, "fetched": 3}


async def test_refresh_detects_cost_change() -> None:
    fake_fetcher = AsyncMock()
    fake_fetcher.fetch_models = AsyncMock(return_value=AA_RESPONSE_BASELINE["data"])
    refresher = RegistryRefresher(fetcher=fake_fetcher)
    first = await refresher.refresh(current=ModelRegistry([]))

    # Simulate AA returning a different price for opus
    bumped = [dict(m) for m in AA_RESPONSE_BASELINE["data"]]
    bumped[0]["price_input_per_m"] = 7.50
    fake_fetcher.fetch_models = AsyncMock(return_value=bumped)
    second = await refresher.refresh(current=first.new_registry)
    assert len(second.updated) == 1
    assert second.updated[0].model_id == "claude-opus-4-7"
    assert "cost_per_m_input" in second.updated[0].fields_changed


async def test_refresh_skips_malformed_records() -> None:
    """Malformed records (no model_id) shouldn't crash the whole refresh."""
    fake_fetcher = AsyncMock()
    payload = [
        AA_RESPONSE_BASELINE["data"][0],   # good record
        {"provider": "no-id"},             # malformed: no model_id
    ]
    fake_fetcher.fetch_models = AsyncMock(return_value=payload)
    refresher = RegistryRefresher(fetcher=fake_fetcher)
    update = await refresher.refresh(current=ModelRegistry([]))
    # The malformed record is dropped during normalize; only good one survives.
    # `fetched_count` now reflects post-normalize entries (one per source).
    assert update.fetched_count == 1
    assert len(update.added) == 1
    assert update.added[0].model_id == "claude-opus-4-7"


async def test_refresh_requires_fetcher() -> None:
    # Post-2.8.3: constructor raises immediately if no source/fetcher given.
    with pytest.raises(ValueError, match="sources.*fetcher|fetcher.*sources"):
        RegistryRefresher(fetcher=None)


# ---------------------------------------------------------------------------
# render_update_log
# ---------------------------------------------------------------------------

def test_render_update_log_contains_counts() -> None:
    from datetime import datetime, timezone

    from modelmeld.scout.benchmarks.refresher import RegistryUpdate

    update = RegistryUpdate(
        new_registry=ModelRegistry([_entry("a")]),
        added=[_entry("a")],
        updated=[ModelDelta(model_id="b", fields_changed=["cost_per_m_input"], before=_entry("b"), after=_entry("b"))],
        removed=[_entry("c")],
        fetched_count=3,
        timestamp=datetime.now(timezone.utc),
    )
    text = render_update_log(update)
    assert "Added:   1" in text
    assert "Removed: 1" in text
    assert "Updated: 1" in text
    assert "+ a" in text
    assert "- c" in text
    assert "~ b" in text
    assert "cost_per_m_input" in text

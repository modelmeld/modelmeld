# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""RegistryRefresher — orchestrates multi-source fetch → merge → diff.

Composes multiple `BenchmarkSource` implementations, merges their contributions
per model (highest-weighted source supplies metadata; task_scores are weight-
averaged across sources that have data), and produces a `RegistryUpdate`.

The refresher does NOT write to disk or mutate global state — the caller
decides what to do with the result.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from modelmeld.scout.benchmarks.artificial_analysis import (
    ArtificialAnalysisFetcher,
    normalize_aa_model,
)
from modelmeld.scout.benchmarks.base import DEFAULT_SOURCE_WEIGHTS, BenchmarkSource
from modelmeld.scout.registry import ModelEntry, ModelRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDelta:
    """Summary of how one model's data changed between refreshes."""

    model_id: str
    fields_changed: list[str]  # e.g. ["cost_per_m_input", "task_scores.coding"]
    before: ModelEntry
    after: ModelEntry


@dataclass(frozen=True)
class RegistryUpdate:
    """Frozen result of one refresh cycle."""

    new_registry: ModelRegistry
    added: list[ModelEntry] = field(default_factory=list)
    removed: list[ModelEntry] = field(default_factory=list)
    updated: list[ModelDelta] = field(default_factory=list)
    fetched_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    per_source_counts: dict[str, int] = field(default_factory=dict)
    per_source_failures: dict[str, str] = field(default_factory=dict)

    def all_sources_failed(self) -> bool:
        """True iff every configured source raised an exception."""
        return bool(self.per_source_failures) and not self.per_source_counts

    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.updated)

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "updated": len(self.updated),
            "fetched": self.fetched_count,
        }


class RegistryRefresher:
    """Pulls latest data from one or more sources and produces a `RegistryUpdate`.

    Accepts EITHER:
      - `sources=[BenchmarkSource, ...]` (multi-source composite)
      - `fetcher=<ArtificialAnalysisFetcher>` (legacy single-source — backwards compat)

    `source_weights` lets you override the default per-source weights; see
    `DEFAULT_SOURCE_WEIGHTS` for the baseline.
    """

    def __init__(
        self,
        sources: list[BenchmarkSource] | None = None,
        *,
        fetcher: ArtificialAnalysisFetcher | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if sources is None and fetcher is None:
            raise ValueError("RegistryRefresher needs either `sources` or `fetcher`")
        if sources is None:
            # Legacy single-source mode — wrap the AAFetcher
            sources = [_LegacyAAFetcherSource(fetcher)] if fetcher is not None else []
        if not sources:
            raise ValueError("RegistryRefresher needs at least one source")
        self.sources = sources
        self.weights = source_weights or dict(DEFAULT_SOURCE_WEIGHTS)

    async def refresh(self, current: ModelRegistry) -> RegistryUpdate:
        per_source_entries, per_source_counts, per_source_failures = await self._fetch_all()

        # Hard-fail: every configured source raised. We don't want to silently
        # blow away a healthy registry because the network had a bad minute.
        if per_source_failures and not per_source_counts:
            first_err = next(iter(per_source_failures.values()))
            raise RefreshError(
                f"all sources failed; first error: {first_err}",
                per_source_failures=per_source_failures,
            )

        merged_entries = _merge_sources(per_source_entries, self.weights)
        new_registry = ModelRegistry(merged_entries)
        added, removed, updated = _diff(current, new_registry)
        return RegistryUpdate(
            new_registry=new_registry,
            added=added,
            removed=removed,
            updated=updated,
            fetched_count=sum(per_source_counts.values()),
            per_source_counts=per_source_counts,
            per_source_failures=per_source_failures,
        )

    async def _fetch_all(
        self,
    ) -> tuple[dict[str, list[ModelEntry]], dict[str, int], dict[str, str]]:
        """Fetch from every source; tolerate per-source failures.

        Returns: (entries_per_source, success_counts, failure_messages)
        — failures are isolated to their source, never blow up the others.
        """
        per_source: dict[str, list[ModelEntry]] = {}
        counts: dict[str, int] = {}
        failures: dict[str, str] = {}
        for source in self.sources:
            try:
                entries = await source.fetch()
            except Exception as e:  # noqa: BLE001 — broad: per-source isolation
                logger.exception("benchmark source failed: %s", source.name)
                failures[source.name] = f"{type(e).__name__}: {e}"
                continue
            per_source[source.name] = entries
            counts[source.name] = len(entries)
        return per_source, counts, failures


class RefreshError(RuntimeError):
    """Raised when refresh cannot produce a usable registry (e.g. all sources failed)."""

    def __init__(self, message: str, *, per_source_failures: dict[str, str]) -> None:
        super().__init__(message)
        self.per_source_failures = per_source_failures


class _LegacyAAFetcherSource(BenchmarkSource):
    """Adapter so the pre-2.8.3 `fetcher=` API keeps working unchanged."""

    name = "artificial_analysis"

    def __init__(self, fetcher: ArtificialAnalysisFetcher) -> None:
        self.fetcher = fetcher

    async def fetch(self) -> list[ModelEntry]:
        raw = await self.fetcher.fetch_models()
        out: list[ModelEntry] = []
        for record in raw:
            try:
                out.append(normalize_aa_model(record))
            except (ValueError, KeyError):
                continue
        return out

    async def close(self) -> None:
        if hasattr(self.fetcher, "close"):
            await self.fetcher.close()


# ----------------------------------------------------------------------
# Source merging
# ----------------------------------------------------------------------

def _merge_sources(
    per_source: dict[str, list[ModelEntry]],
    weights: dict[str, float],
) -> list[ModelEntry]:
    """Combine per-source contributions into one ModelEntry per model_id.

    Metadata (provider / cost / context_window) is taken from the
    highest-weighted source that has non-default values. Task scores are
    weight-averaged across all sources that provided a value for that task.
    """
    contributions: dict[str, list[tuple[str, ModelEntry]]] = defaultdict(list)
    for source_name, entries in per_source.items():
        for entry in entries:
            contributions[entry.model_id].append((source_name, entry))

    merged: list[ModelEntry] = []
    for model_id, parts in contributions.items():
        merged.append(_merge_one_model(model_id, parts, weights))
    return merged


def _merge_one_model(
    model_id: str,
    parts: list[tuple[str, ModelEntry]],
    weights: dict[str, float],
) -> ModelEntry:
    parts_sorted = sorted(
        parts,
        key=lambda p: weights.get(p[0], 1.0),
        reverse=True,
    )

    provider = "unknown"
    context_window = 0
    cost_in = 0.0
    cost_out = 0.0
    last_updated = ""
    for _source_name, entry in parts_sorted:
        if provider == "unknown" and entry.provider and entry.provider != "unknown":
            provider = entry.provider
        if context_window == 0 and entry.context_window > 0:
            context_window = entry.context_window
        if cost_in == 0.0 and entry.cost_per_m_input > 0:
            cost_in = entry.cost_per_m_input
        if cost_out == 0.0 and entry.cost_per_m_output > 0:
            cost_out = entry.cost_per_m_output
        if not last_updated and entry.last_updated:
            last_updated = entry.last_updated

    task_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for source_name, entry in parts:
        w = weights.get(source_name, 1.0)
        for task, score in entry.task_scores.items():
            task_buckets[task].append((score, w))

    task_scores: dict[str, float] = {}
    for task, samples in task_buckets.items():
        weight_total = sum(w for _, w in samples)
        if weight_total > 0:
            task_scores[task] = sum(s * w for s, w in samples) / weight_total

    sources_joined = ",".join(sorted({name for name, _ in parts}))

    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores=task_scores,
        last_updated=last_updated or datetime.now(timezone.utc).isoformat(),
        source=sources_joined,
    )


# ----------------------------------------------------------------------
# Diff between two registries
# ----------------------------------------------------------------------

def _diff(
    current: ModelRegistry, new: ModelRegistry
) -> tuple[list[ModelEntry], list[ModelEntry], list[ModelDelta]]:
    """Compute (added, removed, updated) between two registries."""
    cur_ids = {e.model_id for e in current.all_entries()}
    new_ids = {e.model_id for e in new.all_entries()}

    added_entries = [new.get(mid) for mid in (new_ids - cur_ids)]
    added = [e for e in added_entries if e is not None]

    removed_entries = [current.get(mid) for mid in (cur_ids - new_ids)]
    removed = [e for e in removed_entries if e is not None]

    updated: list[ModelDelta] = []
    for mid in cur_ids & new_ids:
        before = current.get(mid)
        after = new.get(mid)
        if before is None or after is None:
            continue
        changed = _changed_fields(before, after)
        if changed:
            updated.append(ModelDelta(model_id=mid, fields_changed=changed, before=before, after=after))

    return added, removed, updated


def _changed_fields(before: ModelEntry, after: ModelEntry) -> list[str]:
    """List of field paths that differ between two ModelEntry instances."""
    diffs: list[str] = []
    if before.provider != after.provider:
        diffs.append("provider")
    if before.context_window != after.context_window:
        diffs.append("context_window")
    if abs(before.cost_per_m_input - after.cost_per_m_input) > 1e-6:
        diffs.append("cost_per_m_input")
    if abs(before.cost_per_m_output - after.cost_per_m_output) > 1e-6:
        diffs.append("cost_per_m_output")
    for task in set(before.task_scores) | set(after.task_scores):
        b = before.task_scores.get(task)
        a = after.task_scores.get(task)
        if b is None or a is None:
            diffs.append(f"task_scores.{task}")
        elif abs(b - a) > 1e-4:
            diffs.append(f"task_scores.{task}")
    return diffs


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render_update_log(update: RegistryUpdate) -> str:
    """Multi-line log string for stdout / audit / GitHub Action summary."""
    lines = [
        f"Benchmark refresh @ {update.timestamp.isoformat()}",
        f"  Fetched {update.fetched_count} models from upstream",
    ]
    if update.per_source_counts:
        for source_name, count in sorted(update.per_source_counts.items()):
            lines.append(f"    {source_name}: {count}")
    lines.extend([
        f"  Added:   {len(update.added)}",
        f"  Removed: {len(update.removed)}",
        f"  Updated: {len(update.updated)}",
    ])
    if update.added:
        lines.append("  --- Added ---")
        for entry in update.added:
            lines.append(f"    + {entry.model_id} ({entry.provider})")
    if update.removed:
        lines.append("  --- Removed ---")
        for entry in update.removed:
            lines.append(f"    - {entry.model_id} ({entry.provider})")
    if update.updated:
        lines.append("  --- Updated ---")
        for delta in update.updated:
            lines.append(
                f"    ~ {delta.model_id}: {', '.join(delta.fields_changed)}"
            )
    return "\n".join(lines)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""ModelRegistry — capability-aware model selection driven by benchmark scores.

Replaces the LOCAL/CLOUD tier heuristic with specific-model picks. Each
`ModelEntry` carries the model's context window, per-million input/output cost,
and a `task_scores` map. `pick()` returns the cheapest model meeting a quality
threshold for a given task category.

The default registry ships with the package (`data/default_registry.json`) and
is refreshed periodically by the BenchmarkRefresher from public
sources (Artificial Analysis API, Aider Polyglot, LiveBench, LMArena).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)
_seed_warning_emitted = False

# Task categories. Adding new ones requires (a) a benchmark source feeding it,
# (b) a way for Scout to detect the category from the request, (c) test fixtures.
TaskCategory = str   # alias for clarity; not enforced as an enum yet

# Cost-weighting assumption: chat workloads are ~60% input / 40% output tokens.
# Configurable per-tenant in enterprise FinOps; fixed here for default scoring.
_DEFAULT_INPUT_WEIGHT = 0.6
_DEFAULT_OUTPUT_WEIGHT = 0.4


@dataclass(frozen=True)
class ModelEntry:
    """One row in the registry — everything we need to pick this model."""

    model_id: str            # canonical id, matches what /v1/models returns
    provider: str            # "openai" | "anthropic" | "vllm" | ...
    context_window: int      # tokens
    cost_per_m_input: float  # USD per 1M input tokens
    cost_per_m_output: float # USD per 1M output tokens
    task_scores: Mapping[str, float] = field(default_factory=dict)
    last_updated: str = ""   # ISO-8601 timestamp string for portability
    source: str = ""         # which benchmark refresh produced this row
    # Provider-specific model identifier. Defaults to "" meaning "use
    # model_id verbatim on the outbound request to this provider." Set
    # when the provider's API expects a different identifier than the
    # canonical model_id (e.g., a vendor-namespaced or HuggingFace-
    # style path). The router uses this as the request.model override
    # when dispatching to the provider.
    provider_model_id: str = ""
    # Whether this model reliably handles OpenAI function-calling /
    # Anthropic tool-use protocols. Default True since most modern
    # frontier + ≥30B open-weights models support it; set False on
    # models known to fail or partial-support (e.g., phi-4, small
    # 3-7B parameter models per the SLM-for-Agents survey: arxiv
    # 2510.03847 reports multi-step tool chains fail consistently
    # below 7B parameters). CapabilityScout filters by this field
    # when the incoming request has `tools=[...]`.
    supports_tools: bool = True
    # --- Capability metadata for substitution-time feature reconciliation (B-3) ---
    # When alias/capability routing serves a different model than the client
    # tuned its request for, these tell the reconciler whether to forward,
    # translate, or drop the client's model-tuned controls instead of the old
    # blunt "drop everything" behavior.
    #
    # `reasoning_interface` — how this model exposes reasoning on its egress
    # path; drives whether `thinking`/`effort` survive a substitution:
    #   "none"               — no reasoning surface; drop reasoning controls
    #   "anthropic_adaptive" — Claude adaptive thinking (native Anthropic path)
    # Interfaces for OpenAI-compatible backends are added in a later phase.
    # Default "none" is conservative (drop). Sourced from provider catalog
    # parameter lists, or known constants for the native Anthropic models.
    reasoning_interface: str = "none"
    # Per-model max output tokens; clamps an inbound `max_tokens` that exceeds
    # the served model's ceiling (a common substitution 400). None = no clamp.
    max_output_tokens: int | None = None
    # `anthropic-beta` features the served path honors (Anthropic-native only),
    # e.g. context-management/compaction; decides which betas survive a
    # Claude->Claude substitution.
    supported_betas: tuple[str, ...] = ()

    def blended_cost_per_m(
        self,
        input_weight: float = _DEFAULT_INPUT_WEIGHT,
        output_weight: float = _DEFAULT_OUTPUT_WEIGHT,
    ) -> float:
        return (
            self.cost_per_m_input * input_weight
            + self.cost_per_m_output * output_weight
        )

    def meets_threshold(self, task: TaskCategory, threshold: float) -> bool:
        return self.task_scores.get(task, 0.0) >= threshold


class ModelRegistry:
    """In-memory lookup of `ModelEntry` rows; pick() is the routing primitive."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._by_id: dict[str, ModelEntry] = {e.model_id: e for e in entries}

    @classmethod
    def from_json(cls, payload: dict) -> ModelRegistry:
        """Construct from the on-disk JSON format.

        Emits a one-time INFO log when the payload carries a
        `snapshot_release_date` field — that field marks
        a frozen point-in-time snapshot of production-tuned data. The
        OSS bundle ships with a snapshot; the live feed ships without.
        Legacy `seed_only: true` is honored for backward compat.
        Suppressed under pytest to keep test logs clean.
        """
        version = payload.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported registry version: {version}")
        snapshot_date = payload.get("snapshot_release_date")
        if isinstance(snapshot_date, str) and snapshot_date:
            _warn_once_about_snapshot(snapshot_date)
        elif payload.get("seed_only") is True:
            # Legacy flag — older registries that haven't migrated yet.
            _warn_once_about_snapshot(snapshot_date=None)
        entries: list[ModelEntry] = []
        for row in payload.get("models", []):
            entries.append(
                ModelEntry(
                    model_id=row["model_id"],
                    provider=row["provider"],
                    context_window=int(row["context_window"]),
                    cost_per_m_input=float(row["cost_per_m_input"]),
                    cost_per_m_output=float(row["cost_per_m_output"]),
                    task_scores=dict(row.get("task_scores", {})),
                    last_updated=row.get("last_updated", ""),
                    source=row.get("source", ""),
                    provider_model_id=row.get("provider_model_id", ""),
                    supports_tools=bool(row.get("supports_tools", True)),
                    reasoning_interface=str(row.get("reasoning_interface", "none")),
                    max_output_tokens=(
                        int(row["max_output_tokens"])
                        if row.get("max_output_tokens") is not None else None
                    ),
                    supported_betas=tuple(row.get("supported_betas", ())),
                )
            )
        return cls(entries)

    @classmethod
    def from_file(cls, path: str | Path) -> ModelRegistry:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    @classmethod
    def load_default(cls) -> ModelRegistry:
        """Load the registry shipped with the package."""
        # importlib.resources finds the file inside the installed wheel.
        ref = resources.files("modelmeld.scout.data").joinpath("default_registry.json")
        return cls.from_json(json.loads(ref.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelEntry | None:
        return self._by_id.get(model_id)

    def all_entries(self) -> list[ModelEntry]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._by_id

    # ------------------------------------------------------------------
    # The routing primitive
    # ------------------------------------------------------------------

    def pick(
        self,
        task_category: TaskCategory,
        quality_threshold: float = 0.80,
        eligible_providers: frozenset[str] | None = None,
        min_context_window: int = 0,
        require_tool_support: bool = False,
        input_weight: float = _DEFAULT_INPUT_WEIGHT,
        output_weight: float = _DEFAULT_OUTPUT_WEIGHT,
    ) -> ModelEntry | None:
        """Return the cheapest model meeting `quality_threshold` on `task_category`.

        Filters:
        - task_scores[task_category] >= quality_threshold
        - provider in eligible_providers (if set)
        - context_window >= min_context_window
        - supports_tools (if require_tool_support)
        Ties on cost broken by larger context window (prefer safer choice).
        Returns None if no model qualifies.
        """
        candidates: list[ModelEntry] = []
        for entry in self._by_id.values():
            if eligible_providers is not None and entry.provider not in eligible_providers:
                continue
            if entry.context_window < min_context_window:
                continue
            if require_tool_support and not entry.supports_tools:
                continue
            if not entry.meets_threshold(task_category, quality_threshold):
                continue
            candidates.append(entry)

        if not candidates:
            return None

        def _sort_key(e: ModelEntry) -> tuple[float, int]:
            # cheapest blended cost first; tie-break: larger window first (negative for asc sort)
            return (e.blended_cost_per_m(input_weight, output_weight), -e.context_window)

        return min(candidates, key=_sort_key)

    def rank(
        self,
        task_category: TaskCategory,
        quality_threshold: float = 0.0,
        eligible_providers: frozenset[str] | None = None,
        min_context_window: int = 0,
        require_tool_support: bool = False,
    ) -> list[tuple[ModelEntry, float]]:
        """All eligible candidates sorted cheapest-first. Returns (entry, blended_cost) pairs.

        Filters:
          - task_scores[task_category] >= quality_threshold
          - provider in eligible_providers (if set)
          - context_window >= min_context_window
          - supports_tools == True (if require_tool_support)
        """
        result: list[tuple[ModelEntry, float]] = []
        for entry in self._by_id.values():
            if eligible_providers is not None and entry.provider not in eligible_providers:
                continue
            if entry.context_window < min_context_window:
                continue
            if require_tool_support and not entry.supports_tools:
                continue
            if not entry.meets_threshold(task_category, quality_threshold):
                continue
            result.append((entry, entry.blended_cost_per_m()))
        result.sort(key=lambda pair: pair[1])
        return result


# ----------------------------------------------------------------------
# Module-level convenience (so callers can `from modelmeld.scout.registry import default_registry`)
# ----------------------------------------------------------------------

_default_singleton: ModelRegistry | None = None


def default_registry() -> ModelRegistry:
    """Singleton-style accessor for the package-shipped default registry."""
    global _default_singleton
    if _default_singleton is None:
        _default_singleton = ModelRegistry.load_default()
    return _default_singleton


def _warn_once_about_snapshot(snapshot_date: str | None) -> None:
    """One-time INFO log when the gateway loads a snapshot registry.

    Replaces an older "seed warning" with an honest message —
    OSS ships *real* production-tuned scores frozen at release date. The
    snapshot stales over time as the foundation-model market shifts; the
    paid live feed keeps things current.

    Suppressed under pytest so test logs stay clean.
    """
    global _seed_warning_emitted
    if _seed_warning_emitted:
        return
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION"):
        _seed_warning_emitted = True
        return
    if snapshot_date:
        logger.info(
            "ModelRegistry loaded from bundled snapshot (release date: %s). "
            "Scores are production-tuned at that point in time; the foundation-"
            "model market deflates ~50%%/year so this snapshot will stale over "
            "~6 months. Configure MODELMELD_REGISTRY_FEED_URL + a license key "
            "for the live curated feed. See docs/registry-feed.md.",
            snapshot_date,
        )
    else:
        logger.info(
            "ModelRegistry loaded from a bundled snapshot. Scores are "
            "production-tuned at the OSS release date and stale over time. "
            "Configure MODELMELD_REGISTRY_FEED_URL + a license key for the "
            "live curated feed. See docs/registry-feed.md."
        )
    _seed_warning_emitted = True


# Back-compat alias for any external callers that import the old name.
_warn_once_about_seed_data = _warn_once_about_snapshot

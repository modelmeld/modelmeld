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
import statistics
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
    # Whether this row is eligible to be routed to at all. Default True; a
    # row with `enabled=False` stays in the registry (and keeps its scores,
    # cost, latency) but is filtered out of `pick()`/`rank()` candidate
    # selection entirely. This is the disable-not-delete switch: deprecate a
    # model, quarantine a broken/ToS-problematic one, or hold a freshly-added
    # candidate back until it has been validated — all without losing the row
    # or its data. The registry/feed PRODUCER decides the policy that sets it;
    # the gateway only honors the flag.
    enabled: bool = True
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
    # --- Latency signal (D1 of the routing-objective redesign) ---
    # Per-token price selects against the thing that governs the agentic-coding
    # experience: time-to-complete. These two fields let the scout rank on
    # latency-adjusted cost (see `estimated_turn_latency_s` + the scout's D1
    # term). Both default None ("unknown") so a registry/seed without latency
    # data behaves exactly as before — the scout's latency factor is a no-op
    # for entries that lack it.
    #
    # `median_ttft_s` — median time-to-first-token in seconds. Prefill-dominated;
    # the relevant axis for agentic turns (output is tiny, input is ~99%).
    # `median_output_tps` — median output tokens/second (decode throughput).
    # `latency_source` — provenance, e.g. "artificial_analysis@medium" (AA's
    # ~1k-token reference input) or "measured" (first-party gateway telemetry).
    #
    # CAVEAT (load-bearing, see docs/design-routing-objective.md): public AA
    # latency is measured at a small reference input (~1k tokens) on AA's
    # reference provider — a COARSE proxy for our ~61k-token agentic prefill,
    # and NOT per-serving-provider. v1 ships this model-level signal; the
    # per-(model x provider) prefill latency that resolves provider-backend
    # roulette comes from a later first-party-telemetry pass.
    median_ttft_s: float | None = None
    median_output_tps: float | None = None
    latency_source: str = ""

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

    def estimated_turn_latency_s(
        self, input_tokens: int, output_tokens: int,
    ) -> float | None:
        """Estimate wall-clock seconds for one turn — the per-turn latency
        signal the D1 term ranks on.

        Returns None when this row carries no throughput data — callers treat
        that as "no latency signal" (the scout's D1 factor falls back to
        cost-only for such rows, so missing data never makes a model look
        artificially fast).

        v1 model (deliberately conservative):

            median_ttft_s + output_tokens / median_output_tps

        ``input_tokens`` is accepted but NOT used to extrapolate prefill time in
        v1, on purpose. We only have public AA throughput: ``median_output_tps``
        is *decode* speed, and prefill runs at a very different (much faster)
        rate we cannot derive from it — dividing a ~61k-token agentic prefill by
        decode tps over-estimates by 1-2 orders of magnitude and would let the
        latency factor swamp cost. ``median_ttft_s`` (AA's ~1k-reference
        time-to-first-token) is the prefill-onset proxy we *do* have; we use it
        as the fixed base. Producing a real per-(model x provider) prefill
        latency at agentic input sizes is the deferred first-party-telemetry
        pass (see docs/design-routing-objective.md); ``input_tokens`` is kept in
        the signature for that successor model.
        """
        if self.median_output_tps is None or self.median_output_tps <= 0:
            return None
        ttft = self.median_ttft_s or 0.0
        return ttft + output_tokens / self.median_output_tps


def _sort_latency_adjusted(
    result: list[tuple[ModelEntry, float]],
    latency_weight: float,
    ref_input_tokens: int,
    ref_output_tokens: int,
) -> list[tuple[ModelEntry, float]]:
    """Sort ``(entry, blended_cost)`` pairs by latency-adjusted cost (D1).
    Shared by base + multi-provider ``rank()``.

    ``latency_weight <= 0`` → plain cost order (byte-identical to the old
    cost-only ranking; this is what keeps ``-saver`` unchanged). Otherwise the
    key is ``blended_cost * (1 + latency_weight * latency_s)``.

    **Missing-data handling (the load-bearing bit):** a candidate with no
    latency signal is imputed the **median latency of the measured candidates**
    — NOT treated as instant. Earlier this gave unmeasured rows a free pass
    (factor 1.0), which under partial coverage perversely rewarded exactly the
    models we hadn't measured (e.g. the cheapest default pick) and penalized the
    ones we had. Median imputation makes missing data *neutral*: an unmeasured
    model ranks as an average-latency one, never an advantaged one. If NO
    candidate has latency data, falls back to plain cost. The returned pair
    keeps the real blended cost (not the effective key), so cost reporting stays
    truthful; the sort is stable.
    """
    if latency_weight <= 0 or not result:
        return sorted(result, key=lambda pair: pair[1])
    latencies = [
        e.estimated_turn_latency_s(ref_input_tokens, ref_output_tokens)
        for e, _ in result
    ]
    measured = [lat for lat in latencies if lat is not None]
    if not measured:
        return sorted(result, key=lambda pair: pair[1])
    median_lat = statistics.median(measured)

    def _effective(indexed: tuple[int, tuple[ModelEntry, float]]) -> float:
        i, (_, cost) = indexed
        # Bind to a local so the None-narrowing sticks (pyright doesn't retain
        # narrowing across a re-evaluated subscript).
        measured_lat = latencies[i]
        lat = measured_lat if measured_lat is not None else median_lat
        return cost * (1.0 + latency_weight * lat)

    ordered = sorted(enumerate(result), key=_effective)
    return [pair for _, pair in ordered]


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
                    enabled=bool(row.get("enabled", True)),
                    reasoning_interface=str(row.get("reasoning_interface", "none")),
                    max_output_tokens=(
                        int(row["max_output_tokens"])
                        if row.get("max_output_tokens") is not None else None
                    ),
                    supported_betas=tuple(row.get("supported_betas", ())),
                    median_ttft_s=(
                        float(row["median_ttft_s"])
                        if row.get("median_ttft_s") is not None else None
                    ),
                    median_output_tps=(
                        float(row["median_output_tps"])
                        if row.get("median_output_tps") is not None else None
                    ),
                    latency_source=row.get("latency_source", ""),
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
        - enabled (disabled rows are never routed to)
        - task_scores[task_category] >= quality_threshold
        - provider in eligible_providers (if set)
        - context_window >= min_context_window
        - supports_tools (if require_tool_support)
        Ties on cost broken by larger context window (prefer safer choice).
        Returns None if no model qualifies.
        """
        candidates: list[ModelEntry] = []
        for entry in self._by_id.values():
            if not entry.enabled:
                continue
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
        latency_weight: float = 0.0,
        latency_ref_input_tokens: int = 0,
        latency_ref_output_tokens: int = 0,
        agentic_admit_floor: float | None = None,
    ) -> list[tuple[ModelEntry, float]]:
        """All eligible candidates sorted cheapest-first. Returns (entry, blended_cost) pairs.

        Filters:
          - enabled (disabled rows are never routed to)
          - task_scores[task_category] >= quality_threshold
          - provider in eligible_providers (if set)
          - context_window >= min_context_window
          - supports_tools == True (if require_tool_support)

        Ordering (D1 latency term): when ``latency_weight > 0`` candidates are
        sorted by a latency-adjusted *effective* cost
        ``blended_cost * (1 + latency_weight * estimated_turn_latency_s)`` — a
        cheaper-but-slower model loses to a slightly-pricier-but-faster one for
        the agentic shape described by ``latency_ref_*_tokens``. Rows without
        latency data fall back to plain cost (no reward/penalty), so a partial
        registry never makes an unmeasured model look artificially fast. The
        returned pair's second element is always the *real* blended cost (not
        the effective key), so callers' cost reporting stays truthful.
        ``latency_weight == 0`` (the default) is byte-identical to the old
        cost-only ranking — this is what keeps ``-saver`` ranking unchanged.

        ``agentic_admit_floor`` (agentic-axis routes): when set, a model is ALSO
        admitted if it has NO ``task_category`` score at all but its in-house
        ``agentic_coding`` (RO-3) score is >= this floor. A model proven to
        converge on real agentic-coding tasks is a capable coder even when no
        external benchmark has scored it — so it isn't benched for lack of a
        ``coding`` number. This deliberately does NOT admit a model the benchmark
        DID score below threshold (it was measured and found wanting). None
        (default) keeps the pure-threshold gate.
        """
        result: list[tuple[ModelEntry, float]] = []
        for entry in self._by_id.values():
            if not entry.enabled:
                continue
            if eligible_providers is not None and entry.provider not in eligible_providers:
                continue
            if entry.context_window < min_context_window:
                continue
            if require_tool_support and not entry.supports_tools:
                continue
            admitted = entry.meets_threshold(task_category, quality_threshold) or (
                # RO-3 admission applies ONLY to models the benchmark hasn't
                # scored on this category (the coverage gap) — NOT to ones it
                # measured below threshold (those were tested and found wanting).
                agentic_admit_floor is not None
                and task_category not in entry.task_scores
                and entry.task_scores.get("agentic_coding", 0.0) >= agentic_admit_floor
            )
            if not admitted:
                continue
            result.append((entry, entry.blended_cost_per_m()))
        return _sort_latency_adjusted(
            result, latency_weight,
            latency_ref_input_tokens, latency_ref_output_tokens,
        )


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

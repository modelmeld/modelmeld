# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""MultiProviderModelRegistry — extends ModelRegistry to support multiple
ModelEntry rows per model_id, keyed by ``(model_id, provider)``.

The base ``ModelRegistry`` keeps a one-row-per-model_id shape (the
``_by_id`` dict collapses duplicates by overwriting). That works fine
for routing when each canonical model has a single canonical upstream
(e.g., ``claude-opus-4-7`` lives on Anthropic; ``qwen2-5-coder-7b`` lives
on a self-hosted vLLM endpoint).

In practice, the same open-weight model is served by multiple
upstream providers — ``qwen2-5-coder-7b-instruct`` is available via
Fireworks, Together, OpenRouter, AND self-hosted vLLM. Each provider
has its own ``provider_model_id`` (the path the upstream expects),
cost per million tokens, and reliability profile.

``MultiProviderModelRegistry`` stores all of them by ``(model_id,
provider)`` composite key. Callers that need provider-level
enumeration use ``entries_for(model_id)``. Callers that just want one
canonical entry per model continue to use the base class's
``get(model_id)``.

Duplicate ``(model_id, provider)`` rows keep the LAST-inserted entry
and emit a warning — registries are merged from overlay files at boot,
and a later overlay should win over a base default.
"""

from __future__ import annotations

import json
import logging
from importlib import resources

from modelmeld.scout.registry import (
    ModelEntry,
    ModelRegistry,
    _sort_latency_adjusted,
)

logger = logging.getLogger(__name__)


class MultiProviderModelRegistry(ModelRegistry):
    """A ``ModelRegistry`` that indexes entries by ``(model_id, provider)``.

    The base class's ``_by_id`` map (one entry per model_id, last write
    wins) is preserved for back-compat callers that just want "any"
    entry for a given model. Multi-provider callers should use
    ``get_by_key()`` or ``entries_for()``.

    Filtering methods inherited from the base (``pick()``, ``rank()``)
    operate over ``_by_id.values()`` — for multi-provider-aware
    filtering, callers should iterate ``all_entries_multi()``.
    """

    def __init__(self, entries: list[ModelEntry]) -> None:
        # Build the composite-key map first so duplicate handling
        # happens in a deterministic, observable way.
        self._by_key: dict[tuple[str, str], ModelEntry] = {}
        for entry in entries:
            key = (entry.model_id, entry.provider)
            if key in self._by_key:
                logger.warning(
                    "MultiProviderModelRegistry: duplicate "
                    "(model_id=%r, provider=%r) — keeping the last "
                    "entry, dropping the earlier one",
                    entry.model_id,
                    entry.provider,
                )
            self._by_key[key] = entry
        # Base class's ``_by_id`` collapses to one-entry-per-model_id
        # (later entries with same model_id overwrite earlier). That's
        # fine for back-compat ``get(model_id)`` callers; multi-
        # provider callers should use ``entries_for()``.
        super().__init__(entries)

    # ------------------------------------------------------------------
    # Multi-provider accessors
    # ------------------------------------------------------------------

    def get_by_key(self, model_id: str, provider: str) -> ModelEntry | None:
        """Return the specific ``(model_id, provider)`` entry, or None."""
        return self._by_key.get((model_id, provider))

    def entries_for(self, model_id: str) -> list[ModelEntry]:
        """Return ALL entries for a model_id, one per provider."""
        return [e for e in self._by_key.values() if e.model_id == model_id]

    def all_entries_multi(self) -> list[ModelEntry]:
        """All rows, including the multi-provider ones the base class's
        ``_by_id`` collapsed to one-per-model_id."""
        return list(self._by_key.values())

    def providers_for(self, model_id: str) -> frozenset[str]:
        """The set of providers that serve a given model_id."""
        return frozenset(provider for (mid, provider) in self._by_key if mid == model_id)

    def __len__(self) -> int:
        """Total row count (multi-provider aware). Differs from base's
        ``_by_id`` length when the same model_id has multiple rows."""
        return len(self._by_key)

    # ------------------------------------------------------------------
    # Override filtering methods — iterate over ``_by_key`` so multi-
    # provider rows are visible to ``eligible_providers`` filtering.
    # ------------------------------------------------------------------

    def rank(
        self,
        task_category: str,
        quality_threshold: float = 0.0,
        eligible_providers: frozenset[str] | None = None,
        min_context_window: int = 0,
        require_tool_support: bool = False,
        latency_weight: float = 0.0,
        latency_ref_input_tokens: int = 0,
        latency_ref_output_tokens: int = 0,
    ) -> list[tuple[ModelEntry, float]]:
        """Multi-provider rank: iterates over every ``(model_id, provider)``
        row, not just the base's collapsed ``_by_id`` representatives.

        Required for ``eligible_providers`` filtering to actually pick the
        right provider when a model has multiple rows. Without this
        override, the base class only sees one entry per model_id (the
        last-inserted one) — so a customer who configured only one upstream
        would have a model's other-provider rows hidden if a different
        provider was inserted last in the registry.

        Latency (D1) ordering is identical to the base class — see
        ``ModelRegistry.rank`` and ``_effective_cost_key``. Because each
        ``(model_id, provider)`` row carries its own latency fields, the
        latency term ranks *per provider* once per-provider latency data
        lands; with model-level seed data the rows for one model share a
        latency and the term reduces to model-level ordering.
        """
        result: list[tuple[ModelEntry, float]] = []
        for entry in self._by_key.values():
            if eligible_providers is not None and entry.provider not in eligible_providers:
                continue
            if entry.context_window < min_context_window:
                continue
            if require_tool_support and not entry.supports_tools:
                continue
            if not entry.meets_threshold(task_category, quality_threshold):
                continue
            result.append((entry, entry.blended_cost_per_m()))
        return _sort_latency_adjusted(
            result, latency_weight,
            latency_ref_input_tokens, latency_ref_output_tokens,
        )

    def pick(
        self,
        task_category: str,
        quality_threshold: float = 0.80,
        eligible_providers: frozenset[str] | None = None,
        min_context_window: int = 0,
        require_tool_support: bool = False,
        input_weight: float = 0.6,
        output_weight: float = 0.4,
    ) -> ModelEntry | None:
        """Multi-provider pick: cheapest of every ``(model_id, provider)``
        candidate that meets the filters. Mirrors base ``pick()`` semantics
        but iterates ``_by_key`` for visibility into multi-provider rows.
        """
        candidates: list[ModelEntry] = []
        for entry in self._by_key.values():
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
            return (e.blended_cost_per_m(input_weight, output_weight), -e.context_window)

        return min(candidates, key=_sort_key)

    # ------------------------------------------------------------------
    # Default-registry loader (base + overlay merge)
    # ------------------------------------------------------------------

    @classmethod
    def load_default(
        cls,
        base: ModelRegistry | None = None,
    ) -> MultiProviderModelRegistry:
        """Build the multi-provider default by merging base + overlay.

        The base ``default_registry.json`` carries the canonical
        per-model task_scores and the single-provider entries
        (typically ``provider: vllm`` for OSS models). The
        ``default_overlay.json`` adds availability rows on additional
        upstream providers, each with its own cost and
        ``provider_model_id``.

        Overlay rows inherit:
          * ``task_scores`` — from the base entry with the same
            ``model_id``. Without inheritance, the picker's quality
            threshold would exclude all overlay rows.
          * ``supports_tools`` — conservatively ANDed with the base
            value. If either base or overlay row says the model
            lacks tool support, the resulting row also lacks it.

        Overlay rows that reference a model_id not in the base are
        kept but logged as a warning (their task_scores stay empty
        and the picker will skip them at any quality threshold > 0).
        """
        if base is None:
            from modelmeld.scout.registry import default_registry

            base = default_registry()
        base_entries = list(base.all_entries())

        # Index base entries for inheritance lookups.
        scores_by_model: dict[str, dict[str, float]] = {}
        supports_tools_by_model: dict[str, bool] = {}
        reasoning_by_model: dict[str, str] = {}
        max_output_by_model: dict[str, int | None] = {}
        betas_by_model: dict[str, tuple[str, ...]] = {}
        for entry in base_entries:
            if entry.model_id not in scores_by_model and entry.task_scores:
                scores_by_model[entry.model_id] = dict(entry.task_scores)
            if entry.model_id not in supports_tools_by_model:
                supports_tools_by_model[entry.model_id] = entry.supports_tools
            if entry.model_id not in reasoning_by_model and entry.reasoning_interface != "none":
                reasoning_by_model[entry.model_id] = entry.reasoning_interface
            if entry.model_id not in max_output_by_model and entry.max_output_tokens is not None:
                max_output_by_model[entry.model_id] = entry.max_output_tokens
            if entry.model_id not in betas_by_model and entry.supported_betas:
                betas_by_model[entry.model_id] = entry.supported_betas

        # Load the overlay JSON shipped with the package. The overlay is an
        # optional example of multi-provider availability; if it isn't present
        # (e.g. a deployment that ships only the base registry), fall back to a
        # base-only registry rather than failing to start.
        try:
            overlay_payload = json.loads(
                resources.files("modelmeld.scout.data")
                .joinpath("default_overlay.json")
                .read_text(encoding="utf-8"),
            )
        except (FileNotFoundError, ModuleNotFoundError):
            logger.info(
                "default_overlay.json not present; using base registry only "
                "(no multi-provider availability rows)",
            )
            overlay_payload = {"models": []}
        overlay_entries: list[ModelEntry] = []
        for row in overlay_payload.get("models", []):
            model_id = row["model_id"]
            # Merge task_scores: start from the base entry's scores (by
            # model_id) and let the overlay row OVERRIDE per-key. This lets a
            # row supply a measured per-(model, provider) score (e.g. a
            # tool_use score from the eval harness) without dropping the other
            # categories it would otherwise inherit from base — a bare
            # {"tool_use": x} row previously REPLACED all inherited scores,
            # silently zeroing coding/reasoning for that row.
            base_scores = scores_by_model.get(model_id, {})
            row_scores = dict(row.get("task_scores", {}))
            scores = {**base_scores, **row_scores}
            if not scores:
                logger.warning(
                    "default_overlay: no task_scores available for %s "
                    "(neither overlay row nor base registry provides "
                    "them); routing picker will exclude this row at "
                    "any quality threshold > 0",
                    model_id,
                )
            # Inherit supports_tools: AND the two values conservatively.
            row_supports_tools = bool(row.get("supports_tools", True))
            base_supports_tools = supports_tools_by_model.get(model_id, True)
            final_supports_tools = row_supports_tools and base_supports_tools
            # Capability metadata: overlay row OVERRIDES base; inherit base when
            # the row omits it (mirrors task_scores / supports_tools).
            reasoning = row.get("reasoning_interface") or reasoning_by_model.get(
                model_id, "none",
            )
            max_output = (
                int(row["max_output_tokens"])
                if row.get("max_output_tokens") is not None
                else max_output_by_model.get(model_id)
            )
            betas = tuple(row["supported_betas"]) if row.get(
                "supported_betas",
            ) is not None else betas_by_model.get(model_id, ())
            overlay_entries.append(
                ModelEntry(
                    model_id=model_id,
                    provider=row["provider"],
                    context_window=int(row["context_window"]),
                    cost_per_m_input=float(row["cost_per_m_input"]),
                    cost_per_m_output=float(row["cost_per_m_output"]),
                    task_scores=scores,
                    last_updated=row.get("last_updated", ""),
                    source=row.get("source", "default_overlay"),
                    provider_model_id=row.get("provider_model_id", ""),
                    supports_tools=final_supports_tools,
                    reasoning_interface=reasoning,
                    max_output_tokens=max_output,
                    supported_betas=betas,
                )
            )

        return cls(base_entries + overlay_entries)


# ----------------------------------------------------------------------
# Module-level convenience accessor — mirrors registry.default_registry()
# ----------------------------------------------------------------------

_default_multi_provider_singleton: MultiProviderModelRegistry | None = None


def default_multi_provider_registry() -> MultiProviderModelRegistry:
    """Singleton-style accessor for the package-shipped multi-provider registry.

    Loads ``default_registry.json`` (canonical task_scores) + merges
    ``default_overlay.json`` (multi-provider availability rows). Cached
    after first call.
    """
    global _default_multi_provider_singleton
    if _default_multi_provider_singleton is None:
        _default_multi_provider_singleton = MultiProviderModelRegistry.load_default()
    return _default_multi_provider_singleton


__all__ = [
    "MultiProviderModelRegistry",
    "default_multi_provider_registry",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""BenchmarkSource ABC — the contract every benchmark feed implements.

A source produces a list of `ModelEntry` records, each with whatever metadata
and task_scores that source can provide. The `RegistryRefresher` composes
multiple sources and merges their contributions per model.

Sources do NOT have to provide all task_scores or all metadata fields. The
merger handles partial data — Aider Polyglot only contributes `coding`;
LMArena only contributes a subjective preference signal; AA provides nearly
everything. The merge picks the highest-weighted source's metadata and
weight-averages task_scores across sources that have data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from modelmeld.scout.registry import ModelEntry

# Default per-source weights. Tuned to taste; configurable per refresher.
#   AA               — well-curated, multi-benchmark, paid tier exists
#   Aider Polyglot   — coding gold standard; lower noise than AA for code
#   LiveBench        — continuous; multiple categories
#   LMArena          — subjective user preference; gameable; light weight
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "artificial_analysis": 1.0,
    "aider_polyglot": 1.5,
    "livebench": 1.0,
    "lmarena": 0.3,
}


class BenchmarkSource(ABC):
    """A pluggable feed of `ModelEntry` records with task_scores."""

    # Each subclass sets this to a unique identifier used in DEFAULT_SOURCE_WEIGHTS.
    name: str

    @abstractmethod
    async def fetch(self) -> list[ModelEntry]:
        """Pull current data from upstream and return normalized ModelEntries.

        Each entry's `source` should equal the source's name; `task_scores`
        may be a subset of the canonical categories (only fill what the
        upstream actually measures).
        """

    async def close(self) -> None:
        """Release held resources. Default no-op."""

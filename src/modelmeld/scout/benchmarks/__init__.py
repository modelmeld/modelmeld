# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Benchmark ingestion for the ModelRegistry.

Public surface:
    BenchmarkSource            — ABC every source implements
    DEFAULT_SOURCE_WEIGHTS     — per-source weights for the merge
    ArtificialAnalysisFetcher  — fetches model + benchmark data from AA API
    AiderPolyglotFetcher       — Aider polyglot leaderboard (YAML)
    LiveBenchFetcher           — LiveBench monthly leaderboard (JSON)
    LMArenaFetcher             — LMArena daily snapshot (community mirror)
    normalize_aa_model         — maps one AA-shaped record to a ModelEntry
    TASK_BENCHMARK_MAP         — mapping from our task categories to AA benchmark names
    RegistryRefresher          — multi-source fetch + diff + update orchestration
    RegistryUpdate             — frozen result of one refresh cycle
"""

from __future__ import annotations

from modelmeld.scout.benchmarks.aider_polyglot import AiderPolyglotFetcher
from modelmeld.scout.benchmarks.artificial_analysis import (
    TASK_BENCHMARK_MAP,
    ArtificialAnalysisFetcher,
    canonicalize_model_id,
    normalize_aa_model,
)
from modelmeld.scout.benchmarks.base import (
    DEFAULT_SOURCE_WEIGHTS,
    BenchmarkSource,
)
from modelmeld.scout.benchmarks.livebench import LiveBenchFetcher
from modelmeld.scout.benchmarks.lmarena import LMArenaFetcher
from modelmeld.scout.benchmarks.refresher import (
    ModelDelta,
    RegistryRefresher,
    RegistryUpdate,
    render_update_log,
)

__all__ = [
    "AiderPolyglotFetcher",
    "ArtificialAnalysisFetcher",
    "BenchmarkSource",
    "DEFAULT_SOURCE_WEIGHTS",
    "LMArenaFetcher",
    "LiveBenchFetcher",
    "ModelDelta",
    "RegistryRefresher",
    "RegistryUpdate",
    "TASK_BENCHMARK_MAP",
    "canonicalize_model_id",
    "normalize_aa_model",
    "render_update_log",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Artificial Analysis API fetcher + normalizer.

Pulls model + benchmark data from <https://artificialanalysis.ai/documentation>
(free tier: 1000 req/day, requires `x-api-key` header, attribution required —
attribution lives in our generated registry's `notes` field on each refresh).

The shape we expect from AA is approximated from their public documentation as
of May 2026. If the API changes, only this module needs updating — the
ModelRegistry, refresher, and downstream consumers are insulated.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from modelmeld.scout.benchmarks.base import BenchmarkSource
from modelmeld.scout.registry import ModelEntry

# Mapping from our task categories to AA benchmark names. Each task averages
# the available scores from its listed AA benchmarks. Hand-curated — refine as
# we observe real AA data. Benchmarks not present in a given model's AA record
# are simply skipped (the average is over what's available).
#
# AA benchmarks (Intelligence Index v4.0 components as documented May 2026):
#   GPQA Diamond, AA-LCR, AA-Omniscience, HLE, IFBench, Terminal-Bench Hard,
#   SciCode, τ²-Bench Telecom, CritPt, GDPval-AA, LiveCodeBench
TASK_BENCHMARK_MAP: dict[str, list[str]] = {
    "coding": ["Terminal-Bench Hard", "SciCode", "LiveCodeBench"],
    "reasoning": ["GPQA Diamond", "AA-LCR", "HLE", "CritPt"],
    "simple_qa": ["IFBench", "GDPval-AA"],
    "summarization": ["AA-Omniscience"],
    "tool_use": ["τ²-Bench Telecom", "Terminal-Bench Hard"],
}

# AA scores are typically 0–100; our task_scores are 0.0–1.0. Convert.
_AA_SCORE_DIVISOR = 100.0

_DEFAULT_BASE_URL = "https://api.artificialanalysis.ai/v2"


def canonicalize_model_id(raw: str) -> str:
    """Normalize a model id for consistent comparison across sources.

    AA may use "claude-opus-4.7" while we use "claude-opus-4-7" elsewhere.
    Lowercase, dots → dashes, spaces → dashes.
    """
    return raw.strip().lower().replace(".", "-").replace(" ", "-").replace("_", "-")


def normalize_aa_model(aa: dict[str, Any]) -> ModelEntry:
    """Convert one AA model record into a ModelEntry.

    Expected (approximate) AA shape:
        {
            "model_id": "claude-opus-4-7",
            "creator": "Anthropic",
            "context_window": 200000,
            "price_input_per_m": 5.00,
            "price_output_per_m": 25.00,
            "evaluations": {
                "GPQA Diamond": 84.5,
                "AA-LCR": 71.2,
                "Terminal-Bench Hard": 58.3,
                ...
            },
            "last_updated": "2026-05-15T00:00:00Z"
        }
    Missing fields fall back to safe defaults.
    """
    model_id = canonicalize_model_id(
        aa.get("model_id") or aa.get("name") or aa.get("slug") or ""
    )
    if not model_id:
        raise ValueError(f"AA record missing model_id / name / slug: {aa!r}")

    provider = (aa.get("provider") or aa.get("creator") or "unknown").lower()
    context_window = int(aa.get("context_window") or aa.get("context_length") or 0)
    cost_in = float(aa.get("price_input_per_m") or aa.get("input_price_per_m") or 0.0)
    cost_out = float(aa.get("price_output_per_m") or aa.get("output_price_per_m") or 0.0)

    evaluations: dict[str, float] = aa.get("evaluations") or aa.get("benchmarks") or {}
    task_scores = _compute_task_scores(evaluations)

    last_updated = aa.get("last_updated") or datetime.now(timezone.utc).isoformat()

    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores=task_scores,
        last_updated=last_updated,
        source="artificial_analysis",
    )


def _compute_task_scores(evaluations: dict[str, Any]) -> dict[str, float]:
    """Average the AA benchmark scores for each task category, normalized to [0,1]."""
    scores: dict[str, float] = {}
    for task, bench_list in TASK_BENCHMARK_MAP.items():
        applicable: list[float] = []
        for bench_name in bench_list:
            raw = evaluations.get(bench_name)
            if raw is None:
                continue
            try:
                # AA scores are 0–100 (per category). Normalize to 0–1.
                applicable.append(float(raw) / _AA_SCORE_DIVISOR)
            except (TypeError, ValueError):
                continue
        if applicable:
            scores[task] = sum(applicable) / len(applicable)
    return scores


class ArtificialAnalysisError(Exception):
    """Raised when the AA API call fails."""


class ArtificialAnalysisFetcher(BenchmarkSource):
    """Fetches model + benchmark data from the Artificial Analysis REST API.

    Production use requires an API key set in env (`ARTIFICIAL_ANALYSIS_API_KEY`)
    or passed directly. Tests inject a custom `http_client` to mock responses.
    """

    name = "artificial_analysis"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._http = http_client
        self._timeout = timeout
        self._owns_client = http_client is None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def fetch_models(self) -> list[dict[str, Any]]:
        """Fetch the full model list with benchmark scores.

        Returns the raw AA records (list of dicts). Use `normalize_aa_model`
        on each entry to convert into ModelEntry instances.
        """
        if not self.api_key:
            raise ArtificialAnalysisError(
                "ArtificialAnalysisFetcher requires an API key "
                "(pass api_key= or set ARTIFICIAL_ANALYSIS_API_KEY)."
            )

        client = await self._client()
        headers = {"x-api-key": self.api_key, "accept": "application/json"}
        try:
            response = await client.get(f"{self.base_url}/models", headers=headers)
        except httpx.HTTPError as e:
            raise ArtificialAnalysisError(f"AA fetch failed: {e}") from e

        if response.status_code >= 400:
            raise ArtificialAnalysisError(
                f"AA returned {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        # AA wraps results under "data" or "models"; tolerate either.
        if isinstance(payload, dict):
            return payload.get("data") or payload.get("models") or []
        if isinstance(payload, list):
            return payload
        raise ArtificialAnalysisError(f"Unexpected AA payload shape: {type(payload)}")

    async def fetch(self) -> list[ModelEntry]:
        """BenchmarkSource interface: fetch + normalize in one call."""
        raw = await self.fetch_models()
        out: list[ModelEntry] = []
        for record in raw:
            try:
                out.append(normalize_aa_model(record))
            except (ValueError, KeyError):
                continue  # skip malformed entries
        return out

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""LiveBench benchmark source.

LiveBench publishes monthly leaderboard snapshots covering math, coding,
reasoning, language, instruction following, and data analysis. Their public
data file is a JSON dump per release. URL pattern is hardcoded below; if
LiveBench changes their hosting we update this single constant.

Each LiveBench record maps to multiple of our task_scores: math + reasoning →
`reasoning`, coding → `coding`, IF → `simple_qa`, language → `simple_qa`.
We average within each target category.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from modelmeld.scout.benchmarks.base import BenchmarkSource
from modelmeld.scout.registry import ModelEntry

_DEFAULT_URL = "https://livebench.ai/leaderboard.json"

# LiveBench category name → our task category. Multiple LB categories can
# map to the same task; we average them. Items not listed are ignored.
_CATEGORY_TO_TASK: dict[str, str] = {
    "coding": "coding",
    "math": "reasoning",
    "reasoning": "reasoning",
    "data_analysis": "reasoning",
    "instruction_following": "simple_qa",
    "IF": "simple_qa",
    "language": "simple_qa",
}


class LiveBenchError(Exception):
    """Raised when the LiveBench fetch or parse fails."""


class LiveBenchFetcher(BenchmarkSource):
    """Fetch + normalize LiveBench leaderboard JSON into ModelEntries."""

    name = "livebench"

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
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

    async def fetch(self) -> list[ModelEntry]:
        client = await self._client()
        try:
            response = await client.get(self.url, headers={"accept": "application/json"})
        except httpx.HTTPError as e:
            raise LiveBenchError(f"LiveBench fetch failed: {e}") from e
        if response.status_code >= 400:
            raise LiveBenchError(
                f"LiveBench returned {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        # LB might be a bare list or {"models": [...]}
        if isinstance(payload, dict):
            records = payload.get("models") or payload.get("data") or []
        elif isinstance(payload, list):
            records = payload
        else:
            raise LiveBenchError(f"Unexpected LiveBench payload shape: {type(payload)}")

        out: list[ModelEntry] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                entry = _normalize_livebench_record(record)
            except ValueError:
                continue
            if entry is not None:
                out.append(entry)
        return out


def _normalize_livebench_record(record: dict[str, Any]) -> ModelEntry | None:
    """Convert one LiveBench JSON row to a ModelEntry."""
    from modelmeld.scout.benchmarks.artificial_analysis import canonicalize_model_id

    raw_model = record.get("model") or record.get("name") or ""
    if not raw_model:
        return None
    model_id = canonicalize_model_id(str(raw_model))

    # Group LB scores by our task category, average within each group.
    grouped: dict[str, list[float]] = {}
    for lb_cat, our_task in _CATEGORY_TO_TASK.items():
        raw = record.get(lb_cat)
        if raw is None:
            continue
        try:
            score = float(raw) / 100.0   # LB scores are 0-100
        except (TypeError, ValueError):
            continue
        if not (0.0 <= score <= 1.0):
            continue
        grouped.setdefault(our_task, []).append(score)

    task_scores = {task: sum(vals) / len(vals) for task, vals in grouped.items() if vals}
    if not task_scores:
        return None

    return ModelEntry(
        model_id=model_id,
        provider=str(record.get("provider") or record.get("organization") or "unknown").lower(),
        context_window=int(record.get("context_window") or 0),
        cost_per_m_input=0.0,
        cost_per_m_output=0.0,
        task_scores=task_scores,
        last_updated=record.get("evaluation_date") or datetime.now(timezone.utc).isoformat(),
        source="livebench",
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""LMArena (formerly LMSYS Chatbot Arena) benchmark source.

Daily JSON snapshots published by the community mirror at
`oolong-tea-2026/arena-ai-leaderboards`. LMArena scores reflect subjective
user preference (head-to-head pairwise votes), not capability tested
against ground truth — so the default source weight is 0.3 (much lower
than AA / Aider / LiveBench).

We pull from two boards: text_arena (general) and code_arena (coding
preference). The Elo ratings are normalized to 0–1 by a clamped mapping:
Elo 1000 → 0.50, Elo 1300 → 0.90 (approximate ceiling at the May 2026
leaderboard's top). Clamp to [0, 1].
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from modelmeld.scout.benchmarks.base import BenchmarkSource
from modelmeld.scout.registry import ModelEntry

_DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/oolong-tea-2026/arena-ai-leaderboards/main"
)

# Maps which LMArena board contributes to which of our task categories.
# Each board provides ONE score per model; we map it to one or more tasks.
_BOARD_TO_TASKS: dict[str, list[str]] = {
    "text_arena": ["simple_qa"],   # general chat preference
    "code_arena": ["coding"],
}

# Elo → 0–1 calibration. Elo 800 = 0.20, Elo 1300 = 0.90; clamp.
_ELO_FLOOR = 800
_ELO_CEILING = 1300


class LMArenaError(Exception):
    """Raised when the LMArena fetch or parse fails."""


class LMArenaFetcher(BenchmarkSource):
    """Fetch LMArena daily snapshots from the community mirror."""

    name = "lmarena"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        boards: tuple[str, ...] = ("text_arena", "code_arena"),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = http_client
        self._timeout = timeout
        self._owns_client = http_client is None
        self.boards = boards

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def fetch(self) -> list[ModelEntry]:
        # Per-model accumulator: model_id → {task: [scores]}
        per_model_tasks: dict[str, dict[str, list[float]]] = {}
        per_model_meta: dict[str, dict[str, Any]] = {}

        client = await self._client()
        for board in self.boards:
            url = f"{self.base_url}/{board}.json"
            try:
                response = await client.get(url, headers={"accept": "application/json"})
            except httpx.HTTPError:
                continue  # tolerate per-board failures; merge what we got
            if response.status_code >= 400:
                continue

            try:
                payload = response.json()
            except ValueError:
                continue

            records = (
                payload if isinstance(payload, list)
                else payload.get("models", []) if isinstance(payload, dict)
                else []
            )
            target_tasks = _BOARD_TO_TASKS.get(board, [])
            if not target_tasks:
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                normalized = _process_lmarena_record(record, target_tasks)
                if normalized is None:
                    continue
                model_id, task_contributions, meta = normalized
                for task, score in task_contributions.items():
                    per_model_tasks.setdefault(model_id, {}).setdefault(task, []).append(score)
                per_model_meta.setdefault(model_id, meta)

        # Average per task across multiple boards if both contributed.
        out: list[ModelEntry] = []
        for model_id, tasks in per_model_tasks.items():
            task_scores = {task: sum(vals) / len(vals) for task, vals in tasks.items()}
            meta = per_model_meta.get(model_id, {})
            out.append(
                ModelEntry(
                    model_id=model_id,
                    provider=str(meta.get("organization") or "unknown").lower(),
                    context_window=0,
                    cost_per_m_input=0.0,
                    cost_per_m_output=0.0,
                    task_scores=task_scores,
                    last_updated=str(
                        meta.get("date") or datetime.now(timezone.utc).isoformat()
                    ),
                    source="lmarena",
                )
            )
        return out


def _process_lmarena_record(
    record: dict[str, Any], target_tasks: list[str]
) -> tuple[str, dict[str, float], dict[str, Any]] | None:
    """Extract (model_id, {task: score}, meta) from one LMArena record.

    Returns None when the record is unusable (no Elo / no model name).
    """
    from modelmeld.scout.benchmarks.artificial_analysis import canonicalize_model_id

    raw_model = record.get("model") or record.get("name") or ""
    if not raw_model:
        return None
    model_id = canonicalize_model_id(str(raw_model))

    elo_raw = record.get("rating") or record.get("elo") or record.get("score")
    if elo_raw is None:
        return None
    try:
        elo = float(elo_raw)
    except (TypeError, ValueError):
        return None

    normalized = _elo_to_unit(elo)
    contributions = {task: normalized for task in target_tasks}
    meta = {
        "organization": record.get("organization") or record.get("provider"),
        "date": record.get("date") or record.get("snapshot_date"),
    }
    return model_id, contributions, meta


def _elo_to_unit(elo: float) -> float:
    """Linear map [800, 1300] → [0.20, 0.90]; clamp outside."""
    if elo <= _ELO_FLOOR:
        return 0.20
    if elo >= _ELO_CEILING:
        return 0.90
    return 0.20 + (elo - _ELO_FLOOR) / (_ELO_CEILING - _ELO_FLOOR) * (0.90 - 0.20)

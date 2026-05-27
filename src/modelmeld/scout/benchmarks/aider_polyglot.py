# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Aider Polyglot benchmark source.

Fetches the YAML leaderboard from `Aider-AI/aider` repo (canonical location:
`aider/website/_data/edit_leaderboard.yml`). The polyglot benchmark covers
6 languages and 225 Exercism problems — the de facto coding-capability bar
for AI engineering tooling in 2026.

Aider records: model id, pass_rate, edit_format, total_cost, etc. We only
contribute to `task_scores.coding`; all other task categories are left empty
so the composer falls back to other sources.

Model IDs use `provider/model` format (e.g., `anthropic/claude-opus-4-7`).
We canonicalize to our naming convention and extract the provider hint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import yaml

from modelmeld.scout.benchmarks.base import BenchmarkSource
from modelmeld.scout.registry import ModelEntry

_DEFAULT_URL = (
    "https://raw.githubusercontent.com/Aider-AI/aider/main/"
    "aider/website/_data/edit_leaderboard.yml"
)


class AiderPolyglotError(Exception):
    """Raised when the Aider Polyglot fetch or parse fails."""


class AiderPolyglotFetcher(BenchmarkSource):
    """Fetch + normalize Aider's edit-leaderboard YAML into ModelEntries."""

    name = "aider_polyglot"

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
            response = await client.get(self.url, headers={"accept": "application/x-yaml"})
        except httpx.HTTPError as e:
            raise AiderPolyglotError(f"Aider fetch failed: {e}") from e
        if response.status_code >= 400:
            raise AiderPolyglotError(
                f"Aider returned {response.status_code}: {response.text[:200]}"
            )

        try:
            records = yaml.safe_load(response.text)
        except yaml.YAMLError as e:
            raise AiderPolyglotError(f"Aider YAML parse failed: {e}") from e

        if not isinstance(records, list):
            raise AiderPolyglotError(f"Unexpected Aider payload shape: {type(records)}")

        out: list[ModelEntry] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                entry = _normalize_aider_record(record)
            except ValueError:
                continue
            if entry is not None:
                out.append(entry)
        return out


def _normalize_aider_record(record: dict[str, Any]) -> ModelEntry | None:
    """Convert one Aider YAML row to a ModelEntry.

    Returns None when the record lacks a pass_rate (it's a header / placeholder).
    """
    raw_model = (record.get("model") or "").strip()
    if not raw_model:
        return None

    # Aider uses `provider/model` format. Extract.
    if "/" in raw_model:
        provider_hint, model_part = raw_model.split("/", 1)
    else:
        provider_hint, model_part = "unknown", raw_model

    from modelmeld.scout.benchmarks.artificial_analysis import canonicalize_model_id

    model_id = canonicalize_model_id(model_part)

    # Aider's coding score: `pass_rate_2` is the standard field as of 2026; fall back
    # to `pass_rate_1` or `pass_rate`.
    pass_rate = (
        record.get("pass_rate_2")
        or record.get("pass_rate_1")
        or record.get("pass_rate")
    )
    if pass_rate is None:
        return None
    try:
        # Aider scores are 0–100; normalize to 0–1.
        coding_score = float(pass_rate) / 100.0
    except (TypeError, ValueError):
        return None

    if not (0.0 <= coding_score <= 1.0):
        return None

    # Aider also publishes total_cost (USD for the whole 225-case run). Not
    # directly comparable to per-million-token cost, so we leave cost fields at
    # zero — the composer's metadata-merge will prefer AA's pricing.
    return ModelEntry(
        model_id=model_id,
        provider=provider_hint.lower(),
        context_window=0,  # Aider doesn't report; falls back to AA's value on merge
        cost_per_m_input=0.0,
        cost_per_m_output=0.0,
        task_scores={"coding": coding_score},
        last_updated=record.get("released") or datetime.now(timezone.utc).isoformat(),
        source="aider_polyglot",
    )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""In-process request/spend metrics — a basic local observability surface.

Thread-safe collector that accumulates, since process start: request count
(broken down by inbound wire format), summed prompt/completion/total tokens, a
per-routed-model breakdown, and summed *actual* blended cost (the routed
model's registry rate times its tokens). One collector instance lives on
`app.state.metrics`; `GET /metrics` returns its snapshot.

Deliberately minimal and single-process. Multi-tenant, per-API-key, RBAC-gated,
or persisted analytics are out of scope (those are enterprise concerns). This
module imports nothing from the API or enterprise packages.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# Inbound wire formats we break request counts down by.
WIRE_FORMATS: tuple[str, ...] = ("chat_completions", "messages", "responses")


@dataclass(frozen=True)
class ModelMetrics:
    """Accumulated metrics for a single routed model id."""

    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable point-in-time view of the collector."""

    chat_completions_count: int
    messages_count: int
    responses_count: int
    total_request_count: int

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    total_cost_usd: float
    uptime_seconds: float

    per_model: dict[str, ModelMetrics]


class MetricsCollector:
    """Thread-safe in-process metrics accumulator.

    One instance per app (on `app.state.metrics`), so a freshly built app
    starts with zeroed counters — important for test isolation and for
    not leaking state across processes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._created_at = time.monotonic()
        self._wire_counts: dict[str, int] = {fmt: 0 for fmt in WIRE_FORMATS}
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        # model_id -> [count, input_tokens, output_tokens, cost_usd]
        self._model_stats: dict[str, list[float]] = {}

    def record(
        self,
        *,
        wire_format: str,
        input_tokens: int,
        output_tokens: int,
        model_id: str,
        cost_usd: float,
    ) -> None:
        """Record one completed request. `cost_usd` is the caller-computed
        actual blended cost (tokens x the routed model's rate); pass 0.0 when
        no rate is known for the routed model."""
        with self._lock:
            self._wire_counts[wire_format] = self._wire_counts.get(wire_format, 0) + 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cost_usd += cost_usd

            key = model_id or "unknown"
            stats = self._model_stats.get(key)
            if stats is None:
                stats = [0, 0, 0, 0.0]
                self._model_stats[key] = stats
            stats[0] += 1
            stats[1] += input_tokens
            stats[2] += output_tokens
            stats[3] += cost_usd

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            per_model = {
                model_id: ModelMetrics(
                    request_count=int(s[0]),
                    input_tokens=int(s[1]),
                    output_tokens=int(s[2]),
                    total_tokens=int(s[1]) + int(s[2]),
                    cost_usd=float(s[3]),
                )
                for model_id, s in self._model_stats.items()
            }
            uptime = time.monotonic() - self._created_at
            return MetricsSnapshot(
                chat_completions_count=self._wire_counts.get("chat_completions", 0),
                messages_count=self._wire_counts.get("messages", 0),
                responses_count=self._wire_counts.get("responses", 0),
                # Sum ALL buckets (not just the three named) so an unexpected
                # wire_format can't desync the total from the per-format counts.
                total_request_count=sum(self._wire_counts.values()),
                total_input_tokens=self._input_tokens,
                total_output_tokens=self._output_tokens,
                total_tokens=self._input_tokens + self._output_tokens,
                total_cost_usd=self._cost_usd,
                uptime_seconds=uptime,
                per_model=per_model,
            )

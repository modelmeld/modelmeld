# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""GET /metrics — process-local request/spend observability surface.

Returns the current snapshot of the app's in-process MetricsCollector
(`app.state.metrics`) as JSON. Basic single-process surface; no enterprise
features (multi-tenant / per-key / persisted analytics stay enterprise).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from modelmeld.metrics import MetricsCollector, MetricsSnapshot

router = APIRouter()


def _serialize(snap: MetricsSnapshot) -> dict:
    return {
        "chat_completions_count": snap.chat_completions_count,
        "messages_count": snap.messages_count,
        "responses_count": snap.responses_count,
        "total_request_count": snap.total_request_count,
        "total_input_tokens": snap.total_input_tokens,
        "total_output_tokens": snap.total_output_tokens,
        "total_tokens": snap.total_tokens,
        "total_cost_usd": snap.total_cost_usd,
        "uptime_seconds": snap.uptime_seconds,
        "per_model": {
            model_id: {
                "request_count": m.request_count,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "total_tokens": m.total_tokens,
                "cost_usd": m.cost_usd,
            }
            for model_id, m in snap.per_model.items()
        },
    }


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    """Return the current process-local metrics snapshot as JSON."""
    collector: MetricsCollector | None = getattr(request.app.state, "metrics", None)
    if collector is None:
        # Defensive: build_app always installs one, but never 500 if absent.
        return _serialize(MetricsCollector().snapshot())
    return _serialize(collector.snapshot())

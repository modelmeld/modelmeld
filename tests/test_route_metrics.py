# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Integration tests for the GET /metrics observability surface (A-1).

Drives the in-process app with a mock adapter (no real backend) and asserts the
collector reflects requests across wire formats, plus a focused unit test of the
collector's accounting. Cost is asserted as a non-negative float end-to-end (the
exact rate depends on registry contents); the cost arithmetic itself is pinned
in the collector unit test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.metrics import MetricsCollector


class _EchoAdapter(ProviderAdapter):
    name = "mock-echo"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content="echo"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:
            yield  # pragma: no cover

    async def health(self) -> bool:
        return True


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    )


async def test_fresh_app_has_zeroed_metrics() -> None:
    app = build_app(adapter=_EchoAdapter())
    async with _client(app) as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chat_completions_count"] == 0
    assert body["messages_count"] == 0
    assert body["responses_count"] == 0
    assert body["total_request_count"] == 0
    assert body["total_input_tokens"] == 0
    assert body["total_output_tokens"] == 0
    assert body["total_tokens"] == 0
    assert body["total_cost_usd"] == 0.0
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], (int, float))
    assert body["uptime_seconds"] >= 0.0
    assert body["per_model"] == {}


async def test_chat_and_messages_requests_update_metrics() -> None:
    app = build_app(adapter=_EchoAdapter())
    async with _client(app) as client:
        r1 = await client.post("/v1/chat/completions", json={
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": "hi"}],
        })
        r2 = await client.post("/v1/messages", json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hello"}],
        })
        metrics = await client.get("/metrics")

    assert r1.status_code == 200
    assert r2.status_code == 200
    body = metrics.json()

    # Wire-format breakdown reflects one of each.
    assert body["chat_completions_count"] == 1
    assert body["messages_count"] == 1
    assert body["responses_count"] == 0
    assert body["total_request_count"] == 2

    # Token totals are the sum of both echo responses (12 in / 8 out each).
    assert body["total_input_tokens"] == 24
    assert body["total_output_tokens"] == 16
    assert body["total_tokens"] == 40

    # Cost is a non-negative float (actual rate depends on registry contents).
    assert isinstance(body["total_cost_usd"], (int, float))
    assert body["total_cost_usd"] >= 0.0

    # Per-model entry for the served model, with summed tokens across both calls.
    per_model = body["per_model"]
    assert "claude-haiku-4-5-20251001" in per_model
    entry = per_model["claude-haiku-4-5-20251001"]
    assert entry["request_count"] == 2
    assert entry["input_tokens"] == 24
    assert entry["output_tokens"] == 16
    assert entry["total_tokens"] == 40


async def test_metrics_json_shape() -> None:
    app = build_app(adapter=_EchoAdapter())
    async with _client(app) as client:
        body = (await client.get("/metrics")).json()
    for key in (
        "chat_completions_count", "messages_count", "responses_count",
        "total_request_count", "total_input_tokens", "total_output_tokens",
        "total_tokens", "total_cost_usd", "uptime_seconds", "per_model",
    ):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["per_model"], dict)
    assert isinstance(body["uptime_seconds"], (int, float))


async def test_uptime_seconds_tracks_elapsed_time() -> None:
    """uptime_seconds must reflect real elapsed time, not a constant. Read twice
    across a sleep and require a STRICT increase — a hardcoded/zeroed value (the
    obvious regression) would pass a `>=` check but fails this."""
    import asyncio

    app = build_app(adapter=_EchoAdapter())
    async with _client(app) as client:
        r1 = await client.get("/metrics")
        uptime1 = r1.json()["uptime_seconds"]

        await asyncio.sleep(0.05)

        r2 = await client.get("/metrics")
        uptime2 = r2.json()["uptime_seconds"]

    assert isinstance(uptime1, (int, float))
    assert isinstance(uptime2, (int, float))
    assert uptime1 >= 0.0
    # Strictly greater after a real delay: proves it is wired to a live clock.
    assert uptime2 > uptime1


def test_collector_accounting_and_cost() -> None:
    """Unit-test the collector's sums + per-model breakdown directly, including
    the caller-supplied cost (the route computes tokens x rate; here we pin the
    accumulation)."""
    c = MetricsCollector()
    c.record(wire_format="messages", input_tokens=100, output_tokens=50,
             model_id="qwen3-coder-next", cost_usd=0.012)
    c.record(wire_format="messages", input_tokens=10, output_tokens=5,
             model_id="qwen3-coder-next", cost_usd=0.001)
    c.record(wire_format="chat_completions", input_tokens=7, output_tokens=3,
             model_id="claude-haiku-4-5", cost_usd=0.0)

    snap = c.snapshot()
    assert snap.messages_count == 2
    assert snap.chat_completions_count == 1
    assert snap.total_request_count == 3
    assert snap.total_input_tokens == 117
    assert snap.total_output_tokens == 58
    assert snap.total_tokens == 175
    assert abs(snap.total_cost_usd - 0.013) < 1e-9

    qwen = snap.per_model["qwen3-coder-next"]
    assert qwen.request_count == 2
    assert qwen.input_tokens == 110
    assert qwen.output_tokens == 55
    assert qwen.total_tokens == 165
    assert abs(qwen.cost_usd - 0.013) < 1e-9

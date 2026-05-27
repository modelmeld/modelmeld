"""Runtime failover: primary adapter error → fallback to the other tier."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
)
from modelmeld.api.server import build_app
from modelmeld.router import RoutingPolicy, TieredRouter
from modelmeld.scout.base import Scout, ScoutDecision, Tier


class FailingAdapter(ProviderAdapter):
    def __init__(self, name: str, fail_chat: bool = False, fail_stream: bool = False) -> None:
        self._name = name
        self.fail_chat = fail_chat
        self.fail_stream = fail_stream
        self.chat_calls = 0
        self.stream_calls = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.chat_calls += 1
        if self.fail_chat:
            raise AdapterError(f"{self._name} chat fails by config")
        return ChatCompletion(
            id=f"chatcmpl-{self._name}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=f"from {self._name}"),
                    finish_reason="stop",
                )
            ],
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        self.stream_calls += 1
        if self.fail_stream:
            raise AdapterError(f"{self._name} stream fails by config")
        yield ChatCompletionChunk(
            id=f"chatcmpl-{self._name}",
            created=1,
            model=request.model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=f"from {self._name}"),
                    finish_reason="stop",
                )
            ],
        )

    async def health(self) -> bool:
        return True


class FixedScout(Scout):
    name = "fixed"

    def __init__(self, tier: Tier) -> None:
        self._tier = tier

    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        return ScoutDecision(tier=self._tier, confidence=1.0, rationale="fixed")


def _payload() -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_chat_failover_on_primary_error() -> None:
    local = FailingAdapter(name="local-fail", fail_chat=True)
    cloud = FailingAdapter(name="cloud-ok")
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.SCOUT_DRIVEN,
        health_ttl_sec=0.0,
    )
    with TestClient(build_app(router=router)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-cloud-ok"
    assert resp.headers["x-modelmeld-routed-to"] == "cloud-ok"
    assert resp.headers["x-modelmeld-tier"] == "cloud"
    assert resp.headers["x-modelmeld-failover-from"] == "local"
    assert local.chat_calls == 1
    assert cloud.chat_calls == 1


def test_chat_502_when_both_tiers_fail() -> None:
    local = FailingAdapter(name="local-fail", fail_chat=True)
    cloud = FailingAdapter(name="cloud-fail", fail_chat=True)
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        health_ttl_sec=0.0,
    )
    with TestClient(build_app(router=router)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "primary failed" in detail
    assert "fallback failed" in detail


def test_chat_no_failover_when_primary_succeeds() -> None:
    local = FailingAdapter(name="local-ok")
    cloud = FailingAdapter(name="cloud-ok")
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        health_ttl_sec=0.0,
    )
    with TestClient(build_app(router=router)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    assert resp.json()["id"] == "chatcmpl-local-ok"
    assert "x-modelmeld-failover-from" not in resp.headers
    assert local.chat_calls == 1
    assert cloud.chat_calls == 0


def test_single_adapter_router_no_failover_possible() -> None:
    adapter = FailingAdapter(name="solo", fail_chat=True)
    with TestClient(build_app(adapter=adapter)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())
    assert resp.status_code == 502
    assert "solo chat fails" in resp.json()["detail"]


def test_stream_failover_on_primary_open_error() -> None:
    local = FailingAdapter(name="local-fail", fail_stream=True)
    cloud = FailingAdapter(name="cloud-ok")
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        health_ttl_sec=0.0,
    )
    with TestClient(build_app(router=router)) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={**_payload(), "stream": True},
    ) as resp:
        assert resp.headers["x-modelmeld-routed-to"] == "cloud-ok"
        assert resp.headers["x-modelmeld-failover-from"] == "local"
        body = b"".join(resp.iter_bytes()).decode("utf-8")
    assert "data: [DONE]" in body
    assert local.stream_calls == 1
    assert cloud.stream_calls == 1


def test_stream_502_when_both_streams_fail() -> None:
    local = FailingAdapter(name="local-fail", fail_stream=True)
    cloud = FailingAdapter(name="cloud-fail", fail_stream=True)
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        health_ttl_sec=0.0,
    )
    with TestClient(build_app(router=router)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={**_payload(), "stream": True},
        )
    assert resp.status_code == 502


def test_failed_tier_marked_unhealthy_in_cache() -> None:
    """After a chat failure, subsequent requests skip the failed tier.

    F-4: with the default `unhealthy_threshold=3`, a single failure would
    NOT blacklist the tier. This test verifies the *cache mechanism* (not
    the threshold policy), so it pins `unhealthy_threshold=1` to keep the
    one-failure-blacklists semantics. Threshold policy is exercised in
    `test_router_tiered.test_threshold_failures_marks_tier_unhealthy`.
    """
    local = FailingAdapter(name="local-fail", fail_chat=True)
    cloud = FailingAdapter(name="cloud-ok")
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.SCOUT_DRIVEN,
        health_ttl_sec=60.0,
        unhealthy_threshold=1,
    )
    with TestClient(build_app(router=router)) as client:
        # First call fails over to cloud
        first = client.post("/v1/chat/completions", json=_payload())
        # Second call should go straight to cloud (cached unhealthy local)
        second = client.post("/v1/chat/completions", json=_payload())

    assert first.headers.get("x-modelmeld-failover-from") == "local"
    assert second.headers.get("x-modelmeld-failover-from") is None
    assert second.headers["x-modelmeld-tier"] == "cloud"
    # Local was only retried once (the first call); second call skipped it pre-route.
    assert local.chat_calls == 1
    assert cloud.chat_calls == 2

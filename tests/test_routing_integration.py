"""End-to-end: chat route consults router; routing headers exposed on response."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.router import RoutingPolicy, TieredRouter
from modelmeld.scout.base import Scout, ScoutDecision, Tier


class TierTaggedAdapter(ProviderAdapter):
    """Adapter that tags its responses so we can verify which tier was hit."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
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
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
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


class ScoutSays(Scout):
    name = "fixed"

    def __init__(self, tier: Tier) -> None:
        self._tier = tier

    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        return ScoutDecision(tier=self._tier, confidence=1.0, rationale=f"fixed:{self._tier}")


def _tiered_router(scout_tier: Tier, policy: RoutingPolicy = RoutingPolicy.SCOUT_DRIVEN) -> TieredRouter:
    return TieredRouter(
        scout=ScoutSays(scout_tier),
        adapters={
            Tier.LOCAL: TierTaggedAdapter("local-fake"),
            Tier.CLOUD: TierTaggedAdapter("cloud-fake"),
        },
        policy=policy,
    )


def test_scout_local_routes_to_local_adapter() -> None:
    with TestClient(build_app(router=_tiered_router(Tier.LOCAL))) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-local-fake"
    assert resp.headers["x-modelmeld-routed-to"] == "local-fake"
    assert resp.headers["x-modelmeld-tier"] == "local"


def test_scout_cloud_routes_to_cloud_adapter() -> None:
    with TestClient(build_app(router=_tiered_router(Tier.CLOUD))) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["id"] == "chatcmpl-cloud-fake"
    assert resp.headers["x-modelmeld-routed-to"] == "cloud-fake"
    assert resp.headers["x-modelmeld-tier"] == "cloud"


def test_always_local_policy_overrides_scout() -> None:
    router = _tiered_router(Tier.CLOUD, policy=RoutingPolicy.ALWAYS_LOCAL)
    with TestClient(build_app(router=router)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.headers["x-modelmeld-tier"] == "local"


def test_streaming_response_carries_routing_headers() -> None:
    with TestClient(build_app(router=_tiered_router(Tier.LOCAL))) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.headers["x-modelmeld-tier"] == "local"
            assert resp.headers["x-modelmeld-routed-to"] == "local-fake"
            # Consume the body so the connection closes cleanly
            for _ in resp.iter_bytes():
                pass


def test_no_healthy_adapter_returns_503() -> None:
    class UnhealthyAdapter(TierTaggedAdapter):
        async def health(self) -> bool:
            return False

    router = TieredRouter(
        scout=ScoutSays(Tier.LOCAL),
        adapters={
            Tier.LOCAL: UnhealthyAdapter("local-down"),
            Tier.CLOUD: UnhealthyAdapter("cloud-down"),
        },
    )
    with TestClient(build_app(router=router)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503
    assert "No healthy adapter" in resp.json()["detail"]


def test_single_adapter_router_default_path() -> None:
    """build_app() with defaults still works (single-adapter ergonomics preserved)."""
    with TestClient(build_app()) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-routed-to"] == "stub"

"""CapabilityRouter — provider selection + failover."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
    Usage,
    UserMessage,
)
from modelmeld.router import CapabilityRouter, RouterError, RoutingPolicy
from modelmeld.scout import (
    CapabilityScout,
    ModelEntry,
    ModelRegistry,
    Tier,
)


class _FakeAdapter(ProviderAdapter):
    """Adapter that records its name and reports configurable health."""

    is_egress = True

    def __init__(self, name: str, healthy: bool = True) -> None:
        self.name = name
        self._healthy = healthy
        self.calls: list[str] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.calls.append(request.model)
        return ChatCompletion(
            model=request.model,
            choices=[Choice(index=0, message=ResponseMessage(content="ok"), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return self._healthy


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="user-requested",  # CapabilityRouter should override this
        messages=[UserMessage(role="user", content="refactor this code")],
        tools=[],
    )


def _entry(
    model_id: str, provider: str, cost_in: float, cost_out: float, coding: float,
    provider_model_id: str = "",
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, provider=provider, context_window=100000,
        cost_per_m_input=cost_in, cost_per_m_output=cost_out,
        task_scores={"coding": coding},
        last_updated="2026-05-17", source="test",
        provider_model_id=provider_model_id,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_router_returns_chosen_model_and_adapter() -> None:
    registry = ModelRegistry([
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    vllm = _FakeAdapter("vllm")
    anthropic = _FakeAdapter("anthropic")
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"vllm": vllm, "anthropic": anthropic},
    )

    decision = await router.route(_req())
    assert decision.adapter is vllm
    assert decision.model_id_override == "qwen"
    assert decision.policy_applied == RoutingPolicy.CAPABILITY
    assert decision.capability_decision is not None
    # Tier mapping: vllm → LOCAL
    assert decision.tier == Tier.LOCAL


async def test_anthropic_pick_maps_to_cloud_tier() -> None:
    registry = ModelRegistry([_entry("opus", "anthropic", 5.0, 25.0, coding=0.95)])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"anthropic": _FakeAdapter("anthropic")},
    )
    decision = await router.route(_req())
    assert decision.tier == Tier.CLOUD


# ---------------------------------------------------------------------------
# provider_model_id — the provider's own wire slug, distinct from the canonical
# model_id used for attribution. Regression: the canonical id used to leak onto
# the egress wire (the overlay's provider_model_id was never applied), so any
# model whose provider slug differs from its canonical id 4xx'd at the upstream.
# ---------------------------------------------------------------------------

async def test_decision_carries_provider_model_id_for_wire() -> None:
    registry = ModelRegistry([
        _entry("qwen3-coder-480b", "openai", 0.5, 1.0, coding=0.85,
               provider_model_id="qwen/qwen3-coder"),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout, adapters_by_provider={"openai": _FakeAdapter("openai")},
    )
    decision = await router.route(_req())
    # Canonical id is preserved for attribution; the provider slug rides the wire.
    assert decision.model_id_override == "qwen3-coder-480b"
    assert decision.provider_model_id == "qwen/qwen3-coder"


async def test_decision_provider_model_id_none_when_absent() -> None:
    """No provider_model_id on the row → None, so egress sends the canonical id
    verbatim (back-compat for providers that accept the bare name)."""
    registry = ModelRegistry([_entry("gpt", "openai", 1.0, 3.0, coding=0.85)])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout, adapters_by_provider={"openai": _FakeAdapter("openai")},
    )
    decision = await router.route(_req())
    assert decision.model_id_override == "gpt"
    assert decision.provider_model_id is None


async def test_fallback_uses_fallback_entrys_provider_model_id() -> None:
    """A fallback must carry the FALLBACK row's slug, not the primary's."""
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85,
               provider_model_id="vendor/qwen-primary"),         # primary
        _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.83,
               provider_model_id="openai/gpt-mini-slug"),        # 1st fallback
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "vllm": _FakeAdapter("vllm", healthy=False),  # force fallback
            "openai": _FakeAdapter("openai", healthy=True),
        },
    )
    decision = await router.route(_req())
    assert decision.model_id_override == "gpt-mini"
    assert decision.provider_model_id == "openai/gpt-mini-slug"


def test_apply_model_override_puts_provider_slug_on_wire() -> None:
    from modelmeld.api.routes.chat import _apply_model_override
    from modelmeld.router import RoutingDecision

    adapter = _FakeAdapter("openai")
    decision = RoutingDecision(
        tier=Tier.CLOUD, adapter=adapter, scout_decision=None,
        policy_applied=RoutingPolicy.CAPABILITY, rationale="t",
        model_id_override="qwen3-coder-480b",
        provider_model_id="qwen/qwen3-coder",
    )
    out = _apply_model_override(_req(), decision)
    assert out.model == "qwen/qwen3-coder"  # wire = provider slug


def test_apply_model_override_falls_back_to_canonical_without_slug() -> None:
    from modelmeld.api.routes.chat import _apply_model_override
    from modelmeld.router import RoutingDecision

    adapter = _FakeAdapter("openai")
    decision = RoutingDecision(
        tier=Tier.CLOUD, adapter=adapter, scout_decision=None,
        policy_applied=RoutingPolicy.CAPABILITY, rationale="t",
        model_id_override="deepseek-v4-pro",
        provider_model_id=None,
    )
    out = _apply_model_override(_req(), decision)
    assert out.model == "deepseek-v4-pro"  # no slug → canonical verbatim


# ---------------------------------------------------------------------------
# MULTI-PROVIDER wire-slug invariant — the one the gates were BLIND to.
# When a model_id is served by several providers, the egress wire must carry
# the slug of the provider the SCOUT CHOSE (cheapest eligible), NOT whatever
# registry.get(model_id) returns (the last-inserted/default row). Re-deriving
# the slug inside the router via registry.get()/lookup_fallback() passes every
# single-provider fixture above but sends the WRONG provider's slug here. The
# decision must thread the *chosen entry's* slug through from the scout.
# ---------------------------------------------------------------------------

async def test_multi_provider_wire_slug_is_scout_chosen_providers_slug() -> None:
    from modelmeld.api.routes.chat import _apply_model_override
    from modelmeld.scout.multi_provider_registry import MultiProviderModelRegistry

    # Same model_id, two providers, DISTINCT slugs. The PRICEY row is inserted
    # last, so base get(model_id) (last-write-wins) returns it — while pick()
    # (cheapest) returns the CHEAP row. That divergence is what makes the latent
    # bug observable; a single-provider registry can never expose it.
    registry = MultiProviderModelRegistry([
        _entry("glm-5", "cheap-prov", 0.10, 0.20, coding=0.85,
               provider_model_id="cheap-prov/glm-5"),
        _entry("glm-5", "pricey-prov", 5.0, 9.0, coding=0.85,
               provider_model_id="pricey-prov/glm-5"),
    ])
    # Precondition: the two accessors genuinely disagree (else the test is moot).
    default = registry.get("glm-5")
    assert default is not None and default.provider_model_id == "pricey-prov/glm-5"

    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "cheap-prov": _FakeAdapter("cheap-prov"),
            "pricey-prov": _FakeAdapter("pricey-prov"),
        },
    )
    decision = await router.route(_req())

    # Scout picks the cheapest provider; the canonical id is preserved for
    # attribution; the WIRE slug must be the chosen (cheap) provider's, never
    # the get()-default (pricey) one.
    assert decision.adapter.name == "cheap-prov"
    assert decision.model_id_override == "glm-5"
    assert decision.provider_model_id == "cheap-prov/glm-5"
    out = _apply_model_override(_req(), decision)
    assert out.model == "cheap-prov/glm-5"


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------

async def test_falls_back_when_chosen_provider_unhealthy() -> None:
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),     # cheapest
        _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.83),
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "vllm": _FakeAdapter("vllm", healthy=False),
            "openai": _FakeAdapter("openai", healthy=True),
            "anthropic": _FakeAdapter("anthropic", healthy=True),
        },
    )
    decision = await router.route(_req())
    # vllm unhealthy → next fallback (gpt-mini) used
    assert decision.adapter.name == "openai"
    assert decision.model_id_override == "gpt-mini"
    assert "unhealthy" in decision.rationale


async def test_falls_back_when_chosen_provider_not_configured() -> None:
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    # vllm not in adapter map at all
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"anthropic": _FakeAdapter("anthropic")},
    )
    decision = await router.route(_req())
    assert decision.adapter.name == "anthropic"
    assert decision.model_id_override == "opus"
    assert "not_configured" in decision.rationale


async def test_router_error_when_nothing_healthy() -> None:
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "vllm": _FakeAdapter("vllm", healthy=False),
            "anthropic": _FakeAdapter("anthropic", healthy=False),
        },
    )
    with pytest.raises(RouterError, match="No healthy adapter"):
        await router.route(_req())


async def test_router_error_when_no_eligible_model() -> None:
    """The scout's NoEligibleModelError surfaces as a RouterError."""
    registry = ModelRegistry([_entry("weak", "openai", 0.5, 1.0, coding=0.50)])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"openai": _FakeAdapter("openai")},
    )
    with pytest.raises(RouterError, match="capability_scout"):
        await router.route(_req())


# ---------------------------------------------------------------------------
# Failover after request-time failure
# ---------------------------------------------------------------------------

async def test_route_after_failure_picks_next_fallback() -> None:
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),         # primary
        _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.83),   # 1st fallback
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),   # 2nd fallback
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "vllm": _FakeAdapter("vllm"),
            "openai": _FakeAdapter("openai"),
            "anthropic": _FakeAdapter("anthropic"),
        },
    )
    primary = await router.route(_req())
    assert primary.adapter.name == "vllm"

    fallback = await router.route_after_failure(primary, _req())
    assert fallback is not None
    # vllm now in unhealthy cache → next eligible provider (openai)
    assert fallback.adapter.name == "openai"
    assert fallback.model_id_override == "gpt-mini"


async def test_route_after_failure_returns_none_when_exhausted() -> None:
    registry = ModelRegistry([
        _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"vllm": _FakeAdapter("vllm")},
    )
    primary = await router.route(_req())
    # No fallback models in registry → route_after_failure returns None
    assert await router.route_after_failure(primary, _req()) is None


# ---------------------------------------------------------------------------
# Health TTL caching
# ---------------------------------------------------------------------------

async def test_health_check_is_cached() -> None:
    """A second route() call within TTL doesn't re-call adapter.health()."""
    health_calls = {"openai": 0}

    class _CountingAdapter(_FakeAdapter):
        async def health(self) -> bool:
            health_calls[self.name] += 1
            return True

    adapter = _CountingAdapter("openai")
    registry = ModelRegistry([_entry("gpt", "openai", 1.0, 3.0, coding=0.85)])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={"openai": adapter},
        health_ttl_sec=60.0,
    )
    await router.route(_req())
    await router.route(_req())
    assert health_calls["openai"] == 1  # cached on 2nd call


# ---------------------------------------------------------------------------
# Close releases all adapters once
# ---------------------------------------------------------------------------

async def test_close_releases_adapters() -> None:
    closed: list[str] = []

    class _ClosingAdapter(_FakeAdapter):
        async def close(self) -> None:  # type: ignore[override]
            closed.append(self.name)

    registry = ModelRegistry([_entry("a", "openai", 1.0, 3.0, coding=0.85)])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "openai": _ClosingAdapter("openai"),
            "vllm": _ClosingAdapter("vllm"),
        },
    )
    await router.close()
    assert sorted(closed) == ["openai", "vllm"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_empty_adapter_map_rejected() -> None:
    registry = ModelRegistry([])
    scout = CapabilityScout(registry=registry)
    with pytest.raises(ValueError, match="at least one adapter"):
        CapabilityRouter(scout=scout, adapters_by_provider={})

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Routing decision engine — picks an adapter for each request.

Public surface:
    Router               — abstract base class
    RoutingPolicy        — enum (SINGLE / SCOUT_DRIVEN / ALWAYS_LOCAL / ALWAYS_CLOUD)
    RoutingDecision      — frozen result of `Router.route()`
    RouterError          — raised when no adapter can serve the request
    SingleAdapterRouter  — passes everything to one adapter
    TieredRouter         — LOCAL/CLOUD selection with health-aware fallback
    build_router(s, sc)  — factory keyed on GatewaySettings.routing_policy
"""

from __future__ import annotations

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.adapters.stub import StubAdapter
from modelmeld.router.base import (
    Router,
    RouterError,
    RoutingDecision,
    RoutingPolicy,
    SingleAdapterRouter,
    TieredRouter,
)
from modelmeld.router.capability import CapabilityRouter
from modelmeld.scout.base import Scout, Tier


def _build_adapter(provider: str, settings: object) -> ProviderAdapter:
    """Construct an adapter by name from GatewaySettings-shaped config."""
    if provider == "openai":
        # Local import — `openai` is an optional extra.
        from modelmeld.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(
            api_key=getattr(settings, "openai_api_key", None),
            base_url=getattr(settings, "openai_base_url", None),
            served_model=getattr(settings, "openai_served_model", None),
        )
    if provider == "vllm":
        from modelmeld.adapters.vllm_adapter import VLLMAdapter

        return VLLMAdapter(
            endpoint=getattr(settings, "vllm_endpoint", None),
            served_model=getattr(settings, "vllm_served_model", None),
        )
    if provider == "tensorrt_llm":
        from modelmeld.adapters.tensorrt_llm_adapter import TensorRTLLMAdapter

        return TensorRTLLMAdapter(
            endpoint=getattr(settings, "tensorrt_llm_endpoint", None),
        )
    if provider == "anthropic":
        from modelmeld.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(
            api_key=getattr(settings, "anthropic_api_key", None),
            base_url=getattr(settings, "anthropic_base_url", None),
            served_model=getattr(settings, "anthropic_served_model", None),
        )
    if provider == "stub":
        return StubAdapter()
    raise ValueError(f"Unknown adapter provider: {provider}")


def build_router(
    settings: object,
    scout: Scout,
    model_registry: object | None = None,
) -> Router:
    """Construct a Router based on settings.

    `routing_policy="single"` produces SingleAdapterRouter(upstream_provider).
    `routing_policy="capability"` produces CapabilityRouter using the model
    registry. Any other policy produces TieredRouter.
    """
    from modelmeld.config import GatewaySettings

    if not isinstance(settings, GatewaySettings):
        raise TypeError(
            f"build_router expects GatewaySettings, got {type(settings).__name__}"
        )

    if settings.routing_policy == "single":
        adapter = _build_adapter(settings.upstream_provider, settings)
        return SingleAdapterRouter(adapter)

    if settings.routing_policy == "capability":
        return _build_capability_router(settings, model_registry)

    local_adapter = _build_adapter(settings.local_provider, settings)
    cloud_adapter = _build_adapter(settings.cloud_provider, settings)
    policy_map = {
        "scout_driven": RoutingPolicy.SCOUT_DRIVEN,
        "always_local": RoutingPolicy.ALWAYS_LOCAL,
        "always_cloud": RoutingPolicy.ALWAYS_CLOUD,
    }
    return TieredRouter(
        scout=scout,
        adapters={Tier.LOCAL: local_adapter, Tier.CLOUD: cloud_adapter},
        policy=policy_map[settings.routing_policy],
    )


def _build_capability_router(settings: object, model_registry: object | None) -> Router:
    """Build CapabilityRouter from settings + (optional) registry override."""
    from modelmeld.scout.capability import CapabilityScout
    from modelmeld.scout.registry import ModelRegistry, default_registry

    registry: ModelRegistry
    registry = model_registry if isinstance(model_registry, ModelRegistry) else default_registry()

    eligible_list = getattr(settings, "capability_eligible_providers", None)
    eligible_providers = frozenset(eligible_list) if eligible_list else None

    # Build one adapter per provider we can actually serve. We don't read
    # provider names from the registry (which can include providers we have
    # no adapter for); we read them from the eligible_providers config or,
    # if unset, infer from whichever upstream credentials are present.
    if eligible_list:
        providers_to_build = list(eligible_list)
    else:
        providers_to_build = _infer_providers_from_credentials(settings)
        eligible_providers = frozenset(providers_to_build)

    adapters_by_provider: dict[str, ProviderAdapter] = {}
    for provider in providers_to_build:
        adapters_by_provider[provider] = _build_adapter(provider, settings)

    scout = CapabilityScout(
        registry=registry,
        quality_threshold=getattr(settings, "capability_quality_threshold", 0.70),
        eligible_providers=eligible_providers,
        fallback_depth=getattr(settings, "capability_fallback_depth", 5),
    )
    return CapabilityRouter(scout=scout, adapters_by_provider=adapters_by_provider)


def _infer_providers_from_credentials(settings: object) -> list[str]:
    """Pick which provider adapters to instantiate when eligible_providers is unset.

    Default behavior: enable every provider for which we have credentials.
    If nothing is configured, fall back to `stub` so the app can boot
    (useful in dev / tests without keys).
    """
    providers: list[str] = []
    if getattr(settings, "openai_api_key", None):
        providers.append("openai")
    if getattr(settings, "anthropic_api_key", None):
        providers.append("anthropic")
    if getattr(settings, "vllm_endpoint", None):
        providers.append("vllm")
    if getattr(settings, "tensorrt_llm_endpoint", None):
        providers.append("tensorrt_llm")
    return providers or ["stub"]


__all__ = [
    "CapabilityRouter",
    "Router",
    "RouterError",
    "RoutingDecision",
    "RoutingPolicy",
    "SingleAdapterRouter",
    "TieredRouter",
    "build_router",
]

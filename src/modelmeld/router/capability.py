# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""CapabilityRouter — picks an adapter using a CapabilityScout's model choice.

Given the scout's `CapabilityDecision`, the router:
  - Looks up the adapter for `chosen_provider`
  - If that adapter is unhealthy, walks `fallback_model_ids` until it finds
    one whose provider has a healthy adapter
  - Returns a `RoutingDecision` whose `model_id_override` tells the chat
    route to swap `request.model` before delegating to the adapter

Health is cached the same way `TieredRouter` does it — per-provider TTL.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from modelmeld.adapters.base import (
    AdapterError,
    PermanentAdapterError,
    ProviderAdapter,
)
from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.router.base import (
    Router,
    RouterError,
    RoutingDecision,
    RoutingPolicy,
)
from modelmeld.scout.base import Tier
from modelmeld.scout.capability import (
    CapabilityDecision,
    CapabilityScout,
    NoEligibleModelError,
)

if TYPE_CHECKING:
    from modelmeld.api.routing_hints import RoutingHints

logger = logging.getLogger(__name__)


class CapabilityRouter(Router):
    """Capability-based router: pick model first, then find an adapter that serves it."""

    def __init__(
        self,
        scout: CapabilityScout,
        adapters_by_provider: dict[str, ProviderAdapter],
        skip_unhealthy: bool = True,
        health_ttl_sec: float = 30.0,
    ) -> None:
        if not adapters_by_provider:
            raise ValueError("CapabilityRouter needs at least one adapter")
        self.scout = scout
        self.adapters_by_provider = adapters_by_provider
        self.skip_unhealthy = skip_unhealthy
        self.health_ttl = health_ttl_sec
        self._health_cache: dict[str, tuple[bool, float]] = {}

    async def route(
        self,
        request: ChatCompletionRequest,
        hints: "RoutingHints | None" = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision:
        """Route the request to a provider.

        `extra_adapters`: per-request adapter overrides (typically BYOK
        adapters constructed from request headers). These shadow the
        persistent `adapters_by_provider` map for THIS dispatch only —
        the persistent map is never mutated. The route handler is
        responsible for the override adapter's lifecycle (don't put a
        shared client in here).
        """
        # When extra_adapters are provided, derive the BYOK frontier-provider
        # set + pass to the scout so QUALITY/AUTO-escalated policies restrict
        # to providers the customer can actually pay for. Avoids picking
        # gpt-5-mini when the customer only supplied an Anthropic key.
        avail_frontier: frozenset[str] | None = None
        if extra_adapters:
            from modelmeld.api.byok import eligible_providers as byok_eligible
            avail_frontier = frozenset(extra_adapters.keys()) & byok_eligible()

        try:
            decision = await self.scout.choose(
                request, hints=hints,
                available_frontier_providers=avail_frontier,
            )
        except NoEligibleModelError as e:
            raise RouterError(f"capability_scout: {e}") from e

        return await self._resolve_or_fail(
            decision, failover_from=None, extra_adapters=extra_adapters,
        )

    async def route_after_failure(
        self,
        failed: RoutingDecision,
        request: ChatCompletionRequest,  # noqa: ARG002 — signature contract
        error: AdapterError | None = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision | None:
        """Mark the failed provider unhealthy and try the next fallback model.

        F-2: permanent errors (auth failure, model-not-found,
        config mismatch) bubble up - we don't waste a fallback model on a
        problem that's the same across all of them.

        When multi-provider is enabled, also records the
        failure on the per-(model, provider) circuit breaker so the
        picker's NEXT call for the same model excludes the failed
        provider. The breaker will auto-recover after open_duration_sec.
        """
        if isinstance(error, PermanentAdapterError):
            return None

        failed_provider = failed.adapter.name
        self._health_cache[failed_provider] = (False, time.monotonic())

        scout_dec = failed.capability_decision
        if scout_dec is None:
            return None

        try:
            return await self._resolve_or_fail(
                scout_dec, failover_from=failed_provider,
                extra_adapters=extra_adapters,
            )
        except RouterError:
            return None

    async def close(self) -> None:
        seen: set[int] = set()
        for adapter in self.adapters_by_provider.values():
            if id(adapter) not in seen:
                seen.add(id(adapter))
                await adapter.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_adapter(
        self,
        provider: str,
        extra_adapters: dict[str, ProviderAdapter] | None,
    ) -> ProviderAdapter | None:
        """Look up the adapter for a provider. Per-request `extra_adapters`
        (BYOK) shadow the persistent map — letting a customer-supplied
        Anthropic adapter serve a request without ever mutating shared
        state. Returns None if neither map has an adapter for `provider`.
        """
        if extra_adapters is not None and provider in extra_adapters:
            return extra_adapters[provider]
        return self.adapters_by_provider.get(provider)

    async def _resolve_or_fail(
        self,
        decision: CapabilityDecision,
        failover_from: str | None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision:
        """Pick the adapter for `decision`; walk fallbacks if needed.

        `extra_adapters`: per-request BYOK adapter overrides. Used when
        the scout chooses a frontier provider whose key the customer
        supplied via `x-modelmeld-byok-{provider}` headers.
        """
        skipped: list[str] = []
        primary_provider = decision.chosen_provider

        # First, try the chosen model's provider — unless we're explicitly
        # failing over from it.
        if failover_from != primary_provider:
            adapter = self._get_adapter(primary_provider, extra_adapters)
            if adapter is None:
                skipped.append(f"{primary_provider}:not_configured")
            elif not await self._is_healthy(primary_provider, adapter):
                skipped.append(f"{primary_provider}:unhealthy")
            else:
                return self._build_decision(
                    decision, adapter, decision.chosen_model_id, primary_provider,
                    failover_from, skipped,
                )
        else:
            skipped.append(f"{primary_provider}:failover_source")

        # Walk fallbacks in order.
        for fallback_id in decision.fallback_model_ids:
            entry = self.scout.lookup_fallback(fallback_id)
            if entry is None:
                skipped.append(f"{fallback_id}:missing_in_registry")
                continue
            provider = entry.provider
            if provider == failover_from:
                skipped.append(f"{fallback_id}:{provider}:failover_source")
                continue
            adapter = self._get_adapter(provider, extra_adapters)
            if adapter is None:
                skipped.append(f"{fallback_id}:{provider}:not_configured")
                continue
            if not await self._is_healthy(provider, adapter):
                skipped.append(f"{fallback_id}:{provider}:unhealthy")
                continue
            return self._build_decision(
                decision, adapter, fallback_id, provider, failover_from, skipped,
                task_score=entry.task_scores.get(decision.task_category, 0.0),
            )

        raise RouterError(
            f"No healthy adapter available for capability route "
            f"(task={decision.task_category}, "
            f"primary={primary_provider}, skipped={skipped})"
        )

    def _build_decision(
        self,
        cap_decision: CapabilityDecision,
        adapter: ProviderAdapter,
        model_id: str,
        provider: str,
        failover_from: str | None,
        skipped: list[str],
        task_score: float | None = None,
    ) -> RoutingDecision:
        if model_id != cap_decision.chosen_model_id:
            # We took a fallback — update the decision to reflect what we used.
            score = task_score if task_score is not None else cap_decision.task_score
            cap_decision = cap_decision.with_model(model_id, provider, score)

        suffix = f";skipped={','.join(skipped)}" if skipped else ""
        rationale = f"capability;{cap_decision.rationale}{suffix}"

        # Capability routing doesn't fit LOCAL/CLOUD cleanly; vLLM = LOCAL, rest = CLOUD.
        tier = Tier.LOCAL if provider == "vllm" else Tier.CLOUD

        return RoutingDecision(
            tier=tier,
            adapter=adapter,
            scout_decision=None,                # ScoutDecision is the tier-shape; capability uses its own
            policy_applied=RoutingPolicy.CAPABILITY,
            rationale=rationale,
            model_id_override=model_id,
            capability_decision=cap_decision,
        )

    async def _is_healthy(self, provider: str, adapter: ProviderAdapter) -> bool:
        if not self.skip_unhealthy:
            return True
        now = time.monotonic()
        cached = self._health_cache.get(provider)
        if cached is not None:
            healthy, last = cached
            if now - last < self.health_ttl:
                return healthy
        healthy = await adapter.health()
        self._health_cache[provider] = (healthy, now)
        return healthy

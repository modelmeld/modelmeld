# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Routing decision engine.

`TieredRouter` is the main implementation: it consults the Scout (or a fixed
policy) to pick a tier, resolves the tier to an adapter, and falls through to
the other tier if the preferred one is unhealthy. Health is cached with a TTL
so we don't hit upstream on every request.

`SingleAdapterRouter` is the trivial implementation for non-routing setups.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelmeld.adapters.base import (
    AdapterError,
    PermanentAdapterError,
    ProviderAdapter,
)
from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.base import Scout, ScoutDecision, Tier

if TYPE_CHECKING:
    from modelmeld.api.routing_hints import RoutingHints


class RoutingPolicy(str, Enum):
    """How the router picks a tier (or a specific model, for CAPABILITY)."""

    SINGLE = "single"               # no routing; SingleAdapterRouter uses this
    SCOUT_DRIVEN = "scout_driven"   # follow Scout's recommendation (tier-based)
    ALWAYS_LOCAL = "always_local"   # force LOCAL tier
    ALWAYS_CLOUD = "always_cloud"   # force CLOUD tier
    CAPABILITY = "capability"       # registry-driven model pick

    def __str__(self) -> str:
        return self.value


class RouterError(Exception):
    """No adapter could serve the request (none configured / all unhealthy)."""


@dataclass(frozen=True)
class RoutingDecision:
    """Output of `Router.route()`.

    `tier`                 — final tier chosen.
    `adapter`              — the adapter that will service this request.
    `scout_decision`       — what tier-Scout said (None when bypassed).
    `policy_applied`       — the policy that produced this decision.
    `rationale`            — human-readable trace for logs / debugging.
    `model_id_override`    — for CAPABILITY routing: the canonical model_id, used
                             for response attribution + registry lookup.
    `provider_model_id`    — for CAPABILITY routing: the provider's own slug to put
                             on the egress wire (e.g. "qwen/qwen3-coder"). Distinct
                             from `model_id_override`; empty/None → send the
                             canonical id verbatim (back-compat for rows with none).
    `capability_decision`  — for CAPABILITY routing: full CapabilityDecision.
    """

    tier: Tier
    adapter: ProviderAdapter
    scout_decision: ScoutDecision | None
    policy_applied: RoutingPolicy
    rationale: str
    model_id_override: str | None = None
    provider_model_id: str | None = None
    capability_decision: object | None = None
    # ^ typed as object to avoid an import cycle between router/base + scout/capability.


class Router(ABC):
    """Choose an adapter for an incoming request."""

    @abstractmethod
    async def route(
        self,
        request: ChatCompletionRequest,
        hints: RoutingHints | None = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision:
        """Return the routing decision for this request.

        `hints` carries framework-supplied overrides (task_category / agent_role
        / quality_threshold / excluded_providers). Routers that don't support
        capability routing ignore it.

        `extra_adapters` carries per-request adapter overrides (BYOK pattern —
        see modelmeld.api.byok). Routers that don't support multi-provider
        dispatch ignore it.
        """

    async def route_after_failure(
        self,
        failed: RoutingDecision,
        request: ChatCompletionRequest,
        error: AdapterError | None = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision | None:
        """Return a fallback decision after the primary adapter failed.

        `error`, when supplied, is the exception the primary adapter raised.
        Implementations may inspect it (in particular, check whether it is a
        `PermanentAdapterError`) to decide whether failover is appropriate -
        permanent errors should surface to the caller, not be papered over by
        falling back to a different tier (F-2).

        Default implementation gives up (returns None). Tiered routers override
        this to mark the failed tier unhealthy and return the other tier.
        """
        return None

    async def close(self) -> None:
        """Release any held resources (typically adapter clients). Default no-op."""


class SingleAdapterRouter(Router):
    """Send every request to a single configured adapter."""

    def __init__(self, adapter: ProviderAdapter) -> None:
        self.adapter = adapter

    async def route(
        self,
        request: ChatCompletionRequest,
        hints: RoutingHints | None = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision:
        return RoutingDecision(
            tier=Tier.CLOUD,  # nominal — single-adapter mode has no real tier
            adapter=self.adapter,
            scout_decision=None,
            policy_applied=RoutingPolicy.SINGLE,
            rationale=f"single_adapter:{self.adapter.name}",
        )

    async def close(self) -> None:
        await self.adapter.close()


@dataclass
class _HealthState:
    """Per-tier health bookkeeping for TieredRouter (F-4).

    Tracks consecutive failures so a single transient blip doesn't
    blacklist the tier for `health_ttl` seconds. The tier is only marked
    unhealthy after `unhealthy_threshold` consecutive failures.
    """

    healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0


class TieredRouter(Router):
    """LOCAL/CLOUD-tier router with policy and health-aware fallback."""

    def __init__(
        self,
        scout: Scout,
        adapters: dict[Tier, ProviderAdapter],
        policy: RoutingPolicy = RoutingPolicy.SCOUT_DRIVEN,
        skip_unhealthy: bool = True,
        health_ttl_sec: float = 30.0,
        unhealthy_threshold: int = 3,
    ) -> None:
        if policy == RoutingPolicy.SINGLE:
            raise ValueError("TieredRouter does not support RoutingPolicy.SINGLE")
        if unhealthy_threshold < 1:
            raise ValueError(
                f"unhealthy_threshold must be >= 1, got {unhealthy_threshold}",
            )
        self.scout = scout
        self.adapters = adapters
        self.policy = policy
        self.skip_unhealthy = skip_unhealthy
        self.health_ttl = health_ttl_sec
        self.unhealthy_threshold = unhealthy_threshold
        self._health_cache: dict[Tier, _HealthState] = {}

    async def _is_healthy(self, tier: Tier) -> bool:
        """Active reachability probe via `adapter.health()`.

        A probe failure (adapter.health() → False) is a deliberate
        reachability signal and marks the tier unhealthy immediately —
        the F-4 N-consecutive-failures threshold applies to *real call*
        failures via `route_after_failure`, not to active probes.
        """
        adapter = self.adapters.get(tier)
        if adapter is None:
            return False
        if not self.skip_unhealthy:
            return True
        now = time.monotonic()
        state = self._health_cache.get(tier)
        if state is not None and now - state.last_check < self.health_ttl:
            return state.healthy
        # TTL expired (or never probed) — refresh from the adapter.
        healthy_now = await adapter.health()
        if state is None:
            state = _HealthState()
            self._health_cache[tier] = state
        state.last_check = now
        state.healthy = healthy_now
        if healthy_now:
            # Success resets the failure counter — natural recovery path.
            state.consecutive_failures = 0
        return state.healthy

    def record_success(self, tier: Tier) -> None:
        """Notify the router that a real call to this tier just succeeded.

        Resets the consecutive-failure counter — bookkeeping the active
        health-probe path does on its own, but the chat route can also
        call this on every successful adapter response to keep the cache
        fresh between probes.
        """
        state = self._health_cache.get(tier)
        if state is None:
            return
        state.healthy = True
        state.consecutive_failures = 0
        state.last_check = time.monotonic()

    def _other(self, tier: Tier) -> Tier:
        return Tier.CLOUD if tier == Tier.LOCAL else Tier.LOCAL

    async def route(
        self,
        request: ChatCompletionRequest,
        hints: RoutingHints | None = None,
        extra_adapters: dict[str, ProviderAdapter] | None = None,
    ) -> RoutingDecision:
        scout_decision: ScoutDecision | None = None

        if self.policy == RoutingPolicy.ALWAYS_LOCAL:
            preferred = Tier.LOCAL
        elif self.policy == RoutingPolicy.ALWAYS_CLOUD:
            preferred = Tier.CLOUD
        else:  # SCOUT_DRIVEN
            scout_decision = await self.scout.classify(request)
            preferred = scout_decision.tier

        order = [preferred, self._other(preferred)]
        skipped: list[str] = []

        for tier in order:
            adapter = self.adapters.get(tier)
            if adapter is None:
                skipped.append(f"{tier}:not_configured")
                continue
            if not await self._is_healthy(tier):
                skipped.append(f"{tier}:unhealthy")
                continue

            if tier == preferred:
                rationale = f"policy={self.policy};tier={tier}"
            else:
                rationale = (
                    f"policy={self.policy};preferred={preferred};"
                    f"fallback={tier};skipped={','.join(skipped)}"
                )
            return RoutingDecision(
                tier=tier,
                adapter=adapter,
                scout_decision=scout_decision,
                policy_applied=self.policy,
                rationale=rationale,
            )

        raise RouterError(
            f"No healthy adapter available "
            f"(policy={self.policy}, preferred={preferred}, skipped={skipped})"
        )

    async def route_after_failure(
        self,
        failed: RoutingDecision,
        request: ChatCompletionRequest,
        error: AdapterError | None = None,
    ) -> RoutingDecision | None:
        # F-2: permanent errors (auth failure, model not found, config
        # mismatch) must bubble up to the caller, NOT silently fall over to
        # the other tier. A misconfigured cloud adapter should not paper
        # itself over with a local stub response.
        if isinstance(error, PermanentAdapterError):
            return None

        # F-4: increment the consecutive-failure counter rather than
        # immediately blacklisting. A single transient blip shouldn't lock
        # out the tier for `health_ttl` seconds and cascade-fail subsequent
        # traffic. Tier is only marked unhealthy once the counter crosses
        # `unhealthy_threshold` consecutive failures.
        now = time.monotonic()
        state = self._health_cache.get(failed.tier)
        if state is None:
            state = _HealthState()
            self._health_cache[failed.tier] = state
        state.consecutive_failures += 1
        state.last_check = now
        if state.consecutive_failures >= self.unhealthy_threshold:
            state.healthy = False

        other = self._other(failed.tier)
        adapter = self.adapters.get(other)
        if adapter is None or not await self._is_healthy(other):
            return None

        # F-8: don't fail over if the fallback adapter can't serve the
        # requested model. Otherwise a CLOUD→LOCAL failover with
        # model="claude-haiku-..." would hit vLLM and 404 — strictly worse
        # than surfacing the original error to the caller. `serves_model`
        # defaults to True (pass-through) so this only blocks failover when
        # an adapter has been explicitly pinned via `served_model`.
        if not adapter.serves_model(request.model):
            return None

        return RoutingDecision(
            tier=other,
            adapter=adapter,
            scout_decision=failed.scout_decision,
            policy_applied=failed.policy_applied,
            rationale=(
                f"failover;from={failed.tier};to={other};"
                f"policy={self.policy};original={failed.rationale}"
            ),
        )

    async def close(self) -> None:
        seen: set[int] = set()
        for adapter in self.adapters.values():
            if id(adapter) not in seen:
                seen.add(id(adapter))
                await adapter.close()

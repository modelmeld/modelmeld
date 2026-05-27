"""TieredRouter — table-driven over (scout × health × policy)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
)
from modelmeld.router import (
    RouterError,
    RoutingPolicy,
    TieredRouter,
)
from modelmeld.scout.base import Scout, ScoutDecision, Tier

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeAdapter(ProviderAdapter):
    name_str: str = "fake"
    is_healthy: bool = True
    health_calls: int = 0
    chat_calls: list[ChatCompletionRequest] = field(default_factory=list)

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.name_str

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.chat_calls.append(request)
        return ChatCompletion(
            model=request.model,
            choices=[Choice(index=0, message=ResponseMessage(content="ok"), finish_reason="stop")],
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        self.health_calls += 1
        return self.is_healthy


class FixedScout(Scout):
    """Returns a preset tier with confidence 1.0."""

    name = "fixed"

    def __init__(self, tier: Tier) -> None:
        self.tier = tier

    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        return ScoutDecision(tier=self.tier, confidence=1.0, rationale=f"fixed:{self.tier}")


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rejects_single_policy() -> None:
    with pytest.raises(ValueError):
        TieredRouter(
            scout=FixedScout(Tier.LOCAL),
            adapters={Tier.LOCAL: FakeAdapter(), Tier.CLOUD: FakeAdapter()},
            policy=RoutingPolicy.SINGLE,
        )


# ---------------------------------------------------------------------------
# The decision matrix
# ---------------------------------------------------------------------------

# (policy, scout_tier, local_healthy, cloud_healthy, expected_tier_or_None)
# None means RouterError expected.
MATRIX: list[tuple[RoutingPolicy, Tier, bool, bool, Tier | None]] = [
    # SCOUT_DRIVEN — follow scout, fall through if unhealthy
    (RoutingPolicy.SCOUT_DRIVEN, Tier.LOCAL, True,  True,  Tier.LOCAL),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.LOCAL, False, True,  Tier.CLOUD),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.LOCAL, True,  False, Tier.LOCAL),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.LOCAL, False, False, None),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.CLOUD, True,  True,  Tier.CLOUD),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.CLOUD, False, True,  Tier.CLOUD),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.CLOUD, True,  False, Tier.LOCAL),
    (RoutingPolicy.SCOUT_DRIVEN, Tier.CLOUD, False, False, None),
    # ALWAYS_LOCAL — scout result irrelevant; LOCAL preferred; CLOUD is fallback
    (RoutingPolicy.ALWAYS_LOCAL, Tier.LOCAL, True,  True,  Tier.LOCAL),
    (RoutingPolicy.ALWAYS_LOCAL, Tier.LOCAL, False, True,  Tier.CLOUD),
    (RoutingPolicy.ALWAYS_LOCAL, Tier.LOCAL, True,  False, Tier.LOCAL),
    (RoutingPolicy.ALWAYS_LOCAL, Tier.LOCAL, False, False, None),
    # ALWAYS_CLOUD — CLOUD preferred; LOCAL is fallback
    (RoutingPolicy.ALWAYS_CLOUD, Tier.LOCAL, True,  True,  Tier.CLOUD),
    (RoutingPolicy.ALWAYS_CLOUD, Tier.LOCAL, False, True,  Tier.CLOUD),
    (RoutingPolicy.ALWAYS_CLOUD, Tier.LOCAL, True,  False, Tier.LOCAL),
    (RoutingPolicy.ALWAYS_CLOUD, Tier.LOCAL, False, False, None),
]


@pytest.mark.parametrize(
    ("policy", "scout_tier", "local_h", "cloud_h", "expected"),
    MATRIX,
    ids=[f"{p.value}|scout={s}|L={l}|C={c}" for p, s, l, c, _ in MATRIX],
)
async def test_decision_matrix(
    policy: RoutingPolicy,
    scout_tier: Tier,
    local_h: bool,
    cloud_h: bool,
    expected: Tier | None,
) -> None:
    local = FakeAdapter(name_str="local", is_healthy=local_h)
    cloud = FakeAdapter(name_str="cloud", is_healthy=cloud_h)
    router = TieredRouter(
        scout=FixedScout(scout_tier),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=policy,
        health_ttl_sec=0.0,  # never cache — fresh check every call
    )

    if expected is None:
        with pytest.raises(RouterError):
            await router.route(_req())
        return

    decision = await router.route(_req())
    assert decision.tier == expected
    assert decision.policy_applied == policy
    if policy == RoutingPolicy.SCOUT_DRIVEN:
        assert decision.scout_decision is not None
        assert decision.scout_decision.tier == scout_tier
    else:
        assert decision.scout_decision is None


# ---------------------------------------------------------------------------
# Health caching
# ---------------------------------------------------------------------------


async def test_health_check_cached_within_ttl() -> None:
    local = FakeAdapter(name_str="local", is_healthy=True)
    cloud = FakeAdapter(name_str="cloud", is_healthy=True)
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.SCOUT_DRIVEN,
        health_ttl_sec=60.0,
    )
    for _ in range(5):
        await router.route(_req())
    # All 5 requests should reuse the first health-check result
    assert local.health_calls == 1


async def test_skip_unhealthy_disabled_skips_health_check() -> None:
    local = FakeAdapter(name_str="local", is_healthy=False)
    cloud = FakeAdapter(name_str="cloud", is_healthy=False)
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.SCOUT_DRIVEN,
        skip_unhealthy=False,
    )
    # Even though both adapters report unhealthy, with skip_unhealthy=False the
    # router doesn't ask and goes to the scout's preferred tier.
    decision = await router.route(_req())
    assert decision.tier == Tier.LOCAL
    assert local.health_calls == 0


async def test_close_closes_each_adapter_once() -> None:
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
    )
    closed: list[str] = []
    orig_local_close = local.close
    orig_cloud_close = cloud.close

    async def patch_close(name: str, orig):
        async def _c():
            closed.append(name)
            await orig()
        return _c

    local.close = await patch_close("local", orig_local_close)  # type: ignore[assignment,method-assign]
    cloud.close = await patch_close("cloud", orig_cloud_close)  # type: ignore[assignment,method-assign]

    await router.close()
    assert sorted(closed) == ["cloud", "local"]


async def test_close_dedupes_same_adapter() -> None:
    """If both tiers share the same adapter instance, close it only once."""
    shared = FakeAdapter(name_str="shared")
    close_count = 0

    orig = shared.close

    async def counting_close() -> None:
        nonlocal close_count
        close_count += 1
        await orig()

    shared.close = counting_close  # type: ignore[method-assign]

    router = TieredRouter(
        scout=FixedScout(Tier.LOCAL),
        adapters={Tier.LOCAL: shared, Tier.CLOUD: shared},
    )
    await router.close()
    assert close_count == 1


# ---------------------------------------------------------------------------
# F-2: Transient vs permanent error classification in route_after_failure
# ---------------------------------------------------------------------------

from modelmeld.adapters.base import (
    AdapterError,
    PermanentAdapterError,
    TransientAdapterError,
)


async def test_route_after_failure_no_failover_on_permanent_error() -> None:
    """F-2: a PermanentAdapterError (auth fail, model not found, etc.)
    should NOT trigger failover. The original error must bubble up so the
    caller sees the real cause, not a misleading stub response from the
    other tier."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    perm_err = PermanentAdapterError("model not found: claude-3-5-sonnet-latest")
    fallback = await router.route_after_failure(decision, request, error=perm_err)
    assert fallback is None, "permanent errors must NOT trigger failover"


async def test_route_after_failure_failover_on_transient_error() -> None:
    """F-2: TransientAdapterError should still trigger failover (the
    status-quo behavior we want to keep)."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    trans_err = TransientAdapterError("anthropic 529 overloaded")
    fallback = await router.route_after_failure(decision, request, error=trans_err)
    assert fallback is not None
    assert fallback.tier == Tier.LOCAL  # failed over to the other tier


async def test_route_after_failure_failover_when_error_omitted() -> None:
    """Backward compat: when callers don't pass `error`, behave as before
    (assume failover-safe). Preserves existing test/integration patterns."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    fallback = await router.route_after_failure(decision, request)
    assert fallback is not None
    assert fallback.tier == Tier.LOCAL


async def test_route_after_failure_failover_on_base_adapter_error() -> None:
    """A bare AdapterError (not Transient/Permanent) should also trigger
    failover - safest default when classification isn't precise."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    base_err = AdapterError("unclassified failure")
    fallback = await router.route_after_failure(decision, request, error=base_err)
    assert fallback is not None
    assert fallback.tier == Tier.LOCAL


# ---------------------------------------------------------------------------
# F-4: N-consecutive-failures threshold + counter reset on success
# ---------------------------------------------------------------------------


async def test_single_failure_does_not_blacklist_tier() -> None:
    """F-4: one transient failure must NOT mark the tier unhealthy.
    Previously this cascaded into 30s of fallthrough for unrelated
    requests. Now we require 3 consecutive failures by default."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
        unhealthy_threshold=3,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    await router.route_after_failure(
        decision, request, error=TransientAdapterError("blip"),
    )
    state = router._health_cache[Tier.CLOUD]
    assert state.consecutive_failures == 1
    assert state.healthy is True  # NOT blacklisted after one failure


async def test_threshold_failures_marks_tier_unhealthy() -> None:
    """F-4: cross the threshold → tier marked unhealthy."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
        unhealthy_threshold=3,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    for _ in range(3):
        await router.route_after_failure(
            decision, request, error=TransientAdapterError("blip"),
        )
    state = router._health_cache[Tier.CLOUD]
    assert state.consecutive_failures == 3
    assert state.healthy is False  # NOW unhealthy


async def test_record_success_resets_consecutive_failures() -> None:
    """F-4: a successful real call should reset the failure counter."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
        unhealthy_threshold=3,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    # Two failures (still under threshold)
    for _ in range(2):
        await router.route_after_failure(
            decision, request, error=TransientAdapterError("blip"),
        )
    assert router._health_cache[Tier.CLOUD].consecutive_failures == 2

    # A success resets
    router.record_success(Tier.CLOUD)
    assert router._health_cache[Tier.CLOUD].consecutive_failures == 0
    assert router._health_cache[Tier.CLOUD].healthy is True


async def test_health_probe_success_resets_counter() -> None:
    """F-4: when the TTL expires and adapter.health() returns True, the
    counter resets — natural recovery without needing explicit
    record_success."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud", is_healthy=True)
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
        health_ttl_sec=0.0,  # always re-probe
        unhealthy_threshold=3,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    # Accrue some failures
    for _ in range(2):
        await router.route_after_failure(
            decision, request, error=TransientAdapterError("blip"),
        )
    # Cloud reports healthy on next probe (TTL=0 always re-probes)
    assert await router._is_healthy(Tier.CLOUD) is True
    state = router._health_cache[Tier.CLOUD]
    assert state.consecutive_failures == 0


async def test_unhealthy_threshold_validation() -> None:
    """unhealthy_threshold must be >= 1 (zero would blacklist on
    construction)."""
    with pytest.raises(ValueError, match="unhealthy_threshold"):
        TieredRouter(
            scout=FixedScout(Tier.LOCAL),
            adapters={Tier.LOCAL: FakeAdapter(), Tier.CLOUD: FakeAdapter()},
            unhealthy_threshold=0,
        )


async def test_threshold_of_one_preserves_old_aggressive_behavior() -> None:
    """Operators who WANT the old single-failure-blacklists behavior can
    pass unhealthy_threshold=1."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
        unhealthy_threshold=1,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    await router.route_after_failure(
        decision, request, error=TransientAdapterError("blip"),
    )
    assert router._health_cache[Tier.CLOUD].healthy is False


# ---------------------------------------------------------------------------
# F-8: failover skipped when fallback adapter can't serve the model
# ---------------------------------------------------------------------------


async def test_failover_proceeds_when_served_model_substitution_applies() -> None:
    """F-8: when LOCAL has `served_model="Qwen/..."` pinned, failover from
    CLOUD with `model="claude-haiku-..."` STILL proceeds — the adapter
    will substitute the model on the way out and serve the request.
    The whole point of `served_model` is opting into substitution."""
    local = FakeAdapter(name_str="local")
    local.served_model = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"  # pinned
    cloud = FakeAdapter(name_str="cloud")  # served_model=None → pass-through
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(
        model="claude-haiku-4-5-20251001", messages=[],
    )
    decision = await router.route(request)
    fallback = await router.route_after_failure(
        decision, request, error=TransientAdapterError("anthropic 529"),
    )
    assert fallback is not None, (
        "failover must proceed — local will substitute the model and serve"
    )
    assert fallback.tier == Tier.LOCAL


async def test_failover_blocked_when_subclass_implements_strict_mode() -> None:
    """F-8 extensibility: a subclass that overrides serves_model for strict
    matching DOES block failover for non-matching models. This is the
    extensibility hook for compliance-mode adapters."""
    class _StrictFakeAdapter(FakeAdapter):
        def serves_model(self, model_id: str) -> bool:
            if self.served_model is None:
                return True
            return model_id == self.served_model

    local = _StrictFakeAdapter(name_str="local")
    local.served_model = "qwen-strict"
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="claude-anything", messages=[])
    decision = await router.route(request)
    fallback = await router.route_after_failure(
        decision, request, error=TransientAdapterError("anthropic 529"),
    )
    assert fallback is None, "strict-mode subclass must block failover"


async def test_failover_still_happens_when_fallback_can_serve_model() -> None:
    """F-8 happy path: failover proceeds when the fallback IS compatible."""
    local = FakeAdapter(name_str="local")
    local.served_model = "qwen"
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="qwen", messages=[])
    decision = await router.route(request)
    fallback = await router.route_after_failure(
        decision, request, error=TransientAdapterError("anthropic 529"),
    )
    assert fallback is not None
    assert fallback.tier == Tier.LOCAL


async def test_failover_passthrough_default_still_works() -> None:
    """Backward compat: both adapters have served_model=None → failover
    works as before (no compatibility check kicks in)."""
    local = FakeAdapter(name_str="local")  # served_model=None
    cloud = FakeAdapter(name_str="cloud")  # served_model=None
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="anything", messages=[])
    decision = await router.route(request)
    fallback = await router.route_after_failure(
        decision, request, error=TransientAdapterError("blip"),
    )
    assert fallback is not None
    assert fallback.tier == Tier.LOCAL


async def test_permanent_error_does_not_blacklist_failed_tier() -> None:
    """F-2: when failover is suppressed, we should NOT mark the failed
    tier unhealthy either - the issue isn't with the tier itself, it's
    with the specific request (e.g. wrong model name). Marking unhealthy
    would punish unrelated future requests."""
    local = FakeAdapter(name_str="local")
    cloud = FakeAdapter(name_str="cloud")
    router = TieredRouter(
        scout=FixedScout(Tier.CLOUD),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.ALWAYS_CLOUD,
    )
    request = ChatCompletionRequest(model="m", messages=[])
    decision = await router.route(request)

    await router.route_after_failure(
        decision, request,
        error=PermanentAdapterError("401 unauthorized"),
    )
    # Health cache should NOT have an unhealthy state for the cloud tier
    state = router._health_cache.get(Tier.CLOUD)
    if state is not None:
        assert state.healthy is True, "permanent error should not mark tier unhealthy"
        assert state.consecutive_failures == 0, \
            "permanent error should not bump the consecutive-failures counter"

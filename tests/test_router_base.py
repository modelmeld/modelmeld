"""Router ABC, RoutingDecision shape, SingleAdapterRouter."""

from __future__ import annotations

import pytest

from modelmeld.adapters import StubAdapter
from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.router import (
    Router,
    RoutingDecision,
    RoutingPolicy,
    SingleAdapterRouter,
)
from modelmeld.scout.base import Tier


def test_cannot_instantiate_abstract_router() -> None:
    with pytest.raises(TypeError):
        Router()  # type: ignore[abstract]


def test_routing_policy_is_string_enum() -> None:
    assert RoutingPolicy.SCOUT_DRIVEN == "scout_driven"
    assert str(RoutingPolicy.ALWAYS_CLOUD) == "always_cloud"


def test_routing_decision_is_frozen() -> None:
    decision = RoutingDecision(
        tier=Tier.LOCAL,
        adapter=StubAdapter(),
        scout_decision=None,
        policy_applied=RoutingPolicy.SCOUT_DRIVEN,
        rationale="test",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        decision.tier = Tier.CLOUD  # type: ignore[misc]


async def test_single_adapter_router_returns_its_adapter() -> None:
    stub = StubAdapter()
    r = SingleAdapterRouter(stub)
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    decision = await r.route(request)
    assert decision.adapter is stub
    assert decision.policy_applied == RoutingPolicy.SINGLE
    assert decision.scout_decision is None
    assert decision.rationale.startswith("single_adapter:")


async def test_single_adapter_router_close_closes_adapter() -> None:
    class CloseSpy(StubAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    spy = CloseSpy()
    r = SingleAdapterRouter(spy)
    await r.close()
    assert spy.closed is True

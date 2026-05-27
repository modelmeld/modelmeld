"""Scout ABC and decision-shape contract tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.base import Scout, ScoutDecision, Tier


def test_cannot_instantiate_abstract_scout() -> None:
    with pytest.raises(TypeError):
        Scout()  # type: ignore[abstract]


def test_scout_decision_is_frozen() -> None:
    decision = ScoutDecision(tier=Tier.LOCAL, confidence=0.8, rationale="short_prompt")
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        decision.confidence = 0.9  # type: ignore[misc]


def test_scout_decision_signals_default_empty_dict() -> None:
    decision = ScoutDecision(tier=Tier.CLOUD, confidence=0.6, rationale="long_prompt")
    assert isinstance(decision.signals, Mapping)
    assert decision.signals == {}


def test_tier_string_enum() -> None:
    # Tier should serialize as plain strings for logging / JSON.
    assert Tier.LOCAL == "local"
    assert Tier.CLOUD == "cloud"
    assert str(Tier.LOCAL) == "local"


async def test_concrete_implementation_satisfies_contract() -> None:
    class Always(Scout):
        name = "always_local"

        async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
            return ScoutDecision(tier=Tier.LOCAL, confidence=1.0, rationale="forced")

    scout = Always()
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    decision = await scout.classify(request)
    assert decision.tier == Tier.LOCAL
    assert decision.confidence == 1.0

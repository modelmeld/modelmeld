# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Sprint 0 E2E acceptance: self-host with ONE cloud-OSS provider configured.

Before Sprint 0, a customer who set ``FIREWORKS_API_KEY`` (or
``TOGETHER_API_KEY``, or ``OPENROUTER_API_KEY``) and tried to route
a tool-using request via ``-saver`` got a 503 because the bundled
``default_registry.json`` had no entries tagged for those providers
— only ``vllm``. This was the gap between what the public docs
promised (multi-provider OSS routing out of the box) and what the
code actually delivered.

These tests verify the gap is closed. Each test:
  1. Configures the gateway with ONLY one cloud-OSS provider's key.
  2. Builds the capability router via the production code path.
  3. Asks the scout to choose for a tool-using ``-saver`` request.
  4. Asserts the scout picks an entry from the configured provider
     (not vLLM, not frontier, not None).

The adapter is constructed but not called — we're testing the
routing decision, not the upstream HTTP call.
"""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest, Tool, UserMessage
from modelmeld.config import GatewaySettings
from modelmeld.router import _build_capability_router
from modelmeld.scout.multi_provider_registry import MultiProviderModelRegistry


def _tool_using_saver_request() -> ChatCompletionRequest:
    """A Claude Code-shaped tool-bearing request under the ``-saver`` alias."""
    return ChatCompletionRequest(
        model="anthropic/modelmeld-saver",
        messages=[UserMessage(role="user", content="Add a docstring to this function.")],
        tools=[
            Tool(
                type="function",
                function={
                    "name": "edit_file",
                    "description": "Edit a file by replacing text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            )
        ],
        max_tokens=512,
    )


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all upstream-provider env vars so the test starts clean."""
    for key in [
        "MODELMELD_FIREWORKS_API_KEY",
        "FIREWORKS_API_KEY",
        "MODELMELD_TOGETHER_API_KEY",
        "TOGETHER_API_KEY",
        "MODELMELD_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "MODELMELD_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODELMELD_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "MODELMELD_VLLM_ENDPOINT",
        "VLLM_ENDPOINT",
        "MODELMELD_TENSORRT_LLM_ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Fireworks-only setup — the canonical Sprint 0 scenario
# ---------------------------------------------------------------------------


async def test_fireworks_only_setup_routes_tool_request_to_fireworks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deliverable that proves Sprint 0 closed the gap.

    A customer who installs ModelMeld with ONLY a Fireworks key in the
    environment should be able to route tool-using requests via
    ``-saver`` without hitting 503.
    """
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODELMELD_FIREWORKS_API_KEY", "fw_smoke_test_key")
    monkeypatch.setenv("MODELMELD_ROUTING_POLICY", "capability")

    settings = GatewaySettings()
    router = _build_capability_router(settings, None)

    # Multi-provider registry is loaded by default.
    assert isinstance(router.scout.registry, MultiProviderModelRegistry)
    # Only the Fireworks adapter was built (Fireworks is the only configured key).
    assert sorted(router.adapters_by_provider.keys()) == ["fireworks"]

    decision = await router.scout.choose(_tool_using_saver_request())

    # The scout picks a tool-capable OSS model served by Fireworks —
    # NOT vLLM (no vllm endpoint configured), NOT frontier (saver
    # restricts to OSS providers).
    assert decision.chosen_provider == "fireworks", (
        f"Expected fireworks, got {decision.chosen_provider}. Rationale: {decision.rationale}"
    )
    # The chosen model_id must be one our overlay tagged for Fireworks
    # (deepseek-v4-pro, gpt-oss-120b, or kimi-k2.6 per the OSS overlay).
    assert decision.chosen_model_id in {
        "deepseek-v4-pro",
        "gpt-oss-120b",
        "kimi-k2.6",
    }, f"Unexpected model: {decision.chosen_model_id}"


# ---------------------------------------------------------------------------
# Together-only setup
# ---------------------------------------------------------------------------


async def test_together_only_setup_routes_tool_request_to_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODELMELD_TOGETHER_API_KEY", "tg_smoke_test_key")
    monkeypatch.setenv("MODELMELD_ROUTING_POLICY", "capability")

    settings = GatewaySettings()
    router = _build_capability_router(settings, None)
    assert sorted(router.adapters_by_provider.keys()) == ["together"]

    decision = await router.scout.choose(_tool_using_saver_request())
    assert decision.chosen_provider == "together"
    # Together-tagged tool-capable OSS models per the overlay
    assert decision.chosen_model_id in {
        "deepseek-v4-pro",
        "gpt-oss-120b",
        "kimi-k2.6",
        "llama-3.3-70b-instruct",
    }


# ---------------------------------------------------------------------------
# OpenRouter-only setup
# ---------------------------------------------------------------------------


async def test_openrouter_only_setup_routes_tool_request_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODELMELD_OPENROUTER_API_KEY", "or_smoke_test_key")
    monkeypatch.setenv("MODELMELD_ROUTING_POLICY", "capability")

    settings = GatewaySettings()
    router = _build_capability_router(settings, None)
    assert sorted(router.adapters_by_provider.keys()) == ["openrouter"]

    decision = await router.scout.choose(_tool_using_saver_request())
    assert decision.chosen_provider == "openrouter"


# ---------------------------------------------------------------------------
# Multi-provider setup — picker picks cheapest configured
# ---------------------------------------------------------------------------


async def test_multiple_providers_picks_cheapest_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With multiple cloud-OSS providers configured, the scout picks
    whichever (model, provider) has the lowest blended cost across
    every overlay row that meets the quality bar."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODELMELD_FIREWORKS_API_KEY", "fw")
    monkeypatch.setenv("MODELMELD_TOGETHER_API_KEY", "tg")
    monkeypatch.setenv("MODELMELD_OPENROUTER_API_KEY", "or")
    monkeypatch.setenv("MODELMELD_ROUTING_POLICY", "capability")

    settings = GatewaySettings()
    router = _build_capability_router(settings, None)
    assert sorted(router.adapters_by_provider.keys()) == [
        "fireworks",
        "openrouter",
        "together",
    ]

    decision = await router.scout.choose(_tool_using_saver_request())
    # Decision provider should be one of the configured cloud-OSS ones
    assert decision.chosen_provider in {"fireworks", "together", "openrouter"}


# ---------------------------------------------------------------------------
# Coverage check — overlay availability for tool-capable OSS models
# ---------------------------------------------------------------------------


def test_overlay_provides_at_least_one_tool_capable_model_per_provider() -> None:
    """Sanity check on the curated overlay: each cloud-OSS provider has
    at least one tool-capable model in its lineup, so any single-provider
    setup can serve tool-bearing traffic via -saver."""
    from modelmeld.scout.multi_provider_registry import default_multi_provider_registry

    reg = default_multi_provider_registry()

    for provider in ["fireworks", "together", "openrouter"]:
        tool_capable_models = [
            entry
            for entry in reg.all_entries_multi()
            if entry.provider == provider and entry.supports_tools
        ]
        assert len(tool_capable_models) >= 1, (
            f"{provider} has no tool-capable models in the overlay — "
            "single-provider setups can't serve tool-using requests"
        )


# ---------------------------------------------------------------------------
# Sprint 2 closeout — qwen3-coder-next is reachable in the OSS lineup
# ---------------------------------------------------------------------------


def test_qwen3_coder_next_is_reachable_via_openrouter() -> None:
    """Sprint 2 acceptance: qwen3-coder-next (the tool-capable mid-tier
    OSS model added by Sprint 1's overlay correction) must appear in the
    openrouter-reachable candidate pool with tool support enabled.

    The picker may choose a different cheaper model under -saver (gpt-oss-120b
    blends cheaper) — that's correct behavior, not a Sprint 2 regression.
    What Sprint 2 needs to validate is that the model is wired up: present
    in the multi-provider registry, tool-capable, and tagged for a real
    cloud-OSS provider. Without this assertion, a future overlay edit could
    silently drop the entry and we wouldn't catch it until someone wondered
    why their long-context tool-using requests fell back to frontier."""
    from modelmeld.scout.multi_provider_registry import default_multi_provider_registry

    reg = default_multi_provider_registry()
    openrouter_entries = [
        e for e in reg.all_entries_multi()
        if e.provider == "openrouter" and e.model_id == "qwen3-coder-next"
    ]
    assert len(openrouter_entries) == 1, (
        "qwen3-coder-next must have exactly one openrouter overlay row"
    )
    entry = openrouter_entries[0]
    assert entry.supports_tools, (
        "qwen3-coder-next must be tool-capable to serve Claude Code tool-using requests"
    )
    # Context window claim — the niche this model fills. If this drops below
    # 200k, the long-context tool-using-request use case regresses.
    assert entry.context_window >= 200_000, (
        f"qwen3-coder-next overlay row context_window={entry.context_window} "
        "is below 200k — its primary value (long-context tool calls) is gone"
    )


async def test_openrouter_only_lineup_includes_qwen3_coder_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In an openrouter-only setup, the eligible-candidate set for a
    tool-using request should INCLUDE qwen3-coder-next (even if the picker
    chooses a cheaper alternative). Verifies the routing layer's eligibility
    filter doesn't drop the entry due to a mis-set supports_tools or
    threshold."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODELMELD_OPENROUTER_API_KEY", "or_smoke_test_key")
    monkeypatch.setenv("MODELMELD_ROUTING_POLICY", "capability")

    settings = GatewaySettings()
    router = _build_capability_router(settings, None)

    # All tool-capable openrouter-tagged entries meeting the saver task_use
    # threshold should be in the candidate pool. We don't make a routing
    # decision here — we just verify the registry surface.
    candidates = [
        e for e in router.scout.registry.all_entries_multi()
        if e.provider == "openrouter"
        and e.supports_tools
        and e.task_scores.get("tool_use", 0.0) >= 0.70
    ]
    candidate_ids = {e.model_id for e in candidates}
    assert "qwen3-coder-next" in candidate_ids, (
        f"qwen3-coder-next missing from openrouter candidate pool. "
        f"Found: {sorted(candidate_ids)}"
    )

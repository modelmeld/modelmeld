# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""B-3 Phase 2: the client's reasoning intent (output_config.effort / adaptive
thinking) is re-surfaced as `reasoning_effort` on the internal request, so
OpenAI-compatible OSS backends reason instead of having it dropped at
translation. `output_config.effort` wins; adaptive thinking without an explicit
effort maps to a moderate default; no signal → None.
"""

from __future__ import annotations

from modelmeld.api.schemas_anthropic import AnthropicMessage, AnthropicMessagesRequest
from modelmeld.translation import from_anthropic_request


def _req(**extra: object) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="anthropic/modelmeld-auto",
        max_tokens=1024,
        messages=[AnthropicMessage(role="user", content="hi")],
        **extra,
    )


def test_output_config_effort_maps_directly() -> None:
    for level in ("low", "medium", "high"):
        out = from_anthropic_request(_req(output_config={"effort": level}))
        assert out.reasoning_effort == level


def test_effort_above_high_clamps_to_high() -> None:
    for level in ("xhigh", "max"):
        out = from_anthropic_request(_req(output_config={"effort": level}))
        assert out.reasoning_effort == "high"


def test_adaptive_thinking_without_effort_maps_to_medium() -> None:
    out = from_anthropic_request(_req(thinking={"type": "adaptive"}))
    assert out.reasoning_effort == "medium"


def test_explicit_effort_wins_over_thinking() -> None:
    out = from_anthropic_request(
        _req(thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    )
    assert out.reasoning_effort == "low"


def test_no_reasoning_signal_is_none() -> None:
    assert from_anthropic_request(_req()).reasoning_effort is None
    assert from_anthropic_request(_req(thinking={"type": "disabled"})).reasoning_effort is None
    # output_config present but no effort (e.g. only a format) → no reasoning.
    assert from_anthropic_request(_req(output_config={"format": {"type": "json"}})).reasoning_effort is None


def test_unknown_effort_value_is_ignored() -> None:
    # A future/unrecognized effort string falls through to None rather than
    # forwarding a value reasoning_effort can't represent.
    assert from_anthropic_request(_req(output_config={"effort": "ludicrous"})).reasoning_effort is None


def test_reasoning_effort_reaches_openai_egress_params() -> None:
    """End-to-end: Anthropic effort → reasoning_effort → OpenAI egress dict
    (which OpenAI-compatible reasoning-capable backends honor)."""
    from modelmeld.adapters.openai_adapter import OpenAIAdapter

    ad = OpenAIAdapter.__new__(OpenAIAdapter)
    ad.served_model = None

    internal = from_anthropic_request(_req(output_config={"effort": "high"}))
    params = OpenAIAdapter._to_params(ad, internal, stream=False)
    assert params.get("reasoning_effort") == "high"

    # Omitted entirely when no reasoning was signalled (exclude_none).
    plain = from_anthropic_request(_req())
    assert "reasoning_effort" not in OpenAIAdapter._to_params(ad, plain, stream=False)

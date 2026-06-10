# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""B-3 Phase 1: capability-aware reconciliation of model-tuned controls on a
model substitution (`_native_body_for_upstream`).

Replaces the old blunt "drop thinking/effort/output_config/context_management on
every substitution" with: keep the reasoning cluster when the served model can
take it (`reasoning_interface == "anthropic_adaptive"`), always drop
context_management (gateway emulation is a later phase), clamp max_tokens to the
served model's ceiling, and fall back to the blunt drop when the served model is
unknown.
"""

from __future__ import annotations

from modelmeld.api.routes.messages import _native_body_for_upstream
from modelmeld.api.schemas_anthropic import AnthropicMessagesRequest
from modelmeld.scout.registry import ModelEntry


def _body(max_tokens: int = 8000, **extra: object) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="anthropic/modelmeld-auto",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


def _entry(reasoning: str = "anthropic_adaptive", max_out: int | None = None) -> ModelEntry:
    return ModelEntry(
        model_id="served-model", provider="anthropic", context_window=200000,
        cost_per_m_input=3.0, cost_per_m_output=15.0,
        reasoning_interface=reasoning, max_output_tokens=max_out,
    )


def test_no_substitution_returns_body_unchanged() -> None:
    b = _body(thinking={"type": "adaptive"})
    # served_model None or equal to the requested model → passthrough verbatim.
    assert _native_body_for_upstream(b, None) is b
    assert _native_body_for_upstream(b, b.model) is b


def test_reasoning_model_keeps_reasoning_drops_context() -> None:
    b = _body(
        thinking={"type": "adaptive"}, effort="high",
        output_config={"effort": "high"}, context_management={"edits": []},
    )
    out = _native_body_for_upstream(b, "served-model", _entry("anthropic_adaptive"))
    ex = out.model_extra or {}
    assert "thinking" in ex and "effort" in ex and "output_config" in ex
    assert "context_management" not in ex  # always dropped until gateway emulation
    assert out.model == "served-model"


def test_nonreasoning_model_drops_whole_cluster() -> None:
    b = _body(thinking={"type": "adaptive"}, effort="high", context_management={"edits": []})
    out = _native_body_for_upstream(b, "served-model", _entry("none"))
    ex = out.model_extra or {}
    assert not any(k in ex for k in ("thinking", "effort", "output_config", "context_management"))


def test_unknown_entry_falls_back_to_blunt_drop() -> None:
    b = _body(thinking={"type": "adaptive"}, effort="high", context_management={"edits": []})
    out = _native_body_for_upstream(b, "served-model", served_entry=None)
    ex = out.model_extra or {}
    assert not any(k in ex for k in ("thinking", "effort", "context_management"))


def test_max_tokens_clamped_to_served_cap() -> None:
    b = _body(max_tokens=8000, thinking={"type": "adaptive"})
    out = _native_body_for_upstream(b, "served-model", _entry("anthropic_adaptive", max_out=4000))
    assert out.max_tokens == 4000


def test_max_tokens_not_clamped_when_under_cap() -> None:
    b = _body(max_tokens=8000)
    out = _native_body_for_upstream(b, "served-model", _entry("anthropic_adaptive", max_out=64000))
    assert out.max_tokens == 8000


def test_original_body_is_not_mutated() -> None:
    b = _body(thinking={"type": "adaptive"}, context_management={"x": 1})
    _native_body_for_upstream(b, "served-model", _entry("none"))
    ex = b.model_extra or {}
    assert "thinking" in ex and "context_management" in ex  # original untouched

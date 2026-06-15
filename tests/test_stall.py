# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Unit tests for the observe-only stall detector (scout/stall.py)."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)
from modelmeld.scout.stall import (
    StallWeights,
    TurnObservation,
    default_stall_weights,
    detect_stall,
    observations_from_anthropic,
    observations_from_internal,
    stall_shadow_enabled,
)


# --- helpers ---------------------------------------------------------------- #
def _a(*tools: str) -> TurnObservation:
    """An assistant turn calling `tools`."""
    return TurnObservation(role="assistant", tool_names=tools)


def _result(error: bool) -> TurnObservation:
    """A user turn carrying a tool result (errored or not)."""
    return TurnObservation(role="user", has_tool_result=True, tool_error=error)


# --- component signals ------------------------------------------------------ #
def test_loop_alone_does_not_stall() -> None:
    # Two identical Bash turns is a loop (1 structural signal) but not 2 — and
    # patch-before-explore can't fire on a non-edit tool, so this stays calm.
    obs = [_a("Bash"), _result(False), _a("Bash"), _result(False)]
    assert detect_stall(obs).stalled is False


def test_loop_plus_errors_stalls() -> None:
    obs = [_a("Bash"), _result(True), _a("Bash"), _result(True)]
    d = detect_stall(obs)
    assert d.stalled is True
    assert any(s.startswith("tool_loop") for s in d.signals)
    assert any(s.startswith("tool_errors") for s in d.signals)


def test_single_error_does_not_count_as_consecutive() -> None:
    # One errored result is below the error threshold; loop alone won't fire.
    obs = [_a("Bash"), _result(False), _a("Bash"), _result(True)]
    assert detect_stall(obs).stalled is False


def test_patch_before_explore_plus_loop_stalls() -> None:
    obs = [_a("Edit"), _a("Edit")]  # edit, edit — never read first
    d = detect_stall(obs)
    assert d.stalled is True
    assert "patch_before_explore" in d.signals


def test_patch_before_explore_alone_does_not_stall() -> None:
    obs = [_a("Edit"), _a("Read")]  # one structural signal only
    assert detect_stall(obs).stalled is False


def test_read_then_edit_is_not_patch_before_explore() -> None:
    obs = [_a("Read"), _a("Edit"), _a("Read"), _a("Edit")]
    d = detect_stall(obs)
    assert "patch_before_explore" not in d.signals


def test_turn_floor_fires_alone() -> None:
    obs = [_a() for _ in range(8)]  # 8 plain assistant turns, no tools
    d = detect_stall(obs)
    assert d.stalled is True
    assert any(s.startswith("turn_floor") for s in d.signals)


def test_below_turn_floor_does_not_fire() -> None:
    obs = [_a() for _ in range(7)]
    assert detect_stall(obs).stalled is False


def test_repeat_threshold_is_respected() -> None:
    # With repeat_threshold=3, two identical turns no longer count as a loop.
    w = StallWeights(repeat_threshold=3)
    obs = [_a("Bash"), _result(True), _a("Bash"), _result(True)]
    d = detect_stall(obs, w)
    assert not any(s.startswith("tool_loop") for s in d.signals)


# --- extractors ------------------------------------------------------------- #
def test_observations_from_anthropic_reads_tools_and_errors() -> None:
    body = AnthropicMessagesRequest(
        model="anthropic/modelmeld-auto",
        max_tokens=256,
        messages=[
            AnthropicMessage(
                role="assistant",
                content=[AnthropicToolUseBlock(type="tool_use", id="t1", name="Edit")],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicToolResultBlock(
                        type="tool_result",
                        tool_use_id="t1",
                        content=[AnthropicTextBlock(type="text", text="boom")],
                        is_error=True,
                    )
                ],
            ),
        ],
    )
    obs = observations_from_anthropic(body)
    assert obs[0].role == "assistant"
    assert obs[0].tool_names == ("Edit",)
    assert obs[1].has_tool_result is True
    assert obs[1].tool_error is True


def test_observations_from_internal_loses_is_error() -> None:
    req = ChatCompletionRequest(
        model="anthropic/modelmeld-auto",
        messages=[
            UserMessage(role="user", content="fix it"),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function=FunctionCall(name="Edit", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(role="tool", content="AssertionError", tool_call_id="c1"),
        ],
    )
    obs = observations_from_internal(req)
    assert obs[1].tool_names == ("Edit",)
    assert obs[2].has_tool_result is True
    # is_error is dropped in translation, so the internal extractor can't see it.
    assert obs[2].tool_error is False


# --- env gating ------------------------------------------------------------- #
def test_stall_shadow_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_STALL_SHADOW", raising=False)
    assert stall_shadow_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_stall_shadow_enabled_truthy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("MODELMELD_STALL_SHADOW", val)
    assert stall_shadow_enabled() is True


def test_weights_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_STALL_MAX_TURNS", "20")
    monkeypatch.setenv("MODELMELD_STALL_REPEAT_THRESHOLD", "3")
    monkeypatch.setenv("MODELMELD_STALL_ERROR_THRESHOLD", "4")
    w = default_stall_weights()
    assert (w.max_turns, w.repeat_threshold, w.error_threshold) == (20, 3, 4)


def test_weights_env_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_STALL_MAX_TURNS", "not-an-int")
    assert default_stall_weights().max_turns == 8

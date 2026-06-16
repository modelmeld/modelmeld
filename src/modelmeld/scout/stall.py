# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Trajectory stall detector for reactive escalation (SHADOW / observe-only).

Predictive request-time difficulty routing was retired — the open-vs-frontier
gap on agentic coding is small and structurally unpredictable at intake. The
surviving signal for "this run needs help" is RUNTIME non-convergence, which the
gateway can read retrospectively because the full conversation history arrives
client-supplied on every turn:

  - repeated identical tool calls across consecutive turns (a loop);
  - consecutive error-bearing tool results (repair that isn't converging);
  - patch-before-explore (editing before any exploration — correlates with
    failure);
  - a turn-count floor (the run is simply going long).

This module is PURE (no ML, no I/O) so it is safe on the routing hot path. It
emits a `StallDecision` only; in this increment the route handler logs it as a
"would escalate here" shadow signal and **changes nothing about routing**. The
reactive-escalation increment will act on the same decision.

Detection reads the CUMULATIVE history (every turn carries it), so it is almost
stateless; the per-session store exists to de-duplicate telemetry and to carry
the future sticky decision, not to reconstruct the trajectory.

Gated by `MODELMELD_STALL_SHADOW` (default OFF), mirroring
[[scout.difficulty.difficulty_routing_enabled]].
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ToolMessage,
)
from modelmeld.api.schemas_anthropic import (
    AnthropicMessagesRequest,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)

# --- tool classification ---------------------------------------------------- #
# Names a coding agent uses to MODIFY the workspace vs to EXPLORE it. Matched
# case-insensitively against the tool name; explicit sets cover the Claude Code
# tools, substrings cover generic computer-use / framework variants.
_EDIT_TOOLS: frozenset[str] = frozenset(
    {"write", "edit", "multiedit", "notebookedit"}
)
_EDIT_SUBSTRINGS: tuple[str, ...] = ("str_replace", "create_file", "apply_patch", "edit")
_READ_TOOLS: frozenset[str] = frozenset(
    {"read", "grep", "glob", "ls", "notebookread", "cat", "find"}
)
_READ_SUBSTRINGS: tuple[str, ...] = ("read", "grep", "glob", "search")


def _is_edit_tool(name: str) -> bool:
    low = name.lower()
    return low in _EDIT_TOOLS or any(s in low for s in _EDIT_SUBSTRINGS)


def _is_read_tool(name: str) -> bool:
    low = name.lower()
    return low in _READ_TOOLS or any(s in low for s in _READ_SUBSTRINGS)


# --- normalized observation ------------------------------------------------- #
@dataclass(frozen=True)
class TurnObservation:
    """Shape-agnostic view of one message in the conversation history.

    Extracted from either the native Anthropic body (full fidelity, incl.
    `is_error`) or the internal OpenAI-shape request (no `is_error` — it is
    dropped in translation, so `tool_error` is always False there).
    """

    role: str                       # "assistant" | "user" | "tool" | ...
    tool_names: tuple[str, ...] = ()  # tool_use / tool_call names on this turn
    has_tool_result: bool = False    # this turn carries >=1 tool result
    tool_error: bool = False         # >=1 of those results is an error


def observations_from_anthropic(
    body: AnthropicMessagesRequest,
) -> list[TurnObservation]:
    """Full-fidelity observations from the native Anthropic request body."""
    obs: list[TurnObservation] = []
    for msg in body.messages:
        content = msg.content
        if not isinstance(content, list):
            obs.append(TurnObservation(role=msg.role))
            continue
        names: list[str] = []
        has_result = False
        is_error = False
        for block in content:
            if isinstance(block, AnthropicToolUseBlock):
                names.append(block.name)
            elif isinstance(block, AnthropicToolResultBlock):
                has_result = True
                if block.is_error:
                    is_error = True
        obs.append(
            TurnObservation(
                role=msg.role,
                tool_names=tuple(names),
                has_tool_result=has_result,
                tool_error=is_error,
            )
        )
    return obs


def observations_from_internal(
    request: ChatCompletionRequest,
) -> list[TurnObservation]:
    """Observations from the internal OpenAI-shape request.

    `is_error` does not survive Anthropic->OpenAI translation, so the
    consecutive-error signal is unavailable here (tool_error stays False). Used
    for `/v1/chat/completions`, which has no native Anthropic body.
    """
    obs: list[TurnObservation] = []
    for msg in request.messages:
        role = getattr(msg, "role", "")
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            names = tuple(tc.function.name for tc in msg.tool_calls)
            obs.append(TurnObservation(role="assistant", tool_names=names))
        elif isinstance(msg, ToolMessage):
            obs.append(
                TurnObservation(role="tool", has_tool_result=True, tool_error=False)
            )
        else:
            obs.append(TurnObservation(role=role))
    return obs


# --- weights / decision ----------------------------------------------------- #
@dataclass(frozen=True)
class StallWeights:
    """Tunable, precision-biased thresholds. Conservative defaults; a false
    fire would (in the reactive increment) spend frontier budget, so we favour
    precision and let the A/B retune."""

    # consecutive identical-tool turns that count as a loop
    repeat_threshold: int = 2
    # consecutive error-bearing tool results that count as stuck repair
    error_threshold: int = 2
    # assistant-turn count at/above which the run is "going long"
    max_turns: int = 8


@dataclass(frozen=True)
class StallDecision:
    stalled: bool
    signals: tuple[str, ...]
    rationale: str


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return default


def default_stall_weights() -> StallWeights:
    """`StallWeights` with each threshold overridable via `MODELMELD_STALL_*`."""
    return StallWeights(
        repeat_threshold=_int_env("MODELMELD_STALL_REPEAT_THRESHOLD", 2),
        error_threshold=_int_env("MODELMELD_STALL_ERROR_THRESHOLD", 2),
        max_turns=_int_env("MODELMELD_STALL_MAX_TURNS", 8),
    )


def _trailing_identical_run(assistant_obs: list[TurnObservation]) -> int:
    """Length of the trailing run of assistant turns calling the SAME tool set."""
    if not assistant_obs:
        return 0
    last = assistant_obs[-1].tool_names
    if not last:
        return 0
    run = 0
    for o in reversed(assistant_obs):
        if o.tool_names == last:
            run += 1
        else:
            break
    return run


def _trailing_error_run(observations: list[TurnObservation]) -> int:
    """Length of the trailing run of error-bearing tool-result turns.

    Assistant turns interleave tool-result turns, so we look only at the ordered
    sequence of result-bearing turns and count the trailing errors."""
    results = [o.tool_error for o in observations if o.has_tool_result]
    run = 0
    for errored in reversed(results):
        if errored:
            run += 1
        else:
            break
    return run


def _patch_before_explore(observations: list[TurnObservation]) -> bool:
    """True if an edit-class tool was used before any read-class tool appeared."""
    first_edit: int | None = None
    first_read: int | None = None
    for i, o in enumerate(observations):
        for name in o.tool_names:
            if first_read is None and _is_read_tool(name):
                first_read = i
            if first_edit is None and _is_edit_tool(name):
                first_edit = i
    if first_edit is None:
        return False
    return first_read is None or first_edit < first_read


def detect_stall(
    observations: list[TurnObservation],
    weights: StallWeights | None = None,
) -> StallDecision:
    """Decide whether the trajectory in `observations` looks stalled.

    Composite, precision-biased: fire on the turn-count floor alone, OR when at
    least TWO independent structural signals agree. A single weak signal never
    fires."""
    w = weights or DEFAULT_STALL_WEIGHTS
    assistant_obs = [o for o in observations if o.role == "assistant"]
    turn_count = len(assistant_obs)

    signals: list[str] = []

    loop_run = _trailing_identical_run(assistant_obs)
    loop = loop_run >= w.repeat_threshold
    if loop:
        # loop implies a non-empty trailing tool set, so [0] is safe.
        signals.append(f"tool_loop({assistant_obs[-1].tool_names[0]}x{loop_run})")

    error_run = _trailing_error_run(observations)
    consecutive_errors = error_run >= w.error_threshold
    if consecutive_errors:
        signals.append(f"tool_errors(x{error_run})")

    pbe = _patch_before_explore(observations)
    if pbe:
        signals.append("patch_before_explore")

    structural_count = int(loop) + int(consecutive_errors) + int(pbe)

    turn_floor = turn_count >= w.max_turns
    if turn_floor:
        signals.append(f"turn_floor({turn_count})")

    stalled = turn_floor or structural_count >= 2
    if stalled:
        rationale = "stall:" + ",".join(signals)
    else:
        rationale = f"no_stall:turns={turn_count},structural={structural_count}"
    return StallDecision(stalled=stalled, signals=tuple(signals), rationale=rationale)


DEFAULT_STALL_WEIGHTS = StallWeights()


def stall_shadow_enabled() -> bool:
    """Whether `MODELMELD_STALL_SHADOW` enables observe-only stall telemetry.

    Default OFF. When on, the route handler runs `detect_stall` and emits a log
    line + `x-modelmeld-stall-shadow` header but does NOT change routing."""
    return os.environ.get("MODELMELD_STALL_SHADOW", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "DEFAULT_STALL_WEIGHTS",
    "StallDecision",
    "StallWeights",
    "TurnObservation",
    "default_stall_weights",
    "detect_stall",
    "observations_from_anthropic",
    "observations_from_internal",
    "stall_shadow_enabled",
]

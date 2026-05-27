# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""ModelMeld alias-based routing policies.

The gateway exposes three auto-route aliases via `/v1/models`. Each is a
preset of routing-policy parameters the scout reads when the request's
`model` field matches the alias name. Aliases let customers pick a
cost-quality ceiling without writing routing rules:

  - `anthropic/modelmeld-saver`
      Stays inside the OSS provider tier. Frontier rows are filtered out
      of the candidate set entirely. Within OSS, the existing
      shape-biased scout picks cheap-vs-mid-vs-premium based on
      complexity. **Cost ceiling is bounded — customer cannot be
      surprise-billed frontier rates.**

  - `anthropic/modelmeld-auto`
      Starts in OSS tier. Escalates to frontier when the user message
      contains ≥2 distinct "reasoning markers" (e.g., "step by step",
      "explain your reasoning"). Mirrors LiteLLM's Complexity Router
      escalation pattern but with a transparent, operator-tunable marker
      list. Customer message: "Smart default, frontier on demand."

  - `anthropic/modelmeld-quality`
      Starts at frontier (Sonnet 4.6 tier). Downgrades to OSS only via
      the existing autocomplete-shape bias when the request is obviously
      trivial (short input, no tools, small max_tokens). Customer
      message: "Frontier-first with smart cost trimming."

Backwards-compat: the previous 5 aliases (balanced/coding/reasoning/
cheap/frontier-priority) are mapped to the nearest new policy so any
existing integrations keep working. This is purely a positioning
collapse — the underlying registry and scout machinery is unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from enum import Enum


class ModelMeldPolicy(str, Enum):
    """The three customer-facing routing policies."""

    SAVER = "saver"
    AUTO = "auto"
    QUALITY = "quality"


# Provider-tier filters. The OSS / FRONTIER partition is the mechanism
# by which policies select model classes — far more robust than per-task
# score thresholds (which change every time we refresh benchmark data
# and which differ across categories: e.g., Sonnet 4.6 scores 0.80 on
# coding but 0.93 on simple_qa — a fixed numeric threshold misroutes).
#
# OSS providers host the open-weight models we route to (qwen, phi,
# deepseek, granite, gemma). Frontier providers are the closed-model
# vendors. NOTE: if a frontier model later appears on openrouter's
# catalog, we'd need a per-row tier marker — for the current registry
# shape, provider-level partition is correct and avoids schema change.
_OSS_PROVIDERS: frozenset[str] = frozenset({"vllm", "fireworks", "together", "openrouter"})
_FRONTIER_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})


# Alias model_id → policy mapping. The first three are the canonical
# customer-facing surface; the rest are deprecated names mapped to their
# nearest replacement so existing customer code keeps working.
_ALIAS_TO_POLICY: dict[str, ModelMeldPolicy] = {
    # Canonical
    "anthropic/modelmeld-saver": ModelMeldPolicy.SAVER,
    "anthropic/modelmeld-auto": ModelMeldPolicy.AUTO,
    "anthropic/modelmeld-quality": ModelMeldPolicy.QUALITY,
    # Deprecated — kept for backwards compat with any in-flight integrations
    "anthropic/modelmeld-balanced": ModelMeldPolicy.AUTO,
    "anthropic/modelmeld-coding": ModelMeldPolicy.SAVER,
    "anthropic/modelmeld-reasoning": ModelMeldPolicy.AUTO,
    "anthropic/modelmeld-cheap": ModelMeldPolicy.SAVER,
    "anthropic/modelmeld-frontier-priority": ModelMeldPolicy.QUALITY,
}


# Reasoning markers that trigger frontier escalation under AUTO policy.
# Sourced from LiteLLM's Complexity Router and similar systems — these are
# phrases users write when they specifically want careful reasoning.
# We check the USER messages only (not system prompts — those routinely
# carry "think step by step" boilerplate from Claude Code itself, which
# is not a customer-quality-bar signal). Match is case-insensitive and
# substring-based.
#
# Operator override: set MODELMELD_REASONING_MARKERS env var to a
# comma-separated list to replace the default markers (use a leading "+"
# to append instead of replace).
_DEFAULT_REASONING_MARKERS: tuple[str, ...] = (
    "step by step",
    "think through",
    "explain your reasoning",
    "walk me through your logic",
    "carefully consider",
    "show your work",
    "let's reason about",
    "think carefully",
    "reason carefully",
    "break this down",
)

# Number of distinct markers required to escalate. 2+ avoids false
# positives from incidental usage ("I followed the steps step by step"
# shouldn't escalate; "think step by step and show your work" should).
_REASONING_MARKER_ESCALATION_COUNT = 2


# All three policies use the same quality_threshold (the scout's default,
# typically 0.70). Tier selection is done via provider filter, not
# threshold manipulation — see _OSS_PROVIDERS / _FRONTIER_PROVIDERS above.
# Kept as a named constant so future tuning has one place to land.
POLICY_QUALITY_THRESHOLD: float = 0.70


def resolve_policy(model_id: str | None) -> ModelMeldPolicy | None:
    """Return the policy for a ModelMeld alias, or None for non-alias models.

    Non-alias model ids (e.g., `claude-opus-4-7`, `gpt-5`,
    `qwen3-coder-flash`) pass through the scout's default task-category
    routing without policy adjustment. Only aliases trigger policy.
    """
    if not model_id:
        return None
    return _ALIAS_TO_POLICY.get(model_id)


def oss_providers() -> frozenset[str]:
    """The set of providers SAVER restricts the candidate pool to."""
    return _OSS_PROVIDERS


def frontier_providers() -> frozenset[str]:
    """The set of providers AUTO-escalated and QUALITY restrict the pool to.

    Note: QUALITY's autocomplete-shape downgrade is still handled by the
    existing scout shape-bias logic (which lowers threshold + admits OSS
    rows); see `CapabilityScout._apply_shape_bias`. The provider filter
    here is only applied when QUALITY *isn't* downgrading.
    """
    return _FRONTIER_PROVIDERS


def reasoning_markers() -> tuple[str, ...]:
    """Return the active reasoning-marker list (env override or default).

    `MODELMELD_REASONING_MARKERS=foo,bar,baz` replaces the default list.
    Leading "+" appends instead: `MODELMELD_REASONING_MARKERS=+extra,more`.
    Empty / unset env var → default list.
    """
    override = os.environ.get("MODELMELD_REASONING_MARKERS", "").strip()
    if not override:
        return _DEFAULT_REASONING_MARKERS
    if override.startswith("+"):
        extra = tuple(p.strip().lower() for p in override[1:].split(",") if p.strip())
        return _DEFAULT_REASONING_MARKERS + extra
    return tuple(p.strip().lower() for p in override.split(",") if p.strip())


def detect_reasoning_markers(
    text: str, markers: Iterable[str] | None = None,
) -> int:
    """Count distinct reasoning markers present in `text`, case-insensitive.

    Returns 0 for empty text. Each marker in the list is counted at
    most once (we care about diversity, not frequency).
    """
    if not text:
        return 0
    lower = text.lower()
    marker_list = tuple(markers) if markers is not None else reasoning_markers()
    return sum(1 for m in marker_list if m in lower)


def extract_user_text(request) -> str:
    """Concatenate text from USER messages only — system prompts excluded.

    Claude Code's system prompts routinely include "think step by step"
    boilerplate. Counting markers there would trigger spurious frontier
    escalation on every request. Only the actual user turns matter.
    """
    parts: list[str] = []
    for msg in request.messages:
        role = getattr(msg, "role", "")
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    return " ".join(parts)


def should_escalate_to_frontier(request) -> tuple[bool, int]:
    """AUTO policy: should this request escalate to frontier-tier model?

    Returns (escalate, marker_count). Caller uses the bool to bump the
    threshold; marker_count is for audit/log rationale.
    """
    user_text = extract_user_text(request)
    count = detect_reasoning_markers(user_text)
    return (count >= _REASONING_MARKER_ESCALATION_COUNT, count)


__all__ = [
    "POLICY_QUALITY_THRESHOLD",
    "ModelMeldPolicy",
    "detect_reasoning_markers",
    "extract_user_text",
    "frontier_providers",
    "oss_providers",
    "reasoning_markers",
    "resolve_policy",
    "should_escalate_to_frontier",
]

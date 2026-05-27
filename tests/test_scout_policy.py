# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Unit tests for the alias-policy resolver + reasoning-marker classifier.

Task #177. Three policies (saver/auto/quality) each control whether
frontier-tier models are eligible for a request. The detection logic
must be deterministic, false-positive-resistant, and operator-tunable.
"""
from __future__ import annotations

from modelmeld.api.schemas import ChatCompletionRequest, SystemMessage, UserMessage
from modelmeld.scout.policy import (
    POLICY_QUALITY_THRESHOLD,
    ModelMeldPolicy,
    detect_reasoning_markers,
    extract_user_text,
    frontier_providers,
    oss_providers,
    reasoning_markers,
    resolve_policy,
    should_escalate_to_frontier,
)

# ---------------------------------------------------------------------------
# resolve_policy
# ---------------------------------------------------------------------------


def test_resolve_policy_canonical_aliases() -> None:
    assert resolve_policy("anthropic/modelmeld-saver") is ModelMeldPolicy.SAVER
    assert resolve_policy("anthropic/modelmeld-auto") is ModelMeldPolicy.AUTO
    assert resolve_policy("anthropic/modelmeld-quality") is ModelMeldPolicy.QUALITY


def test_resolve_policy_deprecated_aliases_still_work() -> None:
    """Backwards-compat: old 5-alias names still resolve to a sensible policy."""
    assert resolve_policy("anthropic/modelmeld-balanced") is ModelMeldPolicy.AUTO
    assert resolve_policy("anthropic/modelmeld-coding") is ModelMeldPolicy.SAVER
    assert resolve_policy("anthropic/modelmeld-reasoning") is ModelMeldPolicy.AUTO
    assert resolve_policy("anthropic/modelmeld-cheap") is ModelMeldPolicy.SAVER
    assert resolve_policy("anthropic/modelmeld-frontier-priority") is ModelMeldPolicy.QUALITY


def test_resolve_policy_returns_none_for_non_alias() -> None:
    """Non-alias model ids must pass through untouched — the scout's
    default task-category routing handles them."""
    assert resolve_policy("claude-opus-4-7") is None
    assert resolve_policy("gpt-5") is None
    assert resolve_policy("qwen3-coder-flash") is None
    assert resolve_policy(None) is None
    assert resolve_policy("") is None


# ---------------------------------------------------------------------------
# reasoning marker detection
# ---------------------------------------------------------------------------


def test_detect_reasoning_markers_counts_distinct_phrases() -> None:
    text = "Please think step by step and show your work as you reason."
    assert detect_reasoning_markers(text) == 2  # 'step by step' + 'show your work'


def test_detect_reasoning_markers_is_case_insensitive() -> None:
    assert detect_reasoning_markers("Step By Step") == 1


def test_detect_reasoning_markers_returns_zero_for_empty() -> None:
    assert detect_reasoning_markers("") == 0
    assert detect_reasoning_markers("hello world") == 0


def test_detect_reasoning_markers_no_double_count_for_same_phrase() -> None:
    """A single marker repeated 10 times should count as 1, not 10."""
    text = "step by step step by step step by step"
    # We only check `in lower` per marker — counts each marker at most once.
    assert detect_reasoning_markers(text) == 1


def test_reasoning_markers_env_override_replaces_default(monkeypatch) -> None:
    monkeypatch.setenv("MODELMELD_REASONING_MARKERS", "custom phrase,another marker")
    markers = reasoning_markers()
    assert "custom phrase" in markers
    assert "another marker" in markers
    assert "step by step" not in markers  # default replaced


def test_reasoning_markers_env_override_appends_with_plus_prefix(monkeypatch) -> None:
    monkeypatch.setenv("MODELMELD_REASONING_MARKERS", "+extra one,extra two")
    markers = reasoning_markers()
    assert "step by step" in markers  # default kept
    assert "extra one" in markers
    assert "extra two" in markers


# ---------------------------------------------------------------------------
# extract_user_text — system prompts EXCLUDED
# ---------------------------------------------------------------------------


def _req(user_text: str | None = None, system_text: str | None = None) -> ChatCompletionRequest:
    messages: list = []
    if system_text:
        messages.append(SystemMessage(role="system", content=system_text))
    if user_text:
        messages.append(UserMessage(role="user", content=user_text))
    return ChatCompletionRequest(model="anthropic/modelmeld-auto", messages=messages)


def test_extract_user_text_pulls_user_messages_only() -> None:
    req = _req(user_text="hello world", system_text="you are a helpful assistant")
    text = extract_user_text(req)
    assert "hello world" in text
    assert "helpful assistant" not in text  # system excluded


def test_extract_user_text_handles_no_user_message() -> None:
    req = _req(system_text="system only")
    assert extract_user_text(req) == ""


def test_extract_user_text_system_with_reasoning_markers_does_not_escalate() -> None:
    """Critical false-positive test: Claude Code's system prompts often
    contain 'think step by step' boilerplate. That MUST NOT trigger
    frontier escalation — only USER-message markers count."""
    req = _req(
        user_text="hi",
        system_text="You are a careful assistant. Always think step by step and show your work.",
    )
    should_escalate, count = should_escalate_to_frontier(req)
    assert not should_escalate
    assert count == 0


# ---------------------------------------------------------------------------
# should_escalate_to_frontier — the AUTO trigger
# ---------------------------------------------------------------------------


def test_should_escalate_when_two_markers_in_user_text() -> None:
    req = _req(
        user_text="Please think step by step and explain your reasoning as you go."
    )
    should_escalate, count = should_escalate_to_frontier(req)
    assert should_escalate
    assert count == 2


def test_should_not_escalate_with_single_marker() -> None:
    """Single marker is incidental — escalation requires 2+ distinct."""
    req = _req(user_text="Walk me step by step through this problem.")
    should_escalate, count = should_escalate_to_frontier(req)
    assert not should_escalate
    assert count == 1


def test_should_not_escalate_with_no_markers() -> None:
    req = _req(user_text="Write a fizzbuzz program.")
    should_escalate, count = should_escalate_to_frontier(req)
    assert not should_escalate
    assert count == 0


# ---------------------------------------------------------------------------
# provider-tier filters — the actual mechanism behind each policy
# ---------------------------------------------------------------------------


def test_oss_providers_excludes_frontier_providers() -> None:
    providers = oss_providers()
    assert "anthropic" not in providers
    assert "openai" not in providers
    # OSS providers we host on
    assert "openrouter" in providers
    assert "fireworks" in providers


def test_frontier_providers_excludes_oss_providers() -> None:
    providers = frontier_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "openrouter" not in providers
    assert "fireworks" not in providers


def test_oss_and_frontier_providers_are_disjoint() -> None:
    """No provider can be both OSS and frontier — the partition must be clean."""
    assert oss_providers().isdisjoint(frontier_providers())


def test_policy_quality_threshold_is_a_sensible_default() -> None:
    """Single threshold value used for all policies (tier selection is via
    provider filter, not threshold)."""
    assert 0.0 <= POLICY_QUALITY_THRESHOLD <= 1.0

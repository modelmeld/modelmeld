# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for TogetherAdapter + OpenRouterAdapter.

Both are thin OpenAI-compatible subclasses; tests verify the
construction contract + env-var resolution + that they're correctly
exported from the adapters package.
"""

from __future__ import annotations

import pytest

from modelmeld.adapters import (
    FireworksAdapter,
    OpenRouterAdapter,
    TogetherAdapter,
)
from modelmeld.adapters.base import AdapterError


# ---------------------------------------------------------------------------
# TogetherAdapter
# ---------------------------------------------------------------------------

def test_together_constructs_with_explicit_key() -> None:
    adapter = TogetherAdapter(api_key="t_test_key")
    assert adapter.name == "together"
    assert adapter.is_egress is True


def test_together_constructs_from_plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "t_plain")
    TogetherAdapter()


def test_together_constructs_from_modelmeld_prefixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("MODELMELD_TOGETHER_API_KEY", "t_prefixed")
    TogetherAdapter()


def test_together_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("MODELMELD_TOGETHER_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="TogetherAdapter requires an API key"):
        TogetherAdapter()


def test_together_no_served_model_pin() -> None:
    adapter = TogetherAdapter(api_key="t_test")
    assert adapter.served_model is None


# ---------------------------------------------------------------------------
# OpenRouterAdapter
# ---------------------------------------------------------------------------

def test_openrouter_constructs_with_explicit_key() -> None:
    adapter = OpenRouterAdapter(api_key="or_test_key")
    assert adapter.name == "openrouter"
    assert adapter.is_egress is True


def test_openrouter_constructs_from_plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_plain")
    OpenRouterAdapter()


def test_openrouter_constructs_from_modelmeld_prefixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MODELMELD_OPENROUTER_API_KEY", "or_prefixed")
    OpenRouterAdapter()


def test_openrouter_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MODELMELD_OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="OpenRouterAdapter requires an API key"):
        OpenRouterAdapter()


def test_openrouter_no_served_model_pin() -> None:
    adapter = OpenRouterAdapter(api_key="or_test")
    assert adapter.served_model is None


# ---------------------------------------------------------------------------
# All three upstream adapters have distinct provider names so the
# router can key (model_id, provider) cleanly.
# ---------------------------------------------------------------------------

def test_three_upstream_providers_have_distinct_names() -> None:
    fw = FireworksAdapter(api_key="fw")
    tg = TogetherAdapter(api_key="tg")
    or_ = OpenRouterAdapter(api_key="or")
    assert {fw.name, tg.name, or_.name} == {"fireworks", "together", "openrouter"}
    assert fw.is_egress and tg.is_egress and or_.is_egress

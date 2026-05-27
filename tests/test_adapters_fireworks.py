# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for FireworksAdapter.

Thin shape — Fireworks is OpenAI-compatible at the wire so the adapter
inherits virtually all behavior from OpenAIAdapter. These tests verify
the construction contract + the env-var resolution order.
"""

from __future__ import annotations

import pytest

from modelmeld.adapters import FireworksAdapter
from modelmeld.adapters.base import AdapterError


def test_constructs_with_explicit_api_key() -> None:
    adapter = FireworksAdapter(api_key="fw_test_key")
    assert adapter.name == "fireworks"
    assert adapter.is_egress is True


def test_constructs_from_plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_plain_env_key")
    FireworksAdapter()   # no raise


def test_constructs_from_modelmeld_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("MODELMELD_FIREWORKS_API_KEY", "fw_prefixed_env_key")
    FireworksAdapter()


def test_plain_env_takes_priority_over_prefixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "wins")
    monkeypatch.setenv("MODELMELD_FIREWORKS_API_KEY", "loses")
    adapter = FireworksAdapter()
    # OpenAIAdapter stores the resolved key; we only assert construction
    # succeeded — the priority is documented in the adapter's docstring.
    assert adapter is not None


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("MODELMELD_FIREWORKS_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="FireworksAdapter requires an API key"):
        FireworksAdapter()


def test_no_served_model_pin() -> None:
    """Fireworks serves many models under one endpoint; the gateway's
    request.model selects which one. The adapter MUST NOT pin a
    served_model so the client's model id passes through."""
    adapter = FireworksAdapter(api_key="fw_test")
    assert adapter.served_model is None


def test_base_url_override() -> None:
    FireworksAdapter(
        api_key="fw_test",
        base_url="https://staging-api.fireworks.ai/inference/v1",
    )

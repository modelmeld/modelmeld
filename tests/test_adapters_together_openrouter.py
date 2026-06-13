# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for TogetherAdapter + OpenRouterAdapter.

Both are thin OpenAI-compatible subclasses; tests verify the
construction contract + env-var resolution + that they're correctly
exported from the adapters package.
"""

from __future__ import annotations

import httpx
import pytest

import modelmeld.adapters.together_adapter as together_mod
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


def _mock_async_client(handler):
    """Factory replacing together_adapter.httpx.AsyncClient with a mock-transport
    client so health()'s raw GET can be intercepted. Captures the real
    AsyncClient before the monkeypatch so the factory doesn't recurse into the
    patched name."""
    real = httpx.AsyncClient
    def _factory(*_args, **_kwargs):
        return real(transport=httpx.MockTransport(handler))
    return _factory


async def test_together_health_true_on_bare_list_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Together's /models returns a BARE list, not {data:[...]}.
    The inherited SDK models.list() raises on that shape, so health() must use a
    raw GET and treat any 2xx as healthy — otherwise ALL Together routing 503s."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json=[{"id": "a"}, {"id": "b"}])  # bare list
    monkeypatch.setattr(together_mod.httpx, "AsyncClient", _mock_async_client(handler))
    adapter = TogetherAdapter(api_key="t_test")
    assert await adapter.health() is True


async def test_together_health_false_on_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")
    monkeypatch.setattr(together_mod.httpx, "AsyncClient", _mock_async_client(handler))
    adapter = TogetherAdapter(api_key="t_test")
    assert await adapter.health() is False


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

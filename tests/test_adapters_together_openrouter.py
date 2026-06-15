# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for TogetherAdapter + OpenRouterAdapter.

Both are thin OpenAI-compatible subclasses; tests verify the
construction contract + env-var resolution + that they're correctly
exported from the adapters package.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import modelmeld.adapters.together_adapter as together_mod
from modelmeld.adapters import (
    FireworksAdapter,
    OpenRouterAdapter,
    TogetherAdapter,
)
from modelmeld.adapters.base import AdapterError
from modelmeld.api.schemas import ChatCompletionRequest

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


# ---------------------------------------------------------------------------
# OpenRouter provider-routing (cache-stickiness): a deterministic `provider`
# preference pins a session to one backend so the per-backend prompt cache
# accumulates across turns (default load-balancing scatters it across backends).
# ---------------------------------------------------------------------------

def test_openrouter_default_provider_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_OPENROUTER_PROVIDER_SORT", raising=False)
    adapter = OpenRouterAdapter(api_key="or_test")
    assert adapter._extra_body == {"provider": {"sort": "price", "allow_fallbacks": True}}


def test_openrouter_provider_sort_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_OPENROUTER_PROVIDER_SORT", "throughput")
    adapter = OpenRouterAdapter(api_key="or_test")
    assert adapter._extra_body == {"provider": {"sort": "throughput", "allow_fallbacks": True}}


def test_openrouter_provider_routing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_OPENROUTER_PROVIDER_SORT", "")
    adapter = OpenRouterAdapter(api_key="or_test")
    assert adapter._extra_body is None


def test_together_has_no_provider_routing() -> None:
    # Provider routing is OpenRouter-specific; sibling OpenAI-compat adapters unset.
    assert TogetherAdapter(api_key="tg")._extra_body is None


async def test_openrouter_sends_provider_extra_body_on_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider preference must actually reach the wire (extra_body on the
    create call), not just sit on the adapter."""
    monkeypatch.delenv("MODELMELD_OPENROUTER_PROVIDER_SORT", raising=False)
    adapter = OpenRouterAdapter(api_key="or_test")
    fake = MagicMock()
    fake.model_dump.return_value = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_create = AsyncMock(return_value=fake)
    adapter._client.chat.completions.create = mock_create  # type: ignore[method-assign]
    await adapter.chat(ChatCompletionRequest(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
    ))
    assert mock_create.call_args.kwargs.get("extra_body") == {
        "provider": {"sort": "price", "allow_fallbacks": True},
    }

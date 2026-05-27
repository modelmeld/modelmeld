"""F-8: served_model substitution + serves_model() compatibility check."""

from __future__ import annotations

import pytest

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletionRequest,
    UserMessage,
)


class _ConcreteAdapter(ProviderAdapter):
    """Minimal concrete adapter so we can test the base-class helpers
    without depending on Anthropic / OpenAI SDKs."""

    name = "test"

    def __init__(self, served_model: str | None = None) -> None:
        self.served_model = served_model

    async def chat(self, request): raise NotImplementedError
    def stream_chat(self, request): raise NotImplementedError  # pragma: no cover
    async def health(self) -> bool: return True


def _req(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[UserMessage(role="user", content="hi")],
    )


# ---------------------------------------------------------------------------
# serves_model() — accepts any model when served_model is None
# ---------------------------------------------------------------------------

def test_serves_model_passthrough_accepts_anything() -> None:
    """Default served_model=None means 'pass-through' — adapter accepts
    any model id (the upstream provider decides)."""
    a = _ConcreteAdapter()
    assert a.serves_model("claude-haiku-4-5-20251001") is True
    assert a.serves_model("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ") is True
    assert a.serves_model("any-random-string") is True
    assert a.serves_model("") is True


def test_serves_model_pinned_still_serves_via_substitution() -> None:
    """Setting served_model is opting into substitution. The adapter will
    serve any request by rewriting model on the way out, so serves_model()
    returns True for any input. Subclasses can override for strict mode."""
    a = _ConcreteAdapter(served_model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    assert a.serves_model("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ") is True
    assert a.serves_model("claude-haiku-4-5-20251001") is True  # substitution
    assert a.serves_model("") is True


def test_subclass_can_implement_strict_mode() -> None:
    """If an operator wants to reject non-matching model ids outright
    (compliance mode, etc.), they subclass and override serves_model."""
    class _StrictAdapter(ProviderAdapter):
        name = "strict"
        async def chat(self, request): raise NotImplementedError
        def stream_chat(self, request): raise NotImplementedError  # pragma: no cover
        async def health(self) -> bool: return True

        def serves_model(self, model_id: str) -> bool:
            if self.served_model is None:
                return True
            return model_id == self.served_model

    a = _StrictAdapter()
    a.served_model = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    assert a.serves_model("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ") is True
    assert a.serves_model("claude-haiku-4-5-20251001") is False


# ---------------------------------------------------------------------------
# _apply_served_model() — request model substitution
# ---------------------------------------------------------------------------

def test_apply_served_model_pass_through_when_none() -> None:
    a = _ConcreteAdapter()
    req = _req("claude-haiku-4-5-20251001")
    out = a._apply_served_model(req)
    # Same object reference — no copy on the hot path for the common case
    assert out is req


def test_apply_served_model_no_op_when_already_matching() -> None:
    """If client already sent the matching model id, don't copy."""
    a = _ConcreteAdapter(served_model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    req = _req("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    out = a._apply_served_model(req)
    assert out is req
    assert out.model == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"


def test_apply_served_model_substitutes_when_pinned() -> None:
    a = _ConcreteAdapter(served_model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    req = _req("claude-haiku-4-5-20251001")
    out = a._apply_served_model(req)
    assert out.model == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    # Original request must be untouched (immutability for hooks / audit)
    assert req.model == "claude-haiku-4-5-20251001"


def test_apply_served_model_preserves_other_fields() -> None:
    a = _ConcreteAdapter(served_model="qwen")
    req = ChatCompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[UserMessage(role="user", content="hello")],
        max_tokens=128,
        temperature=0.3,
    )
    out = a._apply_served_model(req)
    assert out.model == "qwen"
    assert out.max_tokens == 128
    assert out.temperature == 0.3
    assert len(out.messages) == 1


# ---------------------------------------------------------------------------
# Subclass-override behavior
# ---------------------------------------------------------------------------

def test_subclass_can_override_serves_model_for_richer_matching() -> None:
    """Cloud adapters serving a whole catalog can implement model-family matching."""
    class _CatalogAdapter(ProviderAdapter):
        name = "catalog"
        async def chat(self, request): raise NotImplementedError
        def stream_chat(self, request): raise NotImplementedError  # pragma: no cover
        async def health(self) -> bool: return True

        def serves_model(self, model_id: str) -> bool:
            return model_id.startswith("claude-")

    c = _CatalogAdapter()
    assert c.serves_model("claude-haiku-4-5-20251001") is True
    assert c.serves_model("claude-sonnet-4-6") is True
    assert c.serves_model("Qwen/Qwen2.5-Coder-7B") is False


# ---------------------------------------------------------------------------
# Adapter constructor wiring (smoke-test the three concrete adapters that
# accept served_model)
# ---------------------------------------------------------------------------

def test_vllm_adapter_accepts_served_model(monkeypatch) -> None:
    from modelmeld.adapters.vllm_adapter import VLLMAdapter
    monkeypatch.setenv("MODELMELD_VLLM_ENDPOINT", "http://localhost:8000")
    a = VLLMAdapter(served_model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    assert a.served_model == "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    # With substitution semantics, the adapter will serve any request
    # by rewriting model on the way out.
    assert a.serves_model("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ") is True
    assert a.serves_model("claude-haiku-4-5-20251001") is True


def test_anthropic_adapter_accepts_served_model(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    from modelmeld.adapters.anthropic_adapter import AnthropicAdapter
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    a = AnthropicAdapter(served_model="claude-haiku-4-5-20251001")
    assert a.served_model == "claude-haiku-4-5-20251001"
    assert a.serves_model("claude-haiku-4-5-20251001") is True
    assert a.serves_model("Qwen/Qwen2.5-Coder-7B") is True  # substitution


def test_openai_adapter_accepts_served_model(monkeypatch) -> None:
    pytest.importorskip("openai")
    from modelmeld.adapters.openai_adapter import OpenAIAdapter
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    a = OpenAIAdapter(served_model="gpt-5-mini")
    assert a.served_model == "gpt-5-mini"
    assert a.serves_model("gpt-5-mini") is True
    assert a.serves_model("claude-haiku") is True  # substitution


def test_adapter_default_served_model_is_none(monkeypatch) -> None:
    """Backward compat: existing adapter construction without
    `served_model=` should preserve the pre-F-8 pass-through behavior."""
    pytest.importorskip("anthropic")
    from modelmeld.adapters.anthropic_adapter import AnthropicAdapter
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    a = AnthropicAdapter()
    assert a.served_model is None
    assert a.serves_model("anything") is True

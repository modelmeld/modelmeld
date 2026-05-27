"""TokenCounter — char-based default, LiteLLM backend with fallback."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from modelmeld.api.schemas import (
    SystemMessage,
    UserMessage,
)
from modelmeld.tokens import (
    CharBasedTokenCounter,
    LiteLLMTokenCounter,
    build_token_counter,
)

# ---------------------------------------------------------------------------
# CharBasedTokenCounter — heuristic
# ---------------------------------------------------------------------------

def test_char_based_count_text_uses_4_chars_per_token() -> None:
    c = CharBasedTokenCounter()
    assert c.count_text("") == 0
    assert c.count_text("x") == 1           # min 1 token for non-empty
    assert c.count_text("x" * 4) == 1
    assert c.count_text("x" * 16) == 4
    assert c.count_text("x" * 100) == 25


def test_char_based_count_messages_sums_content() -> None:
    c = CharBasedTokenCounter()
    msgs = [
        SystemMessage(role="system", content="x" * 12),  # 3 tokens
        UserMessage(role="user", content="x" * 20),      # 5 tokens
    ]
    assert c.count_messages(msgs) == 8


def test_char_based_handles_pydantic_and_dict_messages() -> None:
    c = CharBasedTokenCounter()
    msgs = [
        SystemMessage(role="system", content="x" * 12),
        {"role": "user", "content": "x" * 20},
    ]
    assert c.count_messages(msgs) == 8


def test_char_based_handles_multimodal_content() -> None:
    """Multipart content parts contribute their text fields."""
    c = CharBasedTokenCounter()
    msg = UserMessage(role="user", content=[
        {"type": "text", "text": "x" * 12},
        {"type": "image_url", "image_url": {"url": "http://x"}},
        {"type": "text", "text": "y" * 12},
    ])
    # Two text parts × 12 chars = 24 chars → 6 tokens
    assert c.count_messages([msg]) == 6


def test_char_based_ignores_model_param() -> None:
    """The heuristic is model-agnostic."""
    c = CharBasedTokenCounter()
    assert c.count_text("hello", model="claude-opus-4-7") == 1
    assert c.count_text("hello", model="gpt-5") == 1


# ---------------------------------------------------------------------------
# LiteLLMTokenCounter — uses real backend when present
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a fake `litellm` module in sys.modules for the test's duration."""
    fake = MagicMock()
    fake.token_counter = MagicMock()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    yield fake


def test_litellm_counter_delegates_to_token_counter(fake_litellm) -> None:
    fake_litellm.token_counter.return_value = 42
    counter = LiteLLMTokenCounter()
    n = counter.count_text("the quick brown fox", model="gpt-5")
    assert n == 42
    fake_litellm.token_counter.assert_called_once_with(model="gpt-5", text="the quick brown fox")


def test_litellm_counter_falls_back_when_model_missing(fake_litellm) -> None:
    """Without a model name, no litellm call is made; char-based is used."""
    counter = LiteLLMTokenCounter()
    n = counter.count_text("x" * 100, model=None)
    assert n == 25
    fake_litellm.token_counter.assert_not_called()


def test_litellm_counter_falls_back_on_error(fake_litellm) -> None:
    fake_litellm.token_counter.side_effect = RuntimeError("unknown model")
    counter = LiteLLMTokenCounter()
    n = counter.count_text("x" * 100, model="some-future-model")
    # Falls back to char-based silently
    assert n == 25


def test_litellm_counter_messages_uses_dicts(fake_litellm) -> None:
    fake_litellm.token_counter.return_value = 99
    counter = LiteLLMTokenCounter()
    msgs = [
        SystemMessage(role="system", content="hello"),
        UserMessage(role="user", content="world"),
    ]
    n = counter.count_messages(msgs, model="gpt-5")
    assert n == 99
    # Check that litellm got dicts, not pydantic objects
    call_kwargs = fake_litellm.token_counter.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5"
    assert isinstance(call_kwargs["messages"], list)
    assert all(isinstance(m, dict) for m in call_kwargs["messages"])


def test_litellm_counter_messages_falls_back_on_error(fake_litellm) -> None:
    fake_litellm.token_counter.side_effect = RuntimeError("nope")
    counter = LiteLLMTokenCounter()
    msgs = [UserMessage(role="user", content="x" * 100)]
    # Char-based fallback: 100 chars → 25 tokens
    assert counter.count_messages(msgs, model="some-model") == 25


def test_litellm_counter_import_error_when_unavailable(monkeypatch) -> None:
    """When `litellm` isn't installed, constructor raises ImportError."""
    monkeypatch.setitem(sys.modules, "litellm", None)
    # `None` in sys.modules makes the import fail
    with pytest.raises(ImportError, match="modelmeld\\[tokenizer\\]"):
        LiteLLMTokenCounter()


# ---------------------------------------------------------------------------
# build_token_counter — factory
# ---------------------------------------------------------------------------

def test_build_default_returns_char_based() -> None:
    counter = build_token_counter(_FakeSettings(backend="char"))
    assert isinstance(counter, CharBasedTokenCounter)
    assert counter.name == "char"


def test_build_litellm_falls_back_when_extra_not_installed(monkeypatch, caplog) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)
    with caplog.at_level("WARNING"):
        counter = build_token_counter(_FakeSettings(backend="litellm"))
    # Boot succeeds with char-based fallback + warning log
    assert isinstance(counter, CharBasedTokenCounter)
    assert any("falling back" in r.message.lower() or "char-based" in r.message.lower()
               for r in caplog.records)


def test_build_litellm_returns_litellm_counter_when_available(fake_litellm) -> None:
    counter = build_token_counter(_FakeSettings(backend="litellm"))
    assert isinstance(counter, LiteLLMTokenCounter)
    assert counter.name == "litellm"


class _FakeSettings:
    def __init__(self, backend: str) -> None:
        self.token_counter_backend = backend

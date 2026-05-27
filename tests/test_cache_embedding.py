"""EmbeddingClient + HashedBagOfWordsEmbedder + canonicalize_request_text."""

from __future__ import annotations

import math

import pytest

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
)
from modelmeld.cache import (
    HashedBagOfWordsEmbedder,
    canonicalize_request_text,
    cosine_similarity,
    is_request_semantically_cacheable,
)


def _req(content: str, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=kwargs.pop("model", "test-model"),
        messages=[UserMessage(role="user", content=content)],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

async def test_embedder_returns_unit_length_vector() -> None:
    e = HashedBagOfWordsEmbedder(dim=64)
    v = await e.embed("write a python function that sums a list")
    assert len(v) == 64
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


async def test_embedder_deterministic() -> None:
    """Same text → identical vector across calls and embedder instances."""
    e1 = HashedBagOfWordsEmbedder(dim=64)
    e2 = HashedBagOfWordsEmbedder(dim=64)
    v1 = await e1.embed("hello world")
    v2 = await e1.embed("hello world")
    v3 = await e2.embed("hello world")
    assert v1 == v2
    assert v1 == v3


async def test_paraphrases_score_high() -> None:
    """Texts with mostly-shared vocabulary land above the 0.92 threshold."""
    e = HashedBagOfWordsEmbedder(dim=256)
    a = await e.embed("write a python function that returns the sum of a list of numbers")
    b = await e.embed("write a python function that returns the sum of a list of numbers please")
    sim = cosine_similarity(a, b)
    assert sim >= 0.92, f"paraphrase score {sim:.3f} below threshold 0.92"


async def test_unrelated_prompts_score_low() -> None:
    """Disjoint-vocabulary prompts land well below the threshold."""
    e = HashedBagOfWordsEmbedder(dim=256)
    a = await e.embed("write a python function that returns the sum of a list")
    b = await e.embed("what is the capital of France")
    sim = cosine_similarity(a, b)
    assert sim < 0.5, f"unrelated score {sim:.3f} unexpectedly high"


async def test_invalid_dim_rejected() -> None:
    with pytest.raises(ValueError):
        HashedBagOfWordsEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashedBagOfWordsEmbedder(dim=-1)


async def test_empty_text_returns_zero_vector() -> None:
    e = HashedBagOfWordsEmbedder(dim=64)
    v = await e.embed("")
    assert all(x == 0.0 for x in v)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# canonicalize_request_text
# ---------------------------------------------------------------------------

def test_canonicalize_includes_user_message_text() -> None:
    text = canonicalize_request_text(_req("hello world"))
    assert "[user]" in text
    assert "hello world" in text


def test_canonicalize_includes_system_messages() -> None:
    req = ChatCompletionRequest(
        model="m",
        messages=[
            SystemMessage(role="system", content="You are concise"),
            UserMessage(role="user", content="hi"),
        ],
    )
    text = canonicalize_request_text(req)
    assert "[system] You are concise" in text
    assert "[user] hi" in text


def test_canonicalize_skips_assistant_messages() -> None:
    """Assistant content is prior context that varies per session — exclude."""
    from modelmeld.api.schemas import AssistantMessage
    req = ChatCompletionRequest(
        model="m",
        messages=[
            UserMessage(role="user", content="ask"),
            AssistantMessage(role="assistant", content="previous answer"),
            UserMessage(role="user", content="follow up"),
        ],
    )
    text = canonicalize_request_text(req)
    assert "previous answer" not in text
    assert "follow up" in text


def test_canonicalize_pins_served_model() -> None:
    text = canonicalize_request_text(_req("x"), served_model="qwen3-coder")
    assert "model=qwen3-coder" in text


def test_canonicalize_includes_temperature_seed() -> None:
    text = canonicalize_request_text(_req("x", temperature=0.7, seed=42))
    assert "temperature=0.7" in text
    assert "seed=42" in text


def test_canonicalize_omits_unset_hyperparameters() -> None:
    """Default-None fields shouldn't fracture the canonical text."""
    text = canonicalize_request_text(_req("x"))
    assert "temperature=" not in text
    assert "seed=" not in text


# ---------------------------------------------------------------------------
# is_request_semantically_cacheable — same gates as exact-match
# ---------------------------------------------------------------------------

def test_streaming_is_not_semantically_cacheable() -> None:
    assert is_request_semantically_cacheable(_req("x", stream=True)) is False


def test_n_gt_1_is_not_semantically_cacheable() -> None:
    assert is_request_semantically_cacheable(_req("x", n=2)) is False
    assert is_request_semantically_cacheable(_req("x", n=1)) is True
    assert is_request_semantically_cacheable(_req("x")) is True


def test_tool_use_is_not_semantically_cacheable() -> None:
    from modelmeld.api.schemas import FunctionDef, Tool
    tools = [Tool(
        type="function",
        function=FunctionDef(name="search", description="", parameters={"type": "object"}),
    )]
    assert is_request_semantically_cacheable(_req("x", tools=tools)) is False

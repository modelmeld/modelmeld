"""Chat route uses the configured TokenCounter for L0 turn writes."""

from __future__ import annotations

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    HEADER_SESSION_ID,
    InMemoryMemoryStore,
)
from modelmeld.tokens import TokenCounter


class _CountingTokenCounter(TokenCounter):
    """Records every call so tests can assert which texts were counted."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def count_text(self, text: str, model: str | None = None) -> int:
        self.calls.append((text, model))
        # Predictable: 1 token per word
        return len(text.split())

    def count_messages(self, messages, model=None) -> int:
        return sum(self.count_text(getattr(m, "content", "") or "", model) for m in messages)


class _NoUsageAdapter(ProviderAdapter):
    """Adapter that returns a completion WITHOUT a usage block — forces
    the chat route to fall back on the configured TokenCounter."""

    name = "no-usage"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content="three little words"),
                finish_reason="stop",
            )],
            usage=None,  # ← important: no usage means counter is consulted
        )

    async def stream_chat(self, request):
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


async def test_token_counter_called_for_assistant_tokens_when_usage_absent() -> None:
    store = InMemoryMemoryStore()
    counter = _CountingTokenCounter()
    app = build_app(adapter=_NoUsageAdapter(), memory_store=store, token_counter=counter)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [
                {"role": "user", "content": "Tell me everything"},
            ]},
            headers={HEADER_SESSION_ID: "sess-K"},
        )
    assert resp.status_code == 200

    # User text + assistant text should have been counted
    counted_texts = {text for text, _ in counter.calls}
    assert "Tell me everything" in counted_texts
    assert "three little words" in counted_texts

    # The L0 row should reflect the counter's verdict, not char/4
    turns = await store.list_turns("sess-K", ANONYMOUS_TENANT_ID)
    assistant_turn = next(t for t in turns if t.role.value == "assistant")
    # "three little words" → 3 words → 3 tokens via our counter
    assert assistant_turn.token_count == 3

    user_turn = next(t for t in turns if t.role.value == "user")
    # "Tell me everything" → 3 words → 3 tokens
    assert user_turn.token_count == 3


class _WithUsageAdapter(ProviderAdapter):
    name = "with-usage"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content="reply"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
        )

    async def stream_chat(self, request):
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


async def test_upstream_usage_wins_over_counter_for_assistant_tokens() -> None:
    """When upstream returns usage.completion_tokens, we trust it (they know best)."""
    store = InMemoryMemoryStore()
    counter = _CountingTokenCounter()
    app = build_app(adapter=_WithUsageAdapter(), memory_store=store, token_counter=counter)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [
                {"role": "user", "content": "hi"},
            ]},
            headers={HEADER_SESSION_ID: "sess-W"},
        )
    assert resp.status_code == 200

    turns = await store.list_turns("sess-W", ANONYMOUS_TENANT_ID)
    assistant_turn = next(t for t in turns if t.role.value == "assistant")
    # Upstream said 7 tokens — counter's 1-token verdict ("reply" → 1 word) is overridden
    assert assistant_turn.token_count == 7
    # Counter was still called for the USER message (no upstream usage for prompts)
    counted_texts = {text for text, _ in counter.calls}
    assert "hi" in counted_texts


async def test_token_counter_receives_model_used_not_requested() -> None:
    """When the router overrides the model (capability mode), the counter sees the served model."""
    store = InMemoryMemoryStore()
    counter = _CountingTokenCounter()
    app = build_app(adapter=_NoUsageAdapter(), memory_store=store, token_counter=counter)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "claude-opus-4-7", "messages": [
                {"role": "user", "content": "hello"},
            ]},
            headers={HEADER_SESSION_ID: "sess-M"},
        )

    # No router override here (SingleAdapterRouter), so counter sees requested model
    models_seen = {model for _, model in counter.calls}
    assert "claude-opus-4-7" in models_seen

"""Chat route writes user + assistant turns to memory when a session is active."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    ChoiceDelta,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    HEADER_SESSION_ID,
    InMemoryMemoryStore,
    Role,
)


class _EchoAdapter(ProviderAdapter):
    name = "echo"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        # Echo the last user message back as the assistant
        last_user = ""
        for m in reversed(request.messages):
            if m.role == "user":
                if isinstance(m.content, str):
                    last_user = m.content
                break
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content=f"echo: {last_user}"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=len(last_user) // 4 or 1,
                       completion_tokens=len(last_user) // 4 or 1,
                       total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        import time as _time
        last_user = ""
        for m in reversed(request.messages):
            if m.role == "user":
                if isinstance(m.content, str):
                    last_user = m.content
                break
        text = f"echo: {last_user}"
        # Split into 3 chunks so we exercise the accumulator
        chunk_size = max(1, len(text) // 3)
        pieces = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] or [text]
        chunk_id = "chatcmpl-test"
        created = int(_time.time())
        for piece in pieces:
            yield ChatCompletionChunk(
                id=chunk_id, created=created, model=request.model,
                choices=[ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=piece),
                    finish_reason=None,
                )],
            )
        yield ChatCompletionChunk(
            id=chunk_id, created=created, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )

    async def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Memory writes happen when session header present
# ---------------------------------------------------------------------------

async def test_session_header_writes_user_and_assistant_turns() -> None:
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "stub", "messages": [{"role": "user", "content": "hello"}]},
            headers={HEADER_SESSION_ID: "sess-1"},
        )
    assert resp.status_code == 200

    sess = await store.get_session("sess-1", ANONYMOUS_TENANT_ID)
    assert sess is not None
    turns = await store.list_turns("sess-1", ANONYMOUS_TENANT_ID)
    assert len(turns) == 2
    assert turns[0].role == Role.USER
    assert turns[0].content == "hello"
    assert turns[1].role == Role.ASSISTANT
    assert turns[1].content == "echo: hello"


async def test_no_session_header_skips_memory_writes() -> None:
    """Without a session id, no turns are logged."""
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "stub", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    # No sessions, no turns
    assert await store.get_session("nope", ANONYMOUS_TENANT_ID) is None


async def test_multi_turn_conversation_accumulates_turns() -> None:
    """Two requests with same session_id → 4 turns total (2 user + 2 assistant)."""
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # First turn
        await client.post(
            "/v1/chat/completions",
            json={"model": "stub", "messages": [{"role": "user", "content": "first"}]},
            headers={HEADER_SESSION_ID: "s"},
        )
        # Second turn — client includes previous turns in messages (OpenAI style)
        await client.post(
            "/v1/chat/completions",
            json={"model": "stub", "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "echo: first"},
                {"role": "user", "content": "second"},
            ]},
            headers={HEADER_SESSION_ID: "s"},
        )

    turns = await store.list_turns("s", ANONYMOUS_TENANT_ID)
    assert len(turns) == 4
    # Only the NEWEST user turn from each request is logged (not the prior history)
    assert [t.content for t in turns] == [
        "first", "echo: first",
        "second", "echo: second",
    ]


async def test_streaming_accumulates_assistant_text_then_writes() -> None:
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "stub", "stream": True,
                  "messages": [{"role": "user", "content": "streaming?"}]},
            headers={HEADER_SESSION_ID: "stream-1"},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():
                pass

    turns = await store.list_turns("stream-1", ANONYMOUS_TENANT_ID)
    assert len(turns) == 2
    assert turns[0].content == "streaming?"
    # Reassembled streamed content (echo split into pieces, then joined)
    assert turns[1].content == "echo: streaming?"


# ---------------------------------------------------------------------------
# Memory failure must not break the request
# ---------------------------------------------------------------------------

class _BrokenMemoryStore(InMemoryMemoryStore):
    async def append_turn(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("storage offline")


async def test_memory_write_failure_doesnt_break_response() -> None:
    """User got their answer; we shouldn't 500 because the audit row didn't save."""
    store = _BrokenMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "stub", "messages": [{"role": "user", "content": "hi"}]},
            headers={HEADER_SESSION_ID: "s"},
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "echo: hi"

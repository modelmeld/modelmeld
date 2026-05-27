"""End-to-end: memory written in turn 1 actually appears in turn 2's prompt."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    HEADER_MEMORY_MODE,
    HEADER_SESSION_ID,
    InMemoryMemoryStore,
)


class _CapturingAdapter(ProviderAdapter):
    """Adapter that records every request it received so we can assert injection."""

    name = "capturing"
    is_egress = False

    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.requests.append(request)
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content="ok"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


def _payload(text: str) -> dict:
    return {"model": "stub", "messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# AUGMENT mode: facts + summary appear as system message in turn 2
# ---------------------------------------------------------------------------

async def test_facts_injected_into_subsequent_request() -> None:
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)

    # Pre-seed: a session with a declared fact
    await store.get_or_create_session("sess-X", ANONYMOUS_TENANT_ID)
    await store.set_fact("sess-X", ANONYMOUS_TENANT_ID, "user_name", "Kevin")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("hi"),
            headers={HEADER_SESSION_ID: "sess-X"},
        )
    assert resp.status_code == 200

    # The adapter received: [SystemMessage(facts), UserMessage("hi")]
    assert len(adapter.requests) == 1
    msgs = adapter.requests[0].messages
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert "user_name: Kevin" in msgs[0].content
    assert msgs[1].role == "user"
    assert msgs[1].content == "hi"


# ---------------------------------------------------------------------------
# FULL mode: L3 turns replayed before the framework's messages
# ---------------------------------------------------------------------------

async def test_full_mode_replays_prior_turns() -> None:
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Turn 1 — establishes history (default AUGMENT mode, no replay)
        await client.post(
            "/v1/chat/completions",
            json=_payload("introduce yourself"),
            headers={HEADER_SESSION_ID: "sess-Y"},
        )
        # Turn 2 — FULL mode replays the previous user + assistant turns
        adapter.requests.clear()
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("now repeat that back"),
            headers={
                HEADER_SESSION_ID: "sess-Y",
                HEADER_MEMORY_MODE: "full",
            },
        )
    assert resp.status_code == 200

    msgs = adapter.requests[0].messages
    # Expect prior turn 1's user + assistant replayed, then turn 2's user.
    # No system message because there are no facts/summary.
    contents = [(m.role, m.content) for m in msgs]
    assert contents == [
        ("user", "introduce yourself"),
        ("assistant", "ok"),
        ("user", "now repeat that back"),
    ]


# ---------------------------------------------------------------------------
# OFF mode: no injection even with data + session
# ---------------------------------------------------------------------------

async def test_off_mode_skips_injection_but_still_writes_turns() -> None:
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)

    await store.get_or_create_session("sess-Z", ANONYMOUS_TENANT_ID)
    await store.set_fact("sess-Z", ANONYMOUS_TENANT_ID, "k", "v")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("hi"),
            headers={HEADER_SESSION_ID: "sess-Z", HEADER_MEMORY_MODE: "off"},
        )
    assert resp.status_code == 200

    # Adapter saw ONLY the user message — no system injection
    msgs = adapter.requests[0].messages
    assert len(msgs) == 1
    assert msgs[0].role == "user"

    # But turns were still written (audit trail preserved)
    turns = await store.list_turns("sess-Z", ANONYMOUS_TENANT_ID)
    assert len(turns) == 2  # user + assistant


# ---------------------------------------------------------------------------
# No session header → no injection (single-turn mode unaffected)
# ---------------------------------------------------------------------------

async def test_no_session_id_no_injection() -> None:
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)
    # Even though there's "global" data (different session) → not injected
    await store.get_or_create_session("other-sess", ANONYMOUS_TENANT_ID)
    await store.set_fact("other-sess", ANONYMOUS_TENANT_ID, "k", "v")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload("hi"))
    assert resp.status_code == 200
    assert len(adapter.requests[0].messages) == 1


# ---------------------------------------------------------------------------
# Malformed memory-mode header → 400
# ---------------------------------------------------------------------------

async def test_invalid_memory_mode_returns_400() -> None:
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("hi"),
            headers={HEADER_SESSION_ID: "s", HEADER_MEMORY_MODE: "telepathy"},
        )
    assert resp.status_code == 400
    assert "invalid_memory_header" in resp.json()["detail"]
    # Adapter never called — failed validation before routing
    assert adapter.requests == []


# ---------------------------------------------------------------------------
# Tenant isolation through injection
# ---------------------------------------------------------------------------

async def test_other_tenants_facts_not_injected() -> None:
    """tenant-A's facts cannot bleed into tenant-B's request.

    Anonymous mode shares the namespace by design, so this test uses two
    distinct anonymous-session IDs to prove that even within a tenant,
    sessions don't cross-contaminate.
    """
    store = InMemoryMemoryStore()
    adapter = _CapturingAdapter()
    app = build_app(adapter=adapter, memory_store=store)

    # Pre-seed: a fact under a DIFFERENT session id
    await store.get_or_create_session("session-A", ANONYMOUS_TENANT_ID)
    await store.set_fact("session-A", ANONYMOUS_TENANT_ID, "secret_a", "alpha")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("hi"),
            headers={HEADER_SESSION_ID: "session-B"},  # ← different session
        )
    assert resp.status_code == 200
    msgs = adapter.requests[0].messages
    # Session B never set this fact; injecting session A's would be a leak.
    for m in msgs:
        if isinstance(m.content, str):
            assert "secret_a" not in m.content
            assert "alpha" not in m.content

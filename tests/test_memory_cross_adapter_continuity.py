"""Memory context flows across adapter switches.

The active-tiered-memory pivot named cross-model conversation continuity
as the moat. Existing tests cover memory injection with a *single*
adapter (test_chat_memory_injection.py) and write-back through the chat
route (test_chat_memory_integration.py), but NONE verify the case
where the moat actually fires: a multi-turn conversation that gets
served by *different adapters* across turns. That's what this file
locks in.

Specifically:

  - AUGMENT mode: pre-seeded facts must be injected as a system message
    on every turn regardless of which adapter the scout routed to
  - FULL mode: L3 turns written in turn N must be REPLAYED into the
    request to whichever adapter serves turn N+1, even when that
    adapter is different from the one that served turn N
  - Failover within a single turn (CLOUD → LOCAL via TieredRouter's
    transient-error failover) must preserve the memory injection on
    the fallback adapter — the customer must not lose context just
    because the primary tier had a blip

The router used is `TieredRouter` because that's what production scout-
driven deployments use. The scout used is `_AlternatingScout` so we get
deterministic adapter alternation per turn without depending on the
heuristic correctly classifying handcrafted prompts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from modelmeld.adapters.base import (
    AdapterError,
    ProviderAdapter,
    TransientAdapterError,
)
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
from modelmeld.router import RoutingPolicy, TieredRouter
from modelmeld.scout.base import Scout, ScoutDecision, Tier

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CapturingAdapter(ProviderAdapter):
    """Records every request received + returns a response that identifies
    which adapter served it. Lets tests assert (a) which adapter served
    each turn and (b) what context that adapter received."""

    is_egress = False

    def __init__(self, name: str, *, fail_with: AdapterError | None = None) -> None:
        self._name = name
        self._fail_with = fail_with
        # Snapshot of each request received. Deep-copy via model_copy so the
        # captured value isn't mutated by downstream code.
        self.requests: list[ChatCompletionRequest] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.requests.append(request.model_copy(deep=True))
        if self._fail_with is not None:
            raise self._fail_with
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content=f"{self._name}-response"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


class _AlternatingScout(Scout):
    """Returns LOCAL, then CLOUD, then LOCAL, then CLOUD, ..."""

    name = "alternating"

    def __init__(self, start_tier: Tier = Tier.LOCAL) -> None:
        self._next_tier = start_tier

    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        tier = self._next_tier
        self._next_tier = Tier.CLOUD if tier == Tier.LOCAL else Tier.LOCAL
        return ScoutDecision(
            tier=tier,
            confidence=1.0,
            rationale="alternating-test-scout",
            signals={},
        )


def _payload(text: str) -> dict:
    return {"model": "stub", "messages": [{"role": "user", "content": text}]}


def _build_routed_app(
    store: InMemoryMemoryStore,
    local: _CapturingAdapter,
    cloud: _CapturingAdapter,
    scout: Scout | None = None,
):
    """Build a FastAPI app with a real TieredRouter wiring two adapters."""
    router = TieredRouter(
        scout=scout or _AlternatingScout(start_tier=Tier.LOCAL),
        adapters={Tier.LOCAL: local, Tier.CLOUD: cloud},
        policy=RoutingPolicy.SCOUT_DRIVEN,
    )
    return build_app(router=router, memory_store=store)


def _system_message_text(req: ChatCompletionRequest) -> str:
    """Concatenated text of every system message in the request."""
    out: list[str] = []
    for m in req.messages:
        if m.role == "system":
            content = m.content
            if isinstance(content, str):
                out.append(content)
            else:
                out.extend(p.text for p in content if hasattr(p, "text"))
    return "\n".join(out)


def _all_text(req: ChatCompletionRequest) -> str:
    """All textual content across all messages in the request."""
    out: list[str] = []
    for m in req.messages:
        content = m.content
        if isinstance(content, str):
            out.append(content)
        elif content is not None:
            out.extend(p.text for p in content if hasattr(p, "text"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# AUGMENT mode — pre-seeded facts injected on every turn regardless of adapter
# ---------------------------------------------------------------------------


async def test_facts_injected_on_both_local_and_cloud_adapters() -> None:
    """The moat: pre-seeded facts in the session must appear in the system
    message on every turn, no matter which adapter served the turn."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    # Pre-seed: a fact known to the gateway.
    await store.get_or_create_session("sess-1", ANONYMOUS_TENANT_ID)
    await store.set_fact("sess-1", ANONYMOUS_TENANT_ID, "user_name", "Alice")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # Three turns. Scout alternates LOCAL → CLOUD → LOCAL.
        await client.post(
            "/v1/chat/completions", json=_payload("turn 1"),
            headers={HEADER_SESSION_ID: "sess-1"},
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn 2"),
            headers={HEADER_SESSION_ID: "sess-1"},
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn 3"),
            headers={HEADER_SESSION_ID: "sess-1"},
        )

    # LOCAL served turns 1 + 3; CLOUD served turn 2.
    assert len(local.requests) == 2
    assert len(cloud.requests) == 1

    # Every one of those requests must contain the Alice fact in the
    # system-message system message.
    for req in (*local.requests, *cloud.requests):
        sys_text = _system_message_text(req)
        assert "user_name: Alice" in sys_text, (
            f"adapter {req.model} received no Alice fact:\n"
            f"system text: {sys_text!r}"
        )


async def test_facts_set_in_turn_1_visible_in_turn_2_on_different_adapter() -> None:
    """Set a fact *during* a turn served by LOCAL; verify it's injected
    on the next turn served by CLOUD. Catches the case where memory
    write-back lands in the store but the next read doesn't see it."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    # Turn 1 routes to LOCAL via the alternating scout. During the request,
    # we directly write a fact (simulating a future "summarizer extracted
    # this from turn 1" path). The next turn routes to CLOUD.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/v1/chat/completions", json=_payload("introduce alice"),
            headers={HEADER_SESSION_ID: "sess-mid"},
        )
        # Set the fact after turn 1 completed (so it's in the store
        # before turn 2 fires the context build).
        await store.set_fact(
            "sess-mid", ANONYMOUS_TENANT_ID, "topic", "alice queue",
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn 2"),
            headers={HEADER_SESSION_ID: "sess-mid"},
        )

    # Turn 2 served by CLOUD must see the fact set after turn 1.
    assert len(cloud.requests) == 1
    sys_text = _system_message_text(cloud.requests[0])
    assert "topic: alice queue" in sys_text


# ---------------------------------------------------------------------------
# FULL mode — L3 turns from turn N replay into turn N+1 served by another
# adapter
# ---------------------------------------------------------------------------


async def test_full_mode_replays_prior_turn_across_adapter_switch() -> None:
    """The high-stakes case: in FULL mode the gateway should replay
    recent verbatim turns from L3. When turn 1 lands at LOCAL and turn 2
    lands at CLOUD, the CLOUD adapter MUST receive the prior user-message
    and the prior assistant-response as message history before its own
    new turn. This is the cross-model continuity primitive."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    full_headers = {
        HEADER_SESSION_ID: "sess-full",
        HEADER_MEMORY_MODE: "full",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # Turn 1 → LOCAL. The user says something memorable.
        await client.post(
            "/v1/chat/completions",
            json=_payload("my favorite color is cerulean"),
            headers=full_headers,
        )
        # Turn 2 → CLOUD. The user asks a follow-up that requires turn 1
        # context. Without memory injection, CLOUD would have no idea.
        await client.post(
            "/v1/chat/completions", json=_payload("what was the color?"),
            headers=full_headers,
        )

    assert len(local.requests) == 1
    assert len(cloud.requests) == 1

    # CLOUD's request must include the prior user message ("cerulean")
    # AND the prior assistant response ("local-response") as replay.
    cloud_all_text = _all_text(cloud.requests[0])
    assert "cerulean" in cloud_all_text, (
        f"CLOUD adapter didn't receive prior user message:\n{cloud_all_text!r}"
    )
    assert "local-response" in cloud_all_text, (
        f"CLOUD adapter didn't receive prior assistant message:\n"
        f"{cloud_all_text!r}"
    )


async def test_full_mode_four_turn_back_and_forth_preserves_context() -> None:
    """Longer cross-adapter conversation. Each turn switches adapter and
    each turn's adapter sees the accumulated L3 history."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    full_headers = {
        HEADER_SESSION_ID: "sess-4",
        HEADER_MEMORY_MODE: "full",
    }
    prompts = [
        "marker-alpha-1",   # turn 1 → LOCAL
        "marker-bravo-2",   # turn 2 → CLOUD
        "marker-charlie-3", # turn 3 → LOCAL
        "marker-delta-4",   # turn 4 → CLOUD
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        for prompt in prompts:
            await client.post(
                "/v1/chat/completions", json=_payload(prompt),
                headers=full_headers,
            )

    # Each adapter served 2 turns.
    assert len(local.requests) == 2
    assert len(cloud.requests) == 2

    # Turn 4 (the last CLOUD request) should have all prior user prompts
    # (alpha, bravo, charlie) replayed via L3 — plus its own delta prompt.
    turn4 = cloud.requests[1]
    text4 = _all_text(turn4)
    for prior_marker in ("marker-alpha-1", "marker-bravo-2", "marker-charlie-3"):
        assert prior_marker in text4, (
            f"Turn 4 (CLOUD) missing prior user marker {prior_marker!r}:\n"
            f"received text:\n{text4!r}"
        )
    # And its own new prompt
    assert "marker-delta-4" in text4


# ---------------------------------------------------------------------------
# OFF mode — turns get written to memory but no context injection happens
# ---------------------------------------------------------------------------


async def test_off_mode_writes_turns_but_does_not_inject_across_adapters() -> None:
    """Verifies the OFF escape hatch behaves the same regardless of
    routing — turns get logged (so audit + future analysis works) but
    no system message gets injected on the next adapter."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    await store.get_or_create_session("sess-off", ANONYMOUS_TENANT_ID)
    await store.set_fact("sess-off", ANONYMOUS_TENANT_ID, "user_name", "Alice")

    headers = {HEADER_SESSION_ID: "sess-off", HEADER_MEMORY_MODE: "off"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/v1/chat/completions", json=_payload("turn 1"), headers=headers,
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn 2"), headers=headers,
        )

    # Neither adapter should have seen the Alice fact.
    for req in (*local.requests, *cloud.requests):
        sys_text = _system_message_text(req)
        assert "Alice" not in sys_text, (
            "OFF mode injected a fact that should have been suppressed"
        )

    # But turns were still recorded.
    turns = await store.list_turns("sess-off", ANONYMOUS_TENANT_ID)
    assert len(turns) == 4  # 2 user + 2 assistant


# ---------------------------------------------------------------------------
# Failover within a single turn — memory still flows to the fallback
# ---------------------------------------------------------------------------


async def test_failover_to_local_still_injects_memory_context() -> None:
    """When CLOUD raises a TransientAdapterError and TieredRouter fails
    over to LOCAL, the LOCAL adapter must still receive the memory-
    injected context. Otherwise, customers lose continuity precisely
    when they need it most (during a provider outage)."""
    store = InMemoryMemoryStore()
    # CLOUD always fails with a transient error
    cloud = _CapturingAdapter(
        "cloud",
        fail_with=TransientAdapterError("simulated 529 overloaded"),
    )
    local = _CapturingAdapter("local")

    # Force the scout to ALWAYS route to CLOUD first; failover will then
    # try LOCAL. This way we know the fallback path is exercised, not
    # natural LOCAL routing.
    class _AlwaysCloudScout(Scout):
        name = "always-cloud"

        async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
            return ScoutDecision(
                tier=Tier.CLOUD, confidence=1.0,
                rationale="always-cloud-test-scout", signals={},
            )

    app = _build_routed_app(store, local, cloud, scout=_AlwaysCloudScout())

    await store.get_or_create_session("sess-fail", ANONYMOUS_TENANT_ID)
    await store.set_fact(
        "sess-fail", ANONYMOUS_TENANT_ID, "user_name", "Alice",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("hello"),
            headers={HEADER_SESSION_ID: "sess-fail"},
        )
    assert resp.status_code == 200

    # CLOUD was tried first (and failed); LOCAL served via failover.
    assert len(cloud.requests) == 1
    assert len(local.requests) == 1

    # Critical: the LOCAL fallback received the SAME memory-injected
    # context CLOUD did — Alice fact in the system message.
    local_sys = _system_message_text(local.requests[0])
    assert "user_name: Alice" in local_sys, (
        f"Failover lost the memory context. LOCAL system text:\n{local_sys!r}"
    )


# ---------------------------------------------------------------------------
# Session isolation — Alice's session never sees Bob's session, even when
# both ride the same alternating adapter pattern.
# ---------------------------------------------------------------------------


async def test_two_sessions_alternating_adapters_dont_cross_contaminate() -> None:
    """Sessions A and B in the same tenant, interleaved. Adapters
    capture every request — assert no session-B context ever appears in
    a session-A request and vice versa."""
    store = InMemoryMemoryStore()
    local = _CapturingAdapter("local")
    cloud = _CapturingAdapter("cloud")
    app = _build_routed_app(store, local, cloud)

    await store.get_or_create_session("sess-A", ANONYMOUS_TENANT_ID)
    await store.get_or_create_session("sess-B", ANONYMOUS_TENANT_ID)
    await store.set_fact(
        "sess-A", ANONYMOUS_TENANT_ID, "user_name", "Alice",
    )
    await store.set_fact("sess-B", ANONYMOUS_TENANT_ID, "user_name", "Bob")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post(
            "/v1/chat/completions", json=_payload("turn A1"),
            headers={HEADER_SESSION_ID: "sess-A"},
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn B1"),
            headers={HEADER_SESSION_ID: "sess-B"},
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn A2"),
            headers={HEADER_SESSION_ID: "sess-A"},
        )
        await client.post(
            "/v1/chat/completions", json=_payload("turn B2"),
            headers={HEADER_SESSION_ID: "sess-B"},
        )

    # Map each captured request to its session by inspecting the user
    # message marker. Then assert no session-X request carries session-Y's
    # fact.
    all_requests = local.requests + cloud.requests
    for req in all_requests:
        text = _all_text(req)
        sys_text = _system_message_text(req)
        if "turn A" in text:
            # Session A → must have Alice, must NOT have Bob.
            assert "Alice" in sys_text, f"sess-A turn missing Alice:\n{sys_text!r}"
            assert "Bob" not in sys_text, f"sess-A turn leaked Bob:\n{sys_text!r}"
        elif "turn B" in text:
            assert "Bob" in sys_text, f"sess-B turn missing Bob:\n{sys_text!r}"
            assert "Alice" not in sys_text, f"sess-B turn leaked Alice:\n{sys_text!r}"
        else:  # pragma: no cover
            pytest.fail(f"unexpected request, can't classify session: {text!r}")

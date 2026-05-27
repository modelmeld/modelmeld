# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Integration tests for the /v1/messages Anthropic-compatible route
(non-streaming path).

Uses httpx.ASGITransport against the in-process FastAPI app with a
mock adapter (no real backend calls). Validates that:

  - Anthropic-shape requests get translated, routed, and serialized
    back into Anthropic shape end-to-end
  - The same x-modelmeld-* response headers as /v1/chat/completions
    are emitted (D-4)
  - Memory write-back works with session headers
  - Tool-use round-trips through the route
  - Error cases (missing max_tokens, image blocks, stream=true) hit
    the right HTTP status codes
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    HEADER_SESSION_ID,
    InMemoryMemoryStore,
    Role,
)

# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------

class _EchoAdapter(ProviderAdapter):
    """Echoes the last user message as the assistant. No tool support."""
    name = "mock-echo"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
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
            usage=Usage(
                prompt_tokens=12, completion_tokens=8, total_tokens=20,
            ),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        # Not exercised in chunk 5 (route returns 501 for streaming).
        if False:
            yield  # pragma: no cover

    async def health(self) -> bool:
        return True


class _ToolCallingAdapter(ProviderAdapter):
    """When the request has `tools`, returns a tool_call. Otherwise plain text."""
    name = "mock-tooluse"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        if request.tools:
            # Choose the first tool and call it with a canned input
            tool = request.tools[0]
            return ChatCompletion(
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=ResponseMessage(
                        content="Calling tool.",
                        tool_calls=[ToolCall(
                            id="tu_test_1",
                            type="function",
                            function=FunctionCall(
                                name=tool.function.name,
                                arguments=json.dumps({"q": "test"}),
                            ),
                        )],
                    ),
                    finish_reason="tool_calls",
                )],
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            )
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content="no-tools"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:
            yield  # pragma: no cover

    async def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------

async def test_plain_request_returns_anthropic_shape() -> None:
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hello"}],
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["id"].startswith("msg_")
    assert body["stop_reason"] == "end_turn"
    # echo: hello
    assert len(body["content"]) == 1
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "echo: hello"
    # Usage translated
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 8}


async def test_model_field_echoes_request_not_internal_completion_model() -> None:
    """Capability routing may rewrite the internal model; the client must
    see the model they asked for."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-6"


async def test_system_prompt_is_visible_to_adapter() -> None:
    """The Anthropic top-level `system` field should land in the internal
    request as a leading SystemMessage and be observable downstream."""
    captured: dict = {}

    class _SystemCapturingAdapter(ProviderAdapter):
        name = "mock-syscap"
        is_egress = False

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            captured["messages"] = request.messages
            return ChatCompletion(
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=ResponseMessage(content="ok"),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        async def stream_chat(self, request):  # type: ignore[no-untyped-def]
            if False: yield  # pragma: no cover

        async def health(self) -> bool: return True

    app = build_app(adapter=_SystemCapturingAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 8,
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "x"}],
        })

    msgs = captured["messages"]
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[0].content == "You are concise."


async def test_multi_turn_request_preserves_conversation_history() -> None:
    captured: dict = {}

    class _CapAdapter(ProviderAdapter):
        name = "mock-capture"
        is_egress = False

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            captured["messages"] = request.messages
            return ChatCompletion(
                model=request.model,
                choices=[Choice(
                    index=0, message=ResponseMessage(content="ack"),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        async def stream_chat(self, request):  # type: ignore[no-untyped-def]
            if False: yield  # pragma: no cover

        async def health(self) -> bool: return True

    app = build_app(adapter=_CapAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first-reply"},
                {"role": "user", "content": "second"},
            ],
        })

    msgs = captured["messages"]
    # No system → exactly 3 messages
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0].content == "first"
    assert msgs[1].content == "first-reply"
    assert msgs[2].content == "second"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

async def test_missing_max_tokens_returns_422() -> None:
    """D-1: max_tokens is required by Pydantic schema. FastAPI maps Pydantic
    validation errors to 422."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 422


async def test_image_content_block_returns_400_translation_error() -> None:
    """Per v1 scope: image content blocks deferred. Translation raises
    TranslationError → 400 with clear detail."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "url", "url": "https://example.com/cat.png"}},
                ],
            }],
        })
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "translation_error" in detail
    assert "image" in detail.lower()


async def test_stream_true_returns_text_event_stream() -> None:
    """Stream=true returns 200 with content-type: text/event-stream.
    (Chunk 6 wired streaming; full SSE-format coverage lives in
    test_route_messages_streaming.py.)"""
    # _EchoAdapter has no stream_chat impl that yields anything, so use a
    # tiny inline streaming adapter to keep this assertion meaningful.
    class _MiniStream(ProviderAdapter):
        name = "mini-stream"
        is_egress = False
        async def chat(self, request):  # type: ignore[no-untyped-def]
            raise NotImplementedError
        async def stream_chat(self, request):  # type: ignore[no-untyped-def]
            from modelmeld.api.schemas import (
                ChatCompletionChunk as _C,
            )
            from modelmeld.api.schemas import (
                ChoiceDelta as _CD,
            )
            from modelmeld.api.schemas import (
                ChunkChoice as _CC,
            )
            from modelmeld.api.schemas import (
                Usage as _U,
            )
            yield _C(id="c1", object="chat.completion.chunk", created=1,
                    model=request.model,
                    choices=[_CC(index=0, delta=_CD(role="assistant", content="x"))])
            yield _C(id="c1", object="chat.completion.chunk", created=1,
                    model=request.model,
                    choices=[_CC(index=0, delta=_CD(), finish_reason="stop")],
                    usage=_U(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        async def health(self) -> bool:
            return True

    app = build_app(adapter=_MiniStream())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client, client.stream("POST", "/v1/messages", json={
        "model": "m",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "x"}],
    }) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Drain so the connection closes cleanly
        async for _ in resp.aiter_text():
            pass


# ---------------------------------------------------------------------------
# Tool-use round-trip
# ---------------------------------------------------------------------------

async def test_request_with_tools_round_trips_to_tool_use_block() -> None:
    """End-to-end: Anthropic tool def → internal Tool → adapter returns
    tool_calls → translated back to Anthropic tool_use block."""
    app = build_app(adapter=_ToolCallingAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "search for stuff"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            ],
        })

    assert resp.status_code == 200
    body = resp.json()
    # Two content blocks: a leading text + the tool_use
    assert len(body["content"]) == 2
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Calling tool."
    tool_use = body["content"][1]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "tu_test_1"
    assert tool_use["name"] == "search"
    assert tool_use["input"] == {"q": "test"}
    assert body["stop_reason"] == "tool_use"


async def test_subsequent_request_with_tool_result_translates_correctly() -> None:
    """The next round-trip: client returns the tool_result; gateway
    translates to a ToolMessage in the internal pipeline."""
    captured: dict = {}

    class _Cap(ProviderAdapter):
        name = "mock-cap"
        is_egress = False

        async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
            captured["messages"] = request.messages
            return ChatCompletion(
                model=request.model,
                choices=[Choice(
                    index=0, message=ResponseMessage(content="Got it."),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            )

        async def stream_chat(self, request):  # type: ignore[no-untyped-def]
            if False: yield  # pragma: no cover

        async def health(self) -> bool: return True

    app = build_app(adapter=_Cap())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "search for stuff"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Calling tool."},
                    {"type": "tool_use", "id": "tu_1",
                     "name": "search", "input": {"q": "test"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": "results: 3 hits"},
                ]},
            ],
        })

    assert resp.status_code == 200
    msgs = captured["messages"]
    # user, assistant (with tool_calls), tool, [nothing else — the second
    # user/text was empty so no extra UserMessage]
    roles = [m.role for m in msgs]
    assert roles == ["user", "assistant", "tool"]
    # The tool message carries the result content
    tool_msg = msgs[2]
    assert tool_msg.tool_call_id == "tu_1"
    assert tool_msg.content == "results: 3 hits"


# ---------------------------------------------------------------------------
# Headers + observability
# ---------------------------------------------------------------------------

async def test_routing_headers_appear_on_response() -> None:
    """Same x-modelmeld-* response headers as the chat route (D-4)."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 200
    # Routing headers consistent with /v1/chat/completions
    assert "x-modelmeld-routed-to" in resp.headers
    assert "x-modelmeld-tier" in resp.headers


# ---------------------------------------------------------------------------
# Memory write-back
# ---------------------------------------------------------------------------

async def test_session_header_triggers_memory_writes() -> None:
    """Same x-modelmeld-session-id header as the chat route (D-4).
    Memory store should record user + assistant turns just like chat."""
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "m",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello via anthropic"}],
            },
            headers={HEADER_SESSION_ID: "anthropic-sess-1"},
        )
    assert resp.status_code == 200

    turns = await store.list_turns("anthropic-sess-1", ANONYMOUS_TENANT_ID)
    assert len(turns) == 2
    assert turns[0].role == Role.USER
    assert turns[0].content == "hello via anthropic"
    assert turns[1].role == Role.ASSISTANT
    assert turns[1].content == "echo: hello via anthropic"


async def test_no_session_header_skips_memory_writes() -> None:
    store = InMemoryMemoryStore()
    app = build_app(adapter=_EchoAdapter(), memory_store=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 200
    assert await store.get_session("anything", ANONYMOUS_TENANT_ID) is None


# ---------------------------------------------------------------------------
# ID normalization (carries through from translator)
# ---------------------------------------------------------------------------

async def test_response_id_normalized_to_msg_prefix() -> None:
    """The internal completion has chatcmpl-* id (default); the Anthropic
    response should rewrite it to msg_* shape."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith("msg_")


# ---------------------------------------------------------------------------
# Anthropic blocks: type=message, role=assistant, ≥1 content block
# ---------------------------------------------------------------------------

async def test_response_shape_matches_anthropic_invariants() -> None:
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "x"}],
        })
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert isinstance(body["content"], list)
    assert len(body["content"]) >= 1
    # No null fields leaked into response (exclude_none on the Pydantic
    # serialization). stop_sequence is None per design and should be absent.
    assert "stop_sequence" not in body


# ---------------------------------------------------------------------------
# /v1/messages tool-capability filter — verified via manual test in pre-launch audit
# ---------------------------------------------------------------------------
# Surfaced 2026-05-25 during Claude Code WSL validation: the
# /v1/messages route had no explicit test asserting that tools=[...]
# triggers scout's supports_tools filter. The translation layer
# DOES propagate request.tools into the internal ChatCompletionRequest,
# so the filter should fire — but we want a pinned regression test
# so a future refactor of from_anthropic_request can't silently
# break this customer-facing protection.

class _RecordingAdapter(ProviderAdapter):
    """Records every chat() invocation so tests can assert which adapter
    handled a request. Returns a minimal valid ChatCompletion."""

    is_egress = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[ChatCompletionRequest] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.calls.append(request)
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content=f"served-by-{self.name}"),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


async def test_messages_route_with_tools_excludes_non_tool_capable_models() -> None:
    """When the /v1/messages caller sends tools=[...], the scout's
    supports_tools filter MUST exclude models marked
    supports_tools=False. Without this, Claude Code traffic on tools
    requests could route to e.g. deepseek-r1-distill-llama-70b which
    OpenRouter then rejects with 404 'no endpoints found that support
    tool use'."""
    from modelmeld.router import CapabilityRouter
    from modelmeld.scout import CapabilityScout, ModelEntry, ModelRegistry

    no_tool_adapter = _RecordingAdapter("non-tool-capable")
    tool_adapter = _RecordingAdapter("tool-capable")

    # Cheaper sub-Haiku-tier model lacks tool support. Pricier model has it.
    registry = ModelRegistry([
        ModelEntry(
            model_id="cheap-no-tools",
            provider="non-tool-capable",
            context_window=131072,
            cost_per_m_input=0.04, cost_per_m_output=0.04,
            task_scores={"coding": 0.85, "tool_use": 0.85},
            supports_tools=False,
        ),
        ModelEntry(
            model_id="pricier-with-tools",
            provider="tool-capable",
            context_window=131072,
            cost_per_m_input=0.30, cost_per_m_output=0.30,
            task_scores={"coding": 0.85, "tool_use": 0.85},
            supports_tools=True,
        ),
    ])
    scout = CapabilityScout(
        registry=registry, quality_threshold=0.80,
        eligible_providers=frozenset({"non-tool-capable", "tool-capable"}),
    )
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "non-tool-capable": no_tool_adapter,
            "tool-capable": tool_adapter,
        },
    )
    app = build_app(router=router, model_registry=registry)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # Claude-Code-shape request: tools=[...] present
        resp = await client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Use the read_file tool."}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            ],
        })
    assert resp.status_code == 200, resp.text
    # Filter MUST have excluded the cheap non-tool-capable adapter
    assert not no_tool_adapter.calls, (
        "Filter regression: non-tool-capable model received tools-bearing request. "
        "supports_tools filter is broken on /v1/messages path."
    )
    assert len(tool_adapter.calls) == 1
    assert resp.headers["x-modelmeld-routed-model"] == "pricier-with-tools"
    # Sanity: the tools survived translation into the internal request shape
    assert tool_adapter.calls[0].tools is not None
    assert len(tool_adapter.calls[0].tools) == 1


async def test_messages_route_without_tools_allows_non_tool_capable_models() -> None:
    """Counter-test: when no tools are sent (e.g., Claude Code's
    background title-generation or summarization calls), the
    supports_tools filter is dormant and the cheaper model wins on
    cost — even if it lacks tool support. This is why Claude Code
    session telemetry showed BOTH deepseek-r1-distill-llama-70b
    (no-tool-support) AND qwen3-coder-flash (with-tool-support) in
    the circuit-breaker registry: Claude Code mixes tool + non-tool
    requests within a session."""
    from modelmeld.router import CapabilityRouter
    from modelmeld.scout import CapabilityScout, ModelEntry, ModelRegistry

    cheap_adapter = _RecordingAdapter("non-tool-capable")
    pricier_adapter = _RecordingAdapter("tool-capable")

    # Provide scores for all default task categories so the test is
    # robust to whichever category the classifier picks for the prompt.
    _all_categories = {
        "coding": 0.85, "reasoning": 0.85, "simple_qa": 0.85,
        "summarization": 0.85, "tool_use": 0.85,
    }
    registry = ModelRegistry([
        ModelEntry(
            model_id="cheap-no-tools",
            provider="non-tool-capable",
            context_window=131072,
            cost_per_m_input=0.04, cost_per_m_output=0.04,
            task_scores=_all_categories,
            supports_tools=False,
        ),
        ModelEntry(
            model_id="pricier-with-tools",
            provider="tool-capable",
            context_window=131072,
            cost_per_m_input=0.30, cost_per_m_output=0.30,
            task_scores=_all_categories,
            supports_tools=True,
        ),
    ])
    scout = CapabilityScout(
        registry=registry, quality_threshold=0.80,
        eligible_providers=frozenset({"non-tool-capable", "tool-capable"}),
    )
    router = CapabilityRouter(
        scout=scout,
        adapters_by_provider={
            "non-tool-capable": cheap_adapter,
            "tool-capable": pricier_adapter,
        },
    )
    app = build_app(router=router, model_registry=registry)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        # No tools field — Claude-Code-background-shape
        resp = await client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "summarize this conversation in 5 words"}],
        })
    assert resp.status_code == 200, resp.text
    # Filter dormant → cheapest wins, regardless of supports_tools
    assert len(cheap_adapter.calls) == 1
    assert not pricier_adapter.calls
    assert resp.headers["x-modelmeld-routed-model"] == "cheap-no-tools"


# ---------------------------------------------------------------------------
# /v1/messages/count_tokens — Anthropic compliance
# ---------------------------------------------------------------------------

async def test_count_tokens_returns_positive_input_tokens() -> None:
    """The endpoint exists, accepts an Anthropic-shape request, and returns
    a positive input_tokens count. Claude Code's cost UI calls this."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Write a Python function "
                 "that returns the longest substring without repeating "
                 "characters."}
            ],
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "input_tokens" in body
    assert isinstance(body["input_tokens"], int)
    assert body["input_tokens"] > 0


async def test_count_tokens_includes_system_prompt() -> None:
    """A request with a heavy system prompt should count more tokens than
    one without — confirms the system block is included in the count."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        without_sys = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        long_system = "You are a coding assistant. " * 50
        with_sys = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "system": long_system,
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert with_sys.json()["input_tokens"] > without_sys.json()["input_tokens"]


async def test_count_tokens_includes_tool_definitions() -> None:
    """Tool schemas consume input tokens too — must be reflected in the count."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        without_tools = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        with_tools = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file from disk and return contents.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path"}
                        },
                    },
                }
            ],
        })
    assert with_tools.json()["input_tokens"] > without_tools.json()["input_tokens"]


async def test_count_tokens_translation_error_returns_400() -> None:
    """Same translation gates as POST /v1/messages — image content blocks
    aren't supported yet (deferred to v2), so the endpoint surfaces a 400."""
    app = build_app(adapter=_EchoAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages/count_tokens", json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBOR...",
                        },
                    }
                ],
            }],
        })
    assert resp.status_code == 400

"""End-to-end: cache hit/miss/bypass headers + adapter bypass on hit."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChoiceDelta,
    ChunkChoice,
    Choice,
    FunctionDef,
    ResponseMessage,
    Tool,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.cache import InMemoryCompletionCache


class _CountingAdapter(ProviderAdapter):
    """Records every call so we can prove cache hits skip the upstream."""

    name = "counting"
    is_egress = False

    def __init__(self, body: str = "hello-from-upstream") -> None:
        self.call_count = 0
        self._body = body

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.call_count += 1
        return ChatCompletion(
            model=request.model,
            choices=[Choice(
                index=0,
                message=ResponseMessage(content=self._body),
                finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        import time as _time
        self.call_count += 1
        yield ChatCompletionChunk(
            id="x", created=int(_time.time()), model=request.model,
            choices=[ChunkChoice(
                index=0, delta=ChoiceDelta(content=self._body), finish_reason=None,
            )],
        )
        yield ChatCompletionChunk(
            id="x", created=int(_time.time()), model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )

    async def health(self) -> bool:
        return True


def _payload(content: str = "say hi") -> dict:
    return {"model": "test-model", "messages": [{"role": "user", "content": content}]}


# ---------------------------------------------------------------------------
# Miss → upstream call → header "miss" → cache populated
# ---------------------------------------------------------------------------

async def test_first_request_misses_and_populates_cache() -> None:
    adapter = _CountingAdapter()
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-cache"] == "miss"
    assert adapter.call_count == 1
    # Cache populated (we have a key for this request)
    assert cache._size() == 1


# ---------------------------------------------------------------------------
# Hit → adapter NOT called → header "hit" → identical bytes
# ---------------------------------------------------------------------------

async def test_second_identical_request_hits_cache() -> None:
    adapter = _CountingAdapter("specific-body-text")
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/v1/chat/completions", json=_payload())
        second = await client.post("/v1/chat/completions", json=_payload())

    assert first.status_code == 200 and second.status_code == 200
    assert first.headers["x-modelmeld-cache"] == "miss"
    assert second.headers["x-modelmeld-cache"] == "hit"
    # Adapter called exactly once (miss); the second request skipped it
    assert adapter.call_count == 1
    # Hit returns identical bytes from the adapter's response
    assert first.json()["choices"][0]["message"]["content"] == "specific-body-text"
    assert second.json()["choices"][0]["message"]["content"] == "specific-body-text"


# ---------------------------------------------------------------------------
# Bypass: streaming + tool requests never hit the cache
# ---------------------------------------------------------------------------

async def test_streaming_request_bypasses_cache() -> None:
    adapter = _CountingAdapter("streamed-body")
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Two streaming requests with identical bodies
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "test-model", "stream": True,
                  "messages": [{"role": "user", "content": "stream me"}]},
        ) as resp1:
            assert resp1.headers["x-modelmeld-cache"] == "bypass"
            async for _ in resp1.aiter_lines():
                pass
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "test-model", "stream": True,
                  "messages": [{"role": "user", "content": "stream me"}]},
        ) as resp2:
            async for _ in resp2.aiter_lines():
                pass

    # Both requests hit the adapter — no caching for SSE in 4.1
    assert adapter.call_count == 2
    assert cache._size() == 0


async def test_tool_call_request_bypasses_cache() -> None:
    adapter = _CountingAdapter()
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{
            "type": "function",
            "function": {"name": "search", "description": "", "parameters": {"type": "object"}},
        }],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-cache"] == "bypass"
    assert cache._size() == 0


# ---------------------------------------------------------------------------
# No cache configured → no x-modelmeld-cache header at all
# ---------------------------------------------------------------------------

async def test_no_cache_configured_no_header() -> None:
    """Backwards compat: when no cache is attached, behavior unchanged."""
    adapter = _CountingAdapter()
    app = build_app(adapter=adapter)   # no completion_cache

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    assert "x-modelmeld-cache" not in resp.headers


# ---------------------------------------------------------------------------
# Different requests → different cache entries
# ---------------------------------------------------------------------------

async def test_different_prompts_dont_share_cache() -> None:
    adapter = _CountingAdapter()
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/v1/chat/completions", json=_payload("question A"))
        await client.post("/v1/chat/completions", json=_payload("question B"))
        await client.post("/v1/chat/completions", json=_payload("question A"))  # repeat A

    # A hit once (3rd call), B hit zero times → adapter called twice
    assert adapter.call_count == 2
    assert cache._size() == 2


# ---------------------------------------------------------------------------
# Cache write failure doesn't break the response
# ---------------------------------------------------------------------------

class _BrokenCache(InMemoryCompletionCache):
    async def set(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("redis offline")


async def test_cache_set_failure_doesnt_break_request() -> None:
    """The user got their answer; a cache write failure must not 500."""
    adapter = _CountingAdapter()
    cache = _BrokenCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload())

    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-cache"] == "miss"
    assert resp.json()["choices"][0]["message"]["content"] == "hello-from-upstream"


# ---------------------------------------------------------------------------
# Performance smoke: cache hit beats the adapter by orders of magnitude
# ---------------------------------------------------------------------------

async def test_cache_hit_is_fast() -> None:
    """Smoke check on the p95 <5ms claim from the project plan.
    We don't measure p95 here — that requires a real benchmark harness —
    but a single hit should certainly be sub-millisecond on a no-op adapter.
    """
    import time as _time
    adapter = _CountingAdapter()
    cache = InMemoryCompletionCache()
    app = build_app(adapter=adapter, completion_cache=cache)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Warm the cache
        await client.post("/v1/chat/completions", json=_payload())
        # Time a hit
        start = _time.perf_counter()
        resp = await client.post("/v1/chat/completions", json=_payload())
        elapsed_ms = (_time.perf_counter() - start) * 1000

    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-cache"] == "hit"
    # Generous threshold — testclient overhead dwarfs the cache lookup itself.
    # Real benchmark goes in CI perf harness later.
    assert elapsed_ms < 100, f"cache hit took {elapsed_ms:.1f}ms (expected <100ms)"

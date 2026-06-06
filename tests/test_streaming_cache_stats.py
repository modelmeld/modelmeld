# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Anthropic `cache_control` stats survive the streaming path.

Non-streaming already surfaces `cache_creation_input_tokens` /
`cache_read_input_tokens` (the cache-write + cache-hit counts customers use to
verify their caching works). These tests cover the streaming path, where the
counts have to survive two translations — Anthropic SSE → internal OpenAI
chunks → Anthropic SSE — and Anthropic reports them ONLY in `message_start`.

Three layers:
  - inbound translator carries them off Anthropic's message_start onto the chunk
  - outbound translator re-emits them in the Anthropic message_start it builds
  - end-to-end through /v1/messages streaming
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
    ChoiceDelta,
    ChunkChoice,
    PromptTokensDetails,
    Usage,
)
from modelmeld.api.schemas_anthropic import AnthropicMessageStartEvent
from modelmeld.api.server import build_app
from modelmeld.translation import OpenAIToAnthropicStreamTranslator
from modelmeld.translation.openai_anthropic import AnthropicStreamTranslator

# ---------------------------------------------------------------------------
# Inbound: Anthropic message_start usage → OpenAI chunk prompt_tokens_details
# ---------------------------------------------------------------------------

def test_inbound_carries_cache_stats_onto_role_chunk() -> None:
    t = AnthropicStreamTranslator()
    chunk = t.translate_event({
        "type": "message_start",
        "message": {
            "id": "msg_1", "model": "claude-opus-4-7",
            "usage": {
                "input_tokens": 120,
                "cache_creation_input_tokens": 4933,
                "cache_read_input_tokens": 12,
            },
        },
    })
    assert chunk is not None and chunk.usage is not None
    details = chunk.usage.prompt_tokens_details
    assert details is not None
    assert details.cache_creation_input_tokens == 4933
    assert details.cache_read_input_tokens == 12
    assert details.cached_tokens == 12  # mirrored cross-vendor slot


def test_inbound_no_cache_means_no_prompt_details() -> None:
    t = AnthropicStreamTranslator()
    chunk = t.translate_event({
        "type": "message_start",
        "message": {"id": "x", "model": "m", "usage": {"input_tokens": 10}},
    })
    assert chunk is not None and chunk.usage is not None
    assert chunk.usage.prompt_tokens_details is None


# ---------------------------------------------------------------------------
# Outbound: chunk prompt_tokens_details → Anthropic message_start usage
# ---------------------------------------------------------------------------

def _first_message_start(events) -> AnthropicMessageStartEvent:
    starts = [e for e in events if isinstance(e, AnthropicMessageStartEvent)]
    assert len(starts) == 1
    return starts[0]


def test_outbound_emits_cache_stats_in_message_start() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="claude-opus-4-7", input_tokens=120)
    first = ChatCompletionChunk(
        id="msg_1", created=0, model="claude-opus-4-7",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=""))],
        usage=Usage(
            prompt_tokens=120, completion_tokens=0, total_tokens=120,
            prompt_tokens_details=PromptTokensDetails(
                cache_creation_input_tokens=4933,
                cache_read_input_tokens=12,
                cached_tokens=12,
            ),
        ),
    )
    usage = _first_message_start(t.translate_chunk(first)).message.usage
    assert usage.cache_creation_input_tokens == 4933
    assert usage.cache_read_input_tokens == 12


def test_outbound_no_cache_fields_when_chunk_has_none() -> None:
    t = OpenAIToAnthropicStreamTranslator(request_model="m", input_tokens=5)
    first = ChatCompletionChunk(
        id="msg_2", created=0, model="m",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=""))],
        usage=None,
    )
    usage = _first_message_start(t.translate_chunk(first)).message.usage
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None


# ---------------------------------------------------------------------------
# End-to-end: /v1/messages streaming surfaces the cache counts
# ---------------------------------------------------------------------------

class _CacheStreamAdapter(ProviderAdapter):
    """Streams a first chunk carrying cache stats (what the inbound translator
    produces from Anthropic's message_start), then a final usage chunk."""

    name = "mock-cache-stream"
    is_egress = False

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        raise NotImplementedError("streaming-only fixture")

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        cid = "msg_cache_fixture"
        yield ChatCompletionChunk(
            id=cid, created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"))],
            usage=Usage(
                prompt_tokens=120, completion_tokens=0, total_tokens=120,
                prompt_tokens_details=PromptTokensDetails(
                    cache_creation_input_tokens=4933,
                    cache_read_input_tokens=12,
                    cached_tokens=12,
                ),
            ),
        )
        yield ChatCompletionChunk(
            id=cid, created=1, model=request.model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=Usage(prompt_tokens=120, completion_tokens=5, total_tokens=125),
        )

    async def health(self) -> bool:
        return True


def _parse_sse(raw: str) -> list[dict]:
    out: list[dict] = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


async def test_messages_streaming_surfaces_cache_stats_in_message_start() -> None:
    app = build_app(adapter=_CacheStreamAdapter())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/v1/messages", json={
            "model": "claude-opus-4-7", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    start = next(e for e in events if e.get("type") == "message_start")
    usage = start["message"]["usage"]
    assert usage["cache_creation_input_tokens"] == 4933
    assert usage["cache_read_input_tokens"] == 12

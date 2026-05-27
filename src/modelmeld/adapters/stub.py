# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""StubAdapter — canned response. Used by tests and when no upstream is configured."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import uuid4

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
    Usage,
)

_STUB_REPLY = (
    "ModelMeld stub adapter. Configure an upstream provider (set "
    "MODELMELD_UPSTREAM_PROVIDER=openai and MODELMELD_OPENAI_API_KEY) to forward traffic."
)
_STUB_USAGE = Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25)


class StubAdapter(ProviderAdapter):
    name = "stub"

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        return ChatCompletion(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=_STUB_REPLY),
                    finish_reason="stop",
                )
            ],
            usage=_STUB_USAGE,
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-{uuid4().hex[:24]}"
        created = int(time.time())
        model = request.model

        # role chunk
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=""))],
        )
        # content deltas
        for word in _STUB_REPLY.split():
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=word + " "))],
            )
        # final chunk
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )
        # optional usage chunk
        if request.stream_options and request.stream_options.include_usage:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[],
                usage=_STUB_USAGE,
            )

    async def health(self) -> bool:
        return True

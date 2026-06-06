# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""MemoryProvider — the seam between the request path and the memory engine.

The chat + messages routes talk to a `MemoryProvider`, not to a `MemoryStore`
directly. This decouples *what the gateway does on each request* (retrieve a
context to inject, record the exchange) from *which engine backs it*.

`TieredMemoryProvider` wraps the built-in L0/L1/L2/L3 `MemoryStore` and is the
default — behavior is identical to calling `assemble_context` + the turn-append
helpers directly, which is what the routes did before this seam existed.

Alternative engines (e.g. a Mem0-backed provider) implement the same two-method
contract and slot in via `GatewaySettings.memory_backend`. The protocol is kept
least-common-denominator (record an exchange / retrieve a context) so heavier
engines fit behind it without reshaping it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from modelmeld.api.schemas import ChatCompletionRequest, TextPart, UserMessage
from modelmeld.memory.base import MemoryStore, Role
from modelmeld.memory.context import MemoryContext, assemble_context
from modelmeld.memory.identity import MemoryIdentity
from modelmeld.tokens import TokenCounter

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Per-request memory contract the routes depend on.

    Implementations MUST treat every call as tenant-scoped (the tenant_id lives
    on `mem_identity`) and MUST NOT let a memory failure break the user's
    request — `retrieve` returns an empty context on error; `record` swallows
    and logs. The user already has (or is getting) their answer either way.
    """

    @abstractmethod
    async def retrieve(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
    ) -> MemoryContext:
        """Return the memory context to inject for this request.

        Empty context when memory is inactive/disabled for the request.
        """

    @abstractmethod
    async def record(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
        assistant_text: str,
        *,
        assistant_tokens: int = 0,
        model_used: str | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Persist the just-completed user→assistant exchange. Best-effort."""

    async def close(self) -> None:
        """Release any held resources. Default no-op."""


class TieredMemoryProvider(MemoryProvider):
    """Default provider over the built-in tiered `MemoryStore` (L0/L1/L2/L3).

    Retrieval delegates to `assemble_context`; recording uses the same
    turn-append path the routes used before the seam existed. Behavior-preserving
    by construction.
    """

    def __init__(self, store: MemoryStore | None) -> None:
        self._store = store

    @property
    def store(self) -> MemoryStore | None:
        return self._store

    async def retrieve(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
    ) -> MemoryContext:
        # `request` is unused here: tiered retrieval keys off the session
        # identity, not the current prompt. It stays in the signature for
        # engines that search by query (e.g. Mem0).
        return await assemble_context(self._store, mem_identity)

    async def record(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
        assistant_text: str,
        *,
        assistant_tokens: int = 0,
        model_used: str | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if self._store is None or not mem_identity.active:
            return
        assert mem_identity.session_id is not None  # narrow for type checker
        store = self._store
        used_model = model_used or request.model
        try:
            await store.get_or_create_session(
                session_id=mem_identity.session_id,
                tenant_id=mem_identity.tenant_id,
                user_id=mem_identity.user_id,
            )
            # Append the newest user turn from the incoming request only — prior
            # turns came from earlier requests and are already in L0.
            last_user = _last_user_turn(request, token_counter, used_model)
            if last_user is not None:
                user_text, user_tokens = last_user
                await store.append_turn(
                    session_id=mem_identity.session_id,
                    tenant_id=mem_identity.tenant_id,
                    role=Role.USER,
                    content=user_text,
                    token_count=user_tokens,
                    model_used=used_model,
                )
            await store.append_turn(
                session_id=mem_identity.session_id,
                tenant_id=mem_identity.tenant_id,
                role=Role.ASSISTANT,
                content=assistant_text,
                token_count=assistant_tokens,
                model_used=used_model,
            )
        except Exception:
            logger.exception(
                "memory write failed for session %s", mem_identity.session_id
            )

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()


def _last_user_turn(
    request: ChatCompletionRequest,
    token_counter: TokenCounter | None,
    model_used: str,
) -> tuple[str, int] | None:
    """Find the most recent user message in the request + return (text, tokens)."""
    for msg in reversed(request.messages):
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                text = msg.content
            else:
                text = "".join(
                    part.text for part in msg.content if isinstance(part, TextPart)
                )
            return text, _count_tokens(token_counter, text, model_used)
    return None


def _count_tokens(
    counter: TokenCounter | None, text: str, model: str | None
) -> int:
    """Count tokens via the configured counter; char-based fallback if unset."""
    if counter is not None:
        return counter.count_text(text, model)
    return max(1, len(text) // 4) if text else 0

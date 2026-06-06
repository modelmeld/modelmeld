# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Mem0-backed MemoryProvider.

Wraps the Apache-2.0 mem0 OSS core behind the `MemoryProvider` seam. The
gateway keeps doing request-path injection (via the shared
`inject_into_request`); mem0 does the extraction + semantic retrieval that
would otherwise be ours to build.

Selected with `GatewaySettings.memory_backend = "mem0"`. Requires the optional
dependency: `pip install modelmeld[mem0]`.

Tenant isolation (security-critical): each tenant gets its OWN mem0 vector
collection — `tenant_collection_name(tenant_id)` — NOT a shared collection with
metadata filters. We keep one `AsyncMemory` instance per tenant, created lazily.
Within a tenant, memories are scoped to the session via mem0's `run_id`.

Cost note: `infer=True` (default) runs an LLM extraction call per write.
Point `mem0_base_url` at this gateway so that call is cost-routed by our own
router (and self-hosters on local models pay ~nothing). `infer=False` stores
raw turns with no per-write LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from modelmeld.api.schemas import ChatCompletionRequest, TextPart, UserMessage
from modelmeld.memory.base import (
    Summary,
    new_summary_id,
    tenant_collection_name,
    utc_now,
)
from modelmeld.memory.context import MemoryContext
from modelmeld.memory.identity import MemoryIdentity
from modelmeld.memory.provider import MemoryProvider

if TYPE_CHECKING:
    from modelmeld.config import GatewaySettings

logger = logging.getLogger(__name__)


class Mem0DependencyError(ImportError):
    """Raised when the `mem0` backend is selected but `mem0ai` isn't installed."""

    def __init__(self) -> None:
        super().__init__(
            "memory_backend='mem0' requires the mem0 extra. "
            "Install it with: pip install 'modelmeld[mem0]'"
        )


class Mem0MemoryProvider(MemoryProvider):
    """MemoryProvider backed by mem0's AsyncMemory, one collection per tenant."""

    def __init__(
        self,
        *,
        infer: bool = True,
        top_k: int = 10,
        rerank: bool = False,
        embedding_dims: int = 1536,
        llm_model: str = "gpt-5-mini",
        embedder_model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        vector_store_url: str | None = None,
        vector_store_api_key: str | None = None,
        vector_store_path: str | None = None,
    ) -> None:
        # Disable mem0's posthog telemetry by default (it phones home about
        # usage). setdefault leaves an explicit operator opt-in untouched.
        # Set before importing mem0 so the telemetry client sees it.
        os.environ.setdefault("MEM0_TELEMETRY", "False")

        # Fail loudly + early if the optional dep is missing — at construction,
        # not on the first request.
        try:
            from mem0 import AsyncMemory  # noqa: F401
        except ImportError as e:
            raise Mem0DependencyError() from e

        logger.warning(
            "Mem0 memory backend is EXPERIMENTAL. infer=%s means an LLM "
            "extraction call per write — route it through this gateway via "
            "mem0_base_url so it's cost-optimized. See docs/integrations/memory.md.",
            infer,
        )

        self._infer = infer
        self._top_k = top_k
        self._rerank = rerank
        self._embedding_dims = embedding_dims
        self._llm_model = llm_model
        self._embedder_model = embedder_model
        self._base_url = base_url
        self._api_key = api_key
        self._vector_store_url = vector_store_url
        self._vector_store_api_key = vector_store_api_key
        self._vector_store_path = vector_store_path

        self._mems: dict[str, Any] = {}   # tenant_id -> AsyncMemory
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: GatewaySettings) -> Mem0MemoryProvider:
        return cls(
            infer=settings.mem0_infer,
            top_k=settings.mem0_top_k,
            rerank=settings.mem0_rerank,
            embedding_dims=settings.mem0_embedding_dims,
            llm_model=settings.mem0_llm_model,
            embedder_model=settings.mem0_embedder_model,
            base_url=settings.mem0_base_url,
            api_key=settings.mem0_api_key,
            vector_store_url=settings.mem0_vector_store_url,
            vector_store_api_key=settings.mem0_vector_store_api_key,
            vector_store_path=settings.mem0_vector_store_path,
        )

    # ----- config + per-tenant instance -----------------------------------

    def _build_config(self, tenant_id: str) -> dict[str, Any]:
        collection = tenant_collection_name(tenant_id)

        llm_cfg: dict[str, Any] = {"model": self._llm_model}
        embed_cfg: dict[str, Any] = {
            "model": self._embedder_model,
            "embedding_dims": self._embedding_dims,
        }
        if self._base_url:
            # Route mem0's extraction LLM + embedder through our gateway.
            llm_cfg["openai_base_url"] = self._base_url
            embed_cfg["openai_base_url"] = self._base_url
        if self._api_key:
            llm_cfg["api_key"] = self._api_key
            embed_cfg["api_key"] = self._api_key

        vs_cfg: dict[str, Any] = {
            "collection_name": collection,
            "embedding_model_dims": self._embedding_dims,
        }
        if self._vector_store_url:
            # Shared qdrant server: tenant isolation via the per-tenant
            # collection_name above.
            vs_cfg["url"] = self._vector_store_url
            if self._vector_store_api_key:
                vs_cfg["api_key"] = self._vector_store_api_key
        elif self._vector_store_path:
            # Embedded on-disk qdrant: give each tenant its own directory so
            # the embedded stores never share a file/lock.
            vs_cfg["path"] = os.path.join(self._vector_store_path, collection)

        return {
            "llm": {"provider": "openai", "config": llm_cfg},
            "embedder": {"provider": "openai", "config": embed_cfg},
            "vector_store": {"provider": "qdrant", "config": vs_cfg},
        }

    async def _mem_for(self, tenant_id: str) -> Any:
        """Return (lazily creating) the AsyncMemory instance for this tenant.

        One instance per tenant → one vector collection per tenant. Double-checked
        locking so concurrent first-requests for a tenant don't build two.
        """
        mem = self._mems.get(tenant_id)
        if mem is not None:
            return mem
        async with self._lock:
            mem = self._mems.get(tenant_id)
            if mem is None:
                from mem0 import AsyncMemory

                mem = AsyncMemory.from_config(self._build_config(tenant_id))
                self._mems[tenant_id] = mem
        return mem

    # ----- MemoryProvider contract ----------------------------------------

    async def retrieve(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
    ) -> MemoryContext:
        if not mem_identity.active:
            return MemoryContext()
        assert mem_identity.session_id is not None
        query = _last_user_text(request)
        if not query:
            return MemoryContext()
        try:
            mem = await self._mem_for(mem_identity.tenant_id)
            result = await mem.search(
                query,
                top_k=self._top_k,
                rerank=self._rerank,
                filters={"run_id": mem_identity.session_id},
            )
        except Exception:
            logger.exception(
                "mem0 retrieve failed for session %s (degraded; no context)",
                mem_identity.session_id,
            )
            return MemoryContext()

        memories = _extract_memory_texts(result)
        if not memories:
            return MemoryContext()
        text = "Relevant memories from this session:\n" + "\n".join(
            f"- {m}" for m in memories
        )
        now = utc_now()
        synthetic = Summary(
            summary_id=new_summary_id(),
            session_id=mem_identity.session_id,
            tenant_id=mem_identity.tenant_id,
            text=text,
            last_applied_turn_id=None,
            version=0,
            source_model="mem0",
            created_at=now,
            updated_at=now,
        )
        return MemoryContext(summary=synthetic)

    async def record(
        self,
        mem_identity: MemoryIdentity,
        request: ChatCompletionRequest,
        assistant_text: str,
        *,
        assistant_tokens: int = 0,
        model_used: str | None = None,
        token_counter: Any | None = None,
    ) -> None:
        if not mem_identity.active:
            return
        assert mem_identity.session_id is not None
        user_text = _last_user_text(request)
        messages: list[dict[str, str]] = []
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        if not messages:
            return
        try:
            mem = await self._mem_for(mem_identity.tenant_id)
            await mem.add(
                messages,
                run_id=mem_identity.session_id,
                infer=self._infer,
            )
        except Exception:
            logger.exception(
                "mem0 record failed for session %s", mem_identity.session_id
            )


def _last_user_text(request: ChatCompletionRequest) -> str:
    """Most recent user message text from the request, or '' if none."""
    for msg in reversed(request.messages):
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                return msg.content
            return "".join(
                part.text for part in msg.content if isinstance(part, TextPart)
            )
    return ""


def _extract_memory_texts(result: Any) -> list[str]:
    """Pull memory strings out of a mem0 search result.

    mem0 returns either a list of memory dicts or `{"results": [...]}` depending
    on version/output format. Each entry carries the text under "memory".
    """
    rows = result.get("results", []) if isinstance(result, dict) else result
    texts: list[str] = []
    for row in rows or []:
        if isinstance(row, dict):
            mem_text = row.get("memory") or row.get("text")
            if mem_text:
                texts.append(str(mem_text))
        elif isinstance(row, str):
            texts.append(row)
    return texts

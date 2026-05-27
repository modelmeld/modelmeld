# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tiered memory.

L0/L1/L2/L3 per-session memory that survives model swaps.
The schema + in-memory backend ship in the OSS core; L2 evolving summary,
L3 hot-zone, summarizer worker, cross-tokenizer accounting, and multi-tenant
Qdrant layer on top.

Public surface:
    MemoryStore               — abstract base class
    InMemoryMemoryStore       — dev/test backend (process-local)
    Session, Turn, Fact, Role — schema dataclasses
    MemoryIdentity            — (tenant_id, session_id, user_id) tuple
    extract_memory_identity   — header → identity helper
    ANONYMOUS_TENANT_ID       — sentinel for unauthenticated requests
"""

from __future__ import annotations

from modelmeld.memory.base import (
    ANONYMOUS_TENANT_ID,
    QDRANT_COLLECTION_PREFIX,
    Fact,
    InvalidTenantIdError,
    MemoryStore,
    Role,
    Session,
    Summary,
    SummaryVersionMismatch,
    Turn,
    tenant_collection_name,
    validate_tenant_id,
)
from modelmeld.memory.context import (
    DEFAULT_HOT_ZONE_CAP,
    DEFAULT_MAX_CONTEXT_CHARS,
    MemoryContext,
    assemble_context,
    inject_into_request,
    render_system_message,
)
from modelmeld.memory.identity import (
    HEADER_MEMORY_MODE,
    HEADER_SESSION_ID,
    HEADER_USER_ID_OVERRIDE,
    MemoryHeaderError,
    MemoryIdentity,
    MemoryMode,
    extract_memory_identity,
)
from modelmeld.memory.in_memory import (
    InMemoryMemoryStore,
    TenantMismatchError,
)
from modelmeld.memory.summarizer import (
    SummarizeCallable,
    SummarizerConfig,
    SummarizerWorker,
    adapter_summarize_call,
    build_summarizer_prompt,
    run_for_pending_sessions,
    sanitize_summary,
)
from modelmeld.memory.tiers import (
    needs_summary_refresh,
    summary_freshness,
    turns_since_summary,
)

__all__ = [
    "ANONYMOUS_TENANT_ID",
    "DEFAULT_HOT_ZONE_CAP",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "Fact",
    "InvalidTenantIdError",
    "QDRANT_COLLECTION_PREFIX",
    "tenant_collection_name",
    "validate_tenant_id",
    "HEADER_MEMORY_MODE",
    "HEADER_SESSION_ID",
    "HEADER_USER_ID_OVERRIDE",
    "InMemoryMemoryStore",
    "MemoryContext",
    "MemoryHeaderError",
    "MemoryIdentity",
    "MemoryMode",
    "MemoryStore",
    "Role",
    "Session",
    "Summary",
    "SummarizeCallable",
    "SummarizerConfig",
    "SummarizerWorker",
    "SummaryVersionMismatch",
    "TenantMismatchError",
    "Turn",
    "adapter_summarize_call",
    "assemble_context",
    "build_summarizer_prompt",
    "extract_memory_identity",
    "inject_into_request",
    "needs_summary_refresh",
    "render_system_message",
    "run_for_pending_sessions",
    "sanitize_summary",
    "summary_freshness",
    "turns_since_summary",
]

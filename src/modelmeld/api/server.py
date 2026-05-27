# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from modelmeld import __version__
from modelmeld.adapters import ProviderAdapter
from modelmeld.cache import CompletionCache, SemanticCompletionCache
from modelmeld.config import GatewaySettings
from modelmeld.hooks import HookRegistry
from modelmeld.memory import InMemoryMemoryStore, MemoryStore
from modelmeld.privacy import Scrubber, build_scrubber
from modelmeld.router import Router, SingleAdapterRouter, build_router
from modelmeld.scout import ModelRegistry, Scout, build_scout, default_registry
from modelmeld.tokens import TokenCounter, build_token_counter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        router: Router | None = getattr(app.state, "router", None)
        if router is not None:
            await router.close()
        memory: MemoryStore | None = getattr(app.state, "memory_store", None)
        if memory is not None:
            await memory.close()
        cache: CompletionCache | None = getattr(app.state, "completion_cache", None)
        if cache is not None:
            await cache.close()
        semantic: SemanticCompletionCache | None = getattr(
            app.state, "semantic_cache", None,
        )
        if semantic is not None:
            await semantic.close()


def build_app(
    settings: GatewaySettings | None = None,
    adapter: ProviderAdapter | None = None,
    router: Router | None = None,
    scout: Scout | None = None,
    scrubber: Scrubber | None = None,
    hooks: HookRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    memory_store: MemoryStore | None = None,
    token_counter: TokenCounter | None = None,
    completion_cache: CompletionCache | None = None,
    semantic_cache: SemanticCompletionCache | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    `adapter` is a convenience: when provided (and `router` is not), the adapter
    is wrapped in a SingleAdapterRouter — preserves single-adapter test ergonomics.
    `hooks` defaults to an empty registry; /enterprise-control's
    `build_enterprise_app()` injects an enriched registry with audit handlers.
    `model_registry` defaults to the package-shipped default; the
    CapabilityScout consumes it for actual routing decisions.
    """
    app = FastAPI(
        title="ModelMeld Gateway",
        description="OpenAI-compatible AI gateway with complexity-based routing.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings or GatewaySettings()
    app.state.scout = scout or build_scout(app.state.settings)
    # Allow explicit None to disable scrubbing in tests.
    app.state.scrubber = scrubber if scrubber is not None else build_scrubber(app.state.settings)
    app.state.hooks = hooks or HookRegistry()
    app.state.model_registry = model_registry or default_registry()
    # Tiered memory. Default: in-process backend for
    # dev/tests. Enterprise plugs in a Postgres-backed store later.
    app.state.memory_store = memory_store if memory_store is not None else InMemoryMemoryStore()
    # Token counter. Default char-based; settings can switch to
    # litellm. Pass `token_counter=` to inject a custom impl in tests.
    app.state.token_counter = (
        token_counter if token_counter is not None
        else build_token_counter(app.state.settings)
    )
    # Completion cache. None when disabled in settings; tests
    # pass an explicit instance to verify cache wiring.
    app.state.completion_cache = completion_cache
    # Semantic cache. Consulted AFTER the exact-match cache.
    app.state.semantic_cache = semantic_cache

    if router is not None:
        app.state.router = router
    elif adapter is not None:
        app.state.router = SingleAdapterRouter(adapter)
    else:
        app.state.router = build_router(
            app.state.settings,
            app.state.scout,
            model_registry=app.state.model_registry,
        )

    from modelmeld.api.body_size_limit import BodySizeLimitMiddleware
    from modelmeld.api.routes import chat, healthz, messages, models

    app.include_router(healthz.router)
    app.include_router(models.router, prefix="/v1")
    app.include_router(chat.router, prefix="/v1")
    app.include_router(messages.router, prefix="/v1")

    # Body-size cap (defense against OOM/DoS from oversized payloads). Chat
    # and messages routes may carry large prompts so the default sits at 4 MB;
    # health and listing routes get the same cap (cheap to apply, harmless).
    # Pure-ASGI so SSE streaming responses are unaffected.
    app.add_middleware(
        BodySizeLimitMiddleware,
        default_max_bytes=4 * 1024 * 1024,  # 4 MB
    )
    return app


app = build_app()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from modelmeld import __version__
from modelmeld.adapters import ProviderAdapter
from modelmeld.cache import CompletionCache, SemanticCompletionCache
from modelmeld.config import GatewaySettings
from modelmeld.hooks import HookRegistry
from modelmeld.memory import (
    InMemoryMemoryStore,
    Mem0MemoryProvider,
    MemoryProvider,
    MemoryStore,
    PostgresMemoryStore,
    TieredMemoryProvider,
)
from modelmeld.metrics import MetricsCollector
from modelmeld.privacy import Scrubber, build_scrubber
from modelmeld.router import Router, SingleAdapterRouter, build_router
from modelmeld.scout import ModelRegistry, Scout, build_scout
from modelmeld.scout.feed import RegistryFeedClient
from modelmeld.scout.multi_provider_registry import default_multi_provider_registry
from modelmeld.scout.session_state import SessionStallStore
from modelmeld.tokens import TokenCounter, build_token_counter

logger = logging.getLogger(__name__)


async def _maybe_load_registry_feed(app: FastAPI) -> None:
    """Fetch the curated registry feed (if configured) and route on it.

    No-op when `registry_feed_url` is unset. On a successful, signature-verified
    fetch this swaps `app.state.model_registry` to the live registry and (when
    the router is gateway-owned) rebuilds the router on it. On ANY failure the
    client returns the bundled seed — we keep the default registry already wired
    in `app.state` and never raise. New feeds are picked up on restart.
    """
    settings = app.state.settings
    feed_url = getattr(settings, "registry_feed_url", None)
    if not feed_url:
        return
    client = RegistryFeedClient(
        feed_url=feed_url,
        license_key=getattr(settings, "registry_feed_license_key", None),
        public_key_pem=getattr(settings, "registry_feed_public_key_pem", None),
        cache_path=getattr(settings, "registry_feed_cache_path", None),
        cache_ttl_sec=getattr(settings, "registry_feed_cache_ttl_sec", 3600),
    )
    try:
        result = await client.fetch()
    finally:
        await client.close()
    if result.source == "seed":
        # Network/signature/schema/expiry failure — the client already logged
        # the reason at WARN. Keep the bundled registry already in app.state.
        return
    app.state.model_registry = result.registry
    # The router captured the prior registry at build time, so rebuild it on the
    # fetched feed — but only when the gateway owns the router (an injected
    # router/adapter belongs to the caller).
    if getattr(app.state, "_router_owned", False):
        old_router: Router | None = getattr(app.state, "router", None)
        app.state.router = build_router(
            settings, app.state.scout, model_registry=result.registry,
        )
        if old_router is not None:
            await old_router.close()
    logger.info(
        "registry feed loaded: source=%s feed_version=%s models=%d",
        result.source, result.feed_version, len(result.registry.all_entries()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Loud-fail the #1 self-host footgun: the default config
    # (routing_policy=single + upstream_provider=stub) serves a canned
    # stub reply to EVERY request with no error. A tester who follows a
    # bare `uvicorn modelmeld.api.server:app` lands here and never sees
    # real routing. Warn at startup so the silent no-op is at least loud.
    settings = getattr(app.state, "settings", None)
    if (
        settings is not None
        and getattr(settings, "routing_policy", None) == "single"
        and getattr(settings, "upstream_provider", None) == "stub"
    ):
        logger.warning(
            "ModelMeld is running with routing_policy=single + "
            "upstream_provider=stub: ALL requests return a canned stub "
            "reply, not real model output. Run `modelmeld setup "
            "--self-host` (or set MODELMELD_ROUTING_POLICY=capability + a "
            "provider key such as MODELMELD_OPENROUTER_API_KEY) to enable "
            "real routing. See the README Self-host section.",
        )
    # Curated registry feed (Pro): fetch + verify the live registry and route
    # on it. Safe no-op when unconfigured; safe fallback to the bundled seed on
    # any failure. Runs once at startup (restart-to-reload).
    await _maybe_load_registry_feed(app)

    try:
        yield
    finally:
        router: Router | None = getattr(app.state, "router", None)
        if router is not None:
            await router.close()
        # Closing the provider releases its underlying store (the provider
        # wraps app.state.memory_store), so we don't close the store twice.
        provider: MemoryProvider | None = getattr(app.state, "memory_provider", None)
        if provider is not None:
            await provider.close()
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
    # Default to the MULTI-provider registry (base + default_overlay) so the
    # cloud OSS provider rows (fireworks/together/openrouter) are actually
    # routable. Defaulting to the base registry here left OSS models only on
    # their unreachable `vllm` rows, so capability routing saw no eligible OSS
    # provider and fell back to frontier (-auto→Haiku) or 400'd (-saver). The
    # multi-provider registry is a ModelRegistry subclass and the header
    # resolvers already prefer its `all_entries_multi`, so this is strictly
    # more correct for every consumer.
    app.state.model_registry = model_registry or default_multi_provider_registry()

    # In-process request/spend metrics collector (GET /metrics). One per app so
    # a freshly built app starts zeroed. Updated directly from the route success
    # paths (not via the HookRegistry, which is the enterprise seam).
    app.state.metrics = MetricsCollector()
    # Tiered memory. Default: in-process backend for dev/tests; operators can
    # opt into a SQL-backed store with MODELMELD_MEMORY_BACKEND=postgres.
    if memory_store is not None:
        app.state.memory_store = memory_store
    elif app.state.settings.memory_backend == "postgres":
        app.state.memory_store = PostgresMemoryStore(
            app.state.settings.memory_database_url
        )
    else:
        app.state.memory_store = InMemoryMemoryStore()
    # The routes talk to a MemoryProvider, not the store directly. Default
    # wraps the tiered store above; memory_backend="mem0" swaps in a
    # Mem0-backed provider (optional dep: pip install modelmeld[mem0]).
    if app.state.settings.memory_backend == "mem0":
        app.state.memory_provider = Mem0MemoryProvider.from_settings(
            app.state.settings
        )
    else:
        app.state.memory_provider = TieredMemoryProvider(app.state.memory_store)
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

    # Per-session stall state for reactive escalation. Constructed
    # unconditionally (cheap, in-process TTL map); the SHADOW behaviour that
    # reads it is gated by MODELMELD_STALL_SHADOW so tests can always reach it.
    app.state.session_stall = SessionStallStore()

    # Track whether WE built the router: only then may the registry-feed
    # startup hook rebuild it on the fetched registry. An injected router /
    # adapter is owned by the caller (tests, embedders) — don't touch it.
    app.state._router_owned = router is None and adapter is None
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
    from modelmeld.api.routes import (
        chat,
        healthz,
        messages,
        metrics,
        models,
        responses,
        version,
    )

    app.include_router(healthz.router)
    app.include_router(version.router)
    app.include_router(metrics.router)
    app.include_router(models.router, prefix="/v1")
    app.include_router(chat.router, prefix="/v1")
    app.include_router(messages.router, prefix="/v1")
    app.include_router(responses.router, prefix="/v1")

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

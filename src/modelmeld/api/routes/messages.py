# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Anthropic Messages API surface.

`POST /v1/messages` — accepts Anthropic-format requests, translates to
the internal OpenAI shape, runs through the SAME pipeline used by
`/v1/chat/completions` (router, memory inject, scout, scrubber, cache,
adapter dispatch with failover, memory write-back, hooks), then
translates the result back to Anthropic shape.

This is the surface Claude Code and any anthropic-SDK client talks to
via `ANTHROPIC_BASE_URL=http://<gateway>:<port>`. Covers both
non-streaming and streaming.

Per D-5 in docs/design-anthropic-messages-api.md: v1 imports
shape-agnostic helpers from chat.py rather than duplicating ~500 lines.
A follow-up will lift the shared pieces into a sibling
_shared.py module once both routes are stable. The private-name imports
below are an intentional, time-bound coupling — not a long-term pattern.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from modelmeld.adapters import AdapterError
from modelmeld.adapters.anthropic_adapter import AnthropicAdapter
from modelmeld.api._safe_error_detail import safe_error_detail
from modelmeld.api.auth_detection import classify_authorization
from modelmeld.api.subscription_passthrough import (
    PassthroughVendor,
    resolve_passthrough_router,
)
from modelmeld.api.byok import (
    build_byok_adapters,
    extract_byok_credentials,
)
from modelmeld.api.routes.chat import (
    _apply_model_override,
    _auth_identity,
    _byok_required_detail,
    _chunk_content_tokens,
    _chunk_text_pieces,
    _count_tokens,
    _fire_failure,
    _fire_success,
    _fire_success_stream,
    _is_byok_required_error,
    _maybe_scrub,
    _routing_headers,
    _write_memory_turns,
    _write_memory_turns_streaming,
)
from modelmeld.api.routing_hints import (
    RoutingHintError,
    extract_hints_from_headers,
)
from modelmeld.api.schemas import ChatCompletion, ChatCompletionRequest, TextPart
from modelmeld.api.schemas_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from modelmeld.cache import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    CompletionCache,
    SemanticCompletionCache,
    cache_key_for_request,
    canonicalize_request_text,
    is_request_semantically_cacheable,
)
from modelmeld.hooks import HookRegistry
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    MemoryHeaderError,
    MemoryStore,
    assemble_context,
    extract_memory_identity,
    inject_into_request,
)
from modelmeld.privacy import Scrubber
from modelmeld.router import Router, RouterError
from modelmeld.tokens import TokenCounter
from modelmeld.translation import (
    OpenAIToAnthropicStreamTranslator,
    TranslationError,
    format_anthropic_sse,
    from_anthropic_request,
    to_anthropic_response,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Anthropic protocol headers the customer may set per request that we
# forward verbatim to the upstream Anthropic call. Without this list,
# beta features the customer activates silently fall back at our boundary.
# Lowercase per Starlette's case-insensitive header lookup contract.
_FORWARDED_ANTHROPIC_HEADERS = ("anthropic-beta", "anthropic-version")


def _collect_anthropic_extra_headers(
    request_headers: Any,
) -> dict[str, str]:
    """Extract Anthropic protocol headers from the incoming request for
    pass-through to the upstream Anthropic call."""
    out: dict[str, str] = {}
    for name in _FORWARDED_ANTHROPIC_HEADERS:
        value = request_headers.get(name)
        if value is not None:
            out[name] = value
    return out


async def _dispatch_chat(
    adapter: Any,
    request: ChatCompletionRequest,
    decision: Any,
    body: AnthropicMessagesRequest,
    extra_headers: dict[str, str] | None = None,
) -> ChatCompletion:
    """Route an /v1/messages request to the adapter, preserving the native
    Anthropic shape when the adapter is AnthropicAdapter.

    AnthropicAdapter accepts optional `native_request` + `extra_headers`
    kwargs; native_request lets it build from the original Anthropic body
    (preserving cache_control), and extra_headers forwards caller-set
    Anthropic protocol headers (anthropic-beta, anthropic-version) verbatim.
    Other adapters use the standard signature.
    """
    request = _apply_model_override(request, decision)
    if isinstance(adapter, AnthropicAdapter):
        # When routing through a frontier adapter (typically BYOK), the
        # native_request body still carries the original alias model id
        # (e.g., "anthropic/modelmeld-quality"). Anthropic returns 404 on
        # that. Rewrite the body's model field to the scout's pick if a
        # model_id_override is set.
        native_body = body
        if decision.model_id_override and decision.model_id_override != body.model:
            native_body = body.model_copy(update={"model": decision.model_id_override})
        return await adapter.chat(
            request, native_request=native_body, extra_headers=extra_headers,
        )
    return await adapter.chat(request)


async def _try_open_stream_native(
    adapter: Any,
    request: ChatCompletionRequest,
    body: AnthropicMessagesRequest,
    extra_headers: dict[str, str] | None = None,
    model_override: str | None = None,
):
    """Open a stream with optional native-Anthropic passthrough + extra
    headers forwarding. Same return shape as chat._try_open_stream:
    (aiter, first_chunk, error).

    `model_override` (when set) rewrites the native body's `model` field
    so the upstream Anthropic call uses the scout's chosen model id
    rather than the original alias (which Anthropic would 404 on).
    """
    if isinstance(adapter, AnthropicAdapter):
        native_body = body
        if model_override and model_override != body.model:
            native_body = body.model_copy(update={"model": model_override})
        aiter = adapter.stream_chat(
            request, native_request=native_body, extra_headers=extra_headers,
        ).__aiter__()
    else:
        aiter = adapter.stream_chat(request).__aiter__()
    try:
        first = await aiter.__anext__()
    except StopAsyncIteration:
        return aiter, None, None
    except AdapterError as e:
        return None, None, e
    return aiter, first, None


def _anthropic_json_response(
    payload: AnthropicMessagesResponse, base_response: Response,
) -> JSONResponse:
    """Serialize an Anthropic response model to JSON, omitting null fields
    (Anthropic's wire format leaves absent fields out rather than emitting
    explicit nulls), and propagate any headers we already set on `base_response`."""
    body = payload.model_dump(exclude_none=True)
    return JSONResponse(content=body, headers=dict(base_response.headers))


@router.post("/messages", response_model=None)
async def anthropic_messages(
    body: AnthropicMessagesRequest,
    fastapi_request: Request,
    response: Response,
) -> JSONResponse | StreamingResponse:
    rt: Router = fastapi_request.app.state.router
    scrubber: Scrubber | None = fastapi_request.app.state.scrubber
    hooks: HookRegistry = fastapi_request.app.state.hooks
    memory: MemoryStore | None = getattr(
        fastapi_request.app.state, "memory_store", None,
    )
    token_counter: TokenCounter | None = getattr(
        fastapi_request.app.state, "token_counter", None,
    )
    completion_cache: CompletionCache | None = getattr(
        fastapi_request.app.state, "completion_cache", None,
    )
    semantic_cache: SemanticCompletionCache | None = getattr(
        fastapi_request.app.state, "semantic_cache", None,
    )
    settings = getattr(fastapi_request.app.state, "settings", None)
    cache_ttl = int(getattr(settings, "cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS))
    semantic_threshold = float(getattr(
        settings, "semantic_cache_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD,
    ))
    tenant_resolver = getattr(fastapi_request.app.state, "tenant_config_resolver", None)
    identity = _auth_identity(fastapi_request)

    # Memory identity (same x-modelmeld-* headers as chat.py per D-4).
    try:
        mem_identity = extract_memory_identity(
            fastapi_request.headers,
            auth_tenant_id=(identity or {}).get("tenant_id") if identity else None,
            auth_user_id=(identity or {}).get("user_id") if identity else None,
        )
    except MemoryHeaderError as e:
        raise HTTPException(status_code=400, detail=f"invalid_memory_header: {e}") from e

    # Translate Anthropic wire format → internal OpenAI shape. Translation
    # errors are surfaced as 400 (client sent something we can't process,
    # e.g. an image content block in v1).
    try:
        internal_request = from_anthropic_request(body)
    except TranslationError as e:
        raise HTTPException(
            status_code=400, detail=f"translation_error: {e}",
        ) from e

    # Per-API-key model allowlist (operates on the original Anthropic-side
    # model string — what the customer asked for, not the routed-to id).
    allowlist = getattr(fastapi_request.state, "api_key_model_allowlist", None)
    if allowlist is not None and body.model not in allowlist:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{body.model}' is not in this API key's allowlist",
        )

    started = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    # Forward Anthropic protocol headers (anthropic-beta, anthropic-version)
    # verbatim when we route to an Anthropic upstream. Lets customers
    # activate beta features without our gateway swallowing them.
    extra_anthropic_headers = _collect_anthropic_extra_headers(
        fastapi_request.headers,
    )

    # Routing hints (same x-modelmeld-* hint headers as chat.py).
    try:
        hints = extract_hints_from_headers(fastapi_request.headers)
    except RoutingHintError as e:
        raise HTTPException(
            status_code=400, detail=f"invalid_routing_hint: {e}",
        ) from e

    # BYOK header extraction (mirrors chat.py — see modelmeld/api/byok.py).
    byok_creds = extract_byok_credentials(fastapi_request.headers.items())
    byok_adapters = build_byok_adapters(byok_creds) if not byok_creds.is_empty() else {}

    # Subscription passthrough (Sprint 5). When the inbound Authorization
    # carries an OAuth-bearer JWT and the operator has enabled the opt-in
    # flag, route via AnthropicAdapter in oauth-bearer mode (raw HTTP to
    # api.anthropic.com with Authorization: Bearer, no SDK) instead of
    # the normal capability router. Mirrors the chat.py /v1/chat/completions
    # wiring with vendor=ANTHROPIC instead of CODEX.
    settings_obj = getattr(fastapi_request.app.state, "settings", None)
    auth_classification = classify_authorization(
        fastapi_request.headers.get("authorization"),
    )
    passthrough_router = resolve_passthrough_router(
        auth_classification,
        vendor=PassthroughVendor.ANTHROPIC,
        allow_passthrough=bool(
            getattr(settings_obj, "allow_subscription_passthrough", False),
        ),
    )
    active_router: Router = passthrough_router or rt

    # Routing decision — scout sees the translated OpenAI-shape request.
    try:
        decision = await active_router.route(
            internal_request,
            hints=hints,
            extra_adapters=byok_adapters if byok_adapters else None,
        )
    except RouterError as e:
        await _fire_failure(
            hooks, request_id, started, internal_request, None, [], None,
            e, "router_error", identity,
        )
        if _is_byok_required_error(e, byok_creds):
            raise HTTPException(
                status_code=400,
                detail=_byok_required_detail(e),
            ) from e
        raise HTTPException(status_code=503, detail=safe_error_detail(e)) from e

    # Per-tenant cache TTL override.
    if (
        tenant_resolver is not None
        and mem_identity.tenant_id
        and mem_identity.tenant_id != ANONYMOUS_TENANT_ID
    ):
        try:
            tenant_cfg = await tenant_resolver.get(mem_identity.tenant_id)
            cache_ttl = tenant_cfg.resolved_cache_ttl(cache_ttl)
        except Exception:
            logger.exception(
                "tenant_config_resolver failed for tenant=%s; using default TTL",
                mem_identity.tenant_id,
            )

    # Memory injection. Operates on internal shape;
    # exact same call as the chat route.
    mem_context = await assemble_context(memory, mem_identity)
    internal_request = inject_into_request(internal_request, mem_context)
    outgoing, redactions = _maybe_scrub(internal_request, decision, scrubber)

    # Streaming path (chunk 6). Pre-count input_tokens so the synthesized
    # `message_start` event carries an accurate count (Anthropic wire format
    # puts input_tokens upfront in message_start; output_tokens streams in
    # via message_delta at the end). Falls back to the chat.py helper's
    # char-based estimate when no real token_counter is wired.
    if outgoing.stream:
        cache_status = "bypass" if completion_cache is not None else None
        input_tokens = _estimate_input_tokens(outgoing, token_counter)
        return await _stream_messages_with_failover(
            rt, decision, outgoing, redactions, hooks, request_id, started,
            identity, memory, mem_identity, token_counter,
            request_model=body.model,
            input_tokens=input_tokens,
            cache_status=cache_status,
            native_request=body,
            extra_anthropic_headers=extra_anthropic_headers,
        )

    # Cache lookup (exact-match + semantic). Cache is keyed by internal request
    # shape, so /v1/messages and /v1/chat/completions can share entries when
    # the prompt content is identical — cache hit just gets translated to
    # Anthropic shape on the way out.
    served_model = decision.model_id_override or outgoing.model
    semantic_cacheable = is_request_semantically_cacheable(outgoing)
    cache_key = (
        cache_key_for_request(
            outgoing, tenant_id=mem_identity.tenant_id, served_model=served_model,
        )
        if completion_cache is not None else None
    )

    # 1. Exact-match cache
    if cache_key is not None and completion_cache is not None:
        lookup = await completion_cache.get(cache_key)
        if lookup.hit and lookup.value is not None:
            response.headers["x-modelmeld-cache"] = "hit"
            for key, value in _routing_headers(decision, redactions, None).items():
                response.headers[key] = value
            await _fire_success(
                hooks, request_id, started, outgoing, decision, redactions, None,
                lookup.value, identity, cache_status="hit",
            )
            return _anthropic_json_response(
                to_anthropic_response(lookup.value, request_model=body.model),
                base_response=response,
            )

    # 2. Semantic cache
    if semantic_cache is not None and semantic_cacheable:
        prompt_text = canonicalize_request_text(outgoing, served_model=served_model)
        sem_lookup = await semantic_cache.search(
            prompt_text,
            tenant_id=mem_identity.tenant_id,
            served_model=served_model,
            similarity_threshold=semantic_threshold,
        )
        if sem_lookup.hit and sem_lookup.value is not None:
            response.headers["x-modelmeld-cache"] = "hit-semantic"
            for key, value in _routing_headers(decision, redactions, None).items():
                response.headers[key] = value
            # Backfill exact cache so identical follow-ups skip the embedding hop
            if completion_cache is not None and cache_key is not None:
                try:
                    await completion_cache.set(cache_key, sem_lookup.value, cache_ttl)
                except Exception:
                    logger.exception("backfill of exact cache after semantic hit failed")
            await _fire_success(
                hooks, request_id, started, outgoing, decision, redactions, None,
                sem_lookup.value, identity, cache_status="hit-semantic",
            )
            return _anthropic_json_response(
                to_anthropic_response(sem_lookup.value, request_model=body.model),
                base_response=response,
            )

    if (cache_key is None and completion_cache is not None) or (
        not semantic_cacheable and semantic_cache is not None
    ):
        response.headers["x-modelmeld-cache"] = "bypass"

    # Adapter call with failover (F-2 transient/permanent split).
    # When the chosen adapter is AnthropicAdapter, pass the original Anthropic
    # request body as `native_request` so the adapter can preserve
    # cache_control, image content blocks, and other Anthropic-specific shape
    # that the OpenAI internal form drops. Other adapters use the standard
    # translated form via the base interface.
    failover_from = None
    try:
        completion = await _dispatch_chat(
            decision.adapter, outgoing, decision, body, extra_anthropic_headers,
        )
    except AdapterError as primary:
        fallback = await rt.route_after_failure(decision, outgoing, error=primary)
        if fallback is None:
            await _fire_failure(
                hooks, request_id, started, outgoing, decision, redactions, None,
                primary, "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail=safe_error_detail(primary)) from primary
        failover_from = decision.tier
        decision = fallback
        try:
            completion = await _dispatch_chat(
                decision.adapter, outgoing, decision, body, extra_anthropic_headers,
            )
        except AdapterError as secondary:
            await _fire_failure(
                hooks, request_id, started, outgoing, decision, redactions,
                failover_from, secondary, "adapter_error", identity,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"primary failed: {safe_error_detail(primary)}; "
                    f"fallback failed: {safe_error_detail(secondary)}"
                ),
            ) from secondary

    # Routing + cache-status response headers.
    for key, value in _routing_headers(decision, redactions, failover_from).items():
        response.headers[key] = value

    # Cache write-through.
    if completion_cache is not None and cache_key is not None:
        response.headers["x-modelmeld-cache"] = "miss"
        try:
            await completion_cache.set(cache_key, completion, cache_ttl)
        except Exception:
            logger.exception("cache set failed for key %r", cache_key)
    if semantic_cache is not None and is_request_semantically_cacheable(outgoing):
        prompt_text = canonicalize_request_text(outgoing, served_model=served_model)
        try:
            await semantic_cache.store(
                prompt_text, completion,
                tenant_id=mem_identity.tenant_id, served_model=served_model,
                ttl_seconds=cache_ttl,
            )
        except Exception:
            logger.exception("semantic cache store failed")

    # Hook fire + memory write-back. Both operate on internal shape — no
    # adaptation needed.
    miss_status: str | None = None
    if completion_cache is not None or semantic_cache is not None:
        miss_status = "miss" if cache_key is not None else "bypass"
    await _fire_success(
        hooks, request_id, started, outgoing, decision, redactions, failover_from,
        completion, identity, cache_status=miss_status,
    )
    await _write_memory_turns(
        memory, mem_identity, outgoing, completion, decision, token_counter,
    )

    # Translate internal ChatCompletion → Anthropic response shape.
    # Pass body.model (the original) so capability-routing model overrides
    # don't leak the internal target id to the client.
    return _anthropic_json_response(
        to_anthropic_response(completion, request_model=body.model),
        base_response=response,
    )


# ---------------------------------------------------------------------------
# count_tokens endpoint — Anthropic compliance
# ---------------------------------------------------------------------------
# Anthropic's /v1/messages/count_tokens returns {"input_tokens": N} for a
# request body without actually running inference. Claude Code uses this
# for pre-flight token estimates in its UI (cost display, context-window
# headroom warnings). Without this endpoint we return 404 and Claude
# Code's cost UI breaks or shows wrong numbers.
#
# We compute locally rather than proxying upstream. Reasons:
#   - count_tokens calls aren't billed by Anthropic, but they still cost
#     network round-trips
#   - Our gateway doesn't always have an upstream Anthropic adapter
#     configured (per the monetization model, frontier keys don't touch
#     our infra by default — the customer's local gateway is the one
#     that talks to Anthropic)
#   - Our TokenCounter handles cross-provider tokenization; the count is
#     close enough for UI purposes
#
# Returns slightly different numbers than Anthropic would (different
# tokenizer for non-claude models) — that's intentional, since we may
# end up routing to a non-Anthropic upstream anyway. The number reflects
# what our gateway would charge against, not what Anthropic alone would.

@router.post("/messages/count_tokens")
async def anthropic_count_tokens(
    body: AnthropicMessagesRequest,
    fastapi_request: Request,
) -> JSONResponse:
    """Estimate input tokens for an Anthropic-format request."""
    token_counter: TokenCounter | None = getattr(
        fastapi_request.app.state, "token_counter", None,
    )
    try:
        internal_request = from_anthropic_request(body)
    except TranslationError as e:
        raise HTTPException(
            status_code=400, detail=f"translation_error: {e}",
        ) from e

    # Reuse the same estimator the streaming path uses — handles
    # multimodal TextPart content and falls back to char-based estimate
    # when no real counter is wired.
    input_tokens = _estimate_input_tokens(internal_request, token_counter)

    # Include tool definitions in the count when present — they consume
    # input tokens too (per Anthropic's API behavior).
    if body.tools:
        for tool in body.tools:
            name_tokens = len(tool.name) // 4 + 1
            desc_tokens = (
                _count_tokens(token_counter, tool.description, body.model)
                if tool.description else 0
            )
            schema_tokens = len(str(tool.input_schema)) // 4 + 1
            input_tokens += name_tokens + desc_tokens + schema_tokens

    return JSONResponse(content={"input_tokens": input_tokens})


# ---------------------------------------------------------------------------
# Streaming path (chunk 6)
# ---------------------------------------------------------------------------

def _estimate_input_tokens(
    request: ChatCompletionRequest,
    token_counter: TokenCounter | None,
) -> int:
    """Sum tokens across the request's messages for the streaming `message_start`
    event. Anthropic puts `input_tokens` upfront in `message_start` (unlike
    OpenAI, where prompt_tokens arrives at the end). We compute it here from
    the internal request shape.

    Falls back to a char-based estimate (1 token ≈ 4 chars) when no real
    counter is wired. Best-effort: the goal is "approximately right" so the
    Claude Code UI's "tokens used so far" display isn't misleading. Final
    accuracy lives in `output_tokens` on message_delta.
    """
    model = request.model
    total = 0
    for msg in request.messages:
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            total += _count_tokens(token_counter, content, model)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, TextPart):
                    total += _count_tokens(token_counter, part.text, model)
                # Non-text parts (images, audio) are out of scope for v1
                # input-token estimation. Their contribution is negligible
                # compared to text in typical Claude Code traffic.
    return total


async def _stream_messages_with_failover(
    rt: Router,
    decision: Any,           # RoutingDecision; imported via chat module
    request: ChatCompletionRequest,
    redactions: list,        # list[Redaction]; chat module owns the type
    hooks: HookRegistry,
    request_id: str,
    started: float,
    identity: dict[str, str | None] | None,
    memory: MemoryStore | None,
    mem_identity: Any,       # MemoryIdentity
    token_counter: TokenCounter | None,
    *,
    request_model: str,
    input_tokens: int,
    cache_status: str | None = None,
    native_request: object | None = None,
    extra_anthropic_headers: dict[str, str] | None = None,
) -> StreamingResponse:
    """Mirror of chat.py `_stream_with_failover` but emits Anthropic SSE.

    The failover decision logic is identical (try primary; on
    PermanentAdapterError or stream-open failure, try the routed fallback).
    Only the wire format of the streamed payload differs.

    `native_request` is the original Anthropic-format body; passed through
    so AnthropicAdapter can preserve cache_control breakpoints on upstream
    calls. Non-Anthropic adapters ignore it.
    """
    failover_from: str | None = None
    primary_aiter, first_chunk, primary_error = await _try_open_stream_native(
        decision.adapter,
        _apply_model_override(request, decision),
        native_request,  # type: ignore[arg-type]
        extra_anthropic_headers,
        model_override=decision.model_id_override,
    )

    if first_chunk is None and primary_aiter is None:
        fallback = await rt.route_after_failure(
            decision, request, error=primary_error,
        )
        if fallback is None:
            err = primary_error or AdapterError("primary stream open failed")
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions, None,
                err, "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail=safe_error_detail(err))
        failover_from = str(decision.tier)
        decision = fallback
        primary_aiter, first_chunk, _ = await _try_open_stream_native(
            decision.adapter,
            _apply_model_override(request, decision),
            native_request,  # type: ignore[arg-type]
            extra_anthropic_headers,
            model_override=decision.model_id_override,
        )
        if primary_aiter is None and first_chunk is None:
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions,
                failover_from, AdapterError("fallback stream open failed"),
                "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail="fallback stream open failed")

    headers = _routing_headers(decision, redactions, failover_from)
    if cache_status is not None:
        headers["x-modelmeld-cache"] = cache_status

    return StreamingResponse(
        _sse_anthropic_stream(
            primary_aiter, first_chunk, hooks, request_id, started,
            request, decision, redactions, failover_from, identity,
            memory, mem_identity, token_counter,
            request_model=request_model, input_tokens=input_tokens,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _sse_anthropic_stream(
    aiter,
    first,
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: Any,           # RoutingDecision
    redactions: list,
    failover_from: str | None,
    identity: dict[str, str | None] | None,
    memory: MemoryStore | None,
    mem_identity: Any,       # MemoryIdentity
    token_counter: TokenCounter | None,
    *,
    request_model: str,
    input_tokens: int,
):
    """Async generator producing Anthropic SSE bytes from an OpenAI chunk stream.

    State held across iterations:
      - The translator's block/index machinery (text vs tool_use blocks)
      - Accumulated assistant text for memory write-back at end
      - Running output_tokens estimate (replaced by real count from the
        final usage chunk if upstream emits one)
      - Any exception caught during streaming → reported via fire_failure
    """
    translator = OpenAIToAnthropicStreamTranslator(
        request_model=request_model, input_tokens=input_tokens,
    )
    output_tokens = 0
    accumulated_text: list[str] = []
    last_usage = None
    error: Exception | None = None

    try:
        if first is not None:
            for event in translator.translate_chunk(first):
                yield format_anthropic_sse(event)
            output_tokens += _chunk_content_tokens(first)
            accumulated_text.extend(_chunk_text_pieces(first))
            if first.usage is not None:
                last_usage = first.usage
        if aiter is not None:
            async for chunk in aiter:
                for event in translator.translate_chunk(chunk):
                    yield format_anthropic_sse(event)
                output_tokens += _chunk_content_tokens(chunk)
                accumulated_text.extend(_chunk_text_pieces(chunk))
                if chunk.usage is not None:
                    last_usage = chunk.usage
        # Emit closing events (content_block_stop, message_delta, message_stop)
        for event in translator.finalize():
            yield format_anthropic_sse(event)
        # NOTE: Anthropic SSE does NOT use a `data: [DONE]` terminator like
        # OpenAI. The `message_stop` event itself signals completion.
    except Exception as e:
        error = e

    if error is None:
        await _fire_success_stream(
            hooks, request_id, started, request, decision, redactions, failover_from,
            output_tokens, last_usage, identity,
        )
        await _write_memory_turns_streaming(
            memory, mem_identity, request, "".join(accumulated_text),
            output_tokens, decision, token_counter,
        )
    else:
        await _fire_failure(
            hooks, request_id, started, request, decision, redactions, failover_from,
            error, "stream_error", identity,
        )

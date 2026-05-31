# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from modelmeld.adapters import AdapterError, ProviderAdapter
from modelmeld.api._safe_error_detail import safe_error_detail
from modelmeld.api.byok import (
    build_byok_adapters,
    extract_byok_credentials,
)
from modelmeld.api.routing_hints import (
    RoutingHintError,
    extract_hints_from_headers,
)
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    TextPart,
    UserMessage,
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
from modelmeld.hooks import (
    HookRegistry,
    RedactionRecord,
    RequestCompletedEvent,
)
from modelmeld.memory import (
    ANONYMOUS_TENANT_ID,
    MemoryHeaderError,
    MemoryIdentity,
    MemoryStore,
    Role,
    assemble_context,
    extract_memory_identity,
    inject_into_request,
)
from modelmeld.privacy import Redaction, Scrubber
from modelmeld.router import Router, RouterError, RoutingDecision
from modelmeld.tokens import TokenCounter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    fastapi_request: Request,
    response: Response,
) -> ChatCompletion | StreamingResponse:
    rt: Router = fastapi_request.app.state.router
    scrubber: Scrubber | None = fastapi_request.app.state.scrubber
    hooks: HookRegistry = fastapi_request.app.state.hooks
    memory: MemoryStore | None = getattr(fastapi_request.app.state, "memory_store", None)
    token_counter: TokenCounter | None = getattr(fastapi_request.app.state, "token_counter", None)
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
    # Per-tenant cache TTL override. When enterprise has attached
    # a resolver and the request is tenant-bound, swap in the per-tenant TTL.
    tenant_resolver = getattr(fastapi_request.app.state, "tenant_config_resolver", None)
    identity = _auth_identity(fastapi_request)
    try:
        mem_identity = extract_memory_identity(
            fastapi_request.headers,
            auth_tenant_id=(identity or {}).get("tenant_id") if identity else None,
            auth_user_id=(identity or {}).get("user_id") if identity else None,
        )
    except MemoryHeaderError as e:
        raise HTTPException(status_code=400, detail=f"invalid_memory_header: {e}") from e

    # Enforce per-API-key model allowlist if one is set.
    allowlist = getattr(fastapi_request.state, "api_key_model_allowlist", None)
    if allowlist is not None and request.model not in allowlist:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{request.model}' is not in this API key's allowlist",
        )

    started = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    # Framework-supplied routing hints. Validation errors → 400.
    try:
        hints = extract_hints_from_headers(fastapi_request.headers)
    except RoutingHintError as e:
        raise HTTPException(status_code=400, detail=f"invalid_routing_hint: {e}") from e

    # BYOK header extraction. Customer can pass frontier-API keys via
    # `x-modelmeld-byok-{provider}` headers. Keys transit this request
    # only — never persisted to disk, never logged, never echoed.
    # See modelmeld/api/byok.py for the full contract.
    byok_creds = extract_byok_credentials(fastapi_request.headers.items())
    byok_adapters = build_byok_adapters(byok_creds) if not byok_creds.is_empty() else {}

    try:
        decision = await rt.route(
            request,
            hints=hints,
            extra_adapters=byok_adapters if byok_adapters else None,
        )
    except RouterError as e:
        await _fire_failure(
            hooks, request_id, started, request, None, [], None, e, "router_error", identity
        )
        # If the scout picked a frontier provider and we have no adapter
        # for it (neither persistent nor BYOK), surface a 400 with the
        # exact header the customer needs to set — much friendlier than
        # the generic 503 the underlying RouterError yields.
        if _is_byok_required_error(e, byok_creds):
            raise HTTPException(
                status_code=400,
                detail=_byok_required_detail(e),
            ) from e
        raise HTTPException(status_code=503, detail=safe_error_detail(e)) from e

    # Resolve per-tenant cache TTL override once we know the
    # authenticated tenant_id. Skip for anonymous traffic (no row to look up)
    # and on resolver errors (fail-open with the global default).
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

    # Memory injection. Prepend L1+L2 (and L3 in FULL mode)
    # BEFORE scrub so injected content is also scrubbed on egress.
    mem_context = await assemble_context(memory, mem_identity)
    request = inject_into_request(request, mem_context)
    outgoing, redactions = _maybe_scrub(request, decision, scrubber)
    # Note: model override is applied inside the failover helpers per-attempt so
    # that, on fallback, we re-override with the fallback decision's model_id.

    if outgoing.stream:
        # Streaming requests bypass the cache entirely. The
        # bypass header is threaded through so the StreamingResponse's
        # outgoing headers carry it (setting `response.headers` here would
        # be silently lost — StreamingResponse uses its own headers).
        cache_status = "bypass" if completion_cache is not None else None
        return await _stream_with_failover(
            rt, decision, outgoing, redactions, hooks, request_id, started,
            identity, memory, mem_identity, token_counter,
            cache_status=cache_status,
        )

    # Exact-match cache, then semantic cache fallback.
    # Tools / n>1 / stream bypass both layers.
    served_model = decision.model_id_override or outgoing.model
    semantic_cacheable = is_request_semantically_cacheable(outgoing)
    cache_key = (
        cache_key_for_request(outgoing, tenant_id=mem_identity.tenant_id, served_model=served_model)
        if completion_cache is not None else None
    )

    # 1. Exact-match cache lookup (cheap, no embedding API call).
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
            return lookup.value

    # 2. Semantic cache lookup. Same gates as exact-match.
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
            # Backfill the exact-match cache so identical follow-ups skip
            # the embedding round-trip entirely.
            if completion_cache is not None and cache_key is not None:
                try:
                    await completion_cache.set(cache_key, sem_lookup.value, cache_ttl)
                except Exception:
                    logger.exception("backfill of exact cache after semantic hit failed")
            await _fire_success(
                hooks, request_id, started, outgoing, decision, redactions, None,
                sem_lookup.value, identity, cache_status="hit-semantic",
            )
            return sem_lookup.value

    # Bypass header for uncacheable requests when at least one cache is wired.
    if (cache_key is None and completion_cache is not None) or (
        not semantic_cacheable and semantic_cache is not None
    ):
        response.headers["x-modelmeld-cache"] = "bypass"

    return await _completion_with_failover(
        rt, decision, outgoing, redactions, hooks, request_id, started, response,
        identity, memory, mem_identity, token_counter,
        completion_cache=completion_cache, cache_key=cache_key, cache_ttl=cache_ttl,
        semantic_cache=semantic_cache, served_model=served_model,
    )


_FRONTIER_MODEL_PREFIXES = ("claude-", "gpt-")


def _is_byok_required_error(error: Exception, byok_creds) -> bool:
    """Return True iff the RouterError indicates a frontier model was
    chosen but no BYOK adapter was supplied to dispatch it.

    Two patterns can produce this:
    1. Base CapabilityRouter: `<provider>:not_configured` in skipped list
    2. MultiProviderCapabilityRouter: `primary=<frontier-provider>` plus
       a list of `<frontier-model>:all_providers_blocked` entries (the
       picker's OSS-only allowlist rejects every frontier provider).
    """
    from modelmeld.api.byok import eligible_providers
    msg = str(error)
    if "No healthy adapter available" not in msg:
        return False
    # Pattern 1: explicit not_configured marker
    for provider in eligible_providers():
        if f"{provider}:not_configured" in msg and byok_creds.get(provider) is None:
            return True
    # Pattern 2: primary=<frontier> + frontier models all_providers_blocked
    if any(f"primary={p}" in msg for p in eligible_providers()):
        # The picker rejected all frontier models because they aren't in
        # the OSS upstream allowlist. If the customer didn't supply ANY
        # BYOK for the relevant frontier providers, byok_required is
        # the right error.
        for provider in eligible_providers():
            for prefix in _FRONTIER_MODEL_PREFIXES:
                if f"{prefix}" in msg and ":all_providers_blocked" in msg:
                    if byok_creds.get(provider) is None:
                        return True
    return False


def _byok_required_detail(error: Exception) -> dict[str, str]:
    """Build the 400-response body telling the customer which BYOK header
    they need to set. Falls back to a generic message when we can't
    pinpoint a specific provider."""
    from modelmeld.api.byok import eligible_providers
    msg = str(error)
    primary: str | None = None
    # First try: explicit not_configured marker → that provider
    for provider in eligible_providers():
        if f"{provider}:not_configured" in msg:
            primary = provider
            break
    # Fallback: primary=<provider> in the error message
    if primary is None:
        for provider in eligible_providers():
            if f"primary={provider}" in msg:
                primary = provider
                break
    # Last resort: anthropic is the most common BYOK target
    if primary is None:
        primary = "anthropic"
    return {
        "error": "byok_required",
        "detail": (
            f"This routing decision picked a {primary} model, but no "
            f"{primary} API key was supplied. Set the header "
            f"`x-modelmeld-byok-{primary}: <your-{primary}-api-key>` to "
            f"enable frontier routing, or pick `anthropic/modelmeld-saver` "
            f"to stay on OSS routing only. For Claude Code, set "
            f"ANTHROPIC_CUSTOM_HEADERS='x-modelmeld-byok-{primary}: sk-...'."
        ),
        "provider_missing": primary,
    }


def _auth_identity(fastapi_request: Request) -> dict[str, str | None] | None:
    """Extract tenant/user/api_key identity populated by the enterprise auth middleware."""
    tenant_id = getattr(fastapi_request.state, "tenant_id", None)
    if tenant_id is None:
        return None
    return {
        "tenant_id": tenant_id,
        "user_id": getattr(fastapi_request.state, "user_id", None),
        "api_key_id": getattr(fastapi_request.state, "api_key_id", None),
    }


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------

async def _completion_with_failover(
    rt: Router,
    decision: RoutingDecision,
    request: ChatCompletionRequest,
    redactions: list[Redaction],
    hooks: HookRegistry,
    request_id: str,
    started: float,
    response: Response,
    identity: dict[str, str | None] | None,
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    token_counter: TokenCounter | None,
    *,
    completion_cache: CompletionCache | None = None,
    cache_key: str | None = None,
    cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    semantic_cache: SemanticCompletionCache | None = None,
    served_model: str | None = None,
) -> ChatCompletion:
    failover_from = None
    try:
        completion = await decision.adapter.chat(_apply_model_override(request, decision))
    except AdapterError as primary:
        fallback = await rt.route_after_failure(decision, request, error=primary)
        if fallback is None:
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions, None, primary, "adapter_error", identity
            )
            raise HTTPException(status_code=502, detail=safe_error_detail(primary)) from primary
        failover_from = decision.tier
        decision = fallback
        try:
            completion = await decision.adapter.chat(_apply_model_override(request, decision))
        except AdapterError as secondary:
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions, failover_from, secondary, "adapter_error", identity
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"primary failed: {safe_error_detail(primary)}; "
                    f"fallback failed: {safe_error_detail(secondary)}"
                ),
            ) from secondary

    for key, value in _routing_headers(decision, redactions, failover_from).items():
        response.headers[key] = value
    if completion_cache is not None and cache_key is not None:
        response.headers["x-modelmeld-cache"] = "miss"
        try:
            await completion_cache.set(cache_key, completion, cache_ttl)
        except Exception:
            logger.exception("cache set failed for key %r", cache_key)
    # Semantic cache write — same gating as the read path.
    if semantic_cache is not None and is_request_semantically_cacheable(request):
        prompt_text = canonicalize_request_text(request, served_model=served_model)
        try:
            await semantic_cache.store(
                prompt_text, completion,
                tenant_id=mem_identity.tenant_id, served_model=served_model,
                ttl_seconds=cache_ttl,
            )
        except Exception:
            logger.exception("semantic cache store failed")

    # Miss vs bypass: we only "missed" if we had a viable cache_key to look up.
    # Otherwise (no cache configured, or uncacheable request) it's a bypass.
    miss_status: str | None = None
    if completion_cache is not None or semantic_cache is not None:
        miss_status = "miss" if cache_key is not None else "bypass"
    await _fire_success(
        hooks, request_id, started, request, decision, redactions, failover_from,
        completion, identity, cache_status=miss_status,
    )
    await _write_memory_turns(memory, mem_identity, request, completion, decision, token_counter)
    return completion


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------

async def _stream_with_failover(
    rt: Router,
    decision: RoutingDecision,
    request: ChatCompletionRequest,
    redactions: list[Redaction],
    hooks: HookRegistry,
    request_id: str,
    started: float,
    identity: dict[str, str | None] | None,
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    token_counter: TokenCounter | None,
    *,
    cache_status: str | None = None,
) -> StreamingResponse:
    failover_from: str | None = None
    primary_aiter, first_chunk, primary_error = await _try_open_stream(
        decision.adapter, _apply_model_override(request, decision)
    )

    if first_chunk is None and primary_aiter is None:
        fallback = await rt.route_after_failure(decision, request, error=primary_error)
        if fallback is None:
            err = primary_error or AdapterError("primary stream open failed")
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions, None,
                err, "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail=safe_error_detail(err))
        failover_from = str(decision.tier)
        decision = fallback
        primary_aiter, first_chunk, _ = await _try_open_stream(
            decision.adapter, _apply_model_override(request, decision)
        )
        if primary_aiter is None and first_chunk is None:
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions, failover_from,
                AdapterError("fallback stream open failed"), "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail="fallback stream open failed")

    headers = _routing_headers(decision, redactions, failover_from)
    if cache_status is not None:
        headers["x-modelmeld-cache"] = cache_status
    return StreamingResponse(
        _sse_stream(
            primary_aiter, first_chunk, hooks, request_id, started,
            request, decision, redactions, failover_from, identity,
            memory, mem_identity, token_counter,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _try_open_stream(
    adapter: ProviderAdapter, request: ChatCompletionRequest
) -> tuple[
    AsyncIterator[ChatCompletionChunk] | None,
    ChatCompletionChunk | None,
    AdapterError | None,
]:
    """Open a stream, returning (aiter, first_chunk, error).

    On success: (aiter, first_chunk_or_None, None). On adapter failure:
    (None, None, the_error) - caller uses the error to decide whether
    failover is appropriate (F-2: PermanentAdapterError → no failover).
    """
    aiter = adapter.stream_chat(request).__aiter__()
    try:
        first = await aiter.__anext__()
    except StopAsyncIteration:
        return aiter, None, None
    except AdapterError as e:
        return None, None, e
    return aiter, first, None


async def _sse_stream(
    aiter: AsyncIterator[ChatCompletionChunk] | None,
    first: ChatCompletionChunk | None,
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: RoutingDecision,
    redactions: list[Redaction],
    failover_from: str | None,
    identity: dict[str, str | None] | None,
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    token_counter: TokenCounter | None,
) -> AsyncIterator[str]:
    output_tokens = 0
    last_usage = None
    accumulated_text: list[str] = []
    error: Exception | None = None
    try:
        if first is not None:
            yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
            output_tokens += _chunk_content_tokens(first)
            accumulated_text.extend(_chunk_text_pieces(first))
            if first.usage is not None:
                last_usage = first.usage
        if aiter is not None:
            async for chunk in aiter:
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                output_tokens += _chunk_content_tokens(chunk)
                accumulated_text.extend(_chunk_text_pieces(chunk))
                if chunk.usage is not None:
                    last_usage = chunk.usage
    except Exception as e:
        error = e
    yield "data: [DONE]\n\n"

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


def _chunk_content_tokens(chunk: ChatCompletionChunk) -> int:
    """Cheap token estimate from a chunk's text content (1 token ≈ 4 chars)."""
    total = 0
    for choice in chunk.choices:
        if choice.delta.content:
            total += max(1, len(choice.delta.content) // 4)
    return total


def _chunk_text_pieces(chunk: ChatCompletionChunk) -> list[str]:
    """Pull out the assistant's text content from a streaming chunk."""
    pieces: list[str] = []
    for choice in chunk.choices:
        if choice.delta.content:
            pieces.append(choice.delta.content)
    return pieces


# ---------------------------------------------------------------------------
# Memory writes
# ---------------------------------------------------------------------------

async def _write_memory_turns(
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    request: ChatCompletionRequest,
    completion: ChatCompletion,
    decision: RoutingDecision,
    token_counter: TokenCounter | None,
) -> None:
    """Append the user prompt + assistant response to L0. Best-effort.

    Memory failures NEVER break the request. The user already got their answer;
    losing a turn write is recoverable from L0 elsewhere (eventually) but a
    500 isn't.
    """
    if memory is None or not mem_identity.active:
        return
    assert mem_identity.session_id is not None  # narrow for type checker
    model_used = decision.model_id_override or request.model
    assistant_text = _extract_completion_text(completion)
    if completion.usage and completion.usage.completion_tokens:
        assistant_tokens = completion.usage.completion_tokens
    else:
        assistant_tokens = _count_tokens(token_counter, assistant_text, model_used)
    await _append_request_response_turns(
        memory=memory,
        mem_identity=mem_identity,
        request=request,
        assistant_text=assistant_text,
        assistant_tokens=assistant_tokens,
        model_used=model_used,
        token_counter=token_counter,
    )


async def _write_memory_turns_streaming(
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    request: ChatCompletionRequest,
    assistant_text: str,
    assistant_tokens: int,
    decision: RoutingDecision,
    token_counter: TokenCounter | None,
) -> None:
    if memory is None or not mem_identity.active:
        return
    model_used = decision.model_id_override or request.model
    # Streaming accumulator gave us a char-based count; if we have a real
    # counter, recount the full reassembled text now that the stream is done.
    if token_counter is not None and assistant_text:
        assistant_tokens = token_counter.count_text(assistant_text, model_used)
    await _append_request_response_turns(
        memory=memory,
        mem_identity=mem_identity,
        request=request,
        assistant_text=assistant_text,
        assistant_tokens=assistant_tokens,
        model_used=model_used,
        token_counter=token_counter,
    )


async def _append_request_response_turns(
    memory: MemoryStore,
    mem_identity: MemoryIdentity,
    request: ChatCompletionRequest,
    assistant_text: str,
    assistant_tokens: int,
    model_used: str,
    token_counter: TokenCounter | None,
) -> None:
    """Shared write path for both streaming + non-streaming success."""
    try:
        await memory.get_or_create_session(
            session_id=mem_identity.session_id,  # type: ignore[arg-type]
            tenant_id=mem_identity.tenant_id,
            user_id=mem_identity.user_id,
        )
        # Append the last user turn from the incoming request (the prompt the
        # user just sent). We only log the newest user message — prior turns
        # came from earlier requests and are already in L0.
        last_user = _last_user_turn(request, token_counter, model_used)
        if last_user is not None:
            user_text, user_tokens = last_user
            await memory.append_turn(
                session_id=mem_identity.session_id,  # type: ignore[arg-type]
                tenant_id=mem_identity.tenant_id,
                role=Role.USER,
                content=user_text,
                token_count=user_tokens,
                model_used=model_used,
            )
        await memory.append_turn(
            session_id=mem_identity.session_id,  # type: ignore[arg-type]
            tenant_id=mem_identity.tenant_id,
            role=Role.ASSISTANT,
            content=assistant_text,
            token_count=assistant_tokens,
            model_used=model_used,
        )
    except Exception:
        logger.exception("memory write failed for session %s", mem_identity.session_id)


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
    counter: TokenCounter | None, text: str, model: str | None,
) -> int:
    """Count tokens via the configured counter; char-based fallback if unset."""
    if counter is not None:
        return counter.count_text(text, model)
    return max(1, len(text) // 4) if text else 0


def _extract_completion_text(completion: ChatCompletion) -> str:
    """Pull out the assistant's plain-text content from a non-streaming completion."""
    pieces: list[str] = []
    for choice in completion.choices:
        msg = choice.message
        if msg.content:
            if isinstance(msg.content, str):
                pieces.append(msg.content)
            else:
                pieces.extend(
                    part.text for part in msg.content if isinstance(part, TextPart)  # pyright: ignore[reportGeneralTypeIssues]
                )
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Hook firing helpers
# ---------------------------------------------------------------------------

async def _fire_success(
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: RoutingDecision,
    redactions: list[Redaction],
    failover_from: object | None,
    completion: ChatCompletion,
    identity: dict[str, str | None] | None,
    cache_status: str | None = None,
) -> None:
    if not hooks.subscriber_count:
        return
    usage = completion.usage
    event = _build_event(
        request_id, started, request, decision, redactions, failover_from,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        error=None,
        error_type=None,
        identity=identity,
        cache_status=cache_status,
    )
    await hooks.fire_on_request_complete(event)


async def _fire_success_stream(
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: RoutingDecision,
    redactions: list[Redaction],
    failover_from: object | None,
    estimated_output_tokens: int,
    last_usage: object | None,
    identity: dict[str, str | None] | None,
) -> None:
    if not hooks.subscriber_count:
        return
    if last_usage is not None:
        in_tok = last_usage.prompt_tokens  # type: ignore[attr-defined]
        out_tok = last_usage.completion_tokens  # type: ignore[attr-defined]
        total = last_usage.total_tokens  # type: ignore[attr-defined]
    else:
        in_tok = 0
        out_tok = estimated_output_tokens
        total = estimated_output_tokens
    event = _build_event(
        request_id, started, request, decision, redactions, failover_from,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=total,
        error=None,
        error_type=None,
        identity=identity,
    )
    await hooks.fire_on_request_complete(event)


async def _fire_failure(
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: RoutingDecision | None,
    redactions: list[Redaction],
    failover_from: object | None,
    error: BaseException,
    error_type: str,
    identity: dict[str, str | None] | None,
) -> None:
    if not hooks.subscriber_count:
        return
    event = _build_event(
        request_id, started, request, decision, redactions, failover_from,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        error=str(error),
        error_type=error_type,
        identity=identity,
    )
    await hooks.fire_on_request_complete(event)


def _build_event(
    request_id: str,
    started: float,
    request: ChatCompletionRequest,
    decision: RoutingDecision | None,
    redactions: list[Redaction],
    failover_from: object | None,
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    error: str | None,
    error_type: str | None,
    identity: dict[str, str | None] | None = None,
    cache_status: str | None = None,
) -> RequestCompletedEvent:
    devtool = "unknown"
    devtool_conf = 0.0
    if decision is not None and decision.scout_decision is not None:
        devtool = str(decision.scout_decision.signals.get("devtool", "unknown"))
        devtool_conf = float(decision.scout_decision.signals.get("devtool_confidence", 0.0))

    routed_to = decision.adapter.name if decision is not None else ""
    tier = str(decision.tier) if decision is not None else ""
    # Capability routing fields (FinOps consumes these for savings math).
    model_served = request.model
    task_category = None
    quality_threshold: float | None = None
    if decision is not None:
        if decision.model_id_override:
            model_served = decision.model_id_override
        cap = getattr(decision, "capability_decision", None)
        if cap is not None:
            task_category = getattr(cap, "task_category", None)
            quality_threshold = getattr(cap, "quality_threshold", None)

    return RequestCompletedEvent(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        requested_model=request.model,
        devtool=devtool,
        devtool_confidence=devtool_conf,
        prompt_hash=_prompt_hash(request),
        routed_to=routed_to,
        tier=tier,
        failover_from=str(failover_from) if failover_from is not None else None,
        model_served=model_served,
        task_category=task_category,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        redactions=tuple(RedactionRecord(label=r.label, count=r.count) for r in redactions),
        error=error,
        error_type=error_type,
        tenant_id=(identity or {}).get("tenant_id"),
        user_id=(identity or {}).get("user_id"),
        api_key_id=(identity or {}).get("api_key_id"),
        cache_status=cache_status,
        quality_threshold=quality_threshold,
        requires_tool_use=bool(getattr(request, "tools", None)),
    )


def _prompt_hash(request: ChatCompletionRequest) -> str:
    """SHA-256 of the canonicalized request body. Stable across runs."""
    canonical = json.dumps(
        request.model_dump(exclude_none=True),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# PII + headers helpers
# ---------------------------------------------------------------------------

def _maybe_scrub(
    request: ChatCompletionRequest,
    decision: RoutingDecision,
    scrubber: Scrubber | None,
) -> tuple[ChatCompletionRequest, list[Redaction]]:
    if scrubber is None or not decision.adapter.is_egress:
        return request, []
    return scrubber.scrub_request(request)


def _apply_model_override(
    request: ChatCompletionRequest,
    decision: RoutingDecision,
) -> ChatCompletionRequest:
    """For CAPABILITY routing: rewrite `request.model` to the scout's pick.

    Other policies leave the request untouched.
    """
    if decision.model_id_override is None:
        return request
    if decision.model_id_override == request.model:
        return request
    return request.model_copy(update={"model": decision.model_id_override})


def _routing_headers(
    decision: RoutingDecision,
    redactions: list[Redaction],
    failover_from: object | None,
) -> dict[str, str]:
    headers = {
        "x-modelmeld-routed-to": decision.adapter.name,
        "x-modelmeld-tier": str(decision.tier),
    }
    if decision.model_id_override:
        headers["x-modelmeld-routed-model"] = decision.model_id_override
    cap = getattr(decision, "capability_decision", None)
    if cap is not None:
        headers["x-modelmeld-task-category"] = cap.task_category
        headers["x-modelmeld-task-score"] = f"{cap.task_score:.2f}"
        headers["x-modelmeld-quality-threshold"] = f"{cap.quality_threshold:.2f}"
        # Echo the source so framework authors can verify the hint took effect.
        rationale = cap.rationale or ""
        if "src=hint:task_category" in rationale:
            headers["x-modelmeld-category-source"] = "hint:task_category"
        elif "src=hint:agent_role" in rationale:
            headers["x-modelmeld-category-source"] = "hint:agent_role"
        elif "src=classifier" in rationale:
            headers["x-modelmeld-category-source"] = "classifier"
        # Surface the dev-tool fingerprint + shape bias
        # so the audit trail is observable from the client side. The
        # fingerprint lets operators slice telemetry by detected client
        # (which is real per-tool savings analytics) and the bias header
        # lets customers see "your autocomplete got routed cheap because
        # of shape detection" without having to read the rationale.
        fp = cap.devtool_fingerprint
        if fp is not None and fp.tool.value != "unknown":
            headers["x-modelmeld-devtool"] = f"{fp.tool.value}:{fp.confidence:.2f}"
        if "bias=" in rationale:
            # Extract the bias-name token after `bias=` for the header.
            bias_segment = rationale.split("bias=", 1)[1]
            bias_name = bias_segment.split("(", 1)[0].split(";", 1)[0]
            if bias_name:
                headers["x-modelmeld-bias"] = bias_name
    if redactions:
        headers["x-modelmeld-redactions"] = ",".join(
            f"{r.label}:{r.count}" for r in redactions
        )
    if failover_from is not None:
        headers["x-modelmeld-failover-from"] = str(failover_from)
    return headers

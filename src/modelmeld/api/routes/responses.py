# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""POST /v1/responses — OpenAI Responses API surface (Codex CLI plug-and-play).

The handler reuses the chat pipeline: translate the Responses request to the
internal Chat shape, run the same routing / BYOK / subscription-passthrough /
PII-scrub / capability-routing path, then translate back. Non-streaming goes
through `_completion_with_failover`; streaming emits Responses SSE events via
`ResponsesStreamTranslator`. Audit headers, hooks, and memory write-back are
shared with the chat route.

Streaming covers both assistant text and tool calls: the translator opens a
`message` item for text and a `function_call` item per tool call, emitting
`response.function_call_arguments.*` events as argument fragments arrive.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from modelmeld.adapters import AdapterError
from modelmeld.api._safe_error_detail import safe_error_detail
from modelmeld.api.auth_detection import classify_authorization
from modelmeld.api.byok import build_byok_adapters, extract_byok_credentials
from modelmeld.api.routes.chat import (
    _apply_model_override,
    _auth_identity,
    _byok_required_detail,
    _canonical_model_id_for_header,
    _chunk_content_tokens,
    _chunk_text_pieces,
    _completion_with_failover,
    _fire_failure,
    _fire_success_stream,
    _is_byok_required_error,
    _is_no_eligible_model_error,
    _maybe_scrub,
    _no_eligible_model_detail,
    _routing_headers,
    _try_open_stream,
    _write_memory_turns_streaming,
)
from modelmeld.api.routing_hints import RoutingHintError, extract_hints_from_headers
from modelmeld.api.schemas_responses import ResponsesRequest
from modelmeld.api.subscription_passthrough import (
    PassthroughVendor,
    resolve_passthrough_router,
)
from modelmeld.hooks import HookRegistry
from modelmeld.memory import (
    MemoryContext,
    MemoryHeaderError,
    MemoryProvider,
    extract_memory_identity,
    inject_into_request,
)
from modelmeld.privacy import Scrubber
from modelmeld.router import Router, RouterError
from modelmeld.tokens import TokenCounter
from modelmeld.translation.openai_responses import (
    from_responses_request,
    to_responses_response,
)
from modelmeld.translation.responses_stream import (
    ResponsesStreamTranslator,
    format_responses_sse,
)

router = APIRouter()


@router.post("/responses", response_model=None)
async def responses(
    body: ResponsesRequest,
    fastapi_request: Request,
    response: Response,
) -> JSONResponse | StreamingResponse:
    rt: Router = fastapi_request.app.state.router
    scrubber: Scrubber | None = fastapi_request.app.state.scrubber
    hooks: HookRegistry = fastapi_request.app.state.hooks
    provider: MemoryProvider | None = getattr(
        fastapi_request.app.state, "memory_provider", None,
    )
    token_counter: TokenCounter | None = getattr(
        fastapi_request.app.state, "token_counter", None,
    )
    settings = getattr(fastapi_request.app.state, "settings", None)
    model_registry = getattr(fastapi_request.app.state, "model_registry", None)
    identity = _auth_identity(fastapi_request)

    try:
        mem_identity = extract_memory_identity(
            fastapi_request.headers,
            auth_tenant_id=(identity or {}).get("tenant_id") if identity else None,
            auth_user_id=(identity or {}).get("user_id") if identity else None,
        )
    except MemoryHeaderError as e:
        raise HTTPException(status_code=400, detail=f"invalid_memory_header: {e}") from e

    allowlist = getattr(fastapi_request.state, "api_key_model_allowlist", None)
    if allowlist is not None and body.model not in allowlist:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{body.model}' is not in this API key's allowlist",
        )

    started = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:24]}"

    try:
        hints = extract_hints_from_headers(fastapi_request.headers)
    except RoutingHintError as e:
        raise HTTPException(status_code=400, detail=f"invalid_routing_hint: {e}") from e

    byok_creds = extract_byok_credentials(fastapi_request.headers.items())
    byok_adapters = build_byok_adapters(byok_creds) if not byok_creds.is_empty() else {}

    # OAuth-bearer requests route through the Codex passthrough adapter, the
    # same vendor the /v1/chat/completions path uses.
    auth_classification = classify_authorization(
        fastapi_request.headers.get("authorization"),
    )
    passthrough_router = resolve_passthrough_router(
        auth_classification,
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=bool(getattr(settings, "allow_subscription_passthrough", False)),
    )
    active_router: Router = passthrough_router or rt

    internal_request = from_responses_request(body)

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
            raise HTTPException(status_code=400, detail=_byok_required_detail(e)) from e
        if _is_no_eligible_model_error(e):
            raise HTTPException(status_code=400, detail=_no_eligible_model_detail(e)) from e
        raise HTTPException(status_code=503, detail=safe_error_detail(e)) from e

    # Memory injection before scrub so injected context is scrubbed on egress.
    mem_context = (
        await provider.retrieve(mem_identity, internal_request)
        if provider is not None else MemoryContext()
    )
    internal_request = inject_into_request(internal_request, mem_context)
    outgoing, redactions = _maybe_scrub(internal_request, decision, scrubber)

    if body.stream:
        return await _stream_responses_with_failover(
            rt, decision, outgoing, redactions, hooks, request_id, started,
            identity, provider, mem_identity, token_counter,
            hints=hints, model_registry=model_registry,
        )

    # Non-streaming. Shared dispatch: failover, audit headers (on `response`),
    # hooks, memory write-back, and the canonical-model pin all happen here.
    completion = await _completion_with_failover(
        rt, decision, outgoing, redactions, hooks, request_id, started, response,
        identity, provider, mem_identity, token_counter,
        hints=hints, model_registry=model_registry,
    )

    payload = to_responses_response(completion, model=completion.model)
    return JSONResponse(
        content=payload.model_dump(exclude_none=True),
        headers=dict(response.headers),
    )


# ---------------------------------------------------------------------------
# Streaming (Responses SSE)
# ---------------------------------------------------------------------------

def _estimate_input_tokens(
    request, served_model: str, token_counter: TokenCounter | None,
) -> int:
    """Best-effort prompt-token count for streamed responses whose upstream
    omitted usage. Uses the configured counter; falls back to ≈4 chars/token."""
    messages = getattr(request, "messages", None) or []
    if token_counter is not None:
        return token_counter.count_messages(messages, served_model)
    chars = 0
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, str):
            chars += len(content)
    return max(1, chars // 4) if chars else 0

async def _stream_responses_with_failover(
    rt: Router,
    decision,
    request,
    redactions: list,
    hooks: HookRegistry,
    request_id: str,
    started: float,
    identity: dict[str, str | None] | None,
    provider: MemoryProvider | None,
    mem_identity,
    token_counter: TokenCounter | None,
    *,
    hints=None,
    model_registry=None,
) -> StreamingResponse:
    failover_from: str | None = None
    primary_aiter, first_chunk, primary_error = await _try_open_stream(
        decision.adapter, _apply_model_override(request, decision),
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
            decision.adapter, _apply_model_override(request, decision),
        )
        if primary_aiter is None and first_chunk is None:
            await _fire_failure(
                hooks, request_id, started, request, decision, redactions,
                failover_from, AdapterError("fallback stream open failed"),
                "adapter_error", identity,
            )
            raise HTTPException(status_code=502, detail="fallback stream open failed")

    headers = _routing_headers(
        decision, redactions, failover_from, hints=hints, model_registry=model_registry,
    )
    served_model = (
        _canonical_model_id_for_header(
            decision.model_id_override, decision.adapter.name, model_registry,
        )
        or decision.model_id_override
        or request.model
    )
    return StreamingResponse(
        _sse_responses(
            primary_aiter, first_chunk, hooks, request_id, started, request,
            decision, redactions, failover_from, identity, provider, mem_identity,
            token_counter, served_model,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _sse_responses(
    aiter,
    first,
    hooks: HookRegistry,
    request_id: str,
    started: float,
    request,
    decision,
    redactions: list,
    failover_from: str | None,
    identity: dict[str, str | None] | None,
    provider: MemoryProvider | None,
    mem_identity,
    token_counter: TokenCounter | None,
    served_model: str,
):
    translator = ResponsesStreamTranslator(model=served_model)
    output_tokens = 0
    last_usage = None
    accumulated_text: list[str] = []
    error: Exception | None = None
    try:
        if first is not None:
            for event in translator.translate_chunk(first):
                yield format_responses_sse(event)
            output_tokens += _chunk_content_tokens(first)
            accumulated_text.extend(_chunk_text_pieces(first))
            if first.usage is not None:
                last_usage = first.usage
        if aiter is not None:
            async for chunk in aiter:
                for event in translator.translate_chunk(chunk):
                    yield format_responses_sse(event)
                output_tokens += _chunk_content_tokens(chunk)
                accumulated_text.extend(_chunk_text_pieces(chunk))
                if chunk.usage is not None:
                    last_usage = chunk.usage
    except Exception as e:
        error = e

    if error is None:
        # Responses SSE has no [DONE] terminator; response.completed ends it.
        # Supply an input-token estimate for the case where the upstream stream
        # never reported usage (translator ignores it if real usage arrived).
        input_estimate = _estimate_input_tokens(request, served_model, token_counter)
        for event in translator.finalize(input_tokens=input_estimate):
            yield format_responses_sse(event)
        await _fire_success_stream(
            hooks, request_id, started, request, decision, redactions, failover_from,
            output_tokens, last_usage, identity,
        )
        await _write_memory_turns_streaming(
            provider, mem_identity, request, "".join(accumulated_text),
            output_tokens, decision, token_counter,
        )
    else:
        await _fire_failure(
            hooks, request_id, started, request, decision, redactions, failover_from,
            error, "stream_error", identity,
        )

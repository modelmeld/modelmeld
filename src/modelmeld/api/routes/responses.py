# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""POST /v1/responses — OpenAI Responses API surface (Codex CLI plug-and-play).

Phase 1 is non-streaming. The handler reuses the chat pipeline: translate the
Responses request to the internal Chat shape, run the same routing / BYOK /
subscription-passthrough / PII-scrub / capability-routing path, then translate
the ChatCompletion back to a Responses result. Audit headers, hooks, and memory
write-back come for free via `_completion_with_failover`.

`stream=true` returns 501 until the Responses SSE event stream lands (Phase 2);
Codex CLI streams, so this endpoint isn't yet a full Codex drop-in.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from modelmeld.api._safe_error_detail import safe_error_detail
from modelmeld.api.auth_detection import classify_authorization
from modelmeld.api.byok import build_byok_adapters, extract_byok_credentials
from modelmeld.api.routes.chat import (
    _auth_identity,
    _byok_required_detail,
    _completion_with_failover,
    _fire_failure,
    _is_byok_required_error,
    _is_no_eligible_model_error,
    _maybe_scrub,
    _no_eligible_model_detail,
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

router = APIRouter()


@router.post("/responses", response_model=None)
async def responses(
    body: ResponsesRequest,
    fastapi_request: Request,
    response: Response,
) -> JSONResponse:
    if body.stream:
        raise HTTPException(
            status_code=501,
            detail=(
                "streaming is not yet supported on /v1/responses; "
                "set stream=false (Responses SSE streaming is a follow-up)"
            ),
        )

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

    # Shared dispatch: failover, audit headers (on `response`), hooks, memory
    # write-back, and the canonical-model pin all happen here.
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

"""End-to-end capability-routing through the FastAPI chat route.

Verifies that:
- `request.model` gets overridden to the scout's pick before reaching the adapter
- New response headers are emitted (`x-modelmeld-routed-model`, `task-category`,
  `task-score`, `quality-threshold`)
- Failover walks the fallback list
- Existing PII scrubbing / hooks / failover-from header still work
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.config import GatewaySettings
from modelmeld.router import CapabilityRouter
from modelmeld.scout import CapabilityScout, ModelEntry, ModelRegistry


class _FakeAdapter(ProviderAdapter):
    is_egress = False  # avoid scrubber path in tests

    def __init__(self, name: str, *, healthy: bool = True, raise_on_chat: bool = False) -> None:
        self.name = name
        self._healthy = healthy
        self._raise = raise_on_chat
        self.received_models: list[str] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.received_models.append(request.model)
        if self._raise:
            raise AdapterError(f"{self.name} chat failure")
        return ChatCompletion(
            model=request.model,
            choices=[
                Choice(index=0, message=ResponseMessage(content=f"served-by-{self.name}"),
                       finish_reason="stop")
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return self._healthy


def _entry(
    model_id: str, provider: str, cost_in: float, cost_out: float,
    coding: float = 0.0,
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, provider=provider, context_window=100000,
        cost_per_m_input=cost_in, cost_per_m_output=cost_out,
        task_scores={"coding": coding},
        last_updated="2026-05-17", source="test",
    )


def _build_app_with_capability_router(
    *,
    registry_entries: list[ModelEntry],
    adapters: dict[str, ProviderAdapter],
    quality_threshold: float = 0.80,
):
    registry = ModelRegistry(registry_entries)
    scout = CapabilityScout(
        registry=registry,
        quality_threshold=quality_threshold,
        eligible_providers=frozenset(adapters.keys()),
    )
    router = CapabilityRouter(scout=scout, adapters_by_provider=adapters)
    return build_app(
        settings=GatewaySettings(),
        router=router,
        model_registry=registry,
    )


def _payload(user_text: str, requested_model: str = "claude-opus-4-7") -> dict:
    return {
        "model": requested_model,
        "messages": [{"role": "user", "content": user_text}],
    }


# ---------------------------------------------------------------------------
# Happy path: model is rewritten + headers expose the decision
# ---------------------------------------------------------------------------

async def test_capability_routing_overrides_request_model() -> None:
    cheap = _FakeAdapter("vllm")
    expensive = _FakeAdapter("anthropic")
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen-cheap", "vllm", 0.5, 1.0, coding=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"vllm": cheap, "anthropic": expensive},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("refactor this function"),
        )
    assert resp.status_code == 200
    body = resp.json()
    # The adapter saw the OVERRIDDEN model id, not the user's claude-opus-4-7
    assert cheap.received_models == ["qwen-cheap"]
    assert expensive.received_models == []
    # Response echoes the served model
    assert body["model"] == "qwen-cheap"
    # Headers expose the capability decision
    assert resp.headers["x-modelmeld-routed-to"] == "vllm"
    assert resp.headers["x-modelmeld-routed-model"] == "qwen-cheap"
    assert resp.headers["x-modelmeld-task-category"] == "coding"
    assert resp.headers["x-modelmeld-task-score"] == "0.85"
    assert resp.headers["x-modelmeld-quality-threshold"] == "0.80"


# ---------------------------------------------------------------------------
# When threshold is too high → 503 from chat route (RouterError mapping)
# ---------------------------------------------------------------------------

async def test_no_eligible_model_returns_503() -> None:
    adapter = _FakeAdapter("openai")
    app = _build_app_with_capability_router(
        registry_entries=[_entry("weak", "openai", 0.5, 1.0, coding=0.50)],
        adapters={"openai": adapter},
        quality_threshold=0.80,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload("refactor"))
    assert resp.status_code == 503
    assert "capability_scout" in resp.json()["detail"]
    assert adapter.received_models == []


# ---------------------------------------------------------------------------
# Failover: primary adapter fails mid-request → fallback model used
# ---------------------------------------------------------------------------

async def test_failover_uses_fallback_model_on_adapter_error() -> None:
    failing = _FakeAdapter("vllm", raise_on_chat=True)
    working = _FakeAdapter("anthropic")
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen-cheap", "vllm", 0.5, 1.0, coding=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"vllm": failing, "anthropic": working},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload("refactor this"))
    assert resp.status_code == 200
    # Failover happened: failing adapter saw the call, working adapter served the response
    assert failing.received_models == ["qwen-cheap"]
    assert working.received_models == ["opus"]
    assert resp.headers["x-modelmeld-routed-to"] == "anthropic"
    assert resp.headers["x-modelmeld-routed-model"] == "opus"
    assert resp.headers["x-modelmeld-failover-from"] == "local"  # vllm tier


# ---------------------------------------------------------------------------
# Both providers fail → 502
# ---------------------------------------------------------------------------

async def test_both_adapters_failing_returns_502() -> None:
    failing1 = _FakeAdapter("vllm", raise_on_chat=True)
    failing2 = _FakeAdapter("anthropic", raise_on_chat=True)
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"vllm": failing1, "anthropic": failing2},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload("refactor"))
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "primary failed" in detail
    assert "fallback failed" in detail


# ---------------------------------------------------------------------------
# eligible_providers filter: anthropic-only restricts the registry pick
# ---------------------------------------------------------------------------

async def test_eligible_providers_filter_excludes_cheaper_provider() -> None:
    cheap_vllm = _FakeAdapter("vllm")
    expensive_anthropic = _FakeAdapter("anthropic")
    # We DON'T pass vllm in adapters_by_provider, so it shouldn't be considered
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen-cheap", "vllm", 0.5, 1.0, coding=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"anthropic": expensive_anthropic},  # only anthropic registered
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_payload("refactor"))
    assert resp.status_code == 200
    # Scout's eligible_providers=frozenset(adapter keys) filtered vllm out
    assert expensive_anthropic.received_models == ["opus"]
    assert cheap_vllm.received_models == []
    assert resp.headers["x-modelmeld-routed-model"] == "opus"


# ---------------------------------------------------------------------------
# build_router factory integration
# ---------------------------------------------------------------------------

def test_build_router_with_capability_policy() -> None:
    """Sanity check: passing routing_policy='capability' actually builds a CapabilityRouter."""
    from modelmeld.router import build_router
    from modelmeld.router.capability import CapabilityRouter as CR
    from modelmeld.scout import HeuristicScout

    registry = ModelRegistry([_entry("a", "openai", 1.0, 3.0, coding=0.85)])
    settings = GatewaySettings(
        routing_policy="capability",
        capability_quality_threshold=0.80,
        capability_eligible_providers=["openai"],
        openai_api_key="sk-test-fake",  # so the openai adapter can be built
    )
    router = build_router(settings, HeuristicScout(), model_registry=registry)
    assert isinstance(router, CR)
    assert "openai" in router.adapters_by_provider


# ---------------------------------------------------------------------------
# Audit-trail headers comprehensively pinned
# ---------------------------------------------------------------------------
# These tests enforce the "auditable routing decisions" claim:
# every routing decision (success path + failover path) MUST surface
# enough info via response headers that a customer or auditor can
# reproduce the decision without server-side log access.

async def test_all_audit_headers_present_on_success_path() -> None:
    """Successful capability-routed request emits the full header set."""
    cheap = _FakeAdapter("vllm")
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen-cheap", "vllm", 0.5, 1.0, coding=0.85),
        ],
        adapters={"vllm": cheap},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("refactor this function"),
        )
    assert resp.status_code == 200
    # The "auditable routing" contract: every one of these MUST be set.
    required_headers = {
        "x-modelmeld-routed-to",          # adapter name
        "x-modelmeld-tier",               # local | cloud
        "x-modelmeld-routed-model",       # actual served model id
        "x-modelmeld-task-category",      # classifier's category
        "x-modelmeld-task-score",         # chosen model's score on category
        "x-modelmeld-quality-threshold",  # threshold applied (post-bias)
        "x-modelmeld-category-source",    # classifier | hint:task_category | hint:agent_role
    }
    missing = required_headers - set(resp.headers.keys())
    assert not missing, f"audit-trail headers missing on success: {missing}"


async def test_all_audit_headers_present_on_failover_path() -> None:
    """Failover-routed request also emits the full header set + failover-from."""
    failing = _FakeAdapter("vllm", raise_on_chat=True)
    working = _FakeAdapter("anthropic")
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("qwen-cheap", "vllm", 0.5, 1.0, coding=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"vllm": failing, "anthropic": working},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload("refactor this"),
        )
    assert resp.status_code == 200
    required_headers = {
        "x-modelmeld-routed-to",
        "x-modelmeld-tier",
        "x-modelmeld-routed-model",
        "x-modelmeld-task-category",
        "x-modelmeld-task-score",
        "x-modelmeld-quality-threshold",
        "x-modelmeld-category-source",
        "x-modelmeld-failover-from",      # failover-specific
    }
    missing = required_headers - set(resp.headers.keys())
    assert not missing, f"audit-trail headers missing on failover: {missing}"
    # Failover header value points at the ORIGINAL tier
    assert resp.headers["x-modelmeld-failover-from"] == "local"
    # routed-model is the FINAL choice, not the primary
    assert resp.headers["x-modelmeld-routed-model"] == "opus"


async def test_devtool_header_present_when_fingerprint_detected() -> None:
    """When the fingerprinter identifies a known dev tool, the response
    advertises it via x-modelmeld-devtool so client-side telemetry can
    slice by tool without re-parsing the request."""
    adapter = _FakeAdapter("vllm")
    app = _build_app_with_capability_router(
        registry_entries=[_entry("qwen", "vllm", 0.5, 1.0, coding=0.85)],
        adapters={"vllm": adapter},
    )
    # Strong Cursor signature — multiple Cursor-distinctive strings
    cursor_prompt = (
        "You are a coding assistant in Cursor. "
        "<custom_instructions>Use TypeScript.</custom_instructions> "
        "You are a powerful agentic AI coding assistant. "
        "Refactor my code."
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_payload(cursor_prompt),
        )
    assert resp.status_code == 200
    devtool = resp.headers.get("x-modelmeld-devtool")
    assert devtool is not None, "x-modelmeld-devtool missing when Cursor fingerprint detected"
    assert devtool.startswith("cursor:"), f"expected cursor:..., got {devtool!r}"


async def test_bias_header_present_when_shape_bias_applied() -> None:
    """When the shape-bias logic fires, the response
    advertises which bias was applied for audit/debugging."""
    adapter = _FakeAdapter("vllm")
    app = _build_app_with_capability_router(
        registry_entries=[
            _entry("granite-cheap", "vllm", 0.017, 0.112, coding=0.60),
            _entry("qwen-mid", "vllm", 0.30, 0.30, coding=0.85),
        ],
        adapters={"vllm": adapter},
    )
    # Autocomplete-shape request: short prompt + low max_tokens + no tools
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "def add(a, b):\n    return"}],
                "max_tokens": 64,
            },
        )
    assert resp.status_code == 200
    bias = resp.headers.get("x-modelmeld-bias")
    assert bias == "autocomplete_shape", f"expected bias header for autocomplete, got {bias!r}"
    # Bias dropped threshold to 0.55 → granite-cheap admitted (would have
    # been excluded at 0.80 with score 0.60)
    assert resp.headers["x-modelmeld-routed-model"] == "granite-cheap"
    assert resp.headers["x-modelmeld-quality-threshold"] == "0.55"


async def test_no_bias_header_when_no_bias_applied() -> None:
    """Negative case: when no shape bias fires, x-modelmeld-bias is absent
    (we don't emit empty headers — absence means 'no bias')."""
    adapter = _FakeAdapter("vllm")
    app = _build_app_with_capability_router(
        registry_entries=[_entry("qwen", "vllm", 0.5, 1.0, coding=0.85)],
        adapters={"vllm": adapter},
    )
    # Plain chat: long prompt, no autocomplete shape
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-opus-4-7",
                "messages": [{
                    "role": "user",
                    "content": "Write a long essay on monad transformers." * 50,
                }],
                "max_tokens": 2048,
            },
        )
    assert resp.status_code == 200
    assert "x-modelmeld-bias" not in resp.headers



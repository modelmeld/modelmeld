"""End-to-end: framework routing hints in headers override scout decisions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.routing_hints import (
    HEADER_AGENT_ROLE,
    HEADER_EXCLUDE_PROVIDERS,
    HEADER_QUALITY_THRESHOLD,
    HEADER_TASK_CATEGORY,
)
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
    Usage,
)
from modelmeld.api.server import build_app
from modelmeld.config import GatewaySettings
from modelmeld.router import CapabilityRouter
from modelmeld.scout import CapabilityScout, ModelEntry, ModelRegistry


class _FakeAdapter(ProviderAdapter):
    is_egress = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.received_models: list[str] = []

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.received_models.append(request.model)
        return ChatCompletion(
            model=request.model,
            choices=[Choice(index=0, message=ResponseMessage(content="ok"), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


def _entry(
    model_id: str, provider: str, cost_in: float, cost_out: float,
    coding: float = 0.0, reasoning: float = 0.0, summarization: float = 0.0,
    tool_use: float = 0.0, simple_qa: float = 0.0,
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, provider=provider, context_window=100000,
        cost_per_m_input=cost_in, cost_per_m_output=cost_out,
        task_scores={
            "coding": coding, "reasoning": reasoning,
            "summarization": summarization, "tool_use": tool_use,
            "simple_qa": simple_qa,
        },
        last_updated="2026-05-17", source="test",
    )


def _build_test_app(
    entries: list[ModelEntry],
    adapters: dict[str, ProviderAdapter],
    quality_threshold: float = 0.80,
    settings: GatewaySettings | None = None,
):
    registry = ModelRegistry(entries)
    scout = CapabilityScout(
        registry=registry,
        quality_threshold=quality_threshold,
        eligible_providers=frozenset(adapters.keys()),
    )
    return build_app(
        settings=settings or GatewaySettings(),
        router=CapabilityRouter(scout=scout, adapters_by_provider=adapters),
        model_registry=registry,
    )


def _payload(text: str = "tell me about transformers") -> dict:
    return {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": text}]}


async def _post(app, body: dict, headers: dict[str, str] | None = None) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post("/v1/chat/completions", json=body, headers=headers or {})


# ---------------------------------------------------------------------------
# task_category header overrides the classifier
# ---------------------------------------------------------------------------

async def test_task_category_header_bypasses_classifier() -> None:
    """Prompt looks like simple_qa; framework declares 'reasoning'."""
    cheap = _FakeAdapter("vllm")
    smart = _FakeAdapter("anthropic")
    app = _build_test_app(
        entries=[
            # `cheap` is good at simple_qa, not at reasoning
            _entry("qwen", "vllm", 0.5, 1.0, simple_qa=0.90, reasoning=0.50),
            # `smart` is good at reasoning
            _entry("opus", "anthropic", 5.0, 25.0, reasoning=0.92, simple_qa=0.95),
        ],
        adapters={"vllm": cheap, "anthropic": smart},
        quality_threshold=0.80,
    )

    # Without hint: prompt "what is X" → simple_qa → cheap (qwen) picked
    resp = await _post(app, _payload("what is a transformer?"))
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-task-category"] == "simple_qa"
    assert resp.headers["x-modelmeld-category-source"] == "classifier"
    assert resp.headers["x-modelmeld-routed-model"] == "qwen"

    # With hint: framework declares reasoning → smart (opus) picked
    cheap.received_models.clear()
    smart.received_models.clear()
    resp = await _post(
        app, _payload("what is a transformer?"),
        headers={HEADER_TASK_CATEGORY: "reasoning"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-task-category"] == "reasoning"
    assert resp.headers["x-modelmeld-category-source"] == "hint:task_category"
    assert resp.headers["x-modelmeld-routed-model"] == "opus"
    assert smart.received_models == ["opus"]


# ---------------------------------------------------------------------------
# agent_role header maps to category
# ---------------------------------------------------------------------------

async def test_agent_role_header_maps_to_category() -> None:
    coder_only = _FakeAdapter("openai")
    app = _build_test_app(
        entries=[
            _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.85, simple_qa=0.60),
        ],
        adapters={"openai": coder_only},
        quality_threshold=0.80,
    )

    # Without role: "hello" → no signals → simple_qa → no model meets 0.80 → 400
    # (Was 503; client problem — their threshold is unmet — so 400 is accurate.)
    resp = await _post(app, _payload("hello"))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "no_eligible_model"

    # With agent_role=coder: → coding → gpt-mini (0.85) meets threshold → 200
    resp = await _post(app, _payload("hello"), headers={HEADER_AGENT_ROLE: "coder"})
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-task-category"] == "coding"
    assert resp.headers["x-modelmeld-category-source"] == "hint:agent_role"
    assert resp.headers["x-modelmeld-routed-model"] == "gpt-mini"
    # Echo: agent_role hint is mirrored back so multi-agent frameworks
    # (OpenClaw / AutoGen / CrewAI / LangGraph) can grep response headers
    # and confirm their sub-agent declaration reached the gateway intact.
    assert resp.headers["x-modelmeld-agent-role"] == "coder"


async def test_agent_role_echo_absent_when_no_hint() -> None:
    """No hint sent → no echo header on the response."""
    coder_only = _FakeAdapter("openai")
    app = _build_test_app(
        entries=[
            _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.85, simple_qa=0.85),
        ],
        adapters={"openai": coder_only},
        quality_threshold=0.80,
    )

    resp = await _post(app, _payload("hello"))
    assert resp.status_code == 200
    assert "x-modelmeld-agent-role" not in resp.headers


# ---------------------------------------------------------------------------
# Audit header: `x-modelmeld-routed-model` reflects the registry canonical
# ---------------------------------------------------------------------------


async def test_routed_model_header_uses_canonical_when_dispatch_uses_provider_slug() -> None:
    """Some registries store BOTH a canonical `model_id` and a distinct
    `provider_model_id` (the string the adapter actually dispatches
    against upstream — useful when multiple providers serve the same
    logical model under different catalog conventions). When the two
    differ, the capability router's `model_id_override` carries the
    dispatch form. The audit header should consistently expose the
    canonical instead — that's the value the rest of the audit-headers
    contract keys on, and the value most clients' cost-tracking logic
    expects to round-trip.

    Reverse-lookup matches on (provider, served_id) and returns the
    canonical from the registry entry. Falls back to the served value
    when no registry match (transition-safe).
    """
    adapter = _FakeAdapter("test-provider")
    entry = ModelEntry(
        model_id="canonical-name",
        provider="test-provider",
        provider_model_id="some/dispatch-form",
        context_window=100000,
        cost_per_m_input=0.04, cost_per_m_output=0.04,
        task_scores={"simple_qa": 0.85},
        last_updated="2026-05-31", source="test",
    )
    registry = ModelRegistry([entry])
    scout = CapabilityScout(
        registry=registry,
        quality_threshold=0.80,
        eligible_providers=frozenset({"test-provider"}),
    )
    app = build_app(
        settings=GatewaySettings(),
        router=CapabilityRouter(
            scout=scout, adapters_by_provider={"test-provider": adapter},
        ),
        model_registry=registry,
    )

    resp = await _post(app, _payload("ping"))
    assert resp.status_code == 200
    # Canonical model_id, not the dispatch form
    assert resp.headers["x-modelmeld-routed-model"] == "canonical-name"
    # The dispatch-form string never appears in any header value
    for v in resp.headers.values():
        assert "some/dispatch-form" not in v


# ---------------------------------------------------------------------------
# #47: response BODY must not leak an upstream-provider model slug
# ---------------------------------------------------------------------------

# A provider-specific catalog id an adapter might echo back from upstream
# (e.g. a hosted-pool provider's internal model name). The body must never
# surface this when capability routing substituted the model.
_LEAK_SLUG = "accounts/secret-provider/models/internal-123"


class _LeakyAdapter(_FakeAdapter):
    """Adapter whose RESPONSE carries a provider slug different from the
    requested model — the real #47 leak vector."""

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.received_models.append(request.model)
        return ChatCompletion(
            model=_LEAK_SLUG,
            choices=[Choice(
                index=0, message=ResponseMessage(content="ok"), finish_reason="stop",
            )],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _LeakyStreamingAdapter(_FakeAdapter):
    """Streaming variant: chunks carry the provider slug."""

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        self.received_models.append(request.model)
        yield ChatCompletionChunk(
            id="x", created=0, model=_LEAK_SLUG,
            choices=[ChunkChoice(
                index=0, delta=ChoiceDelta(content="ok"), finish_reason="stop",
            )],
        )


def _capability_app(adapter: ProviderAdapter):
    """Capability-routing app — the gateway substitutes a model the client
    didn't ask for, so model_id_override is set on the decision."""
    registry = ModelRegistry([_entry("canonical-name", "test-provider", 0.04, 0.04, simple_qa=0.85)])
    scout = CapabilityScout(
        registry=registry, quality_threshold=0.80,
        eligible_providers=frozenset({"test-provider"}),
    )
    return build_app(
        settings=GatewaySettings(),
        router=CapabilityRouter(
            scout=scout, adapters_by_provider={"test-provider": adapter},
        ),
        model_registry=registry,
    )


async def test_response_body_does_not_leak_provider_slug() -> None:
    """Non-streaming: the adapter returns a provider slug, but the body echoes
    the client's requested model — the slug never reaches the client."""
    adapter = _LeakyAdapter("test-provider")
    app = _capability_app(adapter)
    payload = _payload("ping")

    resp = await _post(app, payload)

    assert resp.status_code == 200
    body = resp.json()
    # Body shows the canonical routed model (matches the header), never the slug.
    assert body["model"] == "canonical-name"
    assert body["model"] == resp.headers["x-modelmeld-routed-model"]
    assert _LEAK_SLUG not in json.dumps(body)


async def test_streaming_body_does_not_leak_provider_slug() -> None:
    """Streaming: every chunk's model echoes the requested model, and the slug
    appears nowhere in the SSE payload."""
    adapter = _LeakyStreamingAdapter("test-provider")
    app = _capability_app(adapter)
    payload = {**_payload("ping"), "stream": True}

    resp = await _post(app, payload)

    assert resp.status_code == 200
    models_seen = [
        json.loads(line[6:])["model"]
        for line in resp.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    assert models_seen, "expected at least one streamed chunk"
    assert all(m == "canonical-name" for m in models_seen)
    assert _LEAK_SLUG not in resp.text


# ---------------------------------------------------------------------------
# quality_threshold header overrides scout config
# ---------------------------------------------------------------------------

async def test_quality_threshold_header_overrides_scout_default() -> None:
    """Scout's default 0.80 would pick cheap qwen; hint 0.90 forces smart opus."""
    cheap = _FakeAdapter("vllm")
    smart = _FakeAdapter("anthropic")
    app = _build_test_app(
        entries=[
            _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),     # below 0.90
            _entry("opus", "anthropic", 5.0, 25.0, coding=0.95),
        ],
        adapters={"vllm": cheap, "anthropic": smart},
        quality_threshold=0.80,
    )
    resp = await _post(
        app, _payload("refactor this function"),
        headers={HEADER_QUALITY_THRESHOLD: "0.90"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-routed-model"] == "opus"
    assert resp.headers["x-modelmeld-quality-threshold"] == "0.90"


# ---------------------------------------------------------------------------
# exclude_providers header
# ---------------------------------------------------------------------------

async def test_exclude_providers_header_filters_candidate_set() -> None:
    """Cheap vllm normally wins; excluding 'vllm' forces fallback to openai."""
    vllm = _FakeAdapter("vllm")
    openai = _FakeAdapter("openai")
    app = _build_test_app(
        entries=[
            _entry("qwen", "vllm", 0.5, 1.0, coding=0.85),
            _entry("gpt-mini", "openai", 1.0, 3.0, coding=0.83),
        ],
        adapters={"vllm": vllm, "openai": openai},
        quality_threshold=0.80,
    )
    resp = await _post(
        app, _payload("refactor this code"),
        headers={HEADER_EXCLUDE_PROVIDERS: "vllm"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-routed-to"] == "openai"
    assert resp.headers["x-modelmeld-routed-model"] == "gpt-mini"


async def test_excluding_all_eligible_providers_returns_400() -> None:
    """Excluding every eligible provider should fail closed.

    Was 503 pre-fix; reclassified to 400 since the caller's own
    exclusion list is what made the request unfulfillable.
    """
    openai = _FakeAdapter("openai")
    app = _build_test_app(
        entries=[_entry("gpt-mini", "openai", 1.0, 3.0, coding=0.83)],
        adapters={"openai": openai},
        quality_threshold=0.80,
    )
    resp = await _post(
        app, _payload("refactor"),
        headers={HEADER_EXCLUDE_PROVIDERS: "openai,anthropic,vllm"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "no_eligible_model"


# ---------------------------------------------------------------------------
# Combined: agent_role + threshold
# ---------------------------------------------------------------------------

async def test_combined_role_and_threshold() -> None:
    """A reviewer agent (role=reviewer → reasoning) with raised quality bar."""
    opus = _FakeAdapter("anthropic")
    app = _build_test_app(
        entries=[
            _entry("qwen", "vllm", 0.5, 1.0, reasoning=0.85),
            _entry("opus", "anthropic", 5.0, 25.0, reasoning=0.95),
        ],
        adapters={"vllm": _FakeAdapter("vllm"), "anthropic": opus},
        quality_threshold=0.80,
    )
    resp = await _post(
        app, _payload("review this PR"),
        headers={HEADER_AGENT_ROLE: "reviewer", HEADER_QUALITY_THRESHOLD: "0.90"},
    )
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-task-category"] == "reasoning"
    assert resp.headers["x-modelmeld-routed-model"] == "opus"


# ---------------------------------------------------------------------------
# Malformed headers → 400
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("header", "value"),
    [
        (HEADER_TASK_CATEGORY, "code-review"),
        (HEADER_AGENT_ROLE, "vibe_lord"),
        (HEADER_QUALITY_THRESHOLD, "1.5"),
        (HEADER_QUALITY_THRESHOLD, "not-a-number"),
    ],
)
async def test_malformed_header_returns_400(header: str, value: str) -> None:
    app = _build_test_app(
        entries=[_entry("a", "openai", 1.0, 3.0, coding=0.85)],
        adapters={"openai": _FakeAdapter("openai")},
    )
    resp = await _post(app, _payload("x"), headers={header: value})
    assert resp.status_code == 400
    assert "invalid_routing_hint" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Hints are no-ops for non-capability policies
# ---------------------------------------------------------------------------

async def test_hints_ignored_under_single_adapter_router() -> None:
    """`x-modelmeld-task-category: reasoning` is harmless on a non-capability setup."""
    from modelmeld.adapters.stub import StubAdapter
    app = build_app(adapter=StubAdapter())  # SingleAdapterRouter
    resp = await _post(app, _payload("hi"), headers={HEADER_TASK_CATEGORY: "reasoning"})
    assert resp.status_code == 200
    # No capability headers because no CapabilityDecision produced
    assert "x-modelmeld-task-category" not in resp.headers

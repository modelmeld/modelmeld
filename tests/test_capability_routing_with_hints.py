"""End-to-end: framework routing hints in headers override scout decisions."""

from __future__ import annotations

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
):
    registry = ModelRegistry(entries)
    scout = CapabilityScout(
        registry=registry,
        quality_threshold=quality_threshold,
        eligible_providers=frozenset(adapters.keys()),
    )
    return build_app(
        settings=GatewaySettings(),
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

    # Without role: "hello" → no signals → simple_qa → no model meets 0.80 → 503
    resp = await _post(app, _payload("hello"))
    assert resp.status_code == 503

    # With agent_role=coder: → coding → gpt-mini (0.85) meets threshold → 200
    resp = await _post(app, _payload("hello"), headers={HEADER_AGENT_ROLE: "coder"})
    assert resp.status_code == 200
    assert resp.headers["x-modelmeld-task-category"] == "coding"
    assert resp.headers["x-modelmeld-category-source"] == "hint:agent_role"
    assert resp.headers["x-modelmeld-routed-model"] == "gpt-mini"


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


async def test_excluding_all_eligible_providers_returns_503() -> None:
    """Excluding every eligible provider should fail closed."""
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
    assert resp.status_code == 503


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

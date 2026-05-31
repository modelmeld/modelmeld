"""CapabilityScout — registry-driven model selection."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest, UserMessage
from modelmeld.scout import (
    CapabilityDecision,
    CapabilityScout,
    ModelEntry,
    ModelRegistry,
    NoEligibleModelError,
)


def _req(text: str = "refactor this function") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
        tools=[],
    )


def _entry(
    model_id: str, provider: str, cost_in: float, cost_out: float,
    coding: float = 0.0, reasoning: float = 0.0, context_window: int = 100000,
) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores={"coding": coding, "reasoning": reasoning},
        last_updated="2026-05-17",
        source="test",
    )


# ---------------------------------------------------------------------------
# Happy path: cheapest competent model
# ---------------------------------------------------------------------------

async def test_picks_cheapest_competent_coding_model() -> None:
    registry = ModelRegistry([
        _entry("opus-pricey", "anthropic", 5.0, 25.0, coding=0.95),
        _entry("qwen-cheap", "vllm", 0.5, 1.5, coding=0.85),
        _entry("weak-cheap", "vllm", 0.1, 0.3, coding=0.50),   # below threshold
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_req("refactor this code"))
    assert isinstance(decision, CapabilityDecision)
    assert decision.task_category == "coding"
    assert decision.chosen_model_id == "qwen-cheap"
    assert decision.chosen_provider == "vllm"
    # opus is the only fallback above threshold
    assert "opus-pricey" in decision.fallback_model_ids
    assert decision.task_score == 0.85
    assert decision.quality_threshold == 0.80


async def test_fallback_list_is_cost_ordered() -> None:
    registry = ModelRegistry([
        _entry("a", "openai", 10.0, 30.0, coding=0.90),
        _entry("b", "openai", 1.0, 3.0, coding=0.85),
        _entry("c", "openai", 5.0, 15.0, coding=0.92),
        _entry("d", "openai", 0.5, 1.5, coding=0.83),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_req())
    assert decision.chosen_model_id == "d"
    # Fallbacks: b ($1.80/M blended), c, a — cheapest first
    assert decision.fallback_model_ids[0] == "b"
    assert decision.fallback_model_ids[1] == "c"
    assert decision.fallback_model_ids[2] == "a"


# ---------------------------------------------------------------------------
# Quality threshold gating
# ---------------------------------------------------------------------------

async def test_no_eligible_model_raises() -> None:
    registry = ModelRegistry([
        _entry("weak", "openai", 0.5, 1.0, coding=0.50),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    with pytest.raises(NoEligibleModelError) as exc_info:
        await scout.choose(_req())
    assert exc_info.value.task_category == "coding"
    assert exc_info.value.quality_threshold == 0.80


async def test_raising_threshold_shrinks_candidate_set() -> None:
    registry = ModelRegistry([
        _entry("good-cheap", "vllm", 0.5, 1.0, coding=0.82),
        _entry("great-pricey", "anthropic", 5.0, 25.0, coding=0.95),
    ])
    # threshold 0.80: both eligible → cheap wins
    cheap = await CapabilityScout(registry, quality_threshold=0.80).choose(_req())
    assert cheap.chosen_model_id == "good-cheap"
    # threshold 0.90: only great-pricey
    great = await CapabilityScout(registry, quality_threshold=0.90).choose(_req())
    assert great.chosen_model_id == "great-pricey"


# ---------------------------------------------------------------------------
# Eligible providers filter
# ---------------------------------------------------------------------------

async def test_eligible_providers_filter_excludes_other_providers() -> None:
    registry = ModelRegistry([
        _entry("openai-cheap", "openai", 0.5, 1.5, coding=0.85),
        _entry("vllm-cheaper", "vllm", 0.1, 0.3, coding=0.85),
        _entry("anthropic-cheap", "anthropic", 1.0, 3.0, coding=0.85),
    ])
    # Only openai allowed → openai-cheap wins despite vllm being cheaper.
    scout = CapabilityScout(
        registry=registry,
        quality_threshold=0.80,
        eligible_providers=frozenset({"openai"}),
    )
    decision = await scout.choose(_req())
    assert decision.chosen_model_id == "openai-cheap"
    assert decision.chosen_provider == "openai"
    # No anthropic/vllm in fallbacks
    for fid in decision.fallback_model_ids:
        assert registry.get(fid).provider == "openai"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tools → tool_use category
# ---------------------------------------------------------------------------

async def test_tools_route_to_tool_use_category() -> None:
    from modelmeld.api.schemas import FunctionDef, Tool
    registry = ModelRegistry([
        _entry("opus", "anthropic", 5.0, 25.0, coding=0.81),
        # Same model with a tool_use score, but score-by-task is set via kwargs not in _entry helper
    ])
    # Override entry to include tool_use score
    entries = [
        ModelEntry(
            model_id="opus", provider="anthropic", context_window=200000,
            cost_per_m_input=5.0, cost_per_m_output=25.0,
            task_scores={"coding": 0.81, "tool_use": 0.90},
            last_updated="2026-05-17", source="test",
        ),
    ]
    registry = ModelRegistry(entries)
    req = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(role="user", content="search and reply")],
        tools=[
            Tool(
                type="function",
                function=FunctionDef(
                    name="search", description="", parameters={"type": "object"}
                ),
            )
        ],
    )
    decision = await CapabilityScout(registry, quality_threshold=0.85).choose(req)
    assert decision.task_category == "tool_use"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_threshold_rejected() -> None:
    registry = ModelRegistry([])
    with pytest.raises(ValueError):
        CapabilityScout(registry=registry, quality_threshold=1.5)
    with pytest.raises(ValueError):
        CapabilityScout(registry=registry, quality_threshold=-0.1)


def test_negative_fallback_depth_rejected() -> None:
    with pytest.raises(ValueError):
        CapabilityScout(registry=ModelRegistry([]), fallback_depth=-1)


# ---------------------------------------------------------------------------
# Lookup_fallback + with_model helpers
# ---------------------------------------------------------------------------

def test_lookup_fallback_returns_registry_entry() -> None:
    registry = ModelRegistry([_entry("a", "openai", 1.0, 2.0, coding=0.9)])
    scout = CapabilityScout(registry=registry)
    assert scout.lookup_fallback("a") is not None
    assert scout.lookup_fallback("missing") is None


def test_with_model_swaps_chosen_and_appends_rationale() -> None:
    decision = CapabilityDecision(
        chosen_model_id="primary",
        chosen_provider="anthropic",
        task_category="coding",
        task_score=0.95,
        quality_threshold=0.80,
        fallback_model_ids=["b", "c"],
        rationale="category=coding",
    )
    after = decision.with_model("b", "openai", 0.88)
    assert after.chosen_model_id == "b"
    assert after.chosen_provider == "openai"
    assert after.task_score == 0.88
    assert "failover=b" in after.rationale
    # Original frozen, untouched
    assert decision.chosen_model_id == "primary"


# ---------------------------------------------------------------------------
# Classifier injection
# ---------------------------------------------------------------------------

class _FakeReasoningClassifier:
    def classify(self, request: ChatCompletionRequest):
        from modelmeld.scout.task_category import TaskCategoryDecision
        return TaskCategoryDecision(
            category="reasoning",
            confidence=0.99,
            rationale="forced",
            per_category_scores={},
        )


async def test_custom_classifier_overrides_category() -> None:
    registry = ModelRegistry([
        _entry("smart", "anthropic", 5.0, 25.0, coding=0.50, reasoning=0.90),
        _entry("dumb", "openai", 0.5, 1.0, coding=0.90, reasoning=0.50),
    ])
    scout = CapabilityScout(
        registry=registry,
        classifier=_FakeReasoningClassifier(),
        quality_threshold=0.80,
    )
    # Despite "refactor" prompt smelling like coding, classifier says reasoning →
    # picks `smart` model (only one above threshold for reasoning).
    decision = await scout.choose(_req("refactor this code"))
    assert decision.task_category == "reasoning"
    assert decision.chosen_model_id == "smart"


# ---------------------------------------------------------------------------
# Capability filters — tool-call support + context-window
# ---------------------------------------------------------------------------

def _entry_with_tools(
    model_id: str, provider: str, cost_in: float, cost_out: float,
    coding: float = 0.85, supports_tools: bool = True,
    context_window: int = 100000,
) -> ModelEntry:
    # Mirror typical real-world score parity across categories so the
    # category classifier's choice doesn't cause unrelated test failures.
    # The tests are specifically about the supports_tools / context_window
    # filters, not the category classifier.
    return ModelEntry(
        model_id=model_id,
        provider=provider,
        context_window=context_window,
        cost_per_m_input=cost_in,
        cost_per_m_output=cost_out,
        task_scores={
            "coding": coding,
            "reasoning": coding,
            "simple_qa": coding,
            "summarization": coding,
            "tool_use": coding,
        },
        supports_tools=supports_tools,
        last_updated="2026-05-25",
        source="test",
    )


def _req_with_tools(text: str = "use the read_file tool") -> ChatCompletionRequest:
    """A request that DECLARES tool definitions — scout should require
    tool-capable model."""
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk and return its contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    )


async def test_request_with_tools_filters_out_non_tool_capable_models() -> None:
    """The cheapest model lacks tool support → scout MUST skip it and
    pick the next-cheapest that supports tools."""
    registry = ModelRegistry([
        # cheapest but no tool support
        _entry_with_tools("phi-4-mini", "vllm", 0.04, 0.04, coding=0.85, supports_tools=False),
        # second cheapest with tool support
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30, coding=0.85, supports_tools=True),
        # premium with tool support
        _entry_with_tools("opus", "anthropic", 5.0, 25.0, coding=0.95, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_req_with_tools())
    assert decision.chosen_model_id == "qwen-coder-flash"
    # phi-4-mini was cheaper but lacks tool support; not in fallbacks either
    assert "phi-4-mini" not in decision.fallback_model_ids


async def test_request_without_tools_allows_non_tool_capable_models() -> None:
    """Plain chat request with no tools → cheapest model wins regardless
    of supports_tools flag."""
    registry = ModelRegistry([
        _entry_with_tools("phi-4-mini", "vllm", 0.04, 0.04, coding=0.85, supports_tools=False),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30, coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_req("refactor this function"))
    # No tools in request → phi-4-mini wins on cost
    assert decision.chosen_model_id == "phi-4-mini"


async def test_request_with_tools_but_all_models_lack_support_raises() -> None:
    """No tool-capable model meets threshold → NoEligibleModelError."""
    registry = ModelRegistry([
        _entry_with_tools("phi-4", "vllm", 0.05, 0.05, coding=0.85, supports_tools=False),
        _entry_with_tools("phi-4-mini", "vllm", 0.04, 0.04, coding=0.85, supports_tools=False),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    with pytest.raises(NoEligibleModelError):
        await scout.choose(_req_with_tools())


async def test_large_prompt_filters_out_small_context_window_models() -> None:
    """Long input + max_tokens budget exceeds the small-ctx model's
    capacity → scout MUST skip it."""
    registry = ModelRegistry([
        # cheapest but tiny context window
        _entry_with_tools("phi-4", "vllm", 0.05, 0.05, coding=0.85, supports_tools=True, context_window=16384),
        # larger context
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30, coding=0.85, context_window=131072),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    # Build a request with ~60K chars of content (~15K tokens), max_tokens=4096
    # → with 1.2× headroom = (~15K + 4K) × 1.2 = ~22K tokens required
    # → phi-4's 16K won't fit; qwen-coder-flash's 131K will
    big_content = "x" * 60_000
    req = ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=big_content)],
        max_tokens=4096,
    )
    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen-coder-flash"


async def test_small_prompt_allows_small_context_window_models() -> None:
    """Short prompt fits comfortably in 16K → cheapest model wins
    regardless of small context window."""
    registry = ModelRegistry([
        _entry_with_tools("phi-4", "vllm", 0.05, 0.05, coding=0.85, context_window=16384),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30, coding=0.85, context_window=131072),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_req("refactor this 5-line function"))
    assert decision.chosen_model_id == "phi-4"


async def test_filters_combine_tool_support_and_context_window() -> None:
    """Big prompt with tools → must pick a model that BOTH supports tools
    AND has enough context."""
    registry = ModelRegistry([
        # tiny context window: skip on size
        _entry_with_tools("phi-4", "vllm", 0.05, 0.05, coding=0.85, context_window=16384),
        # big context but no tool support: skip on tools
        _entry_with_tools("big-no-tools", "vllm", 0.10, 0.10, coding=0.85,
                          supports_tools=False, context_window=131072),
        # the winner: big context AND tool support
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30, coding=0.85,
                          supports_tools=True, context_window=131072),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    big_content = "x" * 60_000
    req = ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=big_content)],
        max_tokens=4096,
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )
    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen-coder-flash"


# ---------------------------------------------------------------------------
# DevTool shape-bias — autocomplete-shape detection
# ---------------------------------------------------------------------------

def _autocomplete_req(text: str = "def add(a, b):\n    return") -> ChatCompletionRequest:
    """Realistic autocomplete-shape: tiny prompt, low max_tokens, no tools.
    Matches Cursor/Copilot/Continue's FIM autocomplete pattern."""
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
        max_tokens=64,
    )


async def test_autocomplete_shape_lowers_threshold_to_admit_sub_haiku_tier() -> None:
    """Tiny prompt + low max_tokens + no tools → scout drops to the
    sub-Haiku quality threshold (0.55) so cheap models win on cost."""
    registry = ModelRegistry([
        # sub-Haiku: cheap but score=0.60 — only admitted at threshold ≤ 0.60
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
        # OSS-mid: pricier, score=0.85 — would win without the bias if
        # threshold stayed at 0.80
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30,
                          coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    decision = await scout.choose(_autocomplete_req())
    assert decision.chosen_model_id == "granite-4-micro"
    # Rationale carries the bias clause for audit
    assert "bias=autocomplete_shape" in decision.rationale
    assert "threshold:0.80" in decision.rationale
    # The threshold reported on the decision reflects the BIASED value
    assert decision.quality_threshold == 0.55


async def test_long_prompt_does_not_get_autocomplete_bias() -> None:
    """Long input → not autocomplete shape → original threshold applies →
    expensive model wins as designed."""
    registry = ModelRegistry([
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30,
                          coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    req = ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content="x" * 5000)],
        max_tokens=64,
    )
    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen-coder-flash"
    assert "bias=" not in decision.rationale


async def test_high_max_tokens_does_not_get_autocomplete_bias() -> None:
    """Even with short input, a request asking for big response is NOT
    autocomplete shape (that's chat-with-short-prompt)."""
    registry = ModelRegistry([
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30,
                          coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    req = ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content="explain monads")],
        max_tokens=2048,
    )
    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen-coder-flash"
    assert "bias=" not in decision.rationale


async def test_request_with_tools_does_not_get_autocomplete_bias() -> None:
    """Tools present → not autocomplete (autocomplete is single-shot)."""
    registry = ModelRegistry([
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30,
                          coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    req = ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content="def add(a,b):")],
        max_tokens=64,
        tools=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )
    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen-coder-flash"


async def test_explicit_hint_overrides_bias() -> None:
    """When the caller supplies an explicit quality_threshold hint, that
    wins over the scout's shape bias — frameworks deliberately tuning
    routing per-agent shouldn't be undermined."""
    from modelmeld.api.routing_hints import RoutingHints

    registry = ModelRegistry([
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
        _entry_with_tools("qwen-coder-flash", "openrouter", 0.30, 0.30,
                          coding=0.85, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.80)
    hints = RoutingHints(quality_threshold=0.85)
    decision = await scout.choose(_autocomplete_req(), hints=hints)
    assert decision.chosen_model_id == "qwen-coder-flash"
    assert "bias=" not in decision.rationale


async def test_bias_never_raises_threshold() -> None:
    """Defensive: bias is one-way (only lowers). If the configured
    threshold is already below the bias threshold, no bias applies."""
    registry = ModelRegistry([
        _entry_with_tools("granite-4-micro", "openrouter", 0.017, 0.112,
                          coding=0.60, supports_tools=True),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.40)
    decision = await scout.choose(_autocomplete_req())
    assert decision.chosen_model_id == "granite-4-micro"
    assert "bias=" not in decision.rationale
    assert decision.quality_threshold == 0.40


# ---------------------------------------------------------------------------
# Task #177 — alias-policy routing (saver / auto / quality)
# ---------------------------------------------------------------------------


def _policy_req(alias: str, user_text: str = "refactor this function") -> ChatCompletionRequest:
    """Build a request whose `model` field is a ModelMeld alias."""
    return ChatCompletionRequest(
        model=alias,
        messages=[UserMessage(role="user", content=user_text)],
        tools=[],
    )


def _mixed_tier_registry() -> ModelRegistry:
    """Registry with both OSS rows and frontier rows for tier-filter tests.

    Each row gets BOTH coding and reasoning task_scores so the test doesn't
    depend on which category the classifier ends up assigning to a given
    prompt (reasoning-marker prompts often classify as `reasoning`).
    """
    return ModelRegistry([
        # OSS rows (eligible under SAVER + AUTO default)
        _entry("qwen3-flash", "openrouter", 0.30, 0.60, coding=0.78, reasoning=0.74),
        _entry("qwen-coder-32b", "openrouter", 0.80, 0.80, coding=0.85, reasoning=0.80),
        _entry("deepseek-r1-distill", "openrouter", 0.10, 0.50, coding=0.88, reasoning=0.86),
        # Frontier rows (only eligible under AUTO escalation or QUALITY)
        _entry("claude-sonnet-4-6", "anthropic", 3.0, 15.0, coding=0.95, reasoning=0.95),
        _entry("claude-opus-4-7", "anthropic", 5.0, 25.0, coding=0.97, reasoning=0.97),
    ])


async def test_saver_alias_restricts_to_oss_providers() -> None:
    """SAVER must never pick a frontier-provider row, even if it would be
    cheaper on blended cost. The whole point is a predictable cost ceiling."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req("anthropic/modelmeld-saver"))
    # Chosen must be OSS-provider — never anthropic/openai
    assert decision.chosen_provider in {"openrouter", "vllm", "fireworks", "together"}
    assert decision.chosen_model_id in {"qwen3-flash", "qwen-coder-32b", "deepseek-r1-distill"}
    assert "policy=saver" in decision.rationale


async def test_auto_alias_default_stays_in_oss() -> None:
    """AUTO without reasoning markers behaves like SAVER (OSS tier).
    The escalation is what makes it different — by default, no escalation."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req("anthropic/modelmeld-auto", "build a fizzbuzz"))
    assert decision.chosen_provider == "openrouter"  # OSS
    assert "policy=auto" in decision.rationale
    assert "escalated=no" in decision.rationale


async def test_auto_alias_escalates_to_frontier_on_two_reasoning_markers() -> None:
    """AUTO with 2+ reasoning markers in the USER message must escalate
    to a frontier-tier model (≥0.95 task score)."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req(
        "anthropic/modelmeld-auto",
        "Please think step by step and explain your reasoning carefully.",
    ))
    # 2 markers: "step by step" + "explain your reasoning"
    assert decision.chosen_model_id in {"claude-sonnet-4-6", "claude-opus-4-7"}
    assert decision.chosen_provider == "anthropic"
    assert "escalated=frontier" in decision.rationale


async def test_auto_alias_single_marker_does_not_escalate() -> None:
    """Single marker is not enough — must be 2+ distinct."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req(
        "anthropic/modelmeld-auto",
        "Walk me step by step through this problem.",  # 1 marker only
    ))
    assert decision.chosen_provider == "openrouter"  # stayed OSS
    assert "escalated=no" in decision.rationale


async def test_long_context_request_picks_long_context_model_when_short_context_models_filtered() -> None:
    """Sprint 4 (B-3): when a request's input + response budget exceeds
    the smaller OSS models' context windows (131k-256k), the scout's
    _required_context_window filter must select the long-context option
    (llama-4-scout at 1M tokens) instead of failing routing.

    Tests the FILTER, not provider availability — at the time of
    writing, llama-4-scout has no overlay entries for cloud-OSS
    providers, so production-environment routing still requires vllm
    self-hosting for >256k requests. The filter itself works
    correctly; bridging to cloud-OSS is tracked separately.
    """
    from modelmeld.api.schemas import ChatCompletionRequest, UserMessage

    registry = ModelRegistry([
        # Short-context options (~131k) — would be eligible by score
        # but filtered out by context-window requirement
        _entry("llama-3.3-70b", "openrouter", 0.10, 0.32,
               coding=0.78, reasoning=0.74, context_window=131_072),
        _entry("gpt-oss-120b", "openrouter", 0.04, 0.18,
               coding=0.74, reasoning=0.78, context_window=131_072),
        # Medium-context (256k) — also filtered out by a >256k request
        _entry("qwen3-coder-next", "openrouter", 0.11, 0.80,
               coding=0.78, reasoning=0.72, context_window=262_144),
        # Long-context (1M) — the only model that fits
        _entry("llama-4-scout", "vllm", 0.45, 0.45,
               coding=0.74, reasoning=0.78, context_window=1_048_576),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)

    # Build a request whose input + response budget exceeds 256k. Prefix
    # with a coding-shaped opener so the classifier picks 'coding' (the
    # bulk content is opaque to the classifier but the opener tips it).
    # ~1.2M chars / 4 chars-per-token ≈ 300k input tokens; + 2048
    # max_tokens response; × 1.2 headroom ≈ 363k required context window.
    # That fits llama-4-scout (1M) but NOT qwen3-coder-next (256k).
    large_input = "refactor this code into smaller functions:\n" + ("def foo(): return 1\n" * 60_000)
    req = ChatCompletionRequest(
        model="anthropic/modelmeld-saver",  # OSS-only policy
        messages=[UserMessage(role="user", content=large_input)],
        max_tokens=2048,
    )

    decision = await scout.choose(req)

    assert decision.chosen_model_id == "llama-4-scout", (
        f"Expected llama-4-scout (1M context) for the 300k-token input, "
        f"got {decision.chosen_model_id}. Smaller models should be "
        f"filtered out by the context-window requirement. "
        f"Rationale: {decision.rationale}"
    )


async def test_medium_context_request_picks_qwen3_coder_next_not_long_context_model() -> None:
    """Sprint 4 boundary check: a request that fits in 256k should pick
    qwen3-coder-next (cheaper) over llama-4-scout (more expensive but
    overcapacity). The context-window filter is an admission filter,
    not a preference — the picker still chooses cheapest among admitted.
    """
    from modelmeld.api.schemas import ChatCompletionRequest, UserMessage

    registry = ModelRegistry([
        _entry("llama-3.3-70b", "openrouter", 0.10, 0.32,
               coding=0.78, reasoning=0.74, context_window=131_072),
        _entry("qwen3-coder-next", "openrouter", 0.11, 0.80,
               coding=0.78, reasoning=0.72, context_window=262_144),
        _entry("llama-4-scout", "vllm", 0.45, 0.45,
               coding=0.74, reasoning=0.78, context_window=1_048_576),
    ])
    scout = CapabilityScout(registry=registry, quality_threshold=0.70)

    # ~140k input tokens (fits qwen3-coder-next at 256k but not
    # llama-3.3-70b at 131k); cheapest qualifying = qwen3-coder-next.
    medium_input = "refactor this code:\n" + ("def foo(): return 1\n" * 28_000)
    req = ChatCompletionRequest(
        model="anthropic/modelmeld-saver",
        messages=[UserMessage(role="user", content=medium_input)],
        max_tokens=2048,
    )

    decision = await scout.choose(req)
    assert decision.chosen_model_id == "qwen3-coder-next", (
        f"Expected qwen3-coder-next (256k) for medium-context request, "
        f"got {decision.chosen_model_id}. Rationale: {decision.rationale}"
    )


async def test_auto_alias_falls_back_to_oss_reasoner_when_no_frontier_adapter() -> None:
    """Sprint 3 (B-2): AUTO with reasoning markers must NOT 503 when the
    operator has no frontier adapter configured. Should pick the best
    OSS reasoning model instead.

    The router computes `available_frontier_providers` from the union of
    its env-configured adapters + BYOK overrides. When that intersection
    with frontier_providers() is empty, the scout's AUTO branch leaves
    `eligible` as the OSS pool rather than returning an empty set.
    """
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(
        _policy_req(
            "anthropic/modelmeld-auto",
            "Please think step by step and explain your reasoning carefully.",
        ),
        # Operator has NO frontier adapters configured — neither env
        # (Anthropic / OpenAI keys missing) nor BYOK.
        available_frontier_providers=frozenset(),
    )
    # 2 markers tripped escalation, but no frontier adapter → fall back
    # to the OSS pool. Picker chooses the reasoning-capable OSS row.
    assert decision.chosen_provider in {"openrouter", "vllm", "fireworks", "together"}, (
        f"Expected OSS provider on fallback, got {decision.chosen_provider}. "
        f"Rationale: {decision.rationale}"
    )
    # The rationale must call out the fallback explicitly — operators
    # reading the audit log need to see WHY a frontier-shaped request
    # landed on an OSS model.
    assert "no_frontier_adapter" in decision.rationale
    assert "fallback=oss_reasoner" in decision.rationale


async def test_quality_alias_picks_frontier_by_default() -> None:
    """QUALITY restricts to frontier providers (anthropic/openai).
    Even if frontier task_scores are lower than OSS in some category,
    the provider filter forces the pick into the frontier tier."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req("anthropic/modelmeld-quality"))
    assert decision.chosen_provider == "anthropic"
    assert decision.chosen_model_id in {"claude-sonnet-4-6", "claude-opus-4-7"}
    assert "policy=quality" in decision.rationale
    assert "frontier_first" in decision.rationale


async def test_quality_alias_downgrades_on_autocomplete_shape() -> None:
    """QUALITY's promise: frontier-first WITH automatic downgrade on
    obviously-trivial requests (autocomplete shape). Don't bill Opus
    rates for a 100-token autocomplete completion."""
    from modelmeld.api.schemas import ChatCompletionRequest, UserMessage

    # Add a sub-Haiku-tier OSS row to the registry so the bias has
    # something cheap to downgrade to. Autocomplete-shape requests
    # classify as `simple_qa`, so every row needs a simple_qa score
    # ≥ the autocomplete bias threshold (0.55) to qualify.
    registry = ModelRegistry([
        ModelEntry(
            model_id="qwen3-flash", provider="openrouter",
            context_window=100_000, cost_per_m_input=0.30, cost_per_m_output=0.60,
            task_scores={"coding": 0.78, "simple_qa": 0.74},
            source="test",
        ),
        ModelEntry(
            model_id="phi-4-mini", provider="openrouter",
            context_window=100_000, cost_per_m_input=0.05, cost_per_m_output=0.10,
            task_scores={"coding": 0.55, "simple_qa": 0.74, "reasoning": 0.50},
            source="test",
        ),
        ModelEntry(
            model_id="claude-sonnet-4-6", provider="anthropic",
            context_window=200_000, cost_per_m_input=3.0, cost_per_m_output=15.0,
            task_scores={"coding": 0.80, "simple_qa": 0.93},
            source="test",
        ),
    ])
    scout = CapabilityScout(registry=registry)
    # Autocomplete-shape request: short input, max_tokens ≤ 256, no tools.
    autocomplete_req = ChatCompletionRequest(
        model="anthropic/modelmeld-quality",
        messages=[UserMessage(role="user", content="x = ")],
        max_tokens=64,
        tools=[],
    )
    decision = await scout.choose(autocomplete_req)
    # Shape bias lowers threshold so phi-4-mini qualifies. Provider
    # filter NOT applied (QUALITY skipped the frontier filter on
    # autocomplete shape). Cheapest qualifying model wins → phi-4-mini.
    assert decision.chosen_provider == "openrouter"
    assert "downgrade=autocomplete_shape" in decision.rationale


async def test_deprecated_aliases_still_resolve_to_a_policy() -> None:
    """Backwards-compat: old aliases must continue to work."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    # 'coding' was OSS-only behavior → now maps to SAVER
    decision = await scout.choose(_policy_req("anthropic/modelmeld-coding"))
    assert decision.chosen_provider in {"openrouter", "vllm", "fireworks", "together"}
    assert "policy=saver" in decision.rationale
    # 'frontier-priority' was always-frontier → now maps to QUALITY
    decision = await scout.choose(_policy_req("anthropic/modelmeld-frontier-priority"))
    assert decision.chosen_model_id in {"claude-sonnet-4-6", "claude-opus-4-7"}
    assert "policy=quality" in decision.rationale


async def test_quality_alias_works_when_scout_eligible_is_oss_only() -> None:
    """Regression for live validation finding: on the hosted gateway, the
    scout's persistent `eligible_providers` is the OSS upstream pool
    (fireworks/together/openrouter). Naive intersection with frontier
    yields {} → 503. QUALITY must REPLACE eligible, not intersect."""
    scout = CapabilityScout(
        registry=_mixed_tier_registry(),
        eligible_providers=frozenset({"openrouter"}),  # OSS only, like prod
    )
    decision = await scout.choose(_policy_req("anthropic/modelmeld-quality"))
    # Even though scout's default eligible excluded anthropic, the policy
    # opens it up; scout picks a frontier row.
    assert decision.chosen_provider == "anthropic"
    assert decision.chosen_model_id in {"claude-sonnet-4-6", "claude-opus-4-7"}


async def test_auto_escalated_works_when_scout_eligible_is_oss_only() -> None:
    """Mirror of the above for AUTO + reasoning markers."""
    scout = CapabilityScout(
        registry=_mixed_tier_registry(),
        eligible_providers=frozenset({"openrouter"}),  # OSS only
    )
    decision = await scout.choose(_policy_req(
        "anthropic/modelmeld-auto",
        "Please think step by step and explain your reasoning carefully.",
    ))
    assert decision.chosen_provider == "anthropic"


async def test_non_alias_model_id_does_not_trigger_policy() -> None:
    """A literal `claude-opus-4-7` request must NOT pick up an aliased
    policy — the scout's existing task-category routing handles it."""
    scout = CapabilityScout(registry=_mixed_tier_registry())
    decision = await scout.choose(_policy_req("claude-opus-4-7"))
    # No policy=... rationale; scout used its default threshold (0.70)
    assert "policy=" not in decision.rationale
    # At threshold 0.70, the cheapest qualifier is OSS — same as before #177
    assert decision.chosen_provider in {"openrouter", "vllm", "fireworks", "together"}

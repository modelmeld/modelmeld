# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""CapabilityScout — picks a specific model from the registry per request.

The capability routing primitive. Given a request, the scout:
  1. Classifies the prompt into a task category (via TaskCategoryClassifier)
  2. Asks the ModelRegistry for the cheapest model meeting `quality_threshold`
     on that category, filtered by `eligible_providers`
  3. Returns a CapabilityDecision the router can act on

The decision carries a chosen model + fallbacks so the router can fail over
to the next-cheapest qualified model when the chosen provider is unhealthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.devtool import Fingerprint, Fingerprinter
from modelmeld.scout.difficulty import (
    DifficultyClassifier,
    difficulty_routing_enabled,
)
from modelmeld.scout.policy import (
    ModelMeldPolicy,
    agentic_reliability_floor,
    frontier_providers,
    large_context_threshold,
    oss_providers,
    resolve_policy,
    should_escalate_to_frontier,
)
from modelmeld.scout.registry import ModelEntry, ModelRegistry
from modelmeld.scout.task_category import (
    TaskCategoryClassifier,
    TaskCategoryDecision,
)

# Exposed as a module constant so operators can override
# without hunting through code. The 0.70 figure is the current production-
# tuned value — the cheapest model on each task category must clear this
# task_score (0..1 scale) to be a candidate. Higher → stricter quality
# bar, smaller candidate set, higher cost. The *methodology* for deriving
# this threshold lives in `modelmeld_enterprise.routing_tuning`.
DEFAULT_QUALITY_THRESHOLD = 0.70

# When the request has `tools=[...]` or its estimated
# input prompt exceeds the model's context window, the scout's filters
# kick in. The headroom multiplier reserves space for the response on
# top of the input — a model whose context window is exactly equal
# to the input has zero room for output. 1.2× is a reasonable safety
# margin for typical agentic coding workloads (input dominates output
# ~5-10:1 for code).
_CONTEXT_WINDOW_HEADROOM_MULT = 1.2
# Fallback when no token_counter is configured (or fails). 1 token ≈
# 4 chars of English; conservative — actual ratio is closer to 3.5
# for code, 4-5 for natural language.
_CHARS_PER_TOKEN_ESTIMATE = 4

# DevTool fingerprint → routing biases.
#
# The same tool sends very different request shapes: Cursor's
# autocomplete is sub-Haiku territory, but Cursor's chat is OSS-mid.
# Biasing by tool alone is too coarse. The bias logic combines tool
# fingerprint + request SHAPE (max_tokens, input chars, tools presence)
# to detect specific patterns we can route cheap.
#
# Autocomplete-shape detection thresholds. Empirically derived from
# Cursor/Copilot/Continue autocomplete traffic patterns:
#   - max_tokens ≤ 256 (typical autocomplete budget; chat is usually
#     ≥1024)
#   - estimated input ≤ 4000 chars (~1000 tokens; chat with @-mentioned
#     files is 10K+)
#   - no tools= field (autocomplete is single-shot, no agentic
#     tool calls)
# All three must hold to classify as autocomplete-shape and apply the
# sub-Haiku tier bias.
_AUTOCOMPLETE_MAX_TOKENS = 256
_AUTOCOMPLETE_MAX_INPUT_CHARS = 4000
# The quality_threshold to apply when autocomplete-shape detected.
# Lower threshold makes the sub-Haiku tier (granite-4-micro,
# gemma-3-4b, phi-4-mini) eligible. 0.55 is below the SLM capability
# cliff for tool-use but above raw text quality — matches the
# "trivial-fast" tier's task scores.
_AUTOCOMPLETE_QUALITY_THRESHOLD = 0.55

# --- D1 latency term (routing-objective redesign) ---
# Applied ONLY to `-auto` + tool-bearing (agentic) requests; `-saver` stays
# pure cost (predictable cost ceiling is its wedge) and `-quality` is
# frontier-first. The scout ranks eligible candidates by latency-adjusted
# cost: blended_cost * (1 + _AUTO_LATENCY_WEIGHT * estimated_turn_latency_s).
#
# Weight calibration (deliberately modest — see the pressure-test in
# docs/design-routing-objective.md): at ~0.02/s a 15 s/turn model carries a
# ~+30% effective-cost penalty vs ~+14% for a 7 s/turn one. This BREAKS
# NEAR-COST TIES toward the faster option (the per-(model x provider) case:
# same model + same cost on two backends, faster backend wins) but by design
# does NOT override a large genuine cost advantage. It is NOT meant to flip a
# 2x-cheaper-but-slower model on its own — that model's real problem is
# reliability + turns-to-converge (RO-3) and mid-flight escalation (RO-5),
# not per-turn latency. Tuning this weight to force such a flip would be a
# savings claim that doesn't survive pressure-testing.
_AUTO_LATENCY_WEIGHT = 0.02
# Agentic turns emit little output (D-1 telemetry: ~91 output tokens/turn vs
# ~61k input). Input size is taken from the actual request; output uses this
# small fixed reference.
_AGENTIC_REF_OUTPUT_TOKENS = 128

# Capability category `-quality` ranks agentic (tool-bearing) requests by.
# Sourced from AA's Coding Index via the live registry feed (a sustained-
# agentic-coding success-rate prior); absent in the OSS snapshot, where the
# scout falls back per-model to the request's category score.
_AGENTIC_CAPABILITY_CATEGORY = "agentic_coding"

# Optional dependency: framework-supplied hints. Imported under
# TYPE_CHECKING to avoid an import cycle (api.routing_hints imports task_category).
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelmeld.api.routing_hints import RoutingHints


class NoEligibleModelError(RuntimeError):
    """Raised when no model in the registry meets the quality bar for the task.

    The structured fields (`task_category`, `quality_threshold`,
    `eligible_providers`) are preserved on the exception for operator
    introspection; the human-readable message text deliberately omits
    the `eligible_providers` list. The message bubbles into customer-
    facing error responses, where the provider list is an internal
    routing-table detail that isn't actionable for the caller — they
    need to know what to adjust (threshold, BYOK header, alias), not
    which adapters were considered. Operators who need the provider
    list can read `exc.eligible_providers` directly off the exception.
    """

    def __init__(
        self,
        task_category: str,
        quality_threshold: float,
        eligible_providers: frozenset[str] | None,
    ) -> None:
        self.task_category = task_category
        self.quality_threshold = quality_threshold
        self.eligible_providers = eligible_providers
        super().__init__(
            f"No model in registry meets threshold {quality_threshold} "
            f"for task '{task_category}'"
        )


@dataclass(frozen=True)
class CapabilityDecision:
    """Result of CapabilityScout.choose().

    `chosen_model_id`     — the model the router should send the request to
    `chosen_provider`     — provider for adapter lookup (e.g. "openai")
    `task_category`       — what the classifier called this prompt
    `task_score`          — chosen model's task_scores[task_category]
    `quality_threshold`   — threshold the chosen model just exceeded
    `fallback_model_ids`  — next-cheapest qualified models, in cost order
    `category_decision`   — full TaskCategoryDecision (per-category scores, rationale)
    `devtool_fingerprint` — dev-tool fingerprint, for hooks/analytics
    `rationale`           — short trace for logs ("category=coding;chose=qwen3-coder-next;cost=$0.32/M")
    """

    chosen_model_id: str
    chosen_provider: str
    task_category: str
    task_score: float
    quality_threshold: float
    fallback_model_ids: list[str] = field(default_factory=list)
    category_decision: TaskCategoryDecision | None = None
    devtool_fingerprint: Fingerprint | None = None
    rationale: str = ""
    # The provider's own model id (the slug the upstream expects on the wire),
    # distinct from the canonical `chosen_model_id` used for attribution. Empty
    # when the registry row has none (then the canonical id is sent verbatim).
    provider_model_id: str = ""

    def with_model(
        self, model_id: str, provider: str, task_score: float,
        provider_model_id: str = "",
    ) -> CapabilityDecision:
        """Return a copy with a different chosen model (used during failover)."""
        return CapabilityDecision(
            chosen_model_id=model_id,
            chosen_provider=provider,
            task_category=self.task_category,
            task_score=task_score,
            quality_threshold=self.quality_threshold,
            fallback_model_ids=self.fallback_model_ids,
            category_decision=self.category_decision,
            devtool_fingerprint=self.devtool_fingerprint,
            rationale=f"{self.rationale};failover={model_id}",
            provider_model_id=provider_model_id,
        )


class CapabilityScout:
    """Picks the cheapest provably-competent model for each request.

    The constructor takes a registry, a task classifier, a quality threshold
    (0..1; default 0.70), and an optional set of eligible providers. When
    `eligible_providers` is None, all providers in the registry are considered.
    Restricting to providers you actually have adapters for is the caller's
    responsibility — `build_router` does this for us at app-init.
    """

    name = "capability"

    def __init__(
        self,
        registry: ModelRegistry,
        classifier: TaskCategoryClassifier | None = None,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        eligible_providers: frozenset[str] | None = None,
        fingerprinter: Fingerprinter | None = None,
        fallback_depth: int = 5,
    ) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError(f"quality_threshold must be in [0,1], got {quality_threshold}")
        if fallback_depth < 0:
            raise ValueError(f"fallback_depth must be ≥0, got {fallback_depth}")
        self.registry = registry
        self.classifier = classifier or TaskCategoryClassifier()
        # Structural escalate-detector for AUTO (used when
        # MODELMELD_DIFFICULTY_ROUTING is on; else the marker path runs).
        self.difficulty_classifier = DifficultyClassifier()
        self.quality_threshold = quality_threshold
        self.eligible_providers = eligible_providers
        self.fingerprinter = fingerprinter or Fingerprinter()
        self.fallback_depth = fallback_depth

    async def choose(
        self,
        request: ChatCompletionRequest,
        hints: RoutingHints | None = None,
        available_frontier_providers: frozenset[str] | None = None,
    ) -> CapabilityDecision:
        """Pick the best (model, provider) for this request.

        `available_frontier_providers`: when the request carries BYOK
        credentials, this is the set of frontier providers the route
        handler can dispatch to (i.e., the providers for which the
        customer supplied a key). The AUTO-escalated and QUALITY
        policies use this to restrict frontier picks to providers
        the customer can actually pay for — picking gpt-5-mini when
        the customer only supplied an Anthropic BYOK key would just
        cascade to a 503. None means "no BYOK restriction known."
        """
        # Eval/debug model pin (env-gated upstream in routing_hints —
        # `pin_model` is only ever populated when MODELMELD_ALLOW_MODEL_PIN is
        # set, so this is inert on a default/production gateway). When present,
        # bypass capability selection entirely and serve exactly that model, so
        # a specific served model can be benchmarked end-to-end (agentic-
        # efficiency eval) without skewing registry data or per-token routing.
        pinned = hints.pin_model if hints is not None else None
        if pinned:
            return self._pinned_decision(request, pinned)

        # Hint-driven category overrides the classifier. Frameworks know what
        # their agents do; we trust the declaration when present.
        category_decision: TaskCategoryDecision | None = None
        category_source = "classifier"
        if hints is not None and hints.effective_category() is not None:
            category = hints.effective_category()
            assert category is not None  # narrow for type checker
            category_source = (
                "hint:task_category" if hints.task_category else "hint:agent_role"
            )
        else:
            category_decision = self.classifier.classify(request)
            category = category_decision.category
        fingerprint = self.fingerprinter.identify(request)

        # Threshold + provider filter can also be hinted at request time.
        threshold = self.quality_threshold
        eligible = self.eligible_providers
        if hints is not None:
            if hints.quality_threshold is not None:
                threshold = hints.quality_threshold
            if hints.excluded_providers:
                if eligible is not None:
                    eligible = frozenset(eligible) - hints.excluded_providers
                else:
                    eligible = self._all_registry_providers() - hints.excluded_providers
                if not eligible:
                    raise NoEligibleModelError(
                        task_category=category,
                        quality_threshold=threshold,
                        eligible_providers=eligible,
                    )

        # Alias-based policy resolution. When the request's `model`
        # field matches a ModelMeld alias (e.g.,
        # anthropic/modelmeld-saver), apply that policy's provider tier
        # filter. Hints win over policy. Non-alias model ids pass
        # through unaffected. See modelmeld/scout/policy.py.
        #
        # Tier selection is done via provider filter, NOT task_score
        # threshold manipulation. Task scores are tuned per-category and
        # change with every benchmark refresh — using them as policy
        # gates causes silent breakage when scores drift (e.g., Sonnet's
        # coding score happens to sit at 0.80 today, well below where a
        # static "frontier threshold" would naively land).
        policy = resolve_policy(getattr(request, "model", None))
        policy_rationale = ""
        if policy is not None and (hints is None or hints.quality_threshold is None):
            if policy is ModelMeldPolicy.SAVER:
                # OSS only. Frontier-provider rows are filtered entirely
                # → predictable cost ceiling for the customer.
                oss = oss_providers()
                eligible = (frozenset(eligible) & oss) if eligible is not None else oss
                policy_rationale = "policy=saver(tier=oss_only)"
            elif policy is ModelMeldPolicy.AUTO:
                # Default OSS; escalate to frontier on reasoning markers.
                # When escalating, REPLACE eligible (don't intersect):
                # the scout's persistent eligible set is typically the
                # OSS upstream pool, so intersecting with frontier yields
                # an empty set. REPLACE lets the scout pick frontier
                # rows from the registry; the router then resolves the
                # adapter from per-request BYOK overrides.
                #
                # Further restrict to providers the customer actually
                # supplied BYOK keys for — otherwise scout might pick
                # gpt-5-mini when the customer only has an Anthropic key.
                # Escalation signal: the structural difficulty detector when
                # MODELMELD_DIFFICULTY_ROUTING is on (the gap concentrates on
                # structural shapes, not reasoning-keyword phrasing), else the
                # legacy reasoning-marker count. `category` is already resolved
                # above; the detector is category-gated on it.
                if difficulty_routing_enabled():
                    diff = self.difficulty_classifier.classify(request, category)
                    escalate = diff.escalate
                    esc_detail = diff.rationale
                else:
                    escalate, marker_count = should_escalate_to_frontier(request)
                    esc_detail = f"markers={marker_count}"
                # Large-context prior (always on for AUTO; -saver keeps its OSS
                # ceiling). Open-weight models collapse on long-context repair
                # past the threshold, so route up regardless of the signal above
                # — a high-precision, near-deterministic prior, unlike the broad
                # difficulty heuristics.
                lc_threshold = large_context_threshold()
                if lc_threshold > 0:
                    ctx_tokens = self._estimated_input_tokens(request)
                    if ctx_tokens >= lc_threshold:
                        escalate = True
                        esc_detail = f"{esc_detail};large_context({ctx_tokens}tok)"
                if escalate:
                    fr = frontier_providers()
                    if available_frontier_providers is not None:
                        fr = fr & available_frontier_providers
                    if not fr:
                        # Sprint 3 gap fix: AUTO wants to escalate but no
                        # frontier adapter is available (neither env-configured
                        # nor BYOK). Fall back to an OSS reasoning model
                        # rather than 503ing — leave `eligible` as the original
                        # OSS pool so the picker selects whichever OSS model
                        # has the highest reasoning task_score within budget.
                        policy_rationale = (
                            f"policy=auto(escalation_requested;"
                            f"no_frontier_adapter;fallback=oss_reasoner;"
                            f"{esc_detail})"
                        )
                    else:
                        eligible = fr
                        policy_rationale = (
                            f"policy=auto(escalated=frontier;{esc_detail})"
                        )
                else:
                    policy_rationale = f"policy=auto(escalated=no;{esc_detail})"
            elif policy is ModelMeldPolicy.QUALITY:
                # Frontier by default UNLESS the request is detected as
                # autocomplete-shape (handled by the bias logic below).
                # Same BYOK-availability restriction as AUTO-escalated.
                if not self._is_autocomplete_shape(request):
                    fr = frontier_providers()
                    if available_frontier_providers is not None:
                        fr = fr & available_frontier_providers
                    eligible = fr
                    policy_rationale = "policy=quality(tier=frontier_first)"
                else:
                    policy_rationale = "policy=quality(downgrade=autocomplete_shape)"

        # DevTool fingerprint + request shape → routing bias.
        # Only applies when no explicit hint overrode the threshold (hints
        # always win); biases lower the bar to admit cheap-tier models for
        # detected shapes (autocomplete) where premium-quality routing is
        # wasteful. Bias NEVER raises the threshold — defensive: we let
        # cheap models in for cheap-tier shapes; we don't gate routing
        # tighter than the operator-configured threshold.
        bias_rationale = ""
        if hints is None or hints.quality_threshold is None:
            biased = self._apply_shape_bias(request, fingerprint, threshold)
            if biased is not None:
                threshold, bias_rationale = biased

        # Capability filters derived from request shape.
        # Requests carrying `tools=[...]` MUST go to a tool-capable model
        # (small models < ~7B fail multi-step tool chains per the
        # SLM-for-Agents survey). Estimated input + 1.2× headroom MUST fit
        # in the model's context window so the response isn't truncated.
        require_tool_support = bool(request.tools)
        min_ctx_required = self._required_context_window(request)

        # D1 latency term: only `-auto` + tool-bearing (agentic) requests rank
        # on latency-adjusted cost. `-saver` and non-alias requests stay pure
        # cost (latency_weight=0 → byte-identical to the old ranking);
        # `-quality` is frontier-first and not latency-ranked in v1.
        latency_weight = 0.0
        latency_ref_input = 0
        latency_rationale = ""
        if policy is ModelMeldPolicy.AUTO and require_tool_support:
            latency_weight = _AUTO_LATENCY_WEIGHT
            latency_ref_input = self._estimated_input_tokens(request)
            latency_rationale = (
                f"d1=latency(w={_AUTO_LATENCY_WEIGHT};ref_in={latency_ref_input})"
            )

        ranked = self.registry.rank(
            task_category=category,
            quality_threshold=threshold,
            eligible_providers=eligible,
            min_context_window=min_ctx_required,
            require_tool_support=require_tool_support,
            latency_weight=latency_weight,
            latency_ref_input_tokens=latency_ref_input,
            latency_ref_output_tokens=_AGENTIC_REF_OUTPUT_TOKENS,
        )
        if not ranked:
            raise NoEligibleModelError(
                task_category=category,
                quality_threshold=threshold,
                eligible_providers=eligible,
            )

        # `-quality` on agentic (tool-bearing) work: pick the STRONGEST eligible
        # model, not the cheapest one clearing the bar. The default ranking is
        # cost-ascending, so frontier-restricted QUALITY otherwise selects the
        # cheapest frontier model above threshold — a small/fast frontier model
        # that no-ops on sustained agentic loops (the confirmed `-quality`
        # agentic bug: 24/30 turns to a Haiku-tier model, zero edits). Re-rank
        # by capability descending, with cost as the tie-break so equally-capable
        # models still prefer the cheaper one. Scoped to QUALITY + tools: simple
        # QUALITY requests keep cost-first (cheapest frontier clearing the bar is
        # correct there).
        #
        # Capability signal: prefer the `agentic_coding` prior (AA Coding Index
        # — a sustained-agentic-coding SUCCESS-rate signal, the right axis for
        # "won't no-op"), falling back per-model to the request's category score
        # when a model lacks it (e.g. the OSS snapshot ships no agentic_coding;
        # only the live feed carries it). Both are 0..1 capability scores; the
        # mix is monotonic and degrades gracefully to the prior behavior when
        # the prior is absent.
        quality_rationale = ""
        if policy is ModelMeldPolicy.QUALITY and require_tool_support:
            def _capability(entry: ModelEntry) -> float:
                ts = entry.task_scores
                return ts.get(_AGENTIC_CAPABILITY_CATEGORY, ts.get(category, 0.0))

            ranked = sorted(ranked, key=lambda pair: (-_capability(pair[0]), pair[1]))
            signal = (
                _AGENTIC_CAPABILITY_CATEGORY
                if any(_AGENTIC_CAPABILITY_CATEGORY in e.task_scores for e, _ in ranked)
                else category
            )
            quality_rationale = f"quality_agentic=capability_first(by={signal})"

        # -auto agentic routing: drop models MEASURED unreliable on multi-provider
        # agentic correctness, then keep the cheapest survivor (cheapest-RELIABLE,
        # not cheapest-overall — the chronic shortcut-takers are cheap but ship
        # latent multi-provider bugs). Only filters rows carrying a measured
        # `agentic_coding` below the floor; un-probed rows pass through. Falls back
        # to the full ranking if the floor would leave nothing — never 503s on
        # reliability alone.
        auto_reliability_rationale = ""
        if policy is ModelMeldPolicy.AUTO and require_tool_support:
            floor = agentic_reliability_floor()
            if floor > 0:
                reliable = [
                    (e, c) for (e, c) in ranked
                    if e.task_scores.get(_AGENTIC_CAPABILITY_CATEGORY) is None
                    or e.task_scores[_AGENTIC_CAPABILITY_CATEGORY] >= floor
                ]
                dropped = len(ranked) - len(reliable)
                if reliable and dropped:
                    ranked = reliable
                    auto_reliability_rationale = (
                        f"auto_reliability=floor({floor};dropped={dropped})"
                    )

        chosen_entry, chosen_cost = ranked[0]
        fallbacks: list[str] = [
            entry.model_id for entry, _ in ranked[1 : 1 + self.fallback_depth]
        ]

        rationale = (
            f"category={category}(src={category_source});"
            f"score={chosen_entry.task_scores.get(category, 0.0):.2f};"
            f"chose={chosen_entry.model_id};"
            f"cost=${chosen_cost:.3f}/Mblended"
        )
        if policy_rationale:
            rationale = f"{rationale};{policy_rationale}"
        if bias_rationale:
            rationale = f"{rationale};{bias_rationale}"
        if latency_rationale:
            rationale = f"{rationale};{latency_rationale}"
        if quality_rationale:
            rationale = f"{rationale};{quality_rationale}"
        if auto_reliability_rationale:
            rationale = f"{rationale};{auto_reliability_rationale}"

        return CapabilityDecision(
            chosen_model_id=chosen_entry.model_id,
            chosen_provider=chosen_entry.provider,
            task_category=category,
            task_score=chosen_entry.task_scores.get(category, 0.0),
            quality_threshold=threshold,
            fallback_model_ids=fallbacks,
            category_decision=category_decision,
            devtool_fingerprint=fingerprint,
            rationale=rationale,
            provider_model_id=chosen_entry.provider_model_id,
        )

    def _pinned_decision(
        self, request: ChatCompletionRequest, model_id: str,
    ) -> CapabilityDecision:
        """Serve an explicitly pinned model, bypassing capability selection.
        Eval/debug only (env-gated upstream). Raises NoEligibleModelError if the
        pinned id isn't in the registry, so an operator typo fails loudly rather
        than silently falling back to normal routing."""
        entry = self.registry.get(model_id)
        if entry is None:
            raise NoEligibleModelError(
                task_category="pinned",
                quality_threshold=0.0,
                eligible_providers=None,
            )
        return CapabilityDecision(
            chosen_model_id=entry.model_id,
            chosen_provider=entry.provider,
            task_category="pinned",
            task_score=0.0,
            quality_threshold=0.0,
            fallback_model_ids=[],
            category_decision=None,
            devtool_fingerprint=self.fingerprinter.identify(request),
            rationale=f"PIN: model={entry.model_id};scout_bypassed(MODELMELD_ALLOW_MODEL_PIN)",
            provider_model_id=entry.provider_model_id,
        )

    def lookup_fallback(self, model_id: str) -> ModelEntry | None:
        """Resolve a fallback model_id to its registry entry. Used by the router."""
        return self.registry.get(model_id)

    def _all_registry_providers(self) -> frozenset[str]:
        return frozenset(e.provider for e in self.registry.all_entries())

    def _apply_shape_bias(
        self,
        request: ChatCompletionRequest,
        fingerprint: Fingerprint,
        current_threshold: float,
    ) -> tuple[float, str] | None:
        """Return (biased_threshold, rationale) or None if no bias applies.

        Shape-bias logic — combines DevTool fingerprint with
        request shape (max_tokens, input chars, tools presence) to detect
        cheap-tier-appropriate patterns. The first matching bias wins;
        biases NEVER raise the threshold above the operator-configured
        value (defensive — let cheap models in for cheap shapes, don't
        gate routing tighter).

        Currently detects:
          - autocomplete-shape: any tool + max_tokens ≤ 256 + input ≤ 4K
            chars + no tools[] → quality_threshold lowered to 0.55 so
            sub-Haiku tier models (granite-4-micro, gemma-3-4b,
            phi-4-mini) become eligible. Tool fingerprint included in
            rationale for FinOps slicing.

        Future biases (deferred — need real production traffic data):
          - Aider commit-message-shape → cheap tier
          - Claude Code background-pattern-shape → cheap tier
          - Cline plan-mode → premium tier (raise threshold? — would
            need to gate via operator opt-in)
        """
        if self._is_autocomplete_shape(request):
            new_threshold = _AUTOCOMPLETE_QUALITY_THRESHOLD
            # Never raise — only lower
            if new_threshold >= current_threshold:
                return None
            rationale = (
                f"bias=autocomplete_shape("
                f"tool={fingerprint.tool.value};"
                f"threshold:{current_threshold:.2f}→{new_threshold:.2f})"
            )
            return new_threshold, rationale
        return None

    def _is_autocomplete_shape(self, request: ChatCompletionRequest) -> bool:
        """True iff the request matches the autocomplete signature.

        Empirically: autocomplete is short input (≤ ~1000 tokens),
        small max_tokens budget (≤ 256), no tool definitions, and
        usually a single user message. Any of the four conditions
        failing disqualifies — we'd rather miss an autocomplete bias
        than misroute a real chat to sub-Haiku tier and degrade UX.
        """
        if request.tools:
            return False
        if request.max_tokens is None or request.max_tokens > _AUTOCOMPLETE_MAX_TOKENS:
            return False
        # Count input chars from message contents.
        char_count = 0
        for msg in request.messages:
            content = getattr(msg, "content", None)
            if content is None:
                continue
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if text is not None:
                        char_count += len(text)
            if char_count > _AUTOCOMPLETE_MAX_INPUT_CHARS:
                return False  # short-circuit on first overage
        return char_count <= _AUTOCOMPLETE_MAX_INPUT_CHARS

    def _estimated_input_tokens(self, request: ChatCompletionRequest) -> int:
        """Rough input-token estimate: message text + tool-def schemas, at
        ~4 chars/token (conservative-for-code, per the litellm canonical
        pattern). Used by both the context-window filter and the D1 latency
        term (as the per-turn prefill size)."""
        char_count = 0
        for msg in request.messages:
            content = getattr(msg, "content", None)
            if content is None:
                continue
            if isinstance(content, str):
                char_count += len(content)
            elif isinstance(content, list):
                # multimodal: count text parts; image/audio not in tokens here
                for part in content:
                    text = getattr(part, "text", None)
                    if text is not None:
                        char_count += len(text)
        # Include tool definition schemas (Claude Code sends 30-40K
        # tokens of tool defs even before the user types).
        if request.tools:
            for tool in request.tools:
                try:
                    char_count += len(str(tool.function.parameters))
                except (AttributeError, TypeError):
                    char_count += 200  # rough estimate for unknown tool shape
                if hasattr(tool, "function") and hasattr(tool.function, "description"):
                    char_count += len(tool.function.description or "")
        return char_count // _CHARS_PER_TOKEN_ESTIMATE

    def _required_context_window(self, request: ChatCompletionRequest) -> int:
        """Estimate the context window the request demands.

        Input tokens (`_estimated_input_tokens`) + `max_tokens` response
        budget, times 1.2× headroom for prompt-engineering overhead (system
        prompt template, message role tokens, etc.).

        Returns 0 when the request is small enough that the filter is
        a no-op. The filter only kicks in when input + response
        plausibly exceeds smaller models' context windows
        (~16K-32K range).
        """
        input_tokens = self._estimated_input_tokens(request)
        response_budget = request.max_tokens or 1024
        return int((input_tokens + response_budget) * _CONTEXT_WINDOW_HEADROOM_MULT)

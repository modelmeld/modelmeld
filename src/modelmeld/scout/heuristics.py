# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""HeuristicScout — fast, rule-based classifier with no ML dependencies.

Operates on a small set of signals: token-count buckets, dev-tool boilerplate
keyword matches, complex-reasoning markers, and tool-call presence. Tunable
via `HeuristicWeights`. Dev-tool-aware fingerprints layer on
top via `Fingerprinter`.

OSS users get the same tuned default weights we run in production — see
`DEFAULT_HEURISTIC_WEIGHTS`. The *methodology* for deriving these weights
(traffic-fitted A/B harness, periodic retuning) lives in
`modelmeld_enterprise.routing_tuning` and powers the paid registry feed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
)
from modelmeld.scout.base import Scout, ScoutDecision, Tier
from modelmeld.scout.devtool import Fingerprinter

# Patterns chosen to be unambiguous on a coder/dev-tool prompt distribution.
# All matches are case-insensitive (compiled with re.IGNORECASE below).
_SIMPLE_KEYWORDS = (
    r"\bcomplete the (line|function|signature)\b",
    r"\b(format|reformat|prettify)\b",
    r"\b(rename|extract) (variable|function|method)\b",
    r"\b(add|write|insert) (a )?(docstring|type hint|comment)\b",
    r"\bfix (the )?indent",
    r"\bimport sort",
    r"\bwhat (is|does) [\w\.]+(\(.*\))? (do|mean|return)\b",
    r"\bgenerate boilerplate\b",
    r"\bautocomplete\b",
)

_COMPLEX_KEYWORDS = (
    r"\bdesign (a|an) (distributed|scalable|fault-?tolerant|microservice)",
    r"\b(analy[sz]e|reason about|think through) .{0,40}\b(trade-?offs?|architecture|approach)",
    r"\bstep[- ]by[- ]step\b",
    r"\bprove that\b",
    r"\b(plan|outline|design) .{0,40}\b(system|architecture|migration)",
    r"\bderive\b",
    r"\b(explain|describe) .{0,40}\bin detail\b",
    r"\bdebug (this|the) .{0,40}\b(complex|tricky|subtle)",
)

_SIMPLE_RE = re.compile("|".join(_SIMPLE_KEYWORDS), re.IGNORECASE)
_COMPLEX_RE = re.compile("|".join(_COMPLEX_KEYWORDS), re.IGNORECASE)


@dataclass(frozen=True)
class HeuristicWeights:
    """Tunable scoring constants for `HeuristicScout`.

    Local-score arithmetic:

        score = neutral_base
              + (short_prompt_boost  if tokens < short_token_limit)
              - (long_prompt_penalty if tokens > long_token_limit)
              + (simple_keyword_boost   if SIMPLE_RE matches)
              - (complex_keyword_penalty if COMPLEX_RE matches)
              - (has_tools_penalty   if request.tools is set)

        score is clamped to [0.0, 1.0]. score >= confidence_threshold → LOCAL.

    The OSS default (`DEFAULT_HEURISTIC_WEIGHTS`) ships the current
    production-tuned values. Subclass or build a new instance to override
    for organization-specific tuning (e.g. compliance-biased deployments
    that want stronger short-prompt boost to keep more traffic local).
    """

    neutral_base: float = 0.50
    short_prompt_boost: float = 0.25      # <short_token_limit tokens favors local
    long_prompt_penalty: float = 0.30     # >long_token_limit tokens penalizes local
    simple_keyword_boost: float = 0.20
    complex_keyword_penalty: float = 0.25
    has_tools_penalty: float = 0.15
    short_token_limit: int = 200
    long_token_limit: int = 2000


DEFAULT_HEURISTIC_WEIGHTS = HeuristicWeights()


def _estimate_tokens(text: str) -> int:
    """Cheap stand-in for tiktoken. 1 token ≈ 4 chars on English/code averages."""
    return max(1, len(text) // 4)


def _extract_text(request: ChatCompletionRequest) -> str:
    """Concatenate all textual content across messages for scoring."""
    pieces: list[str] = []
    for msg in request.messages:
        if isinstance(msg, (SystemMessage, ToolMessage)):
            content = msg.content
            if isinstance(content, str):
                pieces.append(content)
            else:
                pieces.extend(part.text for part in content)
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                pieces.append(msg.content)
            else:
                for part in msg.content:
                    if isinstance(part, TextPart):
                        pieces.append(part.text)
        # AssistantMessage content also exists but is prior context, not new signal
    return "\n".join(pieces)


class HeuristicScout(Scout):
    name = "heuristic"

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        fingerprinter: Fingerprinter | None = None,
        weights: HeuristicWeights | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0,1], got {confidence_threshold}"
            )
        self.confidence_threshold = confidence_threshold
        self.fingerprinter = fingerprinter or Fingerprinter()
        self.weights = weights or DEFAULT_HEURISTIC_WEIGHTS

    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        text = _extract_text(request)
        tokens = _estimate_tokens(text)
        w = self.weights

        local_score = w.neutral_base
        signals: list[str] = []

        if tokens < w.short_token_limit:
            local_score += w.short_prompt_boost
            signals.append(f"short_prompt({tokens}tok)")
        elif tokens > w.long_token_limit:
            local_score -= w.long_prompt_penalty
            signals.append(f"long_prompt({tokens}tok)")

        if _SIMPLE_RE.search(text):
            local_score += w.simple_keyword_boost
            signals.append("simple_keyword")

        if _COMPLEX_RE.search(text):
            local_score -= w.complex_keyword_penalty
            signals.append("complex_keyword")

        if request.tools:
            local_score -= w.has_tools_penalty
            signals.append("tools_defined")

        fingerprint = self.fingerprinter.identify(request)
        if fingerprint.tool.value != "unknown":
            signals.append(f"tool:{fingerprint.tool}")

        local_score = max(0.0, min(1.0, local_score))

        if local_score >= self.confidence_threshold:
            tier = Tier.LOCAL
            confidence = local_score
        else:
            tier = Tier.CLOUD
            # Confidence in the cloud routing is how far below threshold we landed.
            confidence = 1.0 - local_score

        rationale = ", ".join(signals) if signals else "neutral"
        return ScoutDecision(
            tier=tier,
            confidence=confidence,
            rationale=rationale,
            signals={
                "local_score": local_score,
                "tokens": tokens,
                "threshold": self.confidence_threshold,
                "matched": signals,
                "devtool": fingerprint.tool.value,
                "devtool_confidence": fingerprint.confidence,
            },
        )

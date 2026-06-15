# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Structural escalate-detector — when does a prompt actually need frontier?

Replaces `-auto`'s reasoning-marker counting (`policy.should_escalate_to_frontier`).
Gap-by-prompt-type analysis found the open-vs-frontier quality gap is LUMPY, not
graded: it concentrates on a few *structural* prompt shapes — multi-file /
long-horizon, greenfield / compositional construction, novel-algorithm reasoning
— and is ~absent on the broad middle and on self-contained algorithmic work
(where strong open-weight models match frontier). Reasoning-keyword phrasing is a
poor proxy for that, and extended reasoning can even hurt coding, so the marker
list misfires on exactly the coding traffic `-auto` serves.

Pure-heuristic, no ML dependency (matches `HeuristicScout` /
`TaskCategoryClassifier`), so it is safe on the routing hot path. Category-gated
and tuned for PRECISION: it defaults to ROUTE_OSS and escalates only on a strong
structural combination, because a false escalate burns the predictable-cost
ceiling while a false route is recoverable later via reactive (stall-based)
escalation.

Selected by the `MODELMELD_DIFFICULTY_ROUTING` env flag (default OFF this
increment — the marker path stays the shipped default until the detector's
precision is evaluated against labeled data and the default is flipped).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.policy import extract_user_text


class DifficultyRoute(str, Enum):
    """Where the structural signal points for this prompt."""

    ROUTE_OSS = "route_oss"   # strong OSS matches frontier — stay OSS
    ESCALATE = "escalate"     # frontier has a real edge (the structural shapes)
    NEITHER = "neither"       # beyond-frontier tail; frontier also struggles.
    # NEITHER is reserved for a later increment (it needs difficulty-depth
    # signals not visible from the prompt surface). v1 emits ROUTE_OSS/ESCALATE
    # only; `-auto` would escalate NEITHER anyway (don't silently degrade).


# Only these task categories are eligible to escalate. The broad middle
# (simple_qa / summarization / tool_use) stays OSS — strong open models match
# frontier there, and escalating it is the main false-positive the marker list
# produced.
_ESCALATABLE_CATEGORIES: frozenset[str] = frozenset({"coding", "reasoning"})


@dataclass(frozen=True)
class DifficultyWeights:
    """Tunable thresholds for the structural escalate-detector.

    OSS default ships conservative (precision-biased) values. Override for
    org-specific tuning; a later increment fits these against labeled data.
    """

    # >= this many DISTINCT referenced files (or an explicit "across files"
    # phrase) => multi-file scope, the strongest escalate signal.
    multi_file_threshold: int = 2
    # >= this many interacting constraints (conjunctions + enumerated items)
    # => compositional.
    constraint_threshold: int = 3


DEFAULT_DIFFICULTY_WEIGHTS = DifficultyWeights()


# --- structural feature patterns (case-insensitive) ------------------------- #
_FILE_PATH_RE = re.compile(
    r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|cc|c|h|hpp|rb|php|cs|swift"
    r"|kt|scala|sh|sql|json|ya?ml|toml|md)\b",
    re.IGNORECASE,
)
_ACROSS_FILES_RE = re.compile(
    r"\b(?:across|spanning|multiple|several|each of the)\s+(?:\w+\s+){0,2}"
    r"(?:files|modules|services|packages|components|subsystems)\b",
    re.IGNORECASE,
)
_GREENFIELD_RE = re.compile(
    r"\b(?:create|build|implement|develop|design|scaffold|set up|stand up|write)\b"
    r".{0,40}?\b(?:new|from scratch|app|application|service|system|library|cli"
    r"|api|pipeline|framework|prototype|mvp|module|component|feature|integration)\b",
    re.IGNORECASE | re.DOTALL,
)
_MUST_DISCOVER_RE = re.compile(
    r"\b(?:figure out|work out|determine|come up with|devise|design)\b"
    r".{0,40}?\b(?:algorithm|approach|strategy|how to|the best way|optimal|solution)\b",
    re.IGNORECASE | re.DOTALL,
)
# Interacting-constraint signals.
_CONSTRAINT_RE = re.compile(
    r"\b(?:and also|as well as|in addition|additionally|while also|must(?: also)?"
    r"|should(?: also)?|ensure (?:that|it)|make sure|without breaking|but keep)\b",
    re.IGNORECASE,
)
_ENUM_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")  # bullet / numbered list items


@dataclass(frozen=True)
class DifficultyDecision:
    """Output of `DifficultyClassifier.classify()`."""

    route: DifficultyRoute
    rationale: str                       # short trace for logs / telemetry
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def escalate(self) -> bool:
        """`-auto` escalates ESCALATE and the (reserved) NEITHER tail."""
        return self.route in (DifficultyRoute.ESCALATE, DifficultyRoute.NEITHER)


class DifficultyClassifier:
    """Heuristic structural escalate-detector. See module docstring."""

    name = "structural_difficulty"

    def __init__(self, weights: DifficultyWeights | None = None) -> None:
        self.weights = weights or DEFAULT_DIFFICULTY_WEIGHTS

    def classify(
        self, request: ChatCompletionRequest, category: str,
    ) -> DifficultyDecision:
        # Category gate first: the broad middle never escalates.
        if category not in _ESCALATABLE_CATEGORIES:
            return DifficultyDecision(
                route=DifficultyRoute.ROUTE_OSS,
                rationale=f"category={category}:not_escalatable",
                signals={"category": category},
            )

        # User text only — system prompts carry harness boilerplate that would
        # skew the signal (mirrors policy.extract_user_text's rationale).
        text = extract_user_text(request)
        w = self.weights

        distinct_files = len({m.group(0).lower() for m in _FILE_PATH_RE.finditer(text)})
        multi_file = (
            distinct_files >= w.multi_file_threshold
            or bool(_ACROSS_FILES_RE.search(text))
        )
        greenfield = bool(_GREENFIELD_RE.search(text))
        constraint_count = len(_CONSTRAINT_RE.findall(text)) + len(_ENUM_RE.findall(text))
        compositional = constraint_count >= w.constraint_threshold
        must_discover = bool(_MUST_DISCOVER_RE.search(text))

        signals: dict[str, Any] = {
            "category": category,
            "distinct_files": distinct_files,
            "multi_file": multi_file,
            "greenfield": greenfield,
            "constraint_count": constraint_count,
            "compositional": compositional,
            "must_discover_algorithm": must_discover,
        }

        # HIGH PRECISION: escalate only on a strong combination, never a single
        # weak hit. These are the shapes where frontier showed a real edge.
        fired: list[str] = []
        if multi_file:
            fired.append(f"multi_file(files={distinct_files})")
        if greenfield and compositional:
            fired.append(f"greenfield+compositional(constraints={constraint_count})")
        if must_discover:
            fired.append("must_discover_algorithm")

        if fired:
            return DifficultyDecision(
                route=DifficultyRoute.ESCALATE,
                rationale="escalate:" + ",".join(fired),
                signals=signals,
            )
        return DifficultyDecision(
            route=DifficultyRoute.ROUTE_OSS,
            rationale="route_oss:no_strong_structural_signal",
            signals=signals,
        )


def difficulty_routing_enabled() -> bool:
    """Whether `MODELMELD_DIFFICULTY_ROUTING` selects the structural detector
    over the legacy reasoning-marker escalation for `-auto`. Default OFF."""
    return os.environ.get("MODELMELD_DIFFICULTY_ROUTING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "DEFAULT_DIFFICULTY_WEIGHTS",
    "DifficultyClassifier",
    "DifficultyDecision",
    "DifficultyRoute",
    "DifficultyWeights",
    "difficulty_routing_enabled",
]

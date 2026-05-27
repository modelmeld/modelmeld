# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Routing hints — framework-supplied overrides via request headers.

Multi-agent frameworks (AutoGen, CrewAI, LangGraph, OpenClaw, MetaGPT) already
know what each agent does — "coder", "researcher", "summarizer", "reviewer".
They can tell us directly via headers, skipping the heuristic classifier and
giving us better routing decisions than we'd get from text alone.

Supported headers:
    x-modelmeld-task-category    one of TASK_CATEGORIES (overrides classifier)
    x-modelmeld-agent-role       agent role name (mapped to a task category)
    x-modelmeld-quality-threshold float 0..1 (overrides scout's default)
    x-modelmeld-exclude-providers comma-separated (compliance / cost ceiling)

Header values are validated and rejected with 400 on malformed input. Hints
propagate into the CapabilityDecision.rationale so audit logs show which
came from a framework declaration vs the heuristic classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from modelmeld.scout.task_category import TASK_CATEGORIES

# Header names. Keep lowercase for case-insensitive ASGI / Starlette lookups.
HEADER_TASK_CATEGORY = "x-modelmeld-task-category"
HEADER_AGENT_ROLE = "x-modelmeld-agent-role"
HEADER_QUALITY_THRESHOLD = "x-modelmeld-quality-threshold"
HEADER_EXCLUDE_PROVIDERS = "x-modelmeld-exclude-providers"

# Liberal mapping — frameworks use varied vocabulary. Keys are normalized
# (lowercased, hyphens → underscores). Add to this rather than reject when
# you see a real-world role we don't cover.
AGENT_ROLE_TO_CATEGORY: dict[str, str] = {
    # Coding
    "coder": "coding",
    "programmer": "coding",
    "developer": "coding",
    "engineer": "coding",
    "code_writer": "coding",
    "software_engineer": "coding",
    "debugger": "coding",
    # Reasoning / analysis
    "researcher": "reasoning",
    "analyst": "reasoning",
    "planner": "reasoning",
    "architect": "reasoning",
    "reviewer": "reasoning",
    "critic": "reasoning",
    "evaluator": "reasoning",
    "judge": "reasoning",
    "strategist": "reasoning",
    "mathematician": "reasoning",
    # Summarization
    "summarizer": "summarization",
    "writer": "summarization",
    "editor": "summarization",
    "scribe": "summarization",
    # Tool use
    "executor": "tool_use",
    "tool_caller": "tool_use",
    "actor": "tool_use",
    "agent": "tool_use",
    "operator": "tool_use",
    # Simple chat / Q&A
    "assistant": "simple_qa",
    "chatbot": "simple_qa",
    "responder": "simple_qa",
    "qa": "simple_qa",
}


class RoutingHintError(ValueError):
    """Header value failed validation; surface as 400."""


@dataclass(frozen=True)
class RoutingHints:
    """Parsed routing-hint headers. All fields optional."""

    task_category: str | None = None
    agent_role: str | None = None             # normalized lowercase form
    quality_threshold: float | None = None
    excluded_providers: frozenset[str] | None = None

    @property
    def has_category_hint(self) -> bool:
        """True iff caller supplied either an explicit category or an agent role we recognize."""
        return self.task_category is not None or self.derived_category is not None

    @property
    def derived_category(self) -> str | None:
        """Category implied by the agent role, if any."""
        if self.agent_role is None:
            return None
        return AGENT_ROLE_TO_CATEGORY.get(self.agent_role)

    def effective_category(self) -> str | None:
        """Explicit category wins over role-derived category."""
        return self.task_category or self.derived_category


def extract_hints_from_headers(headers: Mapping[str, str]) -> RoutingHints:
    """Parse the four supported headers from a request. Returns RoutingHints(empty) if none.

    Raises RoutingHintError on any malformed value. The caller (chat route)
    maps RoutingHintError → HTTP 400.
    """
    # Headers in Starlette / FastAPI are case-insensitive but we accept any casing.
    h = {k.lower(): v for k, v in headers.items()}

    task_category = _parse_task_category(h.get(HEADER_TASK_CATEGORY))
    agent_role = _parse_agent_role(h.get(HEADER_AGENT_ROLE))
    quality_threshold = _parse_quality_threshold(h.get(HEADER_QUALITY_THRESHOLD))
    excluded_providers = _parse_excluded_providers(h.get(HEADER_EXCLUDE_PROVIDERS))

    return RoutingHints(
        task_category=task_category,
        agent_role=agent_role,
        quality_threshold=quality_threshold,
        excluded_providers=excluded_providers,
    )


def _parse_task_category(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value not in TASK_CATEGORIES:
        raise RoutingHintError(
            f"{HEADER_TASK_CATEGORY!r}: '{raw}' not in {sorted(TASK_CATEGORIES)}"
        )
    return value


def _parse_agent_role(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().lower().replace("-", "_")
    if not value:
        return None
    if value not in AGENT_ROLE_TO_CATEGORY:
        raise RoutingHintError(
            f"{HEADER_AGENT_ROLE!r}: '{raw}' unknown; see AGENT_ROLE_TO_CATEGORY"
        )
    return value


def _parse_quality_threshold(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as e:
        raise RoutingHintError(
            f"{HEADER_QUALITY_THRESHOLD!r}: '{raw}' is not a number"
        ) from e
    if not 0.0 <= value <= 1.0:
        raise RoutingHintError(
            f"{HEADER_QUALITY_THRESHOLD!r}: '{raw}' must be in [0, 1]"
        )
    return value


def _parse_excluded_providers(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return None
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if not parts:
        return None
    return frozenset(parts)

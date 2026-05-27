# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Scout abstract base class — the contract every classifier implementation honors.

Kept tight so the underlying backend (heuristics today; vLLM Semantic Router,
HuggingFace classifier, or a remote scoring service later) can be swapped
without changing callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modelmeld.api.schemas import ChatCompletionRequest


class Tier(str, Enum):
    """Routing tier. Mixes in `str` so values compare equal to plain strings."""

    LOCAL = "local"
    CLOUD = "cloud"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ScoutDecision:
    """Result of classifying a single request.

    `tier`        — recommended routing destination (already threshold-aware).
    `confidence`  — 0.0–1.0, the classifier's confidence in `tier`.
    `rationale`   — human-readable breadcrumb for logs / debugging.
    `signals`     — adapter-specific extra info (token estimate, matched rules, etc.).
    """

    tier: Tier
    confidence: float
    rationale: str
    signals: dict[str, Any] = field(default_factory=dict)


class Scout(ABC):
    """Classify an OpenAI-shaped chat request as best served local or cloud."""

    name: str

    @abstractmethod
    async def classify(self, request: ChatCompletionRequest) -> ScoutDecision:
        """Return a routing tier decision plus confidence + rationale."""

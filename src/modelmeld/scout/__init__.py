# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Scout — prompt-complexity classifier.

Public surface:
    Scout            — abstract base class
    ScoutDecision    — frozen result type (tier, confidence, rationale)
    Tier             — LOCAL / CLOUD enum
    HeuristicScout   — rule-based classifier (no ML deps, default)
    build_scout(s)   — factory keyed on GatewaySettings.scout_provider
"""

from __future__ import annotations

from modelmeld.scout.base import Scout, ScoutDecision, Tier
from modelmeld.scout.capability import (
    CapabilityDecision,
    CapabilityScout,
    NoEligibleModelError,
)
from modelmeld.scout.devtool import DevTool, Fingerprint, Fingerprinter
from modelmeld.scout.feed import (
    DEFAULT_CACHE_TTL_SEC,
    DEFAULT_TIMEOUT_SEC,
    SIGNATURE_HEADER,
    SUPPORTED_SCHEMA_VERSIONS,
    FeedFetchResult,
    RegistryFeedClient,
)
from modelmeld.scout.heuristics import HeuristicScout
from modelmeld.scout.registry import ModelEntry, ModelRegistry, default_registry
from modelmeld.scout.task_category import (
    TASK_CATEGORIES,
    TaskCategoryClassifier,
    TaskCategoryDecision,
)


def build_scout(settings: object) -> Scout:
    """Construct a Scout based on settings.

    Currently only `heuristic` is supported. vLLM-SR-backed and ML-classifier
    backends will plug in here once we have the dev/test GPU pipeline running.
    """
    # Local import to avoid circular references at module-load time.
    from modelmeld.config import GatewaySettings

    if not isinstance(settings, GatewaySettings):
        raise TypeError(f"build_scout expects GatewaySettings, got {type(settings).__name__}")

    if settings.scout_provider == "heuristic":
        return HeuristicScout(confidence_threshold=settings.scout_confidence_threshold)
    raise ValueError(f"Unknown scout_provider: {settings.scout_provider}")


__all__ = [
    "DEFAULT_CACHE_TTL_SEC",
    "DEFAULT_TIMEOUT_SEC",
    "SIGNATURE_HEADER",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TASK_CATEGORIES",
    "CapabilityDecision",
    "CapabilityScout",
    "DevTool",
    "FeedFetchResult",
    "Fingerprint",
    "Fingerprinter",
    "HeuristicScout",
    "ModelEntry",
    "ModelRegistry",
    "NoEligibleModelError",
    "RegistryFeedClient",
    "Scout",
    "ScoutDecision",
    "TaskCategoryClassifier",
    "TaskCategoryDecision",
    "Tier",
    "build_scout",
    "default_registry",
]

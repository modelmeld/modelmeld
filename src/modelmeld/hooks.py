# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Extension hooks. The seam where /enterprise-control plugs into request lifecycle.

Core-engine fires these events; it does not know which (if any) handlers are
registered. Enterprise-control registers handlers at startup and persists to
its own infrastructure (Postgres, Redis, FinOps dashboard, etc.).

Boundary contract: this module lives in /core-engine and MUST NOT import from
/enterprise-control. Events are plain dataclasses; handlers are external.

Hook failures are logged but never propagate — a broken audit logger must not
break customer requests.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedactionRecord:
    """Local copy of privacy.Redaction so hooks.py doesn't import privacy."""

    label: str
    count: int


@dataclass(frozen=True)
class RequestCompletedEvent:
    """Emitted after every chat-completion request resolves (success or failure)."""

    request_id: str
    timestamp: datetime

    # What the user asked for
    requested_model: str
    devtool: str  # "cursor", "claude_code", etc., or "unknown"
    devtool_confidence: float
    prompt_hash: str  # sha256 hex of the canonicalized scrubbed request

    # What we did
    routed_to: str  # adapter.name
    tier: str  # "local" or "cloud"
    failover_from: str | None  # tier name if we failed over

    # Token usage and timing
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float

    # Privacy / audit
    redactions: tuple[RedactionRecord, ...] = ()

    # Tenancy (None when running standalone without enterprise auth middleware)
    tenant_id: str | None = None
    user_id: str | None = None
    api_key_id: str | None = None

    # Error info (None on success)
    error: str | None = None
    error_type: str | None = None

    # Capability routing (FinOps consumes these for savings math).
    # `model_served` defaults to `""` for backwards compat; consumers should
    # fall back to `requested_model` when it's empty. `task_category` is the
    # classifier's output (or the framework-supplied hint) when capability
    # routing was active.
    model_served: str = ""
    task_category: str | None = None

    # Completion cache outcome. One of "hit", "hit-semantic",
    # "miss", "bypass", or None when no cache was configured for this request.
    # Drives per-dev-tool cache-hit analytics in FinOps rollups.
    cache_status: str | None = None


OnRequestComplete = Callable[[RequestCompletedEvent], Awaitable[None]]


class HookRegistry:
    """Registry of lifecycle hook handlers. Owned by app.state."""

    def __init__(self) -> None:
        self._on_request_complete: list[OnRequestComplete] = []

    def register_on_request_complete(self, handler: OnRequestComplete) -> None:
        self._on_request_complete.append(handler)

    @property
    def subscriber_count(self) -> int:
        return len(self._on_request_complete)

    async def fire_on_request_complete(self, event: RequestCompletedEvent) -> None:
        """Invoke all registered handlers; failures are logged, never raised."""
        for handler in self._on_request_complete:
            try:
                await handler(event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_request_complete hook failed",
                    extra={
                        "request_id": event.request_id,
                        "handler": getattr(handler, "__qualname__", repr(handler)),
                    },
                )

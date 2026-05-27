# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Retry-with-backoff utility for adapter calls (F-5).

Wraps an async adapter call with exponential backoff retry on transient
errors. Permanent errors (auth failure, config mismatch, schema errors)
raise immediately - retrying them just wastes time and exhausts the
provider's rate limit.

Used by `AnthropicAdapter` and `OpenAIAdapter` to absorb provider
throttling and 5xx blips before they reach the `TieredRouter`. With this
in place, the router's failover logic only triggers on outages that
genuinely persist across retries.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from modelmeld.adapters.base import (
    AdapterError,
    PermanentAdapterError,
    TransientAdapterError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that justify a retry. Anything else is treated as
# permanent - retrying a 401 won't make the credentials valid.
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({
    408,  # Request Timeout
    409,  # Conflict (some APIs use for "operation in progress")
    425,  # Too Early
    429,  # Too Many Requests
    500, 502, 503, 504,  # Server-side failures
    529,  # Anthropic-specific Overloaded
})

# Class-name fragments that indicate transience when status_code isn't
# available on the exception (some SDKs raise network errors that wrap
# the underlying httpx/aiohttp exception).
TRANSIENT_CLASS_HINTS: tuple[str, ...] = (
    "ratelimit",
    "overloaded",
    "timeout",
    "connection",
    "apiconnection",
    "internalservererror",
    "serviceunavailable",
)


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy. Defaults aim for ~7 seconds of total backoff over 3 tries.

    max_attempts=3, base_delay=1s, jitter=20% → waits ~1s, ~2s between
    attempts, with up to 20% randomization to avoid thundering herd.
    """

    max_attempts: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 30.0
    jitter: float = 0.2          # ±20% randomization of the computed delay


def is_transient_error(exc: BaseException) -> bool:
    """Classify whether an exception should trigger retry.

    Inspects, in order:
      1. Network-level exception types (asyncio.TimeoutError, ConnectionError)
      2. HTTP status code on the exception (.status_code attribute)
      3. Class-name fragments (fallback for SDKs without status_code)
    """
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS_CODES

    cls_name = type(exc).__name__.lower()
    return any(hint in cls_name for hint in TRANSIENT_CLASS_HINTS)


def _compute_backoff(
    attempt: int, config: RetryConfig, rng: random.Random | None = None,
) -> float:
    """Exponential backoff with optional ±jitter."""
    raw = config.base_delay_sec * (2 ** (attempt - 1))
    capped = min(raw, config.max_delay_sec)
    if config.jitter > 0:
        r = (rng or random).uniform(-config.jitter, config.jitter)
        capped = capped * (1.0 + r)
    return max(capped, 0.0)


async def retry_async(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    *,
    label: str = "adapter call",
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
) -> T:
    """Run an async callable with exponential-backoff retry.

    Args:
        func: zero-arg async callable to invoke. Wrap your call in a lambda
              or nested coroutine def.
        config: retry policy. Defaults to `RetryConfig()`.
        label: human-readable label for log lines (e.g. "anthropic.chat").
        sleep: injectable async sleep. Defaults to `asyncio.sleep`. Tests
               override this to avoid real wall-clock waits.
        rng: injectable RNG for jitter. Tests pass a seeded `random.Random`
             for deterministic backoff timing.

    Returns:
        Whatever `func()` returns on success.

    Raises:
        The last exception, unmodified, after all attempts exhausted.
        Non-transient errors raise immediately (no retry).
    """
    cfg = config or RetryConfig()
    _sleep = sleep or asyncio.sleep
    last_exc: BaseException | None = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            return await func()
        except BaseException as e:
            last_exc = e
            if not is_transient_error(e):
                # Permanent error - bail immediately, no retry.
                raise
            if attempt >= cfg.max_attempts:
                # Out of retries - re-raise the last error.
                raise
            delay = _compute_backoff(attempt, cfg, rng)
            logger.info(
                "[%s] attempt %d/%d failed (%s: %s); retrying in %.2fs",
                label, attempt, cfg.max_attempts,
                type(e).__name__, str(e)[:120], delay,
            )
            await _sleep(delay)

    # Unreachable in practice, but satisfies type checkers.
    assert last_exc is not None
    raise last_exc


def wrap_as_adapter_error(exc: BaseException, prefix: str) -> AdapterError:
    """Wrap an upstream exception in the appropriate AdapterError subclass.

    `TieredRouter` (F-2) branches on the subclass:
      - `TransientAdapterError` → safe to fail over to the other tier
      - `PermanentAdapterError` → bubble up so the caller sees the real error

    Detection mirrors `is_transient_error`. The string carries enough
    detail to debug from logs without exposing the underlying exception
    type leak.
    """
    msg = f"{prefix}: {exc}"
    if is_transient_error(exc):
        return TransientAdapterError(msg)
    return PermanentAdapterError(msg)


# Backward-compat alias for the underscore-prefixed call site in
# anthropic_adapter / openai_adapter. Both spellings are part of the
# adapter-internal contract; tests should prefer `wrap_as_adapter_error`.
_wrap_as_adapter_error = wrap_as_adapter_error

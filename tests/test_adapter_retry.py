"""Retry-with-backoff + transient/permanent classification (F-5, F-2)."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import pytest

from modelmeld.adapters.base import (
    AdapterError,
    PermanentAdapterError,
    TransientAdapterError,
)
from modelmeld.adapters.retry import (
    RetryConfig,
    TRANSIENT_STATUS_CODES,
    _compute_backoff,
    is_transient_error,
    retry_async,
    wrap_as_adapter_error,
)


# ---------------------------------------------------------------------------
# Synthetic exception types that mimic provider SDK shapes
# ---------------------------------------------------------------------------


class _FakeAPIError(Exception):
    """Mimics anthropic.APIStatusError / openai.APIError - has .status_code."""

    def __init__(self, status_code: int, msg: str = "") -> None:
        super().__init__(msg or f"HTTP {status_code}")
        self.status_code = status_code


class _FakeRateLimitError(Exception):
    """SDK error with the word 'RateLimit' in its class name but no status_code."""


class _FakeOverloadedError(Exception):
    """Mimics anthropic.OverloadedError - class-name-based classification."""


class _FakeAuthError(Exception):
    """A permanent failure that should NOT be retried."""

    def __init__(self) -> None:
        super().__init__("bad credentials")
        self.status_code = 401


# ---------------------------------------------------------------------------
# is_transient_error - classification table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(TRANSIENT_STATUS_CODES))
def test_transient_status_codes_classify_as_transient(code: int) -> None:
    assert is_transient_error(_FakeAPIError(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 405, 410, 422])
def test_4xx_other_than_429_classify_as_permanent(code: int) -> None:
    assert is_transient_error(_FakeAPIError(code)) is False


def test_429_is_transient() -> None:
    """Explicit because 429 is the most common transient one."""
    assert is_transient_error(_FakeAPIError(429)) is True


def test_529_is_transient_anthropic_overloaded() -> None:
    assert is_transient_error(_FakeAPIError(529)) is True


def test_network_errors_are_transient() -> None:
    assert is_transient_error(asyncio.TimeoutError()) is True
    assert is_transient_error(ConnectionError("refused")) is True


def test_class_name_fallback_for_sdks_without_status_code() -> None:
    """Some SDKs raise differently-shaped errors; fall back to class-name match."""
    assert is_transient_error(_FakeRateLimitError("hit limit")) is True
    assert is_transient_error(_FakeOverloadedError("server busy")) is True


def test_unrelated_exception_is_permanent() -> None:
    assert is_transient_error(ValueError("bad json")) is False


def test_auth_error_is_permanent() -> None:
    assert is_transient_error(_FakeAuthError()) is False


# ---------------------------------------------------------------------------
# wrap_as_adapter_error - dispatch to the right subclass
# ---------------------------------------------------------------------------


def test_wrap_transient_yields_transient_subclass() -> None:
    err = wrap_as_adapter_error(_FakeAPIError(529), "Anthropic chat call failed")
    assert isinstance(err, TransientAdapterError)
    assert isinstance(err, AdapterError)
    assert "Anthropic chat call failed" in str(err)
    assert "HTTP 529" in str(err)


def test_wrap_permanent_yields_permanent_subclass() -> None:
    err = wrap_as_adapter_error(_FakeAuthError(), "OpenAI chat call failed")
    assert isinstance(err, PermanentAdapterError)
    assert isinstance(err, AdapterError)
    assert "OpenAI chat call failed" in str(err)


def test_wrap_network_error_is_transient() -> None:
    err = wrap_as_adapter_error(ConnectionError("refused"), "OpenAI chat failed")
    assert isinstance(err, TransientAdapterError)


# ---------------------------------------------------------------------------
# _compute_backoff - exponential timing
# ---------------------------------------------------------------------------


def test_backoff_no_jitter() -> None:
    cfg = RetryConfig(base_delay_sec=1.0, jitter=0.0, max_delay_sec=100.0)
    # attempt 1 -> 1s, attempt 2 -> 2s, attempt 3 -> 4s, attempt 4 -> 8s
    assert _compute_backoff(1, cfg) == 1.0
    assert _compute_backoff(2, cfg) == 2.0
    assert _compute_backoff(3, cfg) == 4.0
    assert _compute_backoff(4, cfg) == 8.0


def test_backoff_capped_at_max() -> None:
    cfg = RetryConfig(base_delay_sec=1.0, jitter=0.0, max_delay_sec=5.0)
    assert _compute_backoff(10, cfg) == 5.0


def test_backoff_jitter_within_range() -> None:
    """Jitter randomization stays within +/-jitter%."""
    cfg = RetryConfig(base_delay_sec=1.0, jitter=0.5, max_delay_sec=100.0)
    rng = random.Random(42)
    delays = [_compute_backoff(1, cfg, rng) for _ in range(100)]
    # base=1s, jitter=50% -> range [0.5, 1.5]
    assert all(0.5 <= d <= 1.5 for d in delays)


# ---------------------------------------------------------------------------
# retry_async - the main loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_immediately_on_first_attempt() -> None:
    calls = 0
    async def _ok():
        nonlocal calls
        calls += 1
        return "ok"
    result = await retry_async(_ok, RetryConfig(max_attempts=3))
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt_after_two_transient_failures() -> None:
    calls = 0
    async def _flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _FakeAPIError(529)
        return "ok"
    # Inject a no-op sleep so the test doesn't wall-clock wait.
    sleeps: list[float] = []
    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)
    result = await retry_async(
        _flaky,
        RetryConfig(max_attempts=3, base_delay_sec=0.5, jitter=0.0),
        sleep=_fake_sleep,
    )
    assert result == "ok"
    assert calls == 3
    # Two retries -> two sleeps with exponential backoff
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts_and_reraises_last() -> None:
    calls = 0
    async def _always_fails():
        nonlocal calls
        calls += 1
        raise _FakeAPIError(503, f"attempt {calls}")
    async def _fake_sleep(_d: float) -> None: pass
    with pytest.raises(_FakeAPIError) as ei:
        await retry_async(
            _always_fails,
            RetryConfig(max_attempts=3, base_delay_sec=0.01, jitter=0.0),
            sleep=_fake_sleep,
        )
    assert calls == 3
    assert "attempt 3" in str(ei.value)
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_permanent_error_does_not_retry() -> None:
    """A permanent error (e.g. 401) must raise immediately, no retries."""
    calls = 0
    async def _auth_fail():
        nonlocal calls
        calls += 1
        raise _FakeAuthError()
    with pytest.raises(_FakeAuthError):
        await retry_async(
            _auth_fail,
            RetryConfig(max_attempts=5, base_delay_sec=10.0),
        )
    assert calls == 1, "permanent error should fail on first attempt, not retry"


@pytest.mark.asyncio
async def test_retry_preserves_original_exception_type() -> None:
    """After all retries fail, the LAST exception is raised unmodified."""
    async def _fail():
        raise _FakeAPIError(529, "still overloaded")
    async def _fake_sleep(_d: float) -> None: pass
    with pytest.raises(_FakeAPIError) as ei:
        await retry_async(
            _fail,
            RetryConfig(max_attempts=2, base_delay_sec=0.01, jitter=0.0),
            sleep=_fake_sleep,
        )
    # NOT wrapped in some new exception type
    assert type(ei.value) is _FakeAPIError
    assert ei.value.status_code == 529


@pytest.mark.asyncio
async def test_retry_network_error_is_treated_as_transient() -> None:
    calls = 0
    async def _net_blip():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ConnectionError("connection reset")
        return "ok"
    async def _fake_sleep(_d: float) -> None: pass
    result = await retry_async(
        _net_blip,
        RetryConfig(max_attempts=3, base_delay_sec=0.01, jitter=0.0),
        sleep=_fake_sleep,
    )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_zero_jitter_produces_deterministic_backoff() -> None:
    """Same RetryConfig with jitter=0 -> identical sleep sequence across runs."""
    async def _fail():
        raise _FakeAPIError(503)
    async def _fake_sleep(_d: float) -> None: pass

    sleeps1: list[float] = []
    async def _capture1(d: float) -> None: sleeps1.append(d)
    sleeps2: list[float] = []
    async def _capture2(d: float) -> None: sleeps2.append(d)

    cfg = RetryConfig(max_attempts=4, base_delay_sec=0.5, jitter=0.0, max_delay_sec=100.0)
    for sleep_fn in (_capture1, _capture2):
        with pytest.raises(_FakeAPIError):
            await retry_async(_fail, cfg, sleep=sleep_fn)
    assert sleeps1 == sleeps2 == [0.5, 1.0, 2.0]

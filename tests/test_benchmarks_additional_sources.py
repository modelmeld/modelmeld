"""Tests for AiderPolyglotFetcher, LiveBenchFetcher, LMArenaFetcher."""

from __future__ import annotations

import httpx
import pytest

from modelmeld.scout.benchmarks import (
    AiderPolyglotFetcher,
    LiveBenchFetcher,
    LMArenaFetcher,
)
from modelmeld.scout.benchmarks.aider_polyglot import AiderPolyglotError
from modelmeld.scout.benchmarks.livebench import LiveBenchError
from modelmeld.scout.benchmarks.lmarena import _elo_to_unit
from tests.fixtures.aider_polyglot_fixture import AIDER_YAML_BASELINE, AIDER_YAML_MALFORMED
from tests.fixtures.livebench_fixture import LIVEBENCH_BARE_LIST, LIVEBENCH_BASELINE
from tests.fixtures.lmarena_fixture import LMARENA_CODE, LMARENA_TEXT, LMARENA_TEXT_MALFORMED


def _mock_transport(handler):
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# AiderPolyglotFetcher
# ---------------------------------------------------------------------------

async def test_aider_fetches_and_normalizes_pass_rate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=AIDER_YAML_BASELINE)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = AiderPolyglotFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()

    assert len(entries) == 3
    opus = next(e for e in entries if e.model_id == "claude-opus-4-7")
    # pass_rate_2 of 81.0 → 0.81
    assert abs(opus.task_scores["coding"] - 0.81) < 1e-6
    assert opus.provider == "anthropic"
    assert opus.source == "aider_polyglot"
    # Aider doesn't supply other task categories
    assert set(opus.task_scores.keys()) == {"coding"}


async def test_aider_skips_malformed_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=AIDER_YAML_MALFORMED)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = AiderPolyglotFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()

    # Only the "no-slash" one survives (bare-model-name, pass_rate_2=50)
    # - no-model: dropped (no model field)
    # - no-pass-rate: dropped (no pass_rate)
    # - bad-pass-rate: dropped (non-numeric)
    # - out-of-range: dropped (>100 means >1.0 after normalization)
    # - no-slash: bare-model-name with pass_rate 50 → survives
    assert len(entries) == 1
    assert entries[0].model_id == "bare-model-name"
    assert entries[0].provider == "unknown"
    assert abs(entries[0].task_scores["coding"] - 0.50) < 1e-6


async def test_aider_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = AiderPolyglotFetcher(http_client=client)
    with pytest.raises(AiderPolyglotError, match="404"):
        await fetcher.fetch()
    await client.aclose()


async def test_aider_raises_on_bad_yaml() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Tabs after colons + unclosed brackets — guaranteed YAML parse failure
        return httpx.Response(200, text=":\n\t[unclosed")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = AiderPolyglotFetcher(http_client=client)
    with pytest.raises(AiderPolyglotError):
        await fetcher.fetch()
    await client.aclose()


# ---------------------------------------------------------------------------
# LiveBenchFetcher
# ---------------------------------------------------------------------------

async def test_livebench_fetches_and_groups_categories_into_tasks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LIVEBENCH_BASELINE)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LiveBenchFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()

    assert len(entries) == 3
    opus = next(e for e in entries if e.model_id == "claude-opus-4-7")
    # coding = 73.5 / 100 → 0.735
    assert abs(opus.task_scores["coding"] - 0.735) < 1e-6
    # reasoning = avg(math 78.1, reasoning 82.0, data_analysis 70.4) → 76.83 / 100
    expected_reasoning = (78.1 + 82.0 + 70.4) / 3 / 100
    assert abs(opus.task_scores["reasoning"] - expected_reasoning) < 1e-6
    # simple_qa = avg(IF 91.0, language 84.3) — note IF key is "instruction_following"
    expected_simple_qa = (91.0 + 84.3) / 2 / 100
    assert abs(opus.task_scores["simple_qa"] - expected_simple_qa) < 1e-6
    assert opus.provider == "anthropic"
    assert opus.context_window == 200000
    assert opus.source == "livebench"


async def test_livebench_accepts_bare_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LIVEBENCH_BARE_LIST)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LiveBenchFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()
    assert len(entries) == 3


async def test_livebench_skips_record_with_no_usable_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"model": "x", "unknown_field": 100}]})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LiveBenchFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()
    assert entries == []


async def test_livebench_raises_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream busy")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LiveBenchFetcher(http_client=client)
    with pytest.raises(LiveBenchError, match="503"):
        await fetcher.fetch()
    await client.aclose()


# ---------------------------------------------------------------------------
# LMArenaFetcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("elo", "expected"),
    [
        (800, 0.20),     # floor
        (1300, 0.90),    # ceiling
        (1050, 0.55),    # midpoint
        (500, 0.20),     # below floor → clamped
        (1500, 0.90),    # above ceiling → clamped
    ],
)
def test_elo_to_unit_calibration(elo: float, expected: float) -> None:
    assert abs(_elo_to_unit(elo) - expected) < 1e-4


async def test_lmarena_merges_text_and_code_boards() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("text_arena.json"):
            return httpx.Response(200, json=LMARENA_TEXT)
        if request.url.path.endswith("code_arena.json"):
            return httpx.Response(200, json=LMARENA_CODE)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LMArenaFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()

    # claude-opus-4-7 appears in both boards → has both simple_qa and coding
    opus = next(e for e in entries if e.model_id == "claude-opus-4-7")
    assert "simple_qa" in opus.task_scores
    assert "coding" in opus.task_scores
    # rating 1285 (text) → simple_qa; rating 1310 (code) → coding clamped to 0.90
    assert opus.task_scores["coding"] == pytest.approx(0.90, abs=1e-4)
    assert opus.source == "lmarena"
    assert opus.provider == "anthropic"


async def test_lmarena_tolerates_one_board_failing() -> None:
    """If text_arena 200's but code_arena 500's, we keep what we got."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("text_arena.json"):
            return httpx.Response(200, json=LMARENA_TEXT)
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LMArenaFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()

    # 3 models from text_arena; none of them have coding scores (code_arena failed)
    assert {e.model_id for e in entries} == {"claude-opus-4-7", "gpt-5-4", "gpt-5-mini"}
    for e in entries:
        assert "simple_qa" in e.task_scores
        assert "coding" not in e.task_scores


async def test_lmarena_skips_malformed_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("text_arena.json"):
            return httpx.Response(200, json=LMARENA_TEXT_MALFORMED)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = LMArenaFetcher(http_client=client)
    try:
        entries = await fetcher.fetch()
    finally:
        await client.aclose()
    # All 3 fixture entries are malformed
    assert entries == []


# ---------------------------------------------------------------------------
# All sources implement the BenchmarkSource contract
# ---------------------------------------------------------------------------

def test_all_fetchers_have_unique_names() -> None:
    names = {
        AiderPolyglotFetcher.name,
        LiveBenchFetcher.name,
        LMArenaFetcher.name,
    }
    assert len(names) == 3
    assert names == {"aider_polyglot", "livebench", "lmarena"}

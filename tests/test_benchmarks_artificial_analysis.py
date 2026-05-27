"""ArtificialAnalysisFetcher + normalizer tests."""

from __future__ import annotations

import httpx
import pytest

from modelmeld.scout.benchmarks import (
    TASK_BENCHMARK_MAP,
    ArtificialAnalysisFetcher,
    canonicalize_model_id,
    normalize_aa_model,
)
from modelmeld.scout.benchmarks.artificial_analysis import ArtificialAnalysisError
from tests.fixtures.aa_response import (
    AA_RESPONSE_BARE_LIST,
    AA_RESPONSE_BASELINE,
    AA_RESPONSE_PARTIAL,
)


# ---------------------------------------------------------------------------
# canonicalize_model_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("Claude-Opus-4.7", "claude-opus-4-7"),
        ("GPT-5 Mini", "gpt-5-mini"),
        ("Claude_Sonnet_4_6", "claude-sonnet-4-6"),
        ("  Spaced  Id  ", "spaced--id"),
    ],
)
def test_canonicalize_model_id(raw: str, expected: str) -> None:
    assert canonicalize_model_id(raw) == expected


# ---------------------------------------------------------------------------
# normalize_aa_model
# ---------------------------------------------------------------------------

def test_normalize_extracts_basic_fields() -> None:
    aa = AA_RESPONSE_BASELINE["data"][0]   # claude-opus-4-7
    entry = normalize_aa_model(aa)
    assert entry.model_id == "claude-opus-4-7"
    assert entry.provider == "anthropic"
    assert entry.context_window == 200000
    assert entry.cost_per_m_input == 5.0
    assert entry.cost_per_m_output == 25.0
    assert entry.source == "artificial_analysis"
    assert entry.last_updated == "2026-05-15T00:00:00Z"


def test_normalize_computes_task_scores_via_averages() -> None:
    """Each task category averages the AA benchmarks listed in TASK_BENCHMARK_MAP, normalized to [0,1]."""
    aa = AA_RESPONSE_BASELINE["data"][0]   # opus, has all 11 benchmarks
    entry = normalize_aa_model(aa)
    # coding = avg(Terminal-Bench Hard, SciCode, LiveCodeBench) / 100
    expected_coding = (58.3 + 67.5 + 72.0) / 3 / 100
    assert abs(entry.task_scores["coding"] - expected_coding) < 1e-6
    # reasoning = avg(GPQA Diamond, AA-LCR, HLE, CritPt) / 100
    expected_reasoning = (84.5 + 71.2 + 35.1 + 62.0) / 4 / 100
    assert abs(entry.task_scores["reasoning"] - expected_reasoning) < 1e-6


def test_normalize_missing_benchmarks_skipped_in_average() -> None:
    """If some AA benchmarks for a task aren't reported, average the available subset."""
    aa = AA_RESPONSE_BASELINE["data"][2]   # qwen3-coder-next — only has some benchmarks
    entry = normalize_aa_model(aa)
    # qwen3-coder-next has Terminal-Bench Hard (55), SciCode (60), LiveCodeBench (68)
    expected_coding = (55.0 + 60.0 + 68.0) / 3 / 100
    assert abs(entry.task_scores["coding"] - expected_coding) < 1e-6


def test_normalize_no_evaluations_yields_empty_task_scores() -> None:
    aa = AA_RESPONSE_PARTIAL["data"][0]   # no-benchmarks
    entry = normalize_aa_model(aa)
    assert entry.task_scores == {}


def test_normalize_accepts_alternative_field_names() -> None:
    aa = AA_RESPONSE_PARTIAL["data"][1]   # uses `name` + `creator`
    entry = normalize_aa_model(aa)
    assert entry.model_id == "alt-field-names"
    assert entry.provider == "some lab"


def test_normalize_rejects_record_without_id() -> None:
    aa = AA_RESPONSE_PARTIAL["data"][2]   # no model_id / name / slug
    with pytest.raises(ValueError, match="model_id"):
        normalize_aa_model(aa)


def test_task_scores_in_unit_range() -> None:
    """Sanity: AA scores are 0-100; our normalized output must be 0-1."""
    for aa in AA_RESPONSE_BASELINE["data"]:
        entry = normalize_aa_model(aa)
        for task, score in entry.task_scores.items():
            assert 0.0 <= score <= 1.0, f"{entry.model_id}.{task} = {score}"


def test_task_benchmark_map_covers_required_tasks() -> None:
    required = {"coding", "reasoning", "simple_qa", "summarization", "tool_use"}
    assert required.issubset(TASK_BENCHMARK_MAP.keys())


# ---------------------------------------------------------------------------
# ArtificialAnalysisFetcher — mocked HTTP
# ---------------------------------------------------------------------------

def _mock_transport(handler):
    return httpx.MockTransport(handler)


async def test_fetcher_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
    fetcher = ArtificialAnalysisFetcher()
    with pytest.raises(ArtificialAnalysisError, match="API key"):
        await fetcher.fetch_models()


async def test_fetcher_returns_models_from_data_wrapper() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "test-key"
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json=AA_RESPONSE_BASELINE)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = ArtificialAnalysisFetcher(api_key="test-key", http_client=client)
    try:
        models = await fetcher.fetch_models()
    finally:
        await client.aclose()
    assert len(models) == 3
    assert models[0]["model_id"] == "claude-opus-4-7"


async def test_fetcher_accepts_bare_list_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=AA_RESPONSE_BARE_LIST)

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = ArtificialAnalysisFetcher(api_key="x", http_client=client)
    try:
        models = await fetcher.fetch_models()
    finally:
        await client.aclose()
    assert len(models) == 3


async def test_fetcher_accepts_models_wrapper() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": AA_RESPONSE_BASELINE["data"]})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = ArtificialAnalysisFetcher(api_key="x", http_client=client)
    try:
        models = await fetcher.fetch_models()
    finally:
        await client.aclose()
    assert len(models) == 3


async def test_fetcher_raises_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = ArtificialAnalysisFetcher(api_key="bad-key", http_client=client)
    with pytest.raises(ArtificialAnalysisError, match="401"):
        await fetcher.fetch_models()
    await client.aclose()


async def test_fetcher_raises_on_network_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    fetcher = ArtificialAnalysisFetcher(api_key="x", http_client=client)
    with pytest.raises(ArtificialAnalysisError, match="AA fetch failed"):
        await fetcher.fetch_models()
    await client.aclose()


async def test_fetcher_close_releases_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "x")
    fetcher = ArtificialAnalysisFetcher()
    # touch _client() to instantiate the owned client
    await fetcher._client()
    assert fetcher._http is not None
    await fetcher.close()
    assert fetcher._http is None

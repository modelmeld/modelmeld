"""Structural escalate-detector (scout/difficulty.py)."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.difficulty import (
    DifficultyClassifier,
    DifficultyRoute,
    difficulty_routing_enabled,
)

_clf = DifficultyClassifier()


def _req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": text}])


def _route(text: str, category: str = "coding") -> DifficultyRoute:
    return _clf.classify(_req(text), category).route


# --- category gate: the broad middle never escalates ----------------------- #

def test_non_escalatable_category_never_escalates() -> None:
    # blatantly multi-file, but summarization is not an escalatable category
    d = _clf.classify(_req("summarize the changes across a.py b.py c.py d.py"), "summarization")
    assert d.route is DifficultyRoute.ROUTE_OSS
    assert "not_escalatable" in d.rationale


def test_simple_qa_routes_oss() -> None:
    assert _route("what does this function return?", "simple_qa") is DifficultyRoute.ROUTE_OSS


# --- file/module scope (strongest signal) ---------------------------------- #

def test_multi_file_coding_escalates() -> None:
    d = _clf.classify(
        _req("Fix the auth bug across auth.py, routes.py and handlers.py"), "coding")
    assert d.route is DifficultyRoute.ESCALATE
    assert d.signals["distinct_files"] >= 2
    assert "multi_file" in d.rationale


def test_across_files_phrase_escalates() -> None:
    assert _route("Refactor the error handling across multiple modules") is DifficultyRoute.ESCALATE


def test_single_file_fix_routes_oss() -> None:
    assert _route("Fix the off-by-one in parser.py") is DifficultyRoute.ROUTE_OSS


# --- greenfield + compositional -------------------------------------------- #

def test_greenfield_plus_compositional_escalates() -> None:
    text = (
        "Build a new rate-limiting service. It must support sliding windows, "
        "and also per-tenant quotas, and ensure that it persists across restarts."
    )
    assert _route(text) is DifficultyRoute.ESCALATE


def test_greenfield_alone_routes_oss() -> None:
    # greenfield verb but no interacting constraints → not strong enough
    assert _route("Build a small CLI that echoes its arguments") is DifficultyRoute.ROUTE_OSS


# --- novel-algorithm vs specified ------------------------------------------ #

def test_must_discover_algorithm_escalates() -> None:
    assert _route(
        "Figure out the best algorithm to deduplicate this stream in O(1) memory",
    ) is DifficultyRoute.ESCALATE


def test_specified_algorithm_single_file_routes_oss() -> None:
    # self-contained, algorithm given (LiveCodeBench-shape) → strong OSS holds
    assert _route("Implement binary search over the sorted list in search.py") is DifficultyRoute.ROUTE_OSS


# --- escalate property + flag ---------------------------------------------- #

def test_escalate_property() -> None:
    assert _clf.classify(_req("fix the bug across a.py and b.py"), "coding").escalate is True
    assert _clf.classify(_req("fix the typo in a.py"), "coding").escalate is False


def test_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_DIFFICULTY_ROUTING", raising=False)
    assert difficulty_routing_enabled() is False


@pytest.mark.parametrize(
    ("val", "expected"),
    [("1", True), ("true", True), ("on", True), ("YES", True), ("0", False), ("", False)],
)
def test_flag_reads_env(monkeypatch: pytest.MonkeyPatch, val: str, expected: bool) -> None:
    monkeypatch.setenv("MODELMELD_DIFFICULTY_ROUTING", val)
    assert difficulty_routing_enabled() is expected

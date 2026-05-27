"""Fingerprinter tests — detect Cursor / Claude Code / Aider / Cline / etc."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.scout.devtool import DevTool, Fingerprint, Fingerprinter


def _req(system: str = "", user: str = "hi") -> ChatCompletionRequest:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return ChatCompletionRequest(model="m", messages=messages)


@pytest.fixture
def fingerprinter() -> Fingerprinter:
    return Fingerprinter()


def test_unknown_when_no_signals(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(_req(system="You are a helpful assistant."))
    assert fp.tool == DevTool.UNKNOWN
    assert fp.confidence == 0.0


def test_detects_cursor(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(
        _req(
            system="You are a powerful agentic AI coding assistant integrated with Cursor.",
            user="complete this",
        )
    )
    assert fp.tool == DevTool.CURSOR
    assert fp.confidence >= 0.6


def test_detects_claude_code(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(
        _req(
            system="You are Claude Code, Anthropic's official CLI for Claude.",
            user="<system-reminder>Be concise.</system-reminder> Refactor utils.py.",
        )
    )
    assert fp.tool == DevTool.CLAUDE_CODE
    assert fp.confidence >= 0.6


def test_detects_aider_from_search_replace(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(
        _req(
            system="You output edits as SEARCH/REPLACE blocks.",
            user="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
        )
    )
    assert fp.tool == DevTool.AIDER


def test_detects_cline(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(
        _req(
            system="You are Cline, an AI coding agent.",
            user="<read_file>path/to/file.py</read_file>",
        )
    )
    assert fp.tool == DevTool.CLINE


def test_detects_github_copilot(fingerprinter: Fingerprinter) -> None:
    fp = fingerprinter.identify(
        _req(system="You are GitHub Copilot.", user="autocomplete: def fact(")
    )
    assert fp.tool == DevTool.GITHUB_COPILOT


def test_more_hits_means_higher_confidence(fingerprinter: Fingerprinter) -> None:
    one_hit = fingerprinter.identify(
        _req(system="You are Claude Code.", user="hi")
    )
    multi_hit = fingerprinter.identify(
        _req(
            system="You are Claude Code, Anthropic's official CLI for Claude. <system-reminder>...",
            user="hi",
        )
    )
    assert one_hit.tool == DevTool.CLAUDE_CODE
    assert multi_hit.tool == DevTool.CLAUDE_CODE
    assert multi_hit.confidence > one_hit.confidence


def test_no_cross_contamination(fingerprinter: Fingerprinter) -> None:
    """If only Cursor signals appear, we shouldn't accidentally tag as Aider/Cline."""
    fp = fingerprinter.identify(
        _req(system="You are an assistant integrated with Cursor.", user="hi")
    )
    assert fp.tool == DevTool.CURSOR
    assert all("aider" not in s.lower() for s in fp.matched_signals)
    assert all("cline" not in s.lower() for s in fp.matched_signals)


def test_fingerprint_is_frozen(fingerprinter: Fingerprinter) -> None:
    fp = Fingerprint(tool=DevTool.CURSOR, confidence=0.8)
    with pytest.raises(Exception):
        fp.confidence = 0.9  # type: ignore[misc]

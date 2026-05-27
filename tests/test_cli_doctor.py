# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for `modelmeld doctor` — the post-install diagnostic CLI.

Every failure mode that today's 3-hour manual validation surfaced has a
regression test here. Specifically the silent CR-in-env-var that ate
30 minutes of debugging is the headline test (#test_env_var_with_cr).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelmeld.cli.doctor import (
    _check_cache_file,
    _check_env_vars,
)


# ---------------------------------------------------------------------------
# Env-var checks
# ---------------------------------------------------------------------------


def test_env_clean_passes(monkeypatch) -> None:
    """The happy path — all three env vars present, clean, no AUTH_TOKEN."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.modelmeld.ai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_TestKey12345")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    results = _check_env_vars()
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_env_var_missing_is_flagged(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    results = _check_env_vars()
    base_url_check = next(r for r in results if "ANTHROPIC_BASE_URL" in r.label)
    assert not base_url_check.ok
    assert "not set" in base_url_check.detail


def test_env_var_with_cr_is_flagged(monkeypatch) -> None:
    """The headline test — exact bug from today's 30-min debug session.
    Trailing CR in ANTHROPIC_BASE_URL silently broke Claude Code's
    discovery cache reader's baseUrl != env_var comparison."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com\r")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    results = _check_env_vars()
    base_url_check = next(r for r in results if "ANTHROPIC_BASE_URL" in r.label)
    assert not base_url_check.ok
    assert "CR/LF" in base_url_check.detail
    assert base_url_check.fix and "re-export" in base_url_check.fix


def test_env_var_with_lf_is_flagged(monkeypatch) -> None:
    """Same bug, LF flavor — terminal-wrap during paste can insert LF."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_first\nrest\nof_key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.com")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    results = _check_env_vars()
    api_key_check = next(r for r in results if "ANTHROPIC_API_KEY" in r.label)
    assert not api_key_check.ok
    assert "CR/LF" in api_key_check.detail


def test_env_auth_token_conflict_is_flagged(monkeypatch) -> None:
    """ANTHROPIC_AUTH_TOKEN set alongside ANTHROPIC_API_KEY triggers
    Claude Code's 'auth conflict' warning + silently disables discovery."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "some-oauth-token")
    results = _check_env_vars()
    auth_check = next(r for r in results if "ANTHROPIC_AUTH_TOKEN" in r.label)
    assert not auth_check.ok
    assert "conflict" in auth_check.detail.lower() or "discovery" in auth_check.detail


def test_byok_header_missing_colon_prefix_is_flagged(monkeypatch) -> None:
    """The other 30-min-debugger from today — customer exported just the
    key value into ANTHROPIC_CUSTOM_HEADERS, omitting the `name: ` prefix."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    # Just the key — no `name:` prefix
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "sk-ant-RAWKEYNOPREFIX")
    results = _check_env_vars()
    byok_check = next(r for r in results if "ANTHROPIC_CUSTOM_HEADERS" in r.label)
    assert not byok_check.ok
    assert "prefix" in byok_check.detail.lower() or "colon" in byok_check.detail.lower()


def test_byok_header_with_wrap_pollution_is_flagged(monkeypatch) -> None:
    """Terminal-wrap embedded \\n + spaces in the middle of the key value."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "x-modelmeld-byok-anthropic: sk-ant-part1\n   part2",
    )
    results = _check_env_vars()
    byok_check = next(r for r in results if "ANTHROPIC_CUSTOM_HEADERS" in r.label)
    assert not byok_check.ok


def test_byok_clean_passes(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "gws_x")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "x-modelmeld-byok-anthropic: sk-ant-test1234567890",
    )
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    results = _check_env_vars()
    assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# Cache-file checks
# ---------------------------------------------------------------------------


def _write_valid_cache(path: Path, base_url: str) -> None:
    cache = {
        "baseUrl": base_url,
        "fetchedAt": 1748192400000,
        "models": [
            {"id": "anthropic/modelmeld-saver", "display_name": "Saver"},
            {"id": "anthropic/modelmeld-auto", "display_name": "Auto"},
            {"id": "anthropic/modelmeld-quality", "display_name": "Quality"},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write LF only
    path.write_bytes(json.dumps(cache, indent=2).encode("utf-8"))


def test_cache_clean_passes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # for Path.home() on Windows
    cache_path = tmp_path / ".claude" / "cache" / "gateway-models.json"
    _write_valid_cache(cache_path, "https://gateway.example.com")
    results = _check_cache_file("https://gateway.example.com")
    assert all(r.ok for r in results)


def test_cache_missing_is_flagged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    results = _check_cache_file("https://gateway.example.com")
    assert any(not r.ok and "missing" in r.detail for r in results)


def test_cache_with_crlf_is_flagged(monkeypatch, tmp_path: Path) -> None:
    """File with CRLF line endings — Claude Code v2.1.150 may reject."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cache_path = tmp_path / ".claude" / "cache" / "gateway-models.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {"baseUrl": "https://x.com", "fetchedAt": 0, "models": []}
    cache_path.write_bytes(json.dumps(cache, indent=2).encode("utf-8").replace(b"\n", b"\r\n"))
    results = _check_cache_file("https://x.com")
    assert any(not r.ok and "CR/LF" in r.detail for r in results)


def test_cache_baseurl_mismatch_is_flagged(monkeypatch, tmp_path: Path) -> None:
    """The subtle bug: cache says one URL, env var says another (often
    differing only by a trailing CR). Claude Code's reader does exact
    string match — they MUST be byte-identical."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cache_path = tmp_path / ".claude" / "cache" / "gateway-models.json"
    _write_valid_cache(cache_path, "https://gateway.example.com")
    # Env says something else — the check should flag the mismatch
    results = _check_cache_file("https://different.example.com")
    failed = [r for r in results if not r.ok]
    assert failed, "expected at least one failed check"
    # The label or detail must indicate the mismatch
    assert any(
        "match" in (r.label.lower() + " " + r.detail.lower())
        or ("cache says" in r.detail and "env says" in r.detail)
        for r in failed
    ), f"failed checks: {[(r.label, r.detail) for r in failed]}"


def test_cache_missing_aliases_is_flagged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cache_path = tmp_path / ".claude" / "cache" / "gateway-models.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Valid wrapper but no ModelMeld aliases — picker won't show them
    cache = {
        "baseUrl": "https://x.com",
        "fetchedAt": 0,
        "models": [{"id": "some-other-model", "display_name": "Other"}],
    }
    cache_path.write_bytes(json.dumps(cache, indent=2).encode("utf-8"))
    results = _check_cache_file("https://x.com")
    assert any(not r.ok and "missing" in r.detail.lower() for r in results)


def test_cache_invalid_json_is_flagged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cache_path = tmp_path / ".claude" / "cache" / "gateway-models.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"{this is not valid json")
    results = _check_cache_file("https://x.com")
    assert any(not r.ok and "json" in r.detail.lower() for r in results)

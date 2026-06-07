# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for the `modelmeld setup` CLI.

Task #144. Every footgun we hit during 2026-05-25 manual validation has
a test here so we never regress. Specifically:
  - Customer-pasted secrets with trailing CR/LF/whitespace get normalized
  - Files we write contain ONLY LF line endings (never CRLF, even on Windows)
  - Cache file matches the wrapper format Claude Code's reader expects
  - Env script is sourceable and produces clean env vars
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Mode bits don't apply on Windows (NTFS), so skip those tests there.
# In production the gateway runs on Linux (Render); WSL also supports POSIX
# modes. The .chmod() call is a no-op on Windows but harmless.
skip_on_windows_modes = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits not supported on Windows NTFS",
)

from modelmeld.cli.setup import (
    _atomic_write_text,
    _codex_config_content,
    _normalize,
    _write_claude_code_cache,
    _write_claude_code_env_script,
    _write_codex_config,
    _write_codex_env_script,
)

# ---------------------------------------------------------------------------
# Input normalization — the #1 hours-burner footgun
# ---------------------------------------------------------------------------


def test_normalize_strips_trailing_cr() -> None:
    """A copy-paste from a Windows clipboard often has a trailing \\r.
    This silently breaks `H.baseUrl !== process.env.ANTHROPIC_BASE_URL`
    string-equality checks downstream. Strip it."""
    assert _normalize("gws_abc\r") == "gws_abc"


def test_normalize_strips_trailing_lf() -> None:
    assert _normalize("gws_abc\n") == "gws_abc"


def test_normalize_strips_embedded_crlf() -> None:
    """If a customer's key got terminal-wrapped on paste, CRLF can land
    in the MIDDLE of the value. The downstream signature check fails
    cryptically. Strip any CR/LF, not just trailing."""
    assert _normalize("gws_\r\nmiddle\r\n") == "gws_middle"


def test_normalize_strips_leading_and_trailing_whitespace() -> None:
    assert _normalize("  gws_abc  ") == "gws_abc"


def test_normalize_handles_none() -> None:
    assert _normalize(None) == ""


def test_normalize_handles_empty() -> None:
    assert _normalize("") == ""


# ---------------------------------------------------------------------------
# File writing — binary mode, LF only, post-write validation
# ---------------------------------------------------------------------------


def test_atomic_write_text_writes_lf_only(tmp_path: Path) -> None:
    """The contract that prevented today's pain: never CRLF on disk."""
    target = tmp_path / "test.sh"
    content = "line1\nline2\nline3\n"
    _atomic_write_text(target, content)
    written = target.read_bytes()
    assert b"\r" not in written
    assert written == content.encode("utf-8")


def test_atomic_write_text_normalizes_crlf_input(tmp_path: Path) -> None:
    """Even if caller accidentally passes CRLF content, the file on disk
    has LF only. Defense in depth."""
    target = tmp_path / "test.sh"
    _atomic_write_text(target, "line1\r\nline2\r\n")
    assert target.read_bytes() == b"line1\nline2\n"


def test_atomic_write_text_normalizes_lone_cr(tmp_path: Path) -> None:
    target = tmp_path / "test.sh"
    _atomic_write_text(target, "line1\rline2")
    assert target.read_bytes() == b"line1line2"


@skip_on_windows_modes
def test_atomic_write_text_sets_requested_mode(tmp_path: Path) -> None:
    target = tmp_path / "secret.sh"
    _atomic_write_text(target, "secret content", mode=0o600)
    import stat
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "f.txt"
    _atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_text_removes_temp_on_success(tmp_path: Path) -> None:
    target = tmp_path / "out.sh"
    _atomic_write_text(target, "content")
    # The .tmp sibling should not remain
    assert not (target.parent / (target.name + ".tmp")).exists()


# ---------------------------------------------------------------------------
# Env-script generation — must be sourceable without surprises
# ---------------------------------------------------------------------------


def test_env_script_writes_minimum_set(tmp_path: Path) -> None:
    target = tmp_path / "setup.sh"
    _write_claude_code_env_script(
        target,
        base_url="https://gateway.example.com",
        api_key="gws_TESTKEY",
        byok_anthropic=None,
        byok_openai=None,
    )
    content = target.read_text(encoding="utf-8")
    assert "ANTHROPIC_BASE_URL=https://gateway.example.com" in content
    assert "ANTHROPIC_API_KEY=gws_TESTKEY" in content
    assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1" in content
    assert "unset ANTHROPIC_AUTH_TOKEN" in content
    # No BYOK header set when no BYOK keys provided
    assert "ANTHROPIC_CUSTOM_HEADERS" not in content
    # No CRLF
    assert "\r" not in target.read_bytes().decode("latin-1")


def test_env_script_includes_byok_anthropic(tmp_path: Path) -> None:
    target = tmp_path / "setup.sh"
    _write_claude_code_env_script(
        target,
        base_url="https://gateway.example.com",
        api_key="gws_TESTKEY",
        byok_anthropic="sk-ant-TEST",
        byok_openai=None,
    )
    content = target.read_text(encoding="utf-8")
    # Header line includes the prefix + value
    assert 'ANTHROPIC_CUSTOM_HEADERS="x-modelmeld-byok-anthropic: sk-ant-TEST"' in content


def test_env_script_includes_both_byok_keys(tmp_path: Path) -> None:
    target = tmp_path / "setup.sh"
    _write_claude_code_env_script(
        target,
        base_url="https://gateway.example.com",
        api_key="gws_TESTKEY",
        byok_anthropic="sk-ant-A",
        byok_openai="sk-openai-O",
    )
    content = target.read_text(encoding="utf-8")
    assert "x-modelmeld-byok-anthropic: sk-ant-A" in content
    assert "x-modelmeld-byok-openai: sk-openai-O" in content


def test_env_script_is_lf_only(tmp_path: Path) -> None:
    """The launch-blocker bug we burned hours on. NEVER allow CRLF."""
    target = tmp_path / "setup.sh"
    _write_claude_code_env_script(
        target, "https://x.com", "gws_K", "sk-ant-K", None,
    )
    assert b"\r" not in target.read_bytes()


@skip_on_windows_modes
def test_env_script_has_mode_0600(tmp_path: Path) -> None:
    """Contains secrets — must be owner-only readable."""
    import stat
    target = tmp_path / "setup.sh"
    _write_claude_code_env_script(target, "https://x.com", "gws_K", None, None)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Codex CLI — config.toml provider block + env script
# ---------------------------------------------------------------------------


def test_codex_config_has_responses_provider_block() -> None:
    content = _codex_config_content("https://gateway.example.com")
    assert 'model_provider = "modelmeld"' in content
    assert "[model_providers.modelmeld]" in content
    assert 'base_url = "https://gateway.example.com/v1"' in content
    # Codex speaks the Responses API — the wire_api MUST be responses.
    assert 'wire_api = "responses"' in content
    # Key is read from an env var, never written into the config file.
    assert 'env_key = "MODELMELD_API_KEY"' in content
    assert "gws_" not in content


def test_codex_config_default_model_is_saver() -> None:
    # Predictable cost ceiling by default — OSS-only, no silent frontier.
    content = _codex_config_content("https://x.com")
    assert 'model = "anthropic/modelmeld-saver"' in content


def test_write_codex_config_is_lf_only(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    _write_codex_config(target, "https://x.com")
    assert b"\r" not in target.read_bytes()


def test_codex_env_script_exports_only_the_key(tmp_path: Path) -> None:
    target = tmp_path / "setup-codex.sh"
    _write_codex_env_script(target, "gws_CODEXKEY")
    content = target.read_text(encoding="utf-8")
    assert "export MODELMELD_API_KEY=gws_CODEXKEY" in content
    assert b"\r" not in target.read_bytes()


@skip_on_windows_modes
def test_codex_env_script_has_mode_0600(tmp_path: Path) -> None:
    import stat
    target = tmp_path / "setup-codex.sh"
    _write_codex_env_script(target, "gws_K")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Claude Code cache — exact wrapper shape Claude Code v2.1.150 expects
# ---------------------------------------------------------------------------


def _models_response_fixture() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": "gpt-5-mini", "display_name": "GPT-5 Mini"},
            {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
            {"id": "claude-haiku-4-5-20251001", "display_name": None},
            {
                "id": "anthropic/modelmeld-saver",
                "display_name": "ModelMeld — Saver (OSS-only auto-route)",
            },
            {
                "id": "anthropic/modelmeld-auto",
                "display_name": "ModelMeld — Auto (smart escalation)",
            },
            {
                "id": "anthropic/modelmeld-quality",
                "display_name": "ModelMeld — Quality (frontier-first)",
            },
        ],
    }


def test_cache_wrapper_has_required_keys(tmp_path: Path) -> None:
    """Claude Code's reader does
    `if (!H || H.baseUrl !== process.env.ANTHROPIC_BASE_URL) return [];`
    so the wrapper must have these three top-level keys."""
    target = tmp_path / "cache.json"
    _write_claude_code_cache(
        target, "https://gateway.example.com", _models_response_fixture(),
    )
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert "baseUrl" in parsed
    assert "fetchedAt" in parsed
    assert "models" in parsed


def test_cache_baseurl_matches_input(tmp_path: Path) -> None:
    """If baseUrl in cache doesn't match process.env.ANTHROPIC_BASE_URL,
    reader returns empty array → picker shows nothing. Exact match is
    load-bearing."""
    url = "https://gateway.example.com"
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, url, _models_response_fixture())
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["baseUrl"] == url


def test_cache_fetchedAt_is_number(tmp_path: Path) -> None:
    """Zod schema requires number, not string."""
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, "x", _models_response_fixture())
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(parsed["fetchedAt"], int)


def test_cache_each_model_has_id_and_display_name(tmp_path: Path) -> None:
    """Per the zod model schema: id (required string), display_name (optional)."""
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, "x", _models_response_fixture())
    parsed = json.loads(target.read_text(encoding="utf-8"))
    for m in parsed["models"]:
        assert "id" in m and isinstance(m["id"], str) and m["id"]
        assert "display_name" in m and isinstance(m["display_name"], str)


def test_cache_falls_back_to_id_when_display_name_null(tmp_path: Path) -> None:
    """A null display_name in the source becomes the id in the cache.
    Otherwise Claude Code's reader rejects the row (display_name must be
    string, not null)."""
    target = tmp_path / "cache.json"
    src = {"data": [{"id": "claude-haiku-4-5-20251001", "display_name": None}]}
    _write_claude_code_cache(target, "x", src)
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["models"][0]["display_name"] == "claude-haiku-4-5-20251001"


def test_cache_is_lf_only(tmp_path: Path) -> None:
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, "x", _models_response_fixture())
    assert b"\r" not in target.read_bytes()


@skip_on_windows_modes
def test_cache_has_mode_0600(tmp_path: Path) -> None:
    import stat
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, "x", _models_response_fixture())
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_cache_preserves_unicode_in_display_name(tmp_path: Path) -> None:
    """The em-dash in ModelMeld — Saver must round-trip cleanly."""
    target = tmp_path / "cache.json"
    _write_claude_code_cache(target, "x", _models_response_fixture())
    parsed = json.loads(target.read_text(encoding="utf-8"))
    found = [m for m in parsed["models"] if m["id"] == "anthropic/modelmeld-saver"]
    assert len(found) == 1
    assert "—" in found[0]["display_name"]

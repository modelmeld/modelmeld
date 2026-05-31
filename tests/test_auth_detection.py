# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Shape detection for inbound Authorization headers."""

from __future__ import annotations

from modelmeld.api.auth_detection import (
    AuthKind,
    classify_authorization,
)


# ---------------------------------------------------------------------------
# Empty / missing
# ---------------------------------------------------------------------------


def test_missing_header_returns_missing() -> None:
    result = classify_authorization(None)
    assert result.kind is AuthKind.MISSING
    assert result.token == ""


def test_empty_string_returns_missing() -> None:
    result = classify_authorization("")
    assert result.kind is AuthKind.MISSING


def test_whitespace_only_returns_missing() -> None:
    result = classify_authorization("   ")
    assert result.kind is AuthKind.MISSING


def test_bearer_with_no_token_returns_missing() -> None:
    result = classify_authorization("Bearer ")
    assert result.kind is AuthKind.MISSING


# ---------------------------------------------------------------------------
# Bearer prefix stripping
# ---------------------------------------------------------------------------


def test_strips_bearer_prefix() -> None:
    result = classify_authorization("Bearer sk-test123")
    assert result.kind is AuthKind.OPENAI_API_KEY
    assert result.token == "sk-test123"


def test_strips_bearer_prefix_case_insensitively() -> None:
    """RFC 7235 says scheme is case-insensitive; some clients send 'BEARER' or 'bearer'."""
    for variant in ["bearer sk-test", "BEARER sk-test", "Bearer sk-test", "bEaReR sk-test"]:
        result = classify_authorization(variant)
        assert result.kind is AuthKind.OPENAI_API_KEY, (
            f"Failed to strip prefix from {variant!r}"
        )


def test_accepts_bare_token_without_bearer_prefix() -> None:
    result = classify_authorization("sk-test123")
    assert result.kind is AuthKind.OPENAI_API_KEY
    assert result.token == "sk-test123"


# ---------------------------------------------------------------------------
# Shape detection — known providers
# ---------------------------------------------------------------------------


def test_anthropic_api_key_detected_by_sk_ant_prefix() -> None:
    result = classify_authorization("Bearer sk-ant-api03-abcdef1234567890")
    assert result.kind is AuthKind.ANTHROPIC_API_KEY


def test_openai_api_key_detected_by_sk_prefix() -> None:
    result = classify_authorization("Bearer sk-proj-abc123")
    assert result.kind is AuthKind.OPENAI_API_KEY


def test_anthropic_prefix_takes_precedence_over_openai() -> None:
    """sk-ant-... matches both sk- AND sk-ant-; must classify as Anthropic."""
    # If the ordering were wrong, sk-ant-x would classify as OpenAI.
    result = classify_authorization("Bearer sk-ant-x")
    assert result.kind is AuthKind.ANTHROPIC_API_KEY


def test_modelmeld_license_detected_by_gws_prefix() -> None:
    result = classify_authorization("Bearer gws_abcdef1234567890")
    assert result.kind is AuthKind.MODELMELD_LICENSE


def test_oauth_jwt_detected_by_eyj_prefix() -> None:
    """A JWT's base64-encoded header always starts with 'eyJ' because
    JSON header object always begins with '{"alg":'."""
    # Realistic JWT-shaped token
    jwt_like = (
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNzAwMDAwMDAwfQ."
        "signature_part"
    )
    result = classify_authorization(f"Bearer {jwt_like}")
    assert result.kind is AuthKind.OAUTH_BEARER


# ---------------------------------------------------------------------------
# Unknown shapes
# ---------------------------------------------------------------------------


def test_unknown_token_shape_returns_unknown() -> None:
    """A random string that doesn't match any known prefix."""
    result = classify_authorization("Bearer xoxp-slack-style-token-123")
    assert result.kind is AuthKind.UNKNOWN
    # Token preserved so the caller can still log/inspect.
    assert result.token == "xoxp-slack-style-token-123"


def test_basic_auth_returns_unknown() -> None:
    """We only handle Bearer auth; Basic / Digest / etc. fall to UNKNOWN."""
    result = classify_authorization("Basic dXNlcjpwYXNz")
    assert result.kind is AuthKind.UNKNOWN


# ---------------------------------------------------------------------------
# Redaction safety — display field must NEVER echo the token
# ---------------------------------------------------------------------------


def test_display_redacts_secret_bytes() -> None:
    """The display field is the safe-to-log view. Secret bytes must
    not appear in it."""
    secret = "sk-ant-api03-VERY-SECRET-TOKEN-MATERIAL-DO-NOT-LEAK"
    result = classify_authorization(f"Bearer {secret}")
    assert "VERY-SECRET-TOKEN-MATERIAL" not in result.display
    assert "DO-NOT-LEAK" not in result.display


def test_display_keeps_prefix_for_operator_diagnostics() -> None:
    """Operators looking at a log line should still be able to verify
    the SHAPE of the token (which provider it was for) without seeing
    the secret bytes. We preserve the first 7 chars."""
    result = classify_authorization("Bearer sk-ant-api03-realsecretbytes")
    # First 7 chars of the token = 'sk-ant-' → visible in display
    assert result.display.startswith("sk-ant-")
    # Length preserved so operators can spot truncation / wrong-size issues
    # `sk-ant-api03-realsecretbytes` = 28 chars
    assert "len=28" in result.display


def test_display_handles_short_tokens_gracefully() -> None:
    """Tokens shorter than 7 chars don't crash redaction."""
    result = classify_authorization("Bearer abc")
    # No prefix shown for sub-7-char tokens to avoid leaking the whole token
    assert "abc" not in result.display
    assert "len=3" in result.display


# ---------------------------------------------------------------------------
# Real-world Authorization header examples
# ---------------------------------------------------------------------------


def test_codex_cli_oauth_token_shape() -> None:
    """Smoke-test that the JWT shape Codex CLI sends gets classified
    correctly. (Synthetic; real Codex JWT is much longer.)"""
    codex_jwt = "eyJhbGciOiJFUzI1NiIsImtpZCI6ImNvZGV4LWNsaSJ9.eyJzdWIiOiJjaGF0Z3B0LXVzZXIifQ.sig"
    result = classify_authorization(f"Bearer {codex_jwt}")
    assert result.kind is AuthKind.OAUTH_BEARER
    assert result.token == codex_jwt


def test_claude_code_oauth_token_shape() -> None:
    """Claude Max OAuth tokens are also JWTs starting with eyJ."""
    claude_jwt = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJjbGF1ZGUtbWF4LXVzZXIifQ.sig"
    result = classify_authorization(f"Bearer {claude_jwt}")
    assert result.kind is AuthKind.OAUTH_BEARER

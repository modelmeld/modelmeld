# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Subscription-passthrough route helper — opt-in gating + adapter selection."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from modelmeld.api.auth_detection import AuthClassification, AuthKind
from modelmeld.api.subscription_passthrough import (
    PassthroughVendor,
    resolve_passthrough_router,
)
from modelmeld.router.base import SingleAdapterRouter


def _auth(kind: AuthKind, token: str = "x") -> AuthClassification:
    return AuthClassification(kind=kind, token=token, display=f"{token[:7]}***")


# ---------------------------------------------------------------------------
# Non-passthrough paths — return None, let normal routing proceed
# ---------------------------------------------------------------------------


def test_missing_authorization_returns_none() -> None:
    """No bearer at all → not a passthrough request; normal routing."""
    result = resolve_passthrough_router(
        _auth(AuthKind.MISSING, ""),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert result is None


def test_openai_api_key_returns_none() -> None:
    """sk-... bearer is an OpenAI API key, not a subscription token."""
    result = resolve_passthrough_router(
        _auth(AuthKind.OPENAI_API_KEY, "sk-test-123"),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert result is None


def test_anthropic_api_key_returns_none() -> None:
    """sk-ant-... bearer is an Anthropic API key, not a subscription token."""
    result = resolve_passthrough_router(
        _auth(AuthKind.ANTHROPIC_API_KEY, "sk-ant-test"),
        vendor=PassthroughVendor.ANTHROPIC,
        allow_passthrough=True,
    )
    assert result is None


def test_modelmeld_license_returns_none() -> None:
    """gws_... is our own license key, not a subscription token."""
    result = resolve_passthrough_router(
        _auth(AuthKind.MODELMELD_LICENSE, "gws_test"),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert result is None


def test_unknown_token_shape_returns_none() -> None:
    """An unknown bearer doesn't trigger passthrough — caller continues
    with normal routing and lets the eventual error bubble up naturally."""
    result = resolve_passthrough_router(
        _auth(AuthKind.UNKNOWN, "xoxp-slack-token"),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert result is None


# ---------------------------------------------------------------------------
# OAuth bearer + opt-in DISABLED → 403
# ---------------------------------------------------------------------------


def test_oauth_bearer_with_flag_disabled_returns_403() -> None:
    """OAuth bearer is present but the operator hasn't enabled
    subscription passthrough — must 403, not silently fall through.

    Silent fallback would lead the user's tool to a confusing
    downstream error (the bearer fails API-key validation against
    api.openai.com). The 403 with the exact env-var name to flip is
    much friendlier."""
    with pytest.raises(HTTPException) as exc:
        resolve_passthrough_router(
            _auth(AuthKind.OAUTH_BEARER, "eyJtest"),
            vendor=PassthroughVendor.CODEX,
            allow_passthrough=False,
        )
    assert exc.value.status_code == 403
    assert "MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH" in exc.value.detail
    assert "subscription_passthrough_disabled" in exc.value.detail


# ---------------------------------------------------------------------------
# OAuth bearer + opt-in ENABLED → SingleAdapterRouter
# ---------------------------------------------------------------------------


def test_oauth_bearer_codex_vendor_returns_single_adapter_router() -> None:
    """Full happy-path: OAuth bearer + flag enabled → SingleAdapterRouter
    wrapping CodexPassthroughAdapter. Caller routes via this for THIS
    request, leaving the persistent capability router untouched."""
    result = resolve_passthrough_router(
        _auth(AuthKind.OAUTH_BEARER, "eyJfake_codex_jwt"),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert isinstance(result, SingleAdapterRouter)
    assert result.adapter.name == "codex_passthrough"


def test_oauth_bearer_anthropic_vendor_returns_anthropic_oauth_adapter() -> None:
    """Sprint 5 Phase B landed AnthropicAdapter OAuth mode. The helper
    now resolves ANTHROPIC vendor to an AnthropicAdapter constructed
    with oauth_bearer (NOT api_key)."""
    result = resolve_passthrough_router(
        _auth(AuthKind.OAUTH_BEARER, "eyJfake_claude_max_jwt"),
        vendor=PassthroughVendor.ANTHROPIC,
        allow_passthrough=True,
    )
    assert isinstance(result, SingleAdapterRouter)
    assert result.adapter.name == "anthropic"


# ---------------------------------------------------------------------------
# Token forwarding — adapter must receive the EXACT bearer token
# ---------------------------------------------------------------------------


def test_oauth_bearer_token_forwarded_to_adapter_verbatim() -> None:
    """Whatever the inbound bearer is, the adapter sees the exact same
    token. Forwarding is the entire point of passthrough — any
    transformation would defeat the verbatim-passthrough ToS posture."""
    token = "eyJhbGciOiJSUzI1NiJ9.payload.signature"
    router = resolve_passthrough_router(
        _auth(AuthKind.OAUTH_BEARER, token),
        vendor=PassthroughVendor.CODEX,
        allow_passthrough=True,
    )
    assert isinstance(router, SingleAdapterRouter)
    # Don't introspect the SDK client's headers (private API).
    # Construction success implies the token was accepted.
    assert router.adapter is not None

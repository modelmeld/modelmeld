# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Route-handler helper for subscription-passthrough routing.

Both `/v1/chat/completions` and `/v1/messages` route handlers need the
same pattern when an OAuth-bearer-shaped Authorization header arrives:

  1. If the opt-in flag `allow_subscription_passthrough` is False:
     reject with HTTP 403 (do NOT silently fall through to normal
     routing — that would confuse the user with an unrelated error
     about their bearer token failing API-key validation downstream).
  2. If enabled: build a passthrough adapter for the appropriate
     vendor and return it wrapped in a SingleAdapterRouter. Caller
     uses that router for THIS request only; the persistent
     capability router is unaffected.

The Authorization JWT shape doesn't itself tell us which vendor —
Codex CLI and Claude Code both send `Bearer eyJ...`. The discriminator
is the inbound API surface:
  - `/v1/chat/completions` (OpenAI-shape API) → Codex backend
  - `/v1/messages`         (Anthropic Messages API) → Claude Max
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import HTTPException

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.api.auth_detection import AuthClassification, AuthKind
from modelmeld.router.base import Router, SingleAdapterRouter

logger = logging.getLogger(__name__)


class PassthroughVendor(str, Enum):
    """Which subscription vendor a given API surface targets."""

    CODEX = "codex"        # /v1/chat/completions
    ANTHROPIC = "anthropic"  # /v1/messages


def resolve_passthrough_router(
    auth: AuthClassification,
    *,
    vendor: PassthroughVendor,
    allow_passthrough: bool,
) -> Router | None:
    """Decide whether THIS request should bypass normal routing and use
    a subscription-passthrough adapter instead.

    Returns:
      - None if the request is NOT a subscription passthrough (no OAuth
        bearer detected, or detection inconclusive). Caller continues
        with the normal capability router.
      - A SingleAdapterRouter wrapping the vendor-specific passthrough
        adapter when the request IS a subscription passthrough and the
        opt-in flag permits it. Caller routes via this for this request.

    Raises HTTPException(403) when an OAuth bearer is present but the
    opt-in flag is False — this is intentional: silent fallthrough to
    normal routing would yield a confusing error elsewhere.
    """
    if auth.kind is not AuthKind.OAUTH_BEARER:
        return None

    if not allow_passthrough:
        raise HTTPException(
            status_code=403,
            detail=(
                "subscription_passthrough_disabled: An OAuth bearer token "
                "was supplied (auth shape suggests a Codex CLI or Claude "
                "Max subscription token) but this gateway has "
                "MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=False. Operator "
                "must explicitly enable subscription passthrough; see "
                "docs/subscription-passthrough.md."
            ),
        )

    adapter = _build_passthrough_adapter(auth, vendor)
    if adapter is None:
        # Vendor-specific construction returned None — typically because
        # the adapter's optional dependency isn't installed. Surface a
        # clean error rather than 500-on-construct.
        raise HTTPException(
            status_code=500,
            detail=(
                f"subscription_passthrough_unavailable: vendor={vendor.value} "
                "adapter could not be constructed. Verify that the gateway "
                "has the optional provider extras installed "
                "(pip install 'modelmeld[openai,anthropic]')."
            ),
        )
    logger.info(
        "subscription_passthrough engaged: vendor=%s auth=%s",
        vendor.value,
        auth.display,
    )
    return SingleAdapterRouter(adapter)


def _build_passthrough_adapter(
    auth: AuthClassification, vendor: PassthroughVendor,
) -> ProviderAdapter | None:
    """Vendor-specific adapter construction. Localized so the route helper
    above doesn't need to know about each vendor's adapter class — keeps
    the test surface small.

    Returns None on construction failure (caller raises 500). The auth
    token is passed to the adapter constructor; the adapter itself
    decides how to use it (Bearer header, custom auth flow, etc.).
    """
    try:
        if vendor is PassthroughVendor.CODEX:
            from modelmeld.adapters.codex_passthrough import CodexPassthroughAdapter
            return CodexPassthroughAdapter(access_token=auth.token)
        if vendor is PassthroughVendor.ANTHROPIC:
            # Sprint 5 work: the AnthropicAdapter learns OAuth-bearer
            # mode. Construction signature here will become:
            #   AnthropicAdapter(oauth_bearer=auth.token)
            # Until that extension lands, this branch surfaces a clean
            # "not yet implemented" error rather than half-working code.
            return None
    except AdapterError:
        logger.exception(
            "subscription_passthrough adapter construction failed vendor=%s",
            vendor.value,
        )
        return None
    return None


__all__ = [
    "PassthroughVendor",
    "resolve_passthrough_router",
]

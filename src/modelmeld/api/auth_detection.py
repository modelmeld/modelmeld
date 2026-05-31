# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Inbound Authorization-header shape detection for subscription passthrough.

When a customer points their dev-tool (Codex CLI, Claude Code, etc.) at our
self-hosted gateway, the tool sends its native Authorization header verbatim.
The gateway needs to distinguish:

- `sk-ant-...`            → Anthropic API key (route to AnthropicAdapter)
- `sk-...` (not sk-ant)   → OpenAI API key (route to OpenAIAdapter)
- `gws_...`               → ModelMeld license key (auth/billing identity)
- `eyJ...`                → OAuth JWT (subscription passthrough — Codex CLI
                            with "Sign in with ChatGPT" enabled, or Claude
                            Code with Claude Max OAuth)
- anything else           → unknown (return as-is to caller)

Subscription passthrough routing is gated on a separate opt-in flag
(`MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=1`) — detection here is purely
shape-based and doesn't itself enable the path. See
`docs/subscription-passthrough-codex-feasibility.md` for the ToS
posture details.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthKind(str, Enum):
    """Coarse classification of an inbound Authorization header value."""

    ANTHROPIC_API_KEY = "anthropic_api_key"
    OPENAI_API_KEY = "openai_api_key"
    MODELMELD_LICENSE = "modelmeld_license"
    # An OAuth/JWT bearer. Could be a Codex CLI subscription token
    # (chatgpt.com backend) or a Claude Max OAuth bearer
    # (api.anthropic.com OAuth surface). The caller resolves which
    # subscription provider to route to using a secondary signal —
    # typically the endpoint path (/v1/chat/completions, /v1/messages,
    # /v1/responses) or the request body's model field.
    OAUTH_BEARER = "oauth_bearer"
    # Header present but doesn't match any known shape.
    UNKNOWN = "unknown"
    # No Authorization header at all (or empty after Bearer stripping).
    MISSING = "missing"


@dataclass(frozen=True)
class AuthClassification:
    """The result of inspecting an Authorization header value.

    `kind` is the shape classification. `token` is the raw token bytes
    AFTER stripping the `Bearer ` prefix — NEVER log this, never echo
    in error responses. `display` is a safe-to-log redacted form.
    """

    kind: AuthKind
    token: str
    display: str


def _redact(token: str) -> str:
    """Length-preserving redaction marker. Mirrors api/byok.py's pattern.

    Keeps the first 7 chars (typically the prefix like `sk-ant-`,
    `gws_`, `eyJ`) so operators can verify shape without seeing the
    secret bytes.
    """
    if not token:
        return "***[empty]"
    prefix = token[:7] if len(token) >= 7 else ""
    return f"{prefix}***[len={len(token)}]"


def classify_authorization(authorization_header: str | None) -> AuthClassification:
    """Inspect an Authorization header value and classify the token shape.

    Accepts either the raw header value (`"Bearer sk-..."`) or the bare
    token (`"sk-..."`). Strips a leading `Bearer ` prefix
    case-insensitively. Trims whitespace.

    Never raises — unrecognizable headers return AuthKind.UNKNOWN with
    the token preserved so the caller can decide what to do.
    """
    if not authorization_header:
        return AuthClassification(
            kind=AuthKind.MISSING,
            token="",
            display="***[missing]",
        )
    raw = authorization_header.strip()
    # Strip an optional `bearer` scheme prefix (case-insensitive). The
    # outer strip() above ate any whitespace after the scheme, so we
    # match the scheme word alone here rather than `bearer ` with a
    # trailing space (otherwise inputs like "Bearer " classify wrong).
    if raw.lower().startswith("bearer"):
        raw = raw[6:].lstrip()
    if not raw:
        return AuthClassification(
            kind=AuthKind.MISSING,
            token="",
            display="***[empty]",
        )

    display = _redact(raw)
    # Order matters: sk-ant must be checked before sk- generally
    # because sk-ant-... also starts with sk-.
    if raw.startswith("sk-ant-"):
        kind = AuthKind.ANTHROPIC_API_KEY
    elif raw.startswith("sk-"):
        kind = AuthKind.OPENAI_API_KEY
    elif raw.startswith("gws_"):
        kind = AuthKind.MODELMELD_LICENSE
    elif raw.startswith("eyJ"):
        # JWT header always begins with `{"alg":...}` which base64-encodes
        # to `eyJh...`. The Codex CLI OAuth JWT and Claude Max OAuth JWT
        # both match this prefix. Opaque-bearer providers (if we add any
        # later) won't match — we'd add a separate detection branch.
        kind = AuthKind.OAUTH_BEARER
    else:
        kind = AuthKind.UNKNOWN
    return AuthClassification(kind=kind, token=raw, display=display)


__all__ = [
    "AuthClassification",
    "AuthKind",
    "classify_authorization",
]

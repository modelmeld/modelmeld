# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""BYOK (bring-your-own-key) header extraction for per-request frontier routing.

The customer sends their frontier API keys via custom HTTP headers
(`x-modelmeld-byok-anthropic`, `x-modelmeld-byok-openai`, etc.) on each
request. The gateway extracts them at the route boundary, hands them to
the router as per-request adapter overrides, and forgets them when the
response is sent. **Keys are never persisted to disk, never logged,
never echoed in error responses.**

This is the architectural complement to our auth model: customers pay
us for OSS routing + smart escalation logic, but the credentials for
proprietary models stay with the customer (per-request transit only,
no custody at rest). The pattern matches Claude Code's native
`ANTHROPIC_CUSTOM_HEADERS` env var — customers can wire BYOK by
adding:

    export ANTHROPIC_CUSTOM_HEADERS="x-modelmeld-byok-anthropic: sk-ant-..."

and Claude Code injects that header on every request to our gateway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


# Canonical header prefix. The provider name follows in lowercase
# (matching the registry provider field): `x-modelmeld-byok-anthropic`,
# `x-modelmeld-byok-openai`, etc. Future providers (`x-modelmeld-byok-google`,
# `-mistral`, `-cohere`) plug in without code changes.
_HEADER_PREFIX = "x-modelmeld-byok-"


# Providers that ARE valid BYOK targets — frontier API providers whose
# adapters live in core-engine. Restrict to this allowlist so a stray
# `x-modelmeld-byok-vllm` header doesn't silently override our own
# self-hosted vLLM adapter (which doesn't need BYOK anyway).
_BYOK_ELIGIBLE_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})


@dataclass(frozen=True)
class BYOKCredentials:
    """Per-request frontier-key dictionary keyed by provider name.

    Treat values as ephemeral — never log, never persist, never echo in
    error responses. The `redact_for_log` method returns a safe view.
    """

    _keys: Mapping[str, str]

    def get(self, provider: str) -> str | None:
        return self._keys.get(provider)

    def providers(self) -> frozenset[str]:
        return frozenset(self._keys.keys())

    def is_empty(self) -> bool:
        return not self._keys

    def redact_for_log(self) -> dict[str, str]:
        """Safe-for-logging view: shows which providers were supplied
        but redacts the keys themselves to a length+prefix marker.
        Example: {"anthropic": "sk-ant-***[len=89]"}.
        """
        view: dict[str, str] = {}
        for provider, key in self._keys.items():
            view[provider] = _redact_key(key)
        return view


def extract_byok_credentials(headers: Iterable[tuple[str, str]] | Mapping[str, str]) -> BYOKCredentials:
    """Parse incoming request headers, extracting any BYOK keys.

    Accepts either:
      - an iterable of (name, value) tuples (e.g., Starlette's `request.headers.items()`)
      - a dict-like mapping (e.g., httpx headers)

    Header names are matched case-insensitively. Unknown provider names
    (anything not in _BYOK_ELIGIBLE_PROVIDERS) are silently dropped —
    safer than echoing an "unknown provider" error that could be used
    to enumerate our adapter inventory.
    """
    keys: dict[str, str] = {}
    items: Iterable[tuple[str, str]]
    if isinstance(headers, Mapping):
        items = headers.items()
    else:
        items = headers
    for name, value in items:
        name_lower = name.lower()
        if not name_lower.startswith(_HEADER_PREFIX):
            continue
        provider = name_lower[len(_HEADER_PREFIX):].strip()
        if provider not in _BYOK_ELIGIBLE_PROVIDERS:
            continue
        if not value or not value.strip():
            continue
        keys[provider] = value.strip()
    return BYOKCredentials(_keys=keys)


def redact_byok_headers(
    headers: Iterable[tuple[str, str]] | Mapping[str, str],
) -> list[tuple[str, str]]:
    """Return headers with BYOK values redacted (for log/audit dumps)."""
    items: Iterable[tuple[str, str]]
    if isinstance(headers, Mapping):
        items = headers.items()
    else:
        items = headers
    out: list[tuple[str, str]] = []
    for name, value in items:
        name_lower = name.lower()
        if name_lower.startswith(_HEADER_PREFIX):
            out.append((name, _redact_key(value)))
        else:
            out.append((name, value))
    return out


def _redact_key(key: str) -> str:
    """Replace key with a length-preserving redaction marker.

    Pattern: keeps the first 7 chars (typically the provider prefix
    like `sk-ant-`, `sk-`, `gws_`, `xoxp-`) so operators can verify
    the *shape* of the key while never exposing the secret bytes.
    """
    if not key:
        return "***[empty]"
    prefix = key[:7] if len(key) >= 7 else ""
    return f"{prefix}***[len={len(key)}]"


def eligible_providers() -> frozenset[str]:
    """The set of providers BYOK headers can target."""
    return _BYOK_ELIGIBLE_PROVIDERS


def build_byok_adapters(creds: BYOKCredentials) -> dict[str, "ProviderAdapter"]:
    """Construct per-request adapter instances using the supplied BYOK keys.

    Returns an empty dict when creds is empty. Caller is responsible for
    the adapter lifecycle — these are one-shot, request-scoped objects.
    They share no client state with the persistent adapters in the
    router's adapters_by_provider map.

    Failed construction (e.g., missing optional adapter dependency) is
    silently dropped from the returned dict — the route handler will
    then return a clean BYOK-required 400 rather than crashing the
    request with an import error.
    """
    out: dict[str, ProviderAdapter] = {}
    for provider in creds.providers():
        key = creds.get(provider)
        if not key:
            continue
        try:
            if provider == "anthropic":
                from modelmeld.adapters.anthropic_adapter import AnthropicAdapter
                out[provider] = AnthropicAdapter(api_key=key)
            elif provider == "openai":
                from modelmeld.adapters.openai_adapter import OpenAIAdapter
                out[provider] = OpenAIAdapter(api_key=key)
        except Exception:  # noqa: BLE001 — never let BYOK construction crash the route
            # Use logger.error (not logger.exception) to avoid emitting
            # a traceback whose locals frame would include the `key`
            # variable. Defense in depth — the SDK constructors don't
            # typically echo the key in their exception text, but the
            # traceback formatter can render locals in some configurations.
            logger.error(
                "BYOK adapter construction failed for provider=%s "
                "(optional dependency missing or invalid key shape?); "
                "request will route as if no BYOK key was provided",
                provider,
            )
    return out


# Type-only import for the build_byok_adapters return annotation. We
# can't put this at module top because adapter modules pull in optional
# deps (anthropic, openai libs) that aren't always installed; we want
# `import byok` to be cheap and never raise.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelmeld.adapters.base import ProviderAdapter


__all__ = [
    "BYOKCredentials",
    "extract_byok_credentials",
    "redact_byok_headers",
    "eligible_providers",
    "build_byok_adapters",
]

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for the BYOK header extraction + adapter factory.

Task #178. The BYOK mechanism is launch-critical for the QUALITY and
AUTO-escalated policies — without it, the hosted gateway cannot dispatch
to frontier providers, and those alias policies return 503.
"""
from __future__ import annotations

import pytest

from modelmeld.api.byok import (
    BYOKCredentials,
    build_byok_adapters,
    eligible_providers,
    extract_byok_credentials,
    redact_byok_headers,
)


# ---------------------------------------------------------------------------
# extract_byok_credentials
# ---------------------------------------------------------------------------


def test_extract_returns_empty_when_no_byok_headers() -> None:
    creds = extract_byok_credentials({
        "content-type": "application/json",
        "x-api-key": "gws_abc",
    })
    assert creds.is_empty()
    assert creds.providers() == frozenset()


def test_extract_picks_up_anthropic_byok_header() -> None:
    creds = extract_byok_credentials({
        "x-modelmeld-byok-anthropic": "sk-ant-abc123",
    })
    assert not creds.is_empty()
    assert creds.providers() == frozenset({"anthropic"})
    assert creds.get("anthropic") == "sk-ant-abc123"


def test_extract_picks_up_openai_byok_header() -> None:
    creds = extract_byok_credentials({
        "x-modelmeld-byok-openai": "sk-openai-abc",
    })
    assert creds.providers() == frozenset({"openai"})
    assert creds.get("openai") == "sk-openai-abc"


def test_extract_handles_both_anthropic_and_openai() -> None:
    creds = extract_byok_credentials({
        "x-modelmeld-byok-anthropic": "sk-ant-x",
        "x-modelmeld-byok-openai": "sk-openai-x",
    })
    assert creds.providers() == frozenset({"anthropic", "openai"})


def test_extract_is_case_insensitive_on_header_name() -> None:
    creds = extract_byok_credentials({
        "X-Modelmeld-Byok-Anthropic": "sk-ant-Z",
    })
    assert creds.get("anthropic") == "sk-ant-Z"


def test_extract_drops_unknown_provider_silently() -> None:
    """A stray `x-modelmeld-byok-vllm` header shouldn't silently override
    our self-hosted vLLM adapter — we restrict to the canonical eligible
    set to prevent that exact footgun."""
    creds = extract_byok_credentials({
        "x-modelmeld-byok-vllm": "should-be-ignored",
        "x-modelmeld-byok-totally-fake-provider": "also-ignored",
    })
    assert creds.is_empty()


def test_extract_drops_empty_values() -> None:
    creds = extract_byok_credentials({
        "x-modelmeld-byok-anthropic": "",
        "x-modelmeld-byok-openai": "  ",  # whitespace-only
    })
    assert creds.is_empty()


def test_extract_strips_whitespace_around_value() -> None:
    creds = extract_byok_credentials({
        "x-modelmeld-byok-anthropic": "  sk-ant-abc  ",
    })
    assert creds.get("anthropic") == "sk-ant-abc"


def test_extract_accepts_tuple_iterable_input() -> None:
    """Starlette/HTTPX headers iterate as (name, value) tuples — make sure
    we accept that shape too, not just dicts."""
    creds = extract_byok_credentials([
        ("content-type", "application/json"),
        ("x-modelmeld-byok-anthropic", "sk-ant-tuple"),
    ])
    assert creds.get("anthropic") == "sk-ant-tuple"


# ---------------------------------------------------------------------------
# Redaction (must never leak secret bytes)
# ---------------------------------------------------------------------------


def test_redact_for_log_preserves_prefix_only() -> None:
    creds = BYOKCredentials(_keys={"anthropic": "sk-ant-api03-LONGSECRET12345"})
    view = creds.redact_for_log()
    # Prefix retained, secret bytes stripped
    assert view["anthropic"].startswith("sk-ant-")
    assert "LONGSECRET" not in view["anthropic"]
    assert "[len=" in view["anthropic"]


def test_redact_for_log_handles_short_key() -> None:
    creds = BYOKCredentials(_keys={"openai": "abc"})
    view = creds.redact_for_log()
    # Short keys still get a redaction marker, no secret bytes echoed
    assert "abc" not in view["openai"] or view["openai"].startswith("abc***")


def test_redact_byok_headers_only_touches_byok_entries() -> None:
    redacted = redact_byok_headers({
        "content-type": "application/json",
        "x-api-key": "gws_abc",
        "x-modelmeld-byok-anthropic": "sk-ant-VERYSECRET",
    })
    items = dict(redacted)
    # Non-BYOK headers passed through unchanged
    assert items["content-type"] == "application/json"
    assert items["x-api-key"] == "gws_abc"
    # BYOK value redacted
    assert "VERYSECRET" not in items["x-modelmeld-byok-anthropic"]


def test_redact_byok_headers_accepts_tuple_iterable() -> None:
    redacted = redact_byok_headers([
        ("x-modelmeld-byok-anthropic", "sk-ant-LONGSECRET"),
    ])
    assert "LONGSECRET" not in dict(redacted)["x-modelmeld-byok-anthropic"]


# ---------------------------------------------------------------------------
# build_byok_adapters
# ---------------------------------------------------------------------------


def test_build_byok_adapters_returns_empty_for_no_creds() -> None:
    creds = BYOKCredentials(_keys={})
    adapters = build_byok_adapters(creds)
    assert adapters == {}


def test_build_byok_adapters_constructs_anthropic_adapter() -> None:
    creds = BYOKCredentials(_keys={"anthropic": "sk-ant-test"})
    adapters = build_byok_adapters(creds)
    assert "anthropic" in adapters
    # The adapter exposes its provider name via .name (per ProviderAdapter contract)
    assert adapters["anthropic"].name == "anthropic"


def test_build_byok_adapters_constructs_openai_adapter() -> None:
    creds = BYOKCredentials(_keys={"openai": "sk-openai-test"})
    adapters = build_byok_adapters(creds)
    assert "openai" in adapters


def test_build_byok_adapters_returns_both_when_both_present() -> None:
    creds = BYOKCredentials(_keys={
        "anthropic": "sk-ant-x",
        "openai": "sk-openai-x",
    })
    adapters = build_byok_adapters(creds)
    assert "anthropic" in adapters and "openai" in adapters


# ---------------------------------------------------------------------------
# eligible_providers
# ---------------------------------------------------------------------------


def test_eligible_providers_includes_frontier_only() -> None:
    eligible = eligible_providers()
    assert "anthropic" in eligible
    assert "openai" in eligible
    assert "vllm" not in eligible
    assert "fireworks" not in eligible

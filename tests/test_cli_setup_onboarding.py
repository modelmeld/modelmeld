"""Fresh-customer onboarding smoke: `modelmeld setup --tool claude-code`.

Exercises the actual onboarding command a new user runs, NON-DESTRUCTIVELY:
  - `Path.home()` is redirected to a tmp dir so nothing touches the real
    ~/.modelmeld or ~/.claude (this test runs on dev machines too).
  - The gateway HTTP calls are stubbed, so no network and no live gateway.

Covers the config-generation path the install→setup→use flow depends on:
the sourceable env script and the Claude Code discovery cache. The "send a
real request + routing headers" half of the flow is covered by the route
tests (test_chat_capability_routing, test_capability_routing_with_hints);
the live-gateway smoke against real keys remains a manual release step.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelmeld.cli import setup as setup_mod

_MODELS_RESPONSE = {
    "data": [
        {"id": "anthropic/modelmeld-saver", "display_name": "ModelMeld — Saver"},
        {"id": "anthropic/modelmeld-auto", "display_name": "ModelMeld — Auto"},
        {"id": "anthropic/modelmeld-quality"},
    ]
}


def _fake_http_get(url, headers, timeout=15.0):
    """Stub the gateway's /v1/models so setup can pre-write the cache offline."""
    return 200, {}, json.dumps(_MODELS_RESPONSE).encode("utf-8")


def _args(**overrides):
    base = dict(
        tool="claude-code",
        base_url="http://127.0.0.1:8080",   # loopback → passes host validation
        api_key="gws_smoke-test-key",
        byok_anthropic=None,
        byok_openai=None,
        allow_custom_host=False,
        skip_smoke_test=True,               # skip the live routing smoke (step 4)
        yes=True,                           # non-interactive
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Redirect Path.home() to a tmp dir AND stub gateway GETs. Belt-and-suspenders
    isolation so setup can never write to the developer's real home."""
    monkeypatch.setattr(Path, "home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(setup_mod, "_http_get", _fake_http_get)
    return tmp_path


def test_setup_writes_env_script_and_cache(isolated_home) -> None:
    rc = setup_mod.run_setup(_args())
    assert rc == 0

    env_script = isolated_home / ".modelmeld" / "setup-claude-code.sh"
    assert env_script.exists()
    text = env_script.read_text()
    assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:8080" in text
    assert "export ANTHROPIC_API_KEY=gws_smoke-test-key" in text
    assert "unset ANTHROPIC_AUTH_TOKEN" in text
    # No BYOK supplied → no custom-headers export.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in text

    cache = isolated_home / ".claude" / "cache" / "gateway-models.json"
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert payload["baseUrl"] == "http://127.0.0.1:8080"
    ids = [m["id"] for m in payload["models"]]
    assert "anthropic/modelmeld-saver" in ids
    assert "anthropic/modelmeld-quality" in ids


def test_setup_byok_adds_custom_headers(isolated_home) -> None:
    rc = setup_mod.run_setup(_args(byok_anthropic="sk-ant-test123"))
    assert rc == 0
    text = (isolated_home / ".modelmeld" / "setup-claude-code.sh").read_text()
    assert "ANTHROPIC_CUSTOM_HEADERS" in text
    assert "x-modelmeld-byok-anthropic: sk-ant-test123" in text


def test_setup_noninteractive_without_key_fails_clean(isolated_home) -> None:
    # --yes (non-interactive) with no key must fail rather than prompt or write.
    rc = setup_mod.run_setup(_args(api_key=None))
    assert rc == 2
    assert not (isolated_home / ".modelmeld").exists()


def test_setup_rejects_non_https_remote_host(isolated_home) -> None:
    # A non-loopback http:// base URL must be rejected (no plaintext to remote).
    rc = setup_mod.run_setup(_args(base_url="http://gateway.example.com"))
    assert rc == 2
    assert not (isolated_home / ".modelmeld").exists()

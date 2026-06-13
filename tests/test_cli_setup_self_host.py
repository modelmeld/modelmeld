# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for `modelmeld setup --self-host`.

The self-host wizard turns the README dead-end (the documented `export
ANTHROPIC_API_KEY=…; uvicorn …` path silently lands on the no-op stub
adapter) into a real on-ramp. These tests pin the contract:

  - It writes routing_policy=capability + the supplied provider keys
    (without capability mode the keys are ignored and the gateway stays
    on stub — the exact bug this fixes).
  - It refuses to configure a keyless (stub-only) gateway: no keys → a
    non-zero exit with an on-ramp message, NOT a silent stub.
  - The smoke gate passes ONLY when a real OSS provider served the
    request (x-modelmeld-routed-to != "stub"); a stub response fails it.
  - The client env script points at the local gateway with NO BYOK
    header (self-host frontier is gateway-side).

HTTP + the transient gateway subprocess are stubbed so the test is
offline and non-destructive (Path.home redirected to tmp). The
real-key live smoke remains a manual release step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modelmeld.cli import setup as setup_mod

skip_on_windows_modes = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits not supported on Windows NTFS",
)

_MODELS_RESPONSE = {
    "data": [
        {"id": "anthropic/modelmeld-saver", "display_name": "ModelMeld — Saver"},
        {"id": "anthropic/modelmeld-auto", "display_name": "ModelMeld — Auto"},
        {"id": "anthropic/modelmeld-quality"},
    ]
}


def _args(**overrides: Any) -> SimpleNamespace:
    base = dict(
        tool="claude-code",
        self_host=True,
        demo=False,
        base_url=None,                 # resolves to http://localhost:8080
        api_key=None,
        byok_anthropic=None,
        byok_openai=None,
        openrouter_key=None,
        fireworks_key=None,
        together_key=None,
        vllm_endpoint=None,
        allow_custom_host=False,
        skip_smoke_test=False,
        yes=True,                      # non-interactive
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeProc:
    """Stand-in for the uvicorn subprocess Popen handle."""

    def __init__(self) -> None:
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0

    def kill(self) -> None:
        self._alive = False


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    routed_to: str = "openrouter",
    routed_model: str = "qwen/qwen3-coder",
    messages_status: int = 200,
) -> None:
    """Redirect home + stub the gateway boot and HTTP so the wizard runs
    fully offline. `routed_to` controls what the smoke test sees."""
    monkeypatch.setattr(Path, "home", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        setup_mod, "_boot_transient_gateway",
        lambda env: (_FakeProc(), "http://127.0.0.1:59999"),
    )
    monkeypatch.setattr(setup_mod, "_wait_for_healthz", lambda url, timeout=25.0: True)

    def _fake_get(url: str, headers: dict[str, str], timeout: float = 15.0):
        return 200, {}, json.dumps(_MODELS_RESPONSE).encode("utf-8")

    def _fake_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 30.0):
        model = body.get("model", "")
        # frontier alias → claim a frontier provider so the optional test passes
        if model.endswith("-quality"):
            return 200, {"x-modelmeld-routed-to": "anthropic",
                         "x-modelmeld-routed-model": "claude-sonnet-4-6"}, b'{"content":[]}'
        if messages_status != 200:
            return messages_status, {}, b'{"detail":"boom"}'
        reply = (b'{"content":[{"text":"stub adapter reply"}]}'
                 if routed_to == "stub" else b'{"content":[{"text":"ok"}]}')
        return 200, {"x-modelmeld-routed-to": routed_to,
                     "x-modelmeld-routed-model": routed_model}, reply

    monkeypatch.setattr(setup_mod, "_http_get", _fake_get)
    monkeypatch.setattr(setup_mod, "_http_post_json", _fake_post)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_gateway_env_always_sets_capability() -> None:
    """The whole point: keys are inert without capability mode."""
    env = setup_mod._self_host_gateway_env_vars(
        openrouter_key="sk-or-x", fireworks_key=None, together_key=None,
        vllm_endpoint=None, anthropic_key=None, openai_key=None,
    )
    assert env["MODELMELD_ROUTING_POLICY"] == "capability"
    assert env["MODELMELD_OPENROUTER_API_KEY"] == "sk-or-x"


def test_gateway_env_wires_each_provider() -> None:
    env = setup_mod._self_host_gateway_env_vars(
        openrouter_key="o", fireworks_key="f", together_key="t",
        vllm_endpoint="http://localhost:8000/v1", anthropic_key="sk-ant-a",
        openai_key="sk-o",
    )
    assert env["MODELMELD_FIREWORKS_API_KEY"] == "f"
    assert env["MODELMELD_TOGETHER_API_KEY"] == "t"
    assert env["MODELMELD_VLLM_ENDPOINT"] == "http://localhost:8000/v1"
    assert env["MODELMELD_ANTHROPIC_API_KEY"] == "sk-ant-a"
    assert env["MODELMELD_OPENAI_API_KEY"] == "sk-o"


def test_gateway_env_omits_unset_providers() -> None:
    env = setup_mod._self_host_gateway_env_vars(
        openrouter_key="o", fireworks_key=None, together_key=None,
        vllm_endpoint=None, anthropic_key=None, openai_key=None,
    )
    assert "MODELMELD_FIREWORKS_API_KEY" not in env
    assert "MODELMELD_ANTHROPIC_API_KEY" not in env


def test_write_gateway_env_is_lf_only(tmp_path: Path) -> None:
    target = tmp_path / "modelmeld-gateway.env"
    env = setup_mod._self_host_gateway_env_vars(
        openrouter_key="sk-or-x", fireworks_key=None, together_key=None,
        vllm_endpoint=None, anthropic_key=None, openai_key=None,
    )
    setup_mod._write_self_host_gateway_env(target, env, "http://localhost:8080")
    raw = target.read_bytes()
    assert b"\r" not in raw
    text = target.read_text()
    assert "export MODELMELD_ROUTING_POLICY=capability" in text
    assert "export MODELMELD_OPENROUTER_API_KEY=sk-or-x" in text
    assert "port 8080" in text


def test_arg_value_normalizes_and_blanks() -> None:
    assert setup_mod._arg_value(SimpleNamespace(k="  v\r\n"), "k") == "v"
    assert setup_mod._arg_value(SimpleNamespace(k=""), "k") is None
    assert setup_mod._arg_value(SimpleNamespace(), "missing") is None


def test_prompt_keys_noninteractive_uses_flags_only() -> None:
    args = _args(openrouter_key="sk-or-x", byok_anthropic="sk-ant-a")
    keys = setup_mod._prompt_self_host_keys(args, interactive=False)
    assert keys["openrouter"] == "sk-or-x"
    assert keys["anthropic"] == "sk-ant-a"
    assert keys["fireworks"] is None


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------


def test_no_keys_refuses_stub_and_exits_nonzero(monkeypatch, tmp_path) -> None:
    """No keys must NOT silently configure a stub gateway."""
    monkeypatch.setattr(Path, "home", lambda *a, **k: tmp_path)
    rc = setup_mod.run_setup(_args())
    assert rc == 2
    # Nothing written — we refused rather than persist a no-op config.
    assert not (tmp_path / ".modelmeld" / "modelmeld-gateway.env").exists()


def test_happy_path_writes_capability_config(monkeypatch, tmp_path) -> None:
    _install_fakes(monkeypatch, tmp_path, routed_to="openrouter")
    rc = setup_mod.run_setup(_args(openrouter_key="sk-or-x"))
    assert rc == 0

    gw = tmp_path / ".modelmeld" / "modelmeld-gateway.env"
    assert gw.exists()
    gw_text = gw.read_text()
    assert "MODELMELD_ROUTING_POLICY=capability" in gw_text
    assert "MODELMELD_OPENROUTER_API_KEY=sk-or-x" in gw_text

    client = tmp_path / ".modelmeld" / "setup-claude-code.sh"
    assert client.exists()
    client_text = client.read_text()
    assert "ANTHROPIC_BASE_URL=http://localhost:8080" in client_text
    # Self-host: NO per-request BYOK header in the client config.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in client_text

    cache = tmp_path / ".claude" / "cache" / "gateway-models.json"
    assert cache.exists()
    assert cache.read_text().count("modelmeld-") >= 3


def test_gate_fails_when_routed_to_stub(monkeypatch, tmp_path) -> None:
    """If the served provider is the stub, the gate must fail (rc=1)."""
    _install_fakes(monkeypatch, tmp_path, routed_to="stub")
    rc = setup_mod.run_setup(_args(openrouter_key="sk-or-x"))
    assert rc == 1


def test_gate_fails_on_upstream_http_error(monkeypatch, tmp_path) -> None:
    _install_fakes(monkeypatch, tmp_path, messages_status=502)
    rc = setup_mod.run_setup(_args(openrouter_key="sk-or-x"))
    assert rc == 1


def test_skip_smoke_still_writes_config(monkeypatch, tmp_path) -> None:
    _install_fakes(monkeypatch, tmp_path, routed_to="stub")  # would fail gate
    rc = setup_mod.run_setup(_args(openrouter_key="sk-or-x", skip_smoke_test=True))
    assert rc == 0  # skipping the smoke means the stub routing isn't checked
    assert (tmp_path / ".modelmeld" / "modelmeld-gateway.env").exists()


def test_frontier_key_enables_quality_smoke(monkeypatch, tmp_path) -> None:
    _install_fakes(monkeypatch, tmp_path, routed_to="openrouter")
    rc = setup_mod.run_setup(_args(openrouter_key="sk-or-x", byok_anthropic="sk-ant-a"))
    assert rc == 0
    gw_text = (tmp_path / ".modelmeld" / "modelmeld-gateway.env").read_text()
    assert "MODELMELD_ANTHROPIC_API_KEY=sk-ant-a" in gw_text


def test_frontier_only_self_host_passes_via_quality_gate(monkeypatch, tmp_path) -> None:
    """A frontier-only self-host (no OSS key) must not be blocked by the
    -saver gate — it has nothing OSS to route to. The -quality frontier
    test is the gate instead."""
    # routed_to=stub would fail an OSS gate, but with no OSS key the OSS
    # test is skipped entirely; only -quality (→ anthropic) is checked.
    _install_fakes(monkeypatch, tmp_path, routed_to="stub")
    rc = setup_mod.run_setup(_args(byok_anthropic="sk-ant-a"))
    assert rc == 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_argparse_self_host_defaults_base_url(monkeypatch, tmp_path) -> None:
    """`setup --self-host` should parse and default base_url to localhost."""
    from modelmeld.cli import main

    captured: dict[str, Any] = {}

    def _fake_run(args: Any) -> int:
        captured["base_url"] = args.base_url
        captured["self_host"] = args.self_host
        return 0

    monkeypatch.setattr("modelmeld.cli.run_setup", _fake_run)
    rc = main(["setup", "--self-host", "--openrouter-key", "sk-or-x"])
    assert rc == 0
    assert captured["self_host"] is True
    # base_url default is resolved inside run_setup, so it's None at parse time
    assert captured["base_url"] is None

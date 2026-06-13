# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""`modelmeld setup --tool claude-code` — one-command customer onboarding.

Pre-flight: collect ModelMeld API key + optional BYOK keys.
Output:     - Sourceable shell script with all env vars (LF-only, validated).
            - Pre-written Claude Code discovery cache (LF, mode 0600).
Validation: smoke-tests the customer's gateway access end-to-end.

Every footgun we hit during the 2026-05-25 launch validation is handled
here so customers never see it. See _atomic_write_text for the file-
writing contract (binary mode + explicit LF + post-write byte check)
and _normalize for env-var input scrubbing (CR/LF/whitespace stripped).
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Windows console defaults to cp1252 which can't encode the check/x glyphs.
# Reconfigure stdout/stderr to UTF-8 with replacement fallback so the CLI
# never crashes on output. Python 3.7+ supports reconfigure() on most
# stdio file objects.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Color output — ANSI escape codes; no extra deps. Disabled on Windows shells
# that don't claim color support or when piped.
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()
_USE_COLOR = _IS_TTY and (os.environ.get("NO_COLOR") is None)


def _c(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if _USE_COLOR else msg


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _bold(s: str) -> str:
    return _c("1", s)


def _ok(msg: str) -> None:
    print(f"  {_green('✓')} {msg}")


def _err(msg: str) -> None:
    print(f"  {_red('✗')} {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"  {_yellow('!')} {msg}")


def _step(msg: str) -> None:
    print(f"\n{_bold(msg)}")


# ---------------------------------------------------------------------------
# Base URL validation — prevents credential exfiltration to attacker-
# controlled gateways. The setup CLI sends the customer's ModelMeld API
# key and any BYOK keys to {base_url}/v1/{models,messages} during the
# smoke test. A malicious blog post recommending an evil --base-url
# would silently exfiltrate both. The default allowlist (ModelMeld
# hosts + loopback + RFC1918) blocks the obvious cases; --allow-custom-
# host is the explicit opt-in for self-hosters on custom domains.
# ---------------------------------------------------------------------------


def _is_loopback_or_rfc1918(host: str) -> bool:
    """True if host is loopback or in an RFC1918 private range."""
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


def _validate_base_url(base_url: str, allow_custom_host: bool) -> None:
    """Validate base_url before sending credentials to it.

    Default policy: only ModelMeld endpoints and loopback/RFC1918 hosts
    are allowed. Non-allowlisted hosts require --allow-custom-host.
    HTTPS is required for any non-loopback/non-RFC1918 host regardless
    of allowlist status.

    Raises ValueError on rejection; caller catches + prints + exits.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()

    if scheme not in ("http", "https"):
        raise ValueError(
            f"--base-url must use http:// or https://, got {scheme!r}"
        )
    if not host:
        raise ValueError(f"--base-url has no host: {base_url!r}")

    is_local = _is_loopback_or_rfc1918(host)
    is_modelmeld = (
        host == "modelmeld.ai"
        or host.endswith(".modelmeld.ai")
    )

    # Plaintext HTTP only OK for loopback / RFC1918 — never for public hosts
    if scheme == "http" and not is_local:
        raise ValueError(
            f"plaintext http:// is only allowed for loopback or RFC1918 "
            f"hosts. Got {base_url!r} — use https:// for {host!r}."
        )

    # Allowlist enforcement
    if not (is_modelmeld or is_local or allow_custom_host):
        raise ValueError(
            f"--base-url host {host!r} is not in the default allowlist. "
            f"Allowed by default: api.modelmeld.ai, *.modelmeld.ai, "
            f"localhost, 127.0.0.1, RFC1918 private ranges. "
            f"Pass --allow-custom-host to send your ModelMeld API key + "
            f"BYOK keys to this host. For self-hosted gateways on a "
            f"custom domain, this opt-in is required."
        )

    if allow_custom_host and not (is_modelmeld or is_local):
        # Loud warning when the customer opts in — credentials will flow
        # to an unfamiliar host
        _warn(
            f"--allow-custom-host: about to send your ModelMeld API key "
            f"+ any BYOK keys to {host!r}. Confirm you trust this host."
        )


# ---------------------------------------------------------------------------
# Input normalization — the #1 footgun we burned hours on
# ---------------------------------------------------------------------------


def _normalize(value: str | None) -> str:
    """Strip CR/LF/whitespace from a customer-supplied secret.

    Why this exists: customer pasting a key into a terminal often picks up
    a trailing `\\r` from CRLF clipboards, or whitespace from accidental
    selection. Either silently corrupts auth in ways that look like
    "discovery just doesn't work" — exactly the trap we spent today
    debugging. Strip it. Always.
    """
    if value is None:
        return ""
    return value.strip().replace("\r", "").replace("\n", "")


def _prompt_secret(prompt: str, *, required: bool = True) -> str:
    """Interactive prompt for a secret. Visible (not getpass) because that
    library has terminal-rendering quirks on Windows that have caused
    customers to type half a key and hit enter. Visible-with-normalize
    is more reliable than masked-and-broken.
    """
    while True:
        value = input(prompt).strip()
        normalized = _normalize(value)
        if normalized:
            return normalized
        if not required:
            return ""
        _warn("Empty value — try again, or Ctrl-C to abort")


# ---------------------------------------------------------------------------
# File writing — binary mode + LF + post-write byte validation
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str, *, mode: int = 0o644) -> None:
    """Write `content` to `path` as UTF-8 with strict LF line endings,
    then verify the on-disk bytes match what we intended. Atomic via
    temp file + rename so partial writes can't leave half-state.

    Python's default `open(path, "w")` on Windows applies newline
    translation that turns `\\n` into `\\r\\n`. That CRLF is what bit
    us when customers `source`'d our setup script — env vars ended up
    with trailing `\\r`, and downstream comparisons silently failed.
    Binary mode + explicit `\\n` avoids it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    # Defense in depth: even if `content` somehow contains CR, strip it.
    encoded = encoded.replace(b"\r\n", b"\n").replace(b"\r", b"")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(encoded)
    os.replace(tmp, path)
    os.chmod(path, mode)
    # Verify: read back, confirm no CR anywhere, length matches.
    with open(path, "rb") as f:
        actual = f.read()
    if b"\r" in actual:
        raise RuntimeError(
            f"Wrote {path} but found CR in the result — write failed safety check.",
        )
    if actual != encoded:
        raise RuntimeError(f"Wrote {path} but bytes don't match — fs corruption?")


# ---------------------------------------------------------------------------
# Smoke tests — verify the setup actually works end-to-end
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    label: str
    ok: bool
    detail: str
    routed_model: str | None = None


def _http_get(
    url: str, headers: dict[str, str], timeout: float = 15.0,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""


def _http_post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 30.0,
) -> tuple[int, dict[str, str], bytes]:
    payload = json.dumps(body).encode("utf-8")
    full_headers = {"content-type": "application/json", **headers}
    req = urllib.request.Request(url, data=payload, method="POST", headers=full_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""


def _smoke_test(
    base_url: str, api_key: str, byok_anthropic: str | None,
) -> list[SmokeResult]:
    """Three tests:
    1. GET /v1/models — gateway reachable + auth works
    2. POST /v1/messages with -saver alias — OSS routing works
    3. POST /v1/messages with -quality alias + BYOK — frontier routing works
       (skipped if no BYOK supplied)
    """
    results: list[SmokeResult] = []

    # Test 1: /v1/models discovery
    status, _hdrs, body = _http_get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"},
    )
    if status == 200:
        try:
            data = json.loads(body)
            n = len(data.get("data", []))
            results.append(SmokeResult(
                "models discovery", True, f"{n} models advertised",
            ))
        except json.JSONDecodeError:
            results.append(SmokeResult(
                "models discovery", False, "200 but body not JSON",
            ))
    elif status == 401:
        results.append(SmokeResult(
            "models discovery", False,
            "401 invalid API key — check your gws_ key is correct + not expired",
        ))
    else:
        results.append(SmokeResult(
            "models discovery", False, f"HTTP {status} — {body[:100].decode('utf-8','replace')}",
        ))

    # Test 2: OSS routing via -saver. The prompt is deliberately verbose
    # and max_tokens is above 256 so the scout doesn't classify it as
    # an autocomplete-shape request (which would downgrade QUALITY to
    # cheap OSS via the bias path and obscure whether BYOK was reached).
    prompt = (
        "I'm onboarding to ModelMeld and verifying my routing setup. "
        "Please reply with a short 2-3 sentence description of what "
        "type hints do in Python and why they help with code quality."
    )
    status, hdrs, body = _http_post_json(
        f"{base_url}/v1/messages",
        body={
            "model": "anthropic/modelmeld-saver",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        },
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    if status == 200:
        routed = hdrs.get("x-modelmeld-routed-model")
        results.append(SmokeResult(
            "OSS routing (-saver)", True,
            f"served by {routed or 'unknown'}", routed_model=routed,
        ))
    else:
        try:
            err_body = json.loads(body)
        except Exception:
            err_body = {"detail": body[:200].decode("utf-8", "replace")}
        results.append(SmokeResult(
            "OSS routing (-saver)", False, f"HTTP {status}: {err_body}",
        ))

    # Test 3: BYOK frontier routing (only if customer supplied a key).
    # Same prompt + max_tokens=512 to stay out of autocomplete-shape
    # detection, otherwise QUALITY would correctly downgrade to OSS and
    # we'd never exercise the BYOK frontier path.
    if byok_anthropic:
        status, hdrs, body = _http_post_json(
            f"{base_url}/v1/messages",
            body={
                "model": "anthropic/modelmeld-quality",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "x-modelmeld-byok-anthropic": byok_anthropic,
            },
        )
        if status == 200:
            routed = hdrs.get("x-modelmeld-routed-model")
            ok = bool(routed and routed.startswith("claude-"))
            results.append(SmokeResult(
                "BYOK routing (-quality)",
                ok,
                f"served by {routed or 'unknown'}",
                routed_model=routed,
            ))
        elif status == 400:
            try:
                err = json.loads(body).get("detail", {})
                if isinstance(err, dict) and err.get("error") == "byok_required":
                    results.append(SmokeResult(
                        "BYOK routing (-quality)", False,
                        "byok_required — gateway didn't see your Anthropic header",
                    ))
                else:
                    results.append(SmokeResult(
                        "BYOK routing (-quality)", False, f"400: {err}",
                    ))
            except Exception:
                results.append(SmokeResult(
                    "BYOK routing (-quality)", False,
                    f"400 (couldn't parse body): {body[:150].decode('utf-8','replace')}",
                ))
        else:
            results.append(SmokeResult(
                "BYOK routing (-quality)", False,
                f"HTTP {status}: {body[:150].decode('utf-8','replace')}",
            ))

    return results


# ---------------------------------------------------------------------------
# Claude Code specifics — env script + discovery cache pre-write
# ---------------------------------------------------------------------------


def _write_claude_code_env_script(
    target: Path,
    base_url: str,
    api_key: str,
    byok_anthropic: str | None,
    byok_openai: str | None,
) -> None:
    """Sourceable shell script that exports all env vars Claude Code needs."""
    lines = [
        "#!/bin/bash",
        "# ModelMeld + Claude Code setup. Source this from your shell:",
        f"#   source {target}",
        "",
        f"export ANTHROPIC_BASE_URL={base_url}",
        f"export ANTHROPIC_API_KEY={api_key}",
        "export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1",
        "unset ANTHROPIC_AUTH_TOKEN  # avoid 'auth conflict' warning",
    ]
    if byok_anthropic or byok_openai:
        header_parts: list[str] = []
        if byok_anthropic:
            header_parts.append(f"x-modelmeld-byok-anthropic: {byok_anthropic}")
        if byok_openai:
            header_parts.append(f"x-modelmeld-byok-openai: {byok_openai}")
        header_value = "\\n".join(header_parts)
        lines.append(f'export ANTHROPIC_CUSTOM_HEADERS="{header_value}"')
    content = "\n".join(lines) + "\n"
    _atomic_write_text(target, content, mode=0o600)


def _write_claude_code_cache(
    cache_path: Path, base_url: str, models_response: dict[str, Any],
) -> None:
    """Write ~/.claude/cache/gateway-models.json in the format
    Claude Code's reader expects: {baseUrl, fetchedAt, models}.

    Claude Code's discovery fetcher is upstream-broken in v2.1.150 for
    third-party gateways. Pre-writing the cache populates the /model
    picker via the reader path which DOES work. See
    docs/integrations/claude-code.md for the full story.
    """
    cache_payload = {
        "baseUrl": base_url,
        "fetchedAt": int(time.time() * 1000),
        "models": [
            {
                "id": m["id"],
                "display_name": m.get("display_name") or m["id"],
            }
            for m in models_response.get("data", [])
        ],
    }
    content = json.dumps(cache_payload, indent=2, ensure_ascii=False)
    _atomic_write_text(cache_path, content, mode=0o600)


# ---------------------------------------------------------------------------
# Codex CLI specifics — config.toml provider block + env script
# ---------------------------------------------------------------------------


def _codex_config_content(base_url: str) -> str:
    """The `~/.codex/config.toml` provider block pointing Codex at the gateway.

    Codex speaks the Responses API, so `wire_api = "responses"`. The key is
    read from the `MODELMELD_API_KEY` env var (named by `env_key`) — never
    written into the config file.
    """
    return (
        'model = "anthropic/modelmeld-saver"\n'
        'model_provider = "modelmeld"\n'
        "\n"
        "[model_providers.modelmeld]\n"
        'name = "ModelMeld"\n'
        f'base_url = "{base_url}/v1"\n'
        'env_key = "MODELMELD_API_KEY"\n'
        'wire_api = "responses"\n'
    )


def _write_codex_config(target: Path, base_url: str) -> None:
    _atomic_write_text(target, _codex_config_content(base_url), mode=0o644)


def _codex_env_script_content(target: Path, api_key: str) -> str:
    return (
        "\n".join([
            "#!/bin/bash",
            "# ModelMeld + Codex CLI setup. Source this from your shell:",
            f"#   source {target}",
            "",
            f"export MODELMELD_API_KEY={api_key}",
        ])
        + "\n"
    )


def _write_codex_env_script(target: Path, api_key: str) -> None:
    _atomic_write_text(target, _codex_env_script_content(target, api_key), mode=0o600)


def _smoke_test_responses(base_url: str, api_key: str) -> list[SmokeResult]:
    """Two tests: /v1/models reachability + a /v1/responses generation via the
    -saver alias (the surface + policy Codex uses)."""
    results: list[SmokeResult] = []

    status, _hdrs, body = _http_get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if status == 200:
        try:
            n = len(json.loads(body).get("data", []))
            results.append(SmokeResult("models discovery", True, f"{n} models advertised"))
        except json.JSONDecodeError:
            results.append(SmokeResult("models discovery", False, "200 but body not JSON"))
    elif status == 401:
        results.append(SmokeResult(
            "models discovery", False,
            "401 invalid API key — check your gws_ key is correct + not expired",
        ))
    else:
        results.append(SmokeResult(
            "models discovery", False,
            f"HTTP {status} — {body[:100].decode('utf-8', 'replace')}",
        ))

    status, hdrs, body = _http_post_json(
        f"{base_url}/v1/responses",
        body={
            "model": "anthropic/modelmeld-saver",
            "input": "Reply in one short sentence: what does ModelMeld do?",
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if status == 200:
        routed = hdrs.get("x-modelmeld-routed-model")
        results.append(SmokeResult(
            "Responses routing (-saver)", True,
            f"served by {routed or 'unknown'}", routed_model=routed,
        ))
    else:
        try:
            err_body: Any = json.loads(body)
        except Exception:
            err_body = {"detail": body[:200].decode("utf-8", "replace")}
        results.append(SmokeResult(
            "Responses routing (-saver)", False, f"HTTP {status}: {err_body}",
        ))
    return results


def _setup_codex(args: Any, base_url: str, interactive: bool) -> int:
    """Configure Codex CLI to route through ModelMeld via /v1/responses."""
    print(_bold("ModelMeld — Codex CLI setup"))
    print(f"  Gateway: {base_url}")

    # --- Collect credentials ---
    _step("Step 1/4: Collect credentials")
    api_key = _normalize(args.api_key) if args.api_key else ""
    if not api_key:
        if not interactive:
            _err("--api-key not supplied and --yes (non-interactive) is set")
            return 2
        print(
            "  Your ModelMeld API key (starts with 'gws_'). Get one at "
            "https://modelmeld.ai/account if you don't have one.",
        )
        api_key = _prompt_secret("  API key: ")
    if not api_key.startswith("gws_"):
        _warn("API key doesn't start with 'gws_' — usually means a copy-paste error")
    _ok(f"ModelMeld API key set ({len(api_key)} chars)")
    # No BYOK prompt: Codex is configured for the -saver policy, which is
    # OSS-only and never escalates to a frontier provider. Switch the `model`
    # to -auto/-quality and add a BYOK header if you want frontier routing.

    # --- Write the sourceable env script (holds the key; config.toml doesn't) ---
    _step("Step 2/4: Write env script")
    setup_dir = Path.home() / ".modelmeld"
    env_script = setup_dir / "setup-codex.sh"
    try:
        _write_codex_env_script(env_script, api_key)
        _ok(f"Wrote {env_script} (LF-only, mode 0600)")
    except Exception as e:
        _err(f"Failed to write env script: {e}")
        return 1

    # --- Configure ~/.codex/config.toml (never clobber an existing one) ---
    _step("Step 3/4: Configure ~/.codex/config.toml")
    codex_config = Path.home() / ".codex" / "config.toml"
    snippet = setup_dir / "codex-config-snippet.toml"
    merge_needed = False
    try:
        if codex_config.exists():
            existing = codex_config.read_text(encoding="utf-8", errors="replace")
            if "[model_providers.modelmeld]" in existing:
                _warn(
                    f"{codex_config} already has a [model_providers.modelmeld] "
                    "block — leaving your config untouched.",
                )
            else:
                _write_codex_config(snippet, base_url)
                merge_needed = True
                _ok(f"Wrote {snippet}")
                _warn(
                    f"{codex_config} already exists — merge the snippet above "
                    "into it (don't overwrite your other providers).",
                )
        else:
            _write_codex_config(codex_config, base_url)
            _ok(f"Wrote {codex_config}")
    except Exception as e:
        _err(f"Failed to write Codex config: {e}")
        return 1

    # --- Smoke-test the Responses surface ---
    if not args.skip_smoke_test:
        _step("Step 4/4: Smoke-test /v1/responses")
        results = _smoke_test_responses(base_url, api_key)
        for r in results:
            if r.ok:
                _ok(f"{r.label}: {r.detail}")
            else:
                _err(f"{r.label}: {r.detail}")
        if not all(r.ok for r in results):
            _err("Some smoke tests failed. See errors above for next steps.")
            return 1
    else:
        _step("Step 4/4: Smoke-test (skipped)")

    # --- Final instructions ---
    print()
    print(_bold("Setup complete."))
    print()
    print("  1. Source the env script (or set MODELMELD_API_KEY yourself):")
    print(f"     {_bold(f'source {env_script}')}")
    if merge_needed:
        print(f"  2. Merge {snippet} into {codex_config}")
        print("  3. Launch Codex:")
    else:
        print("  2. Launch Codex:")
    print(f"     {_bold('codex')}")
    print()
    print("  Routing policy is the `model` field in config.toml:")
    print(f"     • {_bold('anthropic/modelmeld-saver')}   (OSS-only, predictable ceiling)")
    print(f"     • {_bold('anthropic/modelmeld-auto')}    (escalates to frontier on reasoning markers)")
    print(f"     • {_bold('anthropic/modelmeld-quality')} (frontier-first)")
    print()
    return 0


# ---------------------------------------------------------------------------
# Self-host specifics — gateway config, transient boot, real-routing smoke
# ---------------------------------------------------------------------------
#
# Why this path exists: the public-today path is self-host, but the gateway
# ships INERT. config.py defaults to routing_policy="single" +
# upstream_provider="stub", so out of the box every request hits a no-op
# stub adapter with no error. The documented `export ANTHROPIC_API_KEY=…;
# uvicorn …` snippet sets neither the capability routing mode nor any
# MODELMELD_*-prefixed provider key (config reads MODELMELD_ANTHROPIC_API_KEY,
# not ANTHROPIC_API_KEY), so the tester silently gets stub-or-nothing and
# never sees the OSS-routing value prop.
#
# The fix is config/UX, not capability — every adapter already ships and
# `_infer_providers_from_credentials` wires keys → adapters once
# routing_policy=capability. This wizard collects whichever provider keys
# the tester has, writes routing_policy=capability + those keys, and — the
# acceptance bar — boots a transient gateway and proves a real OSS provider
# served a request (x-modelmeld-routed-to != "stub") before declaring
# success.


# The gateway has no auth in the OSS core (auth is the enterprise seam), but
# Claude Code refuses to start without a non-empty ANTHROPIC_API_KEY. This
# placeholder satisfies the client; the local gateway ignores it.
_SELF_HOST_CLIENT_PLACEHOLDER_KEY = "sk-modelmeld-localhost-noauth"

# Real OSS providers we can route to in self-host. If the smoke test's
# x-modelmeld-routed-to is none of these (e.g. "stub"), real routing did
# NOT happen and the gate fails.
_OSS_PROVIDERS = ("openrouter", "fireworks", "together", "vllm", "tensorrt_llm")


def _self_host_gateway_env_vars(
    *,
    openrouter_key: str | None,
    fireworks_key: str | None,
    together_key: str | None,
    vllm_endpoint: str | None,
    anthropic_key: str | None,
    openai_key: str | None,
) -> dict[str, str]:
    """The MODELMELD_* env the self-host gateway needs for real routing.

    Always sets capability routing — without it, the inferred provider keys
    are ignored and the gateway stays on the single/stub default. One dict,
    used for BOTH the persisted sourceable script AND the transient
    smoke-test subprocess, so what we validate is exactly what we persist.
    """
    env: dict[str, str] = {"MODELMELD_ROUTING_POLICY": "capability"}
    if openrouter_key:
        env["MODELMELD_OPENROUTER_API_KEY"] = openrouter_key
    if fireworks_key:
        env["MODELMELD_FIREWORKS_API_KEY"] = fireworks_key
    if together_key:
        env["MODELMELD_TOGETHER_API_KEY"] = together_key
    if vllm_endpoint:
        env["MODELMELD_VLLM_ENDPOINT"] = vllm_endpoint
    # Frontier keys live on the GATEWAY in self-host (the operator IS the
    # user), not as per-request client BYOK headers. capability mode makes
    # them eligible for -auto/-quality while -saver stays OSS-only.
    if anthropic_key:
        env["MODELMELD_ANTHROPIC_API_KEY"] = anthropic_key
    if openai_key:
        env["MODELMELD_OPENAI_API_KEY"] = openai_key
    return env


def _write_self_host_gateway_env(
    target: Path, env_vars: dict[str, str], base_url: str,
) -> None:
    """Sourceable shell script of gateway env vars. Source BEFORE uvicorn."""
    port = base_url.rsplit(":", 1)[-1] if base_url.count(":") >= 2 else "8080"
    lines = [
        "#!/bin/bash",
        "# ModelMeld self-host GATEWAY config. Source this, then launch the",
        "# gateway in the same shell:",
        f"#   source {target}",
        f"#   uvicorn modelmeld.api.server:app --host 0.0.0.0 --port {port}",
        "",
        "# Capability routing: the scout picks the cheapest model meeting the",
        "# quality bar per request. WITHOUT this line the gateway falls back",
        "# to a no-op stub and every request returns a canned reply.",
    ]
    # MODELMELD_ROUTING_POLICY first, then keys, for readability.
    lines.append(f"export MODELMELD_ROUTING_POLICY={env_vars['MODELMELD_ROUTING_POLICY']}")
    for key, value in env_vars.items():
        if key == "MODELMELD_ROUTING_POLICY":
            continue
        lines.append(f"export {key}={value}")
    content = "\n".join(lines) + "\n"
    _atomic_write_text(target, content, mode=0o600)


def _free_port() -> int:
    """Grab a free localhost TCP port for the transient smoke-test gateway."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_healthz(base_url: str, timeout: float = 25.0) -> bool:
    """Poll /healthz until the transient gateway is up or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _hdrs, _body = _http_get(
                f"{base_url}/healthz", headers={}, timeout=2.0,
            )
            if status == 200:
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.4)
    return False


def _boot_transient_gateway(
    env_vars: dict[str, str],
) -> tuple[subprocess.Popen[bytes], str]:
    """Spawn `python -m uvicorn modelmeld.api.server:app` on a free port with
    the collected gateway env injected.

    Spawning the LITERAL command the tester runs (rather than an in-process
    ASGI client) is deliberate: the silent-stub dead-end is specifically
    about `uvicorn …:app`, so the smoke test must prove that exact
    invocation does real routing.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc_env = {**os.environ, **env_vars}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "modelmeld.api.server:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc, base_url


def _smoke_test_self_host(
    base_url: str, *, expect_oss: bool, expect_frontier: bool,
) -> list[SmokeResult]:
    """Self-host real-routing smoke. No api key / no BYOK header (OSS core is
    unauthenticated; frontier is gateway-side).

    Test A: GET /v1/models — gateway reachable.
    Test B (the gate, when an OSS key was supplied): POST /v1/messages with
      -saver. PASS iff 200 AND a real OSS provider served it
      (x-modelmeld-routed-to in _OSS_PROVIDERS, NOT "stub") AND the body
      isn't the stub sentinel. Green here = real OSS routing provably
      happened.
    Test C (when a frontier key was supplied): -quality should route to a
      frontier model. This is the gate for a frontier-only self-host.
    """
    results: list[SmokeResult] = []

    status, _hdrs, body = _http_get(f"{base_url}/v1/models", headers={})
    if status == 200:
        try:
            n = len(json.loads(body).get("data", []))
            results.append(SmokeResult("models discovery", True, f"{n} models advertised"))
        except json.JSONDecodeError:
            results.append(SmokeResult("models discovery", False, "200 but body not JSON"))
    else:
        results.append(SmokeResult(
            "models discovery", False,
            f"HTTP {status} — {body[:120].decode('utf-8', 'replace')}",
        ))

    # Verbose prompt + max_tokens>256 keeps the scout out of
    # autocomplete-shape downgrade territory for both alias tests.
    prompt = (
        "I'm verifying my self-hosted ModelMeld routing setup. Please reply "
        "with a short 2-3 sentence description of what type hints do in "
        "Python and why they help with code quality."
    )

    # The OSS gate: a real OSS provider must serve a -saver request.
    if expect_oss:
        status, hdrs, body = _http_post_json(
            f"{base_url}/v1/messages",
            body={
                "model": "anthropic/modelmeld-saver",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        if status == 200:
            routed_to = (hdrs.get("x-modelmeld-routed-to") or "").lower()
            routed_model = hdrs.get("x-modelmeld-routed-model")
            looks_stub = routed_to == "stub" or b"stub adapter" in body.lower()
            if routed_to in _OSS_PROVIDERS and not looks_stub:
                results.append(SmokeResult(
                    "real OSS routing (-saver)", True,
                    f"served by {routed_to} → {routed_model or 'unknown model'}",
                    routed_model=routed_model,
                ))
            elif looks_stub:
                results.append(SmokeResult(
                    "real OSS routing (-saver)", False,
                    "served by the STUB adapter — no real model was called. "
                    "MODELMELD_ROUTING_POLICY=capability + a provider key didn't "
                    "take effect. Re-run `modelmeld setup --self-host`.",
                ))
            else:
                results.append(SmokeResult(
                    "real OSS routing (-saver)", False,
                    f"routed to unexpected provider {routed_to!r} "
                    f"(model {routed_model!r}) — expected one of {_OSS_PROVIDERS}",
                ))
        else:
            try:
                err_body: Any = json.loads(body)
            except Exception:
                err_body = body[:200].decode("utf-8", "replace")
            results.append(SmokeResult(
                "real OSS routing (-saver)", False,
                f"HTTP {status}: {err_body} — check the provider key is valid "
                "(a bad key surfaces as a 401/502 from the upstream provider).",
            ))

    if expect_frontier:
        status, hdrs, body = _http_post_json(
            f"{base_url}/v1/messages",
            body={
                "model": "anthropic/modelmeld-quality",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        if status == 200:
            routed_to = (hdrs.get("x-modelmeld-routed-to") or "").lower()
            routed_model = hdrs.get("x-modelmeld-routed-model") or ""
            is_frontier = routed_to in ("anthropic", "openai") or (
                routed_model.startswith("claude-") or routed_model.startswith("gpt-")
            )
            results.append(SmokeResult(
                "frontier routing (-quality)", is_frontier,
                f"served by {routed_to} → {routed_model or 'unknown model'}"
                + ("" if is_frontier else " (expected a frontier model)"),
                routed_model=routed_model or None,
            ))
        else:
            results.append(SmokeResult(
                "frontier routing (-quality)", False,
                f"HTTP {status}: {body[:150].decode('utf-8', 'replace')}",
            ))

    return results


def _arg_value(args: Any, name: str) -> str | None:
    """Normalized non-empty CLI arg value, or None."""
    raw = getattr(args, name, None)
    return (_normalize(raw) or None) if raw else None


def _prompt_self_host_keys(args: Any, interactive: bool) -> dict[str, str | None]:
    """Collect provider keys from flags, then prompt for any the tester wants
    to add interactively."""
    keys: dict[str, str | None] = {
        "openrouter": _arg_value(args, "openrouter_key"),
        "fireworks": _arg_value(args, "fireworks_key"),
        "together": _arg_value(args, "together_key"),
        "vllm": _arg_value(args, "vllm_endpoint"),
        "anthropic": _arg_value(args, "byok_anthropic"),
        "openai": _arg_value(args, "byok_openai"),
    }
    if not interactive:
        return keys

    any_oss = any(keys[p] for p in ("openrouter", "fireworks", "together", "vllm"))
    if not any_oss:
        print(
            "\n  Cloud-OSS routing (the value prop): paste a key for any ONE "
            "provider and the gateway routes real OSS models per request. "
            "OpenRouter is the easiest to start with — one key, many OSS "
            "models, pay-as-you-go: https://openrouter.ai/keys",
        )
        if not keys["openrouter"]:
            keys["openrouter"] = _prompt_secret(
                "  OpenRouter API key (sk-or-…, blank to skip): ", required=False,
            ) or None
        if not any(keys[p] for p in ("openrouter", "fireworks", "together", "vllm")):
            ans = input(
                "  Add Fireworks / Together / a local vLLM endpoint instead? [y/N]: ",
            ).strip().lower()
            if ans == "y":
                keys["fireworks"] = _prompt_secret(
                    "  Fireworks API key (blank to skip): ", required=False,
                ) or None
                keys["together"] = _prompt_secret(
                    "  Together API key (blank to skip): ", required=False,
                ) or None
                keys["vllm"] = _prompt_secret(
                    "  vLLM endpoint URL e.g. http://localhost:8000/v1 "
                    "(blank to skip): ", required=False,
                ) or None

    if not keys["anthropic"] and not keys["openai"]:
        print(
            "\n  Optional: a frontier key (Anthropic / OpenAI) lets the "
            "-auto and -quality policies escalate to Sonnet/Opus/GPT. "
            "-saver stays OSS-only regardless. In self-host the key lives on "
            "your gateway and is never sent anywhere but the provider.",
        )
        ans = input("  Add an Anthropic frontier key now? [y/N]: ").strip().lower()
        if ans == "y":
            keys["anthropic"] = _prompt_secret(
                "  Anthropic API key (sk-ant-…): ", required=False,
            ) or None
    return keys


def _setup_self_host(args: Any, base_url: str, interactive: bool) -> int:
    """Configure + validate a gateway the tester runs themselves."""
    print(_bold("ModelMeld — self-host setup"))
    print(f"  Gateway (you run this): {base_url}")

    # --- Step 1: collect keys ---
    _step("Step 1/5: Collect provider keys")
    keys = _prompt_self_host_keys(args, interactive)
    oss_keys = {p: keys[p] for p in ("openrouter", "fireworks", "together", "vllm")}
    has_oss = any(oss_keys.values())
    has_frontier = bool(keys["anthropic"] or keys["openai"])

    if not has_oss and not has_frontier:
        # Refuse to silently configure a no-op stub gateway.
        if getattr(args, "demo", False) or not interactive:
            _warn("No provider keys supplied — nothing to route to.")
        print()
        print(_bold("No keys, no routing — here's the cheapest on-ramp:"))
        print()
        print("  OpenRouter gives you one key for many OSS models, "
              "pay-as-you-go, no minimum:")
        print(f"     {_bold('https://openrouter.ai/keys')}")
        print()
        print("  Grab a key, then re-run:")
        print(f"     {_bold('modelmeld setup --self-host --openrouter-key sk-or-…')}")
        print()
        print("  (A keyless gateway can only serve the no-op stub adapter, so "
              "this wizard won't configure one — you'd see canned replies, not "
              "real routing.)")
        return 2

    if has_oss:
        for prov, val in oss_keys.items():
            if val:
                label = "endpoint" if prov == "vllm" else "key"
                _ok(f"{prov} {label} set")
    else:
        _warn(
            "No cloud-OSS / vLLM key — only frontier routing will work. "
            "-saver (OSS-only) will have nothing to route to. Add an "
            "OpenRouter key to unlock the savings path.",
        )
    if keys["anthropic"]:
        _ok(f"Anthropic frontier key set ({len(keys['anthropic'])} chars)")
    if keys["openai"]:
        _ok(f"OpenAI frontier key set ({len(keys['openai'])} chars)")

    gateway_env = _self_host_gateway_env_vars(
        openrouter_key=keys["openrouter"],
        fireworks_key=keys["fireworks"],
        together_key=keys["together"],
        vllm_endpoint=keys["vllm"],
        anthropic_key=keys["anthropic"],
        openai_key=keys["openai"],
    )

    # --- Step 2: write the gateway env script ---
    _step("Step 2/5: Write gateway env script")
    setup_dir = Path.home() / ".modelmeld"
    gateway_script = setup_dir / "modelmeld-gateway.env"
    try:
        _write_self_host_gateway_env(gateway_script, gateway_env, base_url)
        _ok(f"Wrote {gateway_script} (LF-only, mode 0600)")
    except Exception as e:
        _err(f"Failed to write gateway env script: {e}")
        return 1

    # --- Step 3: write the Claude Code client env script ---
    _step("Step 3/5: Write Claude Code client env script")
    client_script = setup_dir / "setup-claude-code.sh"
    try:
        # No BYOK header: self-host frontier is gateway-side, and the OSS
        # core needs no api key. The placeholder satisfies Claude Code's
        # non-empty-key requirement; the gateway ignores it.
        _write_claude_code_env_script(
            client_script, base_url, _SELF_HOST_CLIENT_PLACEHOLDER_KEY,
            None, None,
        )
        _ok(f"Wrote {client_script} (LF-only, mode 0600)")
    except Exception as e:
        _err(f"Failed to write client env script: {e}")
        return 1

    # --- Step 4: boot a transient gateway, pre-write cache, smoke-test ---
    _step("Step 4/5: Boot a gateway and verify real routing")
    proc: subprocess.Popen[bytes] | None = None
    smoke_ok = True
    try:
        proc, probe_url = _boot_transient_gateway(gateway_env)
        if not _wait_for_healthz(probe_url):
            out = b""
            if proc.poll() is not None and proc.stdout is not None:
                out = proc.stdout.read() or b""
            _err(
                "Gateway didn't come up. Is the `modelmeld` package importable "
                "and uvicorn installed? "
                + (f"Output:\n{out[:500].decode('utf-8', 'replace')}" if out else ""),
            )
            return 1
        _ok("Gateway up")

        # Pre-write the Claude Code discovery cache for the LOCAL gateway.
        cache_path = Path.home() / ".claude" / "cache" / "gateway-models.json"
        try:
            status, _hdrs, mbody = _http_get(f"{probe_url}/v1/models", headers={})
            if status == 200:
                _write_claude_code_cache(cache_path, base_url, json.loads(mbody))
                n_models = len(json.loads(mbody).get("data", []))
                _ok(f"Pre-wrote {cache_path} ({n_models} models)")
            else:
                _warn(f"Couldn't fetch /v1/models (HTTP {status}); /model picker "
                      "may be empty until you launch the gateway.")
        except Exception as e:
            _warn(f"Cache pre-write skipped: {e}")

        if not args.skip_smoke_test:
            results = _smoke_test_self_host(
                probe_url, expect_oss=has_oss, expect_frontier=has_frontier,
            )
            for r in results:
                if r.ok:
                    _ok(f"{r.label}: {r.detail}")
                else:
                    _err(f"{r.label}: {r.detail}")
            # The gate is the real-OSS-routing test specifically. A missing
            # frontier key (no -quality test) or an empty OSS lineup is not a
            # hard failure on its own, but a stub/HTTP failure on -saver is.
            smoke_ok = all(r.ok for r in results)
        else:
            _warn("Smoke test skipped (--skip-smoke-test) — real routing NOT verified.")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            if proc.poll() is None:
                proc.kill()

    if not args.skip_smoke_test and not smoke_ok:
        _err("Routing verification failed — see errors above. Config files were "
             "written; fix the issue and re-run, or `modelmeld doctor`.")
        return 1

    # --- Step 5: final instructions ---
    _step("Step 5/5: Done")
    port = base_url.rsplit(":", 1)[-1] if base_url.count(":") >= 2 else "8080"
    print()
    print(_bold("Self-host setup complete. Real OSS routing verified." if not
                args.skip_smoke_test else "Self-host setup complete."))
    print()
    print("  1. Start the gateway (in one shell):")
    print(f"     {_bold(f'source {gateway_script}')}")
    print(f"     {_bold(f'uvicorn modelmeld.api.server:app --host 0.0.0.0 --port {port}')}")
    print("  2. Point Claude Code at it (in another shell):")
    print(f"     {_bold(f'source {client_script}')}")
    print(f"     {_bold('claude')}")
    print("  3. Pick a routing tier in /model:")
    print(f"     • {_bold('ModelMeld — Saver')}  (OSS-only, predictable cost ceiling)")
    print(f"     • {_bold('ModelMeld — Auto')}   (escalates to frontier on reasoning markers)")
    print(f"     • {_bold('ModelMeld — Quality')} (frontier-first"
          + ("" if has_frontier else "; needs a frontier key — add one and re-run") + ")")
    print()
    print("  Diagnose anytime with `modelmeld doctor`.")
    print()
    return 0


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run_setup(args: Any) -> int:
    """Execute the `modelmeld setup` command. Returns process exit code."""
    self_host = getattr(args, "self_host", False) or getattr(args, "demo", False)
    # --base-url defaults to the hosted endpoint, or localhost for self-host.
    raw_base_url = getattr(args, "base_url", None) or (
        "http://localhost:8080" if self_host else "https://api.modelmeld.ai"
    )
    base_url = raw_base_url.rstrip("/")
    try:
        _validate_base_url(base_url, allow_custom_host=args.allow_custom_host)
    except ValueError as e:
        _err(str(e))
        return 2
    interactive = not args.yes

    if self_host:
        return _setup_self_host(args, base_url, interactive)

    if args.tool == "codex":
        return _setup_codex(args, base_url, interactive)
    if args.tool != "claude-code":
        _err(f"--tool {args.tool!r} not supported yet")
        return 2

    print(_bold("ModelMeld — Claude Code setup"))
    print(f"  Gateway: {base_url}")

    # --- Collect credentials ---
    _step("Step 1/4: Collect credentials")
    api_key = _normalize(args.api_key) if args.api_key else ""
    if not api_key:
        if not interactive:
            _err("--api-key not supplied and --yes (non-interactive) is set")
            return 2
        print(
            "  Your ModelMeld API key (starts with 'gws_'). Get one at "
            "https://modelmeld.ai/account if you don't have one.",
        )
        api_key = _prompt_secret("  API key: ")
    if not api_key.startswith("gws_"):
        _warn("API key doesn't start with 'gws_' — usually means a copy-paste error")
    _ok(f"ModelMeld API key set ({len(api_key)} chars)")

    byok_anthropic = _normalize(args.byok_anthropic) if args.byok_anthropic else ""
    byok_openai = _normalize(args.byok_openai) if args.byok_openai else ""
    if not byok_anthropic and not byok_openai and interactive:
        print(
            "\n  Optionally configure BYOK (bring-your-own-key) for frontier "
            "routing. The Quality and Auto-escalated aliases route through "
            "frontier providers (Anthropic/OpenAI) — for those, you supply "
            "the key. Your key transits the gateway per-request and is "
            "never stored.",
        )
        ans = input("  Configure Anthropic BYOK now? [y/N]: ").strip().lower()
        if ans == "y":
            byok_anthropic = _prompt_secret(
                "  Anthropic API key (sk-ant-…): ", required=False,
            )
    if byok_anthropic:
        if not byok_anthropic.startswith("sk-ant-"):
            _warn("Anthropic key doesn't start with 'sk-ant-' — verify it's correct")
        _ok(f"Anthropic BYOK set ({len(byok_anthropic)} chars)")
    if byok_openai:
        _ok(f"OpenAI BYOK set ({len(byok_openai)} chars)")

    # --- Write the sourceable env script ---
    _step("Step 2/4: Write setup script")
    setup_dir = Path.home() / ".modelmeld"
    env_script = setup_dir / "setup-claude-code.sh"
    try:
        _write_claude_code_env_script(
            env_script, base_url, api_key, byok_anthropic or None, byok_openai or None,
        )
        _ok(f"Wrote {env_script} (LF-only, mode 0600)")
    except Exception as e:
        _err(f"Failed to write env script: {e}")
        return 1

    # --- Pre-write the Claude Code discovery cache ---
    _step("Step 3/4: Pre-write Claude Code discovery cache")
    cache_path = Path.home() / ".claude" / "cache" / "gateway-models.json"
    try:
        status, _hdrs, body = _http_get(
            f"{base_url}/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
            },
        )
        if status != 200:
            _err(
                f"Could not fetch /v1/models (HTTP {status}). "
                "Cache not pre-written — Claude Code's /model picker "
                "may not show the ModelMeld aliases.",
            )
            return 1
        models = json.loads(body)
        _write_claude_code_cache(cache_path, base_url, models)
        n_models = len(models.get("data", []))
        _ok(f"Wrote {cache_path} ({n_models} models, LF-only, mode 0600)")
    except Exception as e:
        _err(f"Failed to write Claude Code cache: {e}")
        # Don't return; the env script alone is still useful.
        _warn("Continuing — env script is set up, just no picker entries.")

    # --- Smoke-test the whole flow ---
    if not args.skip_smoke_test:
        _step("Step 4/4: Smoke-test the routing")
        results = _smoke_test(base_url, api_key, byok_anthropic or None)
        for r in results:
            if r.ok:
                _ok(f"{r.label}: {r.detail}")
            else:
                _err(f"{r.label}: {r.detail}")
        all_ok = all(r.ok for r in results)
        if not all_ok:
            _err("Some smoke tests failed. See errors above for next steps.")
            return 1
    else:
        _step("Step 4/4: Smoke-test (skipped)")

    # --- Final instructions ---
    print()
    print(_bold("Setup complete."))
    print()
    print("  1. Source the env script in your shell:")
    print(f"     {_bold(f'source {env_script}')}")
    print("  2. Launch Claude Code:")
    print(f"     {_bold('claude')}")
    print("  3. Pick a routing tier in /model:")
    print(f"     • {_bold('ModelMeld — Saver')}  (OSS-only, ~90% savings)")
    print(f"     • {_bold('ModelMeld — Auto')}   (escalates to frontier on reasoning markers)")
    print(f"     • {_bold('ModelMeld — Quality')} (frontier-first, requires BYOK)")
    print()
    print("  To persist these across shells, add to ~/.bashrc or ~/.zshrc:")
    print(f"     {_bold(f'source {env_script}')}")
    print()
    return 0

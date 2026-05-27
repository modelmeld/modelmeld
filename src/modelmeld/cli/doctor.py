# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""`modelmeld doctor` — diagnose an existing ModelMeld + Claude Code setup.

Companion to `modelmeld setup`. Where setup writes a clean configuration
from scratch, doctor inspects the live state and reports specific
problems with the SPECIFIC fix for each. Every failure mode customers
might hit (env var got dropped from shell rc, cache file got corrupted
or deleted, gws_ key expired, BYOK header malformed, gateway
unreachable) is a check here.

Output is structured for easy scanning:
    [OK]   Env var ANTHROPIC_BASE_URL is set and clean
    [FAIL] Env var ANTHROPIC_API_KEY has trailing \\r — re-export

Final line is overall verdict + total exit code.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Color + glyph output — reuses the setup module's pattern
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


@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str
    fix: str | None = None


def _emit(r: CheckResult) -> None:
    tag = f"  {_green('[OK]')}  " if r.ok else f"  {_red('[FAIL]')} "
    print(f"{tag}{r.label}: {r.detail}")
    if not r.ok and r.fix:
        print(f"         {_yellow('→ Fix:')} {r.fix}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_env_vars() -> list[CheckResult]:
    """Each env var must be set, clean (no CR/LF), and non-empty."""
    results: list[CheckResult] = []
    expected = {
        "ANTHROPIC_BASE_URL": "https://api.modelmeld.ai (or your gateway)",
        "ANTHROPIC_API_KEY": "your ModelMeld API key (gws_...)",
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
    }
    for name, hint in expected.items():
        value = os.environ.get(name, "")
        if not value:
            results.append(CheckResult(
                f"Env var {name}", False, "not set",
                fix=f"export {name}={hint}, OR run `modelmeld setup --tool claude-code`",
            ))
            continue
        # Critical: the env var must not contain CR or LF — these silently
        # corrupt downstream string equality checks (live-validation lesson).
        if "\r" in value or "\n" in value:
            results.append(CheckResult(
                f"Env var {name}", False,
                f"contains CR/LF (length {len(value)}) — invisible char will break "
                f"discovery / auth",
                fix=f"re-export cleanly: `export {name}={value.strip()}` "
                    f"(see also: source ~/.modelmeld/setup-claude-code.sh)",
            ))
            continue
        if value != value.strip():
            results.append(CheckResult(
                f"Env var {name}", False,
                f"has leading/trailing whitespace",
                fix=f"re-export trimmed: `export {name}={value.strip()}`",
            ))
            continue
        # Length check for sanity
        results.append(CheckResult(
            f"Env var {name}", True, f"set ({len(value)} chars)",
        ))
    # ANTHROPIC_AUTH_TOKEN must NOT be set (Claude Code auth conflict)
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        results.append(CheckResult(
            "Env var ANTHROPIC_AUTH_TOKEN", False,
            "is set — Claude Code will show 'Auth conflict' warning + "
            "discovery silently bails",
            fix="unset ANTHROPIC_AUTH_TOKEN",
        ))
    else:
        results.append(CheckResult(
            "Env var ANTHROPIC_AUTH_TOKEN", True,
            "not set (good — would conflict with ANTHROPIC_API_KEY)",
        ))
    # BYOK header is optional but if set must have `name:` prefix
    byok = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    if byok:
        if "\r" in byok or "\n" in byok or "\n   " in byok:
            results.append(CheckResult(
                "Env var ANTHROPIC_CUSTOM_HEADERS", False,
                "contains CR/LF (likely terminal wrap got embedded mid-value)",
                fix="re-export from a single physical line — run "
                    "`modelmeld setup --tool claude-code` to write it cleanly",
            ))
        elif ":" not in byok:
            results.append(CheckResult(
                "Env var ANTHROPIC_CUSTOM_HEADERS", False,
                "missing the `name:` prefix (you exported just the key, "
                "not the full header)",
                fix='export ANTHROPIC_CUSTOM_HEADERS="x-modelmeld-byok-anthropic: ${ANTHROPIC_CUSTOM_HEADERS}"',
            ))
        else:
            # Looks valid — peek at the format
            name_part = byok.split(":", 1)[0].strip().lower()
            if not name_part.startswith("x-modelmeld-byok-"):
                results.append(CheckResult(
                    "Env var ANTHROPIC_CUSTOM_HEADERS", False,
                    f"header name `{name_part}` doesn't match the BYOK pattern",
                    fix="should be `x-modelmeld-byok-anthropic: sk-ant-...` "
                        "(or -openai for OpenAI BYOK)",
                ))
            else:
                results.append(CheckResult(
                    "Env var ANTHROPIC_CUSTOM_HEADERS", True,
                    f"set with `{name_part}` prefix",
                ))
    return results


def _check_cache_file(base_url: str | None) -> list[CheckResult]:
    """The Claude Code discovery cache must exist with the correct
    wrapper format AND baseUrl matching ANTHROPIC_BASE_URL byte-for-byte."""
    results: list[CheckResult] = []
    cache_path = Path.home() / ".claude" / "cache" / "gateway-models.json"
    if not cache_path.exists():
        results.append(CheckResult(
            "Claude Code cache file", False,
            f"missing at {cache_path}",
            fix="run `modelmeld setup --tool claude-code` to pre-write it",
        ))
        return results
    try:
        raw = cache_path.read_bytes()
    except Exception as e:  # noqa: BLE001
        results.append(CheckResult(
            "Claude Code cache file", False, f"unreadable: {e}",
            fix=f"check file permissions on {cache_path}",
        ))
        return results
    # Line-ending check — CRLF silently breaks Claude Code's parsing
    if b"\r" in raw:
        results.append(CheckResult(
            "Claude Code cache file", False,
            f"contains CR/LF line endings at {cache_path} (Claude Code's parser "
            f"may reject)",
            fix="re-run `modelmeld setup --tool claude-code` (writes LF-only)",
        ))
        return results
    try:
        cache = json.loads(raw)
    except json.JSONDecodeError as e:
        results.append(CheckResult(
            "Claude Code cache file", False, f"not valid JSON: {e}",
            fix="re-run `modelmeld setup --tool claude-code`",
        ))
        return results
    # Wrapper-shape checks — Claude Code's reader requires baseUrl, fetchedAt, models
    if "baseUrl" not in cache or "fetchedAt" not in cache or "models" not in cache:
        results.append(CheckResult(
            "Claude Code cache file", False,
            "missing required wrapper keys (baseUrl, fetchedAt, models)",
            fix="re-run `modelmeld setup --tool claude-code`",
        ))
        return results
    # baseUrl exact-match check vs env var (the trailing-CR bug that ate
    # 30 minutes of debugging today)
    cache_url = cache["baseUrl"]
    if base_url and cache_url != base_url:
        results.append(CheckResult(
            "Cache baseUrl matches env", False,
            f"cache says {cache_url!r}, env says {base_url!r}",
            fix="re-run `modelmeld setup --tool claude-code` so both match",
        ))
        return results
    # Picker entries present
    models = cache.get("models", [])
    alias_ids = {m.get("id", "") for m in models}
    needed = {
        "anthropic/modelmeld-saver",
        "anthropic/modelmeld-auto",
        "anthropic/modelmeld-quality",
    }
    missing = needed - alias_ids
    if missing:
        results.append(CheckResult(
            "Cache contains 3-alias lineup", False,
            f"missing: {sorted(missing)}",
            fix="re-run `modelmeld setup --tool claude-code` (regenerates from "
                "the live /v1/models response)",
        ))
        return results
    results.append(CheckResult(
        "Claude Code cache file", True,
        f"valid wrapper, {len(models)} entries, baseUrl matches env",
    ))
    return results


def _check_gateway_reachability(base_url: str, api_key: str) -> list[CheckResult]:
    """Live HTTP probes against the gateway: /healthz + /v1/models + a real
    /v1/messages OSS roundtrip + (if BYOK supplied) frontier roundtrip."""
    results: list[CheckResult] = []
    # /healthz — basic reachability
    try:
        req = urllib.request.Request(f"{base_url}/healthz")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                results.append(CheckResult(
                    "Gateway reachability (/healthz)", True, f"HTTP {resp.status}",
                ))
            else:
                results.append(CheckResult(
                    "Gateway reachability (/healthz)", False,
                    f"unexpected status {resp.status}",
                    fix="check that the gateway is up + your network can reach it",
                ))
                return results
    except (urllib.error.URLError, TimeoutError) as e:
        results.append(CheckResult(
            "Gateway reachability (/healthz)", False, f"{type(e).__name__}: {e}",
            fix=f"verify {base_url} is correct and your network allows outbound HTTPS",
        ))
        return results

    # /v1/models — auth + cache-source verification
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            if resp.status == 200:
                data = json.loads(body)
                results.append(CheckResult(
                    "Auth check (/v1/models)", True,
                    f"HTTP 200, {len(data.get('data', []))} models",
                ))
            else:
                results.append(CheckResult(
                    "Auth check (/v1/models)", False,
                    f"HTTP {resp.status}: {body[:100].decode('utf-8','replace')}",
                ))
                return results
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        if e.code == 401:
            results.append(CheckResult(
                "Auth check (/v1/models)", False, "401 invalid API key",
                fix="check your gws_ key is correct + not revoked, OR get a "
                    "new one at https://modelmeld.ai/account",
            ))
        else:
            results.append(CheckResult(
                "Auth check (/v1/models)", False, f"HTTP {e.code}: {body[:100]}",
            ))
        return results
    except Exception as e:  # noqa: BLE001
        results.append(CheckResult(
            "Auth check (/v1/models)", False, f"{type(e).__name__}: {e}",
        ))
        return results

    # /v1/messages with -saver — OSS routing roundtrip
    body = {
        "model": "anthropic/modelmeld-saver",
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": (
                "This is a setup verification request from modelmeld doctor. "
                "Please reply with a single short sentence confirming receipt."
            ),
        }],
    }
    try:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=payload,
            method="POST",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            routed = resp.headers.get("x-modelmeld-routed-model", "?")
            results.append(CheckResult(
                "OSS routing roundtrip (-saver)", True,
                f"HTTP {resp.status}, served by {routed}",
            ))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        results.append(CheckResult(
            "OSS routing roundtrip (-saver)", False,
            f"HTTP {e.code}: {body[:200]}",
            fix="check your balance via GET /v1/account/balance — if 0, top up",
        ))
        return results
    except Exception as e:  # noqa: BLE001
        results.append(CheckResult(
            "OSS routing roundtrip (-saver)", False, f"{type(e).__name__}: {e}",
        ))
        return results

    # /v1/messages with -quality — BYOK frontier roundtrip, only when set
    byok = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    if not byok or ":" not in byok or "sk-ant-" not in byok:
        results.append(CheckResult(
            "BYOK frontier roundtrip (-quality)", True,
            "skipped (no ANTHROPIC_CUSTOM_HEADERS set — that's fine if "
            "you only need -saver / -auto without escalation)",
        ))
        return results

    # Build the BYOK header from the env var
    try:
        # Parse the `name: value` header out of ANTHROPIC_CUSTOM_HEADERS
        name_part, _, value_part = byok.partition(":")
        body["model"] = "anthropic/modelmeld-quality"
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            name_part.strip(): value_part.strip(),
        }
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=payload,
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            routed = resp.headers.get("x-modelmeld-routed-model", "?")
            if routed.startswith("claude-") or routed.startswith("gpt-"):
                results.append(CheckResult(
                    "BYOK frontier roundtrip (-quality)", True,
                    f"HTTP {resp.status}, served by {routed} (frontier via BYOK)",
                ))
            else:
                # Probably autocomplete-shape downgraded — that's actually OK
                results.append(CheckResult(
                    "BYOK frontier roundtrip (-quality)", True,
                    f"HTTP {resp.status}, served by {routed} (likely "
                    "autocomplete-shape downgrade, not a failure)",
                ))
    except urllib.error.HTTPError as e:
        body_bytes = e.read() if e.fp else b""
        body_text = body_bytes.decode("utf-8", "replace")
        # 400 byok_required means our gateway rejected — but doctor IS sending
        # the BYOK header, so something's still wrong with the value
        if e.code == 400 and "byok_required" in body_text:
            results.append(CheckResult(
                "BYOK frontier roundtrip (-quality)", False,
                "400 byok_required despite BYOK header set — header value "
                "may be malformed",
                fix="re-run `modelmeld setup --tool claude-code` to rewrite "
                    "the env var cleanly",
            ))
        elif e.code == 502 and "401" in body_text:
            results.append(CheckResult(
                "BYOK frontier roundtrip (-quality)", False,
                "502 — upstream Anthropic returned 401 invalid_x_api_key "
                "(your Anthropic key is bad)",
                fix="check your sk-ant- key is correct + active at "
                    "https://console.anthropic.com",
            ))
        else:
            results.append(CheckResult(
                "BYOK frontier roundtrip (-quality)", False,
                f"HTTP {e.code}: {body_text[:200]}",
            ))
    except Exception as e:  # noqa: BLE001
        results.append(CheckResult(
            "BYOK frontier roundtrip (-quality)", False, f"{type(e).__name__}: {e}",
        ))

    return results


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_doctor(args: Any) -> int:
    """Execute the `modelmeld doctor` command. Returns process exit code."""
    if args.tool != "claude-code":
        print(_red(f"--tool {args.tool!r} not supported yet (claude-code only at launch)"),
              file=sys.stderr)
        return 2

    print(_bold("ModelMeld doctor — claude-code setup"))
    print()

    all_results: list[CheckResult] = []

    print(_bold("Section 1/3: Environment variables"))
    env_results = _check_env_vars()
    for r in env_results:
        _emit(r)
    all_results.extend(env_results)

    print()
    print(_bold("Section 2/3: Claude Code cache file"))
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip().replace("\r", "")
    cache_results = _check_cache_file(base_url or None)
    for r in cache_results:
        _emit(r)
    all_results.extend(cache_results)

    print()
    print(_bold("Section 3/3: Live gateway probes"))
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip().replace("\r", "")
    if not base_url or not api_key:
        print(f"  {_yellow('skipped')} — env vars not clean enough to probe; "
              f"fix Section 1 first")
    else:
        probe_results = _check_gateway_reachability(base_url, api_key)
        for r in probe_results:
            _emit(r)
        all_results.extend(probe_results)

    # Summary
    print()
    n_fail = sum(1 for r in all_results if not r.ok)
    if n_fail == 0:
        print(_green(_bold(f"✓ All {len(all_results)} checks passed. Your setup is healthy.")))
        print()
        print("  Run `claude` and pick a ModelMeld tier in /model to start.")
        return 0
    print(_red(_bold(f"✗ {n_fail} of {len(all_results)} checks failed.")))
    print()
    print("  Apply the fix lines above, then re-run `modelmeld doctor`.")
    print("  If the situation is unclear: `modelmeld setup --tool claude-code` "
          "rewrites everything from scratch.")
    return 1

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
# Main flow
# ---------------------------------------------------------------------------


def run_setup(args: Any) -> int:
    """Execute the `modelmeld setup` command. Returns process exit code."""
    if args.tool != "claude-code":
        _err(f"--tool {args.tool!r} not supported yet (Claude Code only at launch)")
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        _validate_base_url(base_url, allow_custom_host=args.allow_custom_host)
    except ValueError as e:
        _err(str(e))
        return 2
    interactive = not args.yes
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

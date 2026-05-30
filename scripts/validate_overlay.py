# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Validate default_overlay.json against each provider's live model catalog.

Reads $FIREWORKS_API_KEY, $TOGETHER_API_KEY from the environment (no
fallback — these are required). OpenRouter is hit unauthenticated.

For each overlay row, queries the provider's /v1/models endpoint and
reports whether the row's provider_model_id is present in the live
catalog. Also surfaces the catalog's context_window + per-token price
so we can compare against what we ship.

Usage:
    $env:FIREWORKS_API_KEY = "fw_..."
    $env:TOGETHER_API_KEY = "..."
    python scripts/validate_overlay.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from importlib import resources

_PROVIDER_ENDPOINTS = {
    "fireworks": (
        "https://api.fireworks.ai/inference/v1/models",
        "FIREWORKS_API_KEY",
    ),
    "together": (
        "https://api.together.xyz/v1/models",
        "TOGETHER_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1/models",
        None,  # unauthenticated
    ),
}


def _fetch_catalog(provider: str) -> dict[str, dict]:
    url, env_var = _PROVIDER_ENDPOINTS[provider]
    headers = {"User-Agent": "modelmeld-overlay-validator/1.0"}
    if env_var is not None:
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(
                f"{env_var} not set; cannot query {provider}. Set it in your shell and re-run."
            )
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"{provider} HTTP {e.code}: {body}") from None
    # Catalog shape varies: OpenRouter + Fireworks wrap in {"data": [...]};
    # Together returns a bare list.
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("data", payload)
    else:
        raise RuntimeError(f"{provider}: unexpected catalog shape: {type(payload)}")
    if not isinstance(items, list):
        raise RuntimeError(f"{provider}: catalog items not a list: {type(items)}")
    return {item["id"]: item for item in items if isinstance(item, dict) and "id" in item}


def main() -> int:
    overlay_path = resources.files("modelmeld.scout.data").joinpath("default_overlay.json")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    rows = overlay.get("models", [])

    print(
        f"Overlay has {len(rows)} rows across {len(set(r['provider'] for r in rows))} providers.\n"
    )

    catalogs: dict[str, dict[str, dict]] = {}
    for provider in sorted({r["provider"] for r in rows}):
        if provider == "vllm":
            continue  # vllm is operator-hosted; no public catalog
        print(f"Fetching {provider} catalog...")
        try:
            catalogs[provider] = _fetch_catalog(provider)
            print(f"  -> {len(catalogs[provider])} models\n")
        except RuntimeError as e:
            print(f"  ERROR: {e}\n")
            catalogs[provider] = {}

    ok = 0
    miss = 0
    miss_rows: list[dict] = []
    for row in rows:
        provider = row["provider"]
        if provider == "vllm":
            continue
        catalog = catalogs.get(provider, {})
        pid = row["provider_model_id"]
        if pid in catalog:
            ok += 1
            item = catalog[pid]
            # Pricing is heterogeneous across providers; try common shapes
            pricing = item.get("pricing") or {}
            ctx = item.get("context_length") or item.get("context_window") or "?"
            in_price = pricing.get("prompt") or pricing.get("input") or "?"
            out_price = pricing.get("completion") or pricing.get("output") or "?"
            print(f"  OK   {row['model_id']:30s} @ {provider:10s} id={pid}")
            print(
                f"       overlay: ctx={row['context_window']} "
                f"in=${row['cost_per_m_input']:.2f}/out=${row['cost_per_m_output']:.2f} per M"
            )
            print(f"       upstream: ctx={ctx} in=${in_price}/out=${out_price} per token")
        else:
            miss += 1
            miss_rows.append(row)
            partial = [k for k in catalog if pid.split("/")[-1].lower() in k.lower()][:3]
            print(f"  MISS {row['model_id']:30s} @ {provider:10s} id={pid}")
            print(f"       partial matches: {partial}")

    print(f"\n=== SUMMARY: {ok} OK, {miss} MISS ===")
    if miss_rows:
        print("\nRows to fix in default_overlay.json:")
        for r in miss_rows:
            print(f"  - {r['model_id']} @ {r['provider']} (current id: {r['provider_model_id']})")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

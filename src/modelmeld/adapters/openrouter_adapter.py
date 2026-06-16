# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""OpenRouterAdapter — pass-through to OpenRouter's OpenAI-compatible endpoint.

OpenRouter is a meta-router that itself proxies to many underlying
providers. From a wire-format perspective it's just another
OpenAI-compatible endpoint at `https://openrouter.ai/api/v1`.

OpenRouter accepts two optional headers (`HTTP-Referer`, `X-Title`) used
for their public leaderboard analytics. This adapter does NOT set them
by default — set them explicitly via custom request headers if you want
your traffic identified on OpenRouter's public dashboard.
"""

from __future__ import annotations

import os

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _provider_routing() -> dict | None:
    """OpenRouter `provider` routing preference, sent on every request.

    OpenRouter load-balances each request across the underlying backends that
    serve a model. Because prompt caching is per-backend, default load-balancing
    scatters a multi-turn session across backends, so the cache never accumulates
    across turns. A deterministic `sort` pins a session's turns to one backend so
    the cache builds; `price` also makes that backend the cheapest (cost-optimal),
    and `allow_fallbacks` preserves availability if it's down.

    Override the sort or disable via `MODELMELD_OPENROUTER_PROVIDER_SORT`
    (e.g. `throughput`, or empty string to disable and restore default
    load-balancing).
    """
    sort = os.environ.get("MODELMELD_OPENROUTER_PROVIDER_SORT", "price").strip()
    if not sort:
        return None
    return {"sort": sort, "allow_fallbacks": True}


class OpenRouterAdapter(OpenAIAdapter):
    name = "openrouter"
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("MODELMELD_OPENROUTER_API_KEY")
        )
        if not resolved_key:
            raise AdapterError(
                "OpenRouterAdapter requires an API key "
                "(pass api_key= or set OPENROUTER_API_KEY)."
            )
        routing = _provider_routing()
        super().__init__(
            api_key=resolved_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            served_model=None,
            extra_body={"provider": routing} if routing else None,
        )

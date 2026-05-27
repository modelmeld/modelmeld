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
        super().__init__(
            api_key=resolved_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            served_model=None,
        )

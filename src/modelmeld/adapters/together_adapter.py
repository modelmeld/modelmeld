# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""TogetherAdapter — pass-through to Together AI's OpenAI-compatible endpoint.

Together AI exposes the OpenAI Chat Completions wire format at
`https://api.together.xyz/v1`. Thin OpenAIAdapter subclass; reads
`TOGETHER_API_KEY` (or `MODELMELD_TOGETHER_API_KEY` as fallback) at
construction.
"""

from __future__ import annotations

import os

import httpx

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherAdapter(OpenAIAdapter):
    name = "together"
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("TOGETHER_API_KEY")
            or os.environ.get("MODELMELD_TOGETHER_API_KEY")
        )
        if not resolved_key:
            raise AdapterError(
                "TogetherAdapter requires an API key "
                "(pass api_key= or set TOGETHER_API_KEY)."
            )
        self._health_base = (base_url or _TOGETHER_BASE_URL).rstrip("/")
        self._health_key = resolved_key
        super().__init__(
            api_key=resolved_key,
            base_url=base_url or _TOGETHER_BASE_URL,
            served_model=None,
        )

    async def health(self) -> bool:
        """Liveness probe for Together.

        Together's `/models` returns a BARE JSON list, not the OpenAI
        `{object:"list", data:[...]}` shape, so the SDK's `models.list()` (which
        the inherited `OpenAIAdapter.health()` calls) raises
        `'list' object has no attribute '_set_private_attributes'` and reports
        the adapter permanently unhealthy. That silently broke ALL Together
        routing — every model whose route resolves to Together 503'd, with no
        failover on the pinned path. A raw GET that treats any 2xx as healthy is
        the correct liveness signal here (chat itself is OpenAI-compatible and
        unaffected)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._health_base}/models",
                    headers={"Authorization": f"Bearer {self._health_key}"},
                )
            return resp.status_code < 400
        except Exception:
            return False

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
        super().__init__(
            api_key=resolved_key,
            base_url=base_url or _TOGETHER_BASE_URL,
            served_model=None,
        )

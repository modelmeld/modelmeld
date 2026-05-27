# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""FireworksAdapter — pass-through to Fireworks AI's OpenAI-compatible endpoint.

Fireworks exposes the OpenAI Chat Completions wire format at
`https://api.fireworks.ai/inference/v1`, so this adapter is a thin
OpenAIAdapter subclass with the base URL pinned and the api key read
from `FIREWORKS_API_KEY` (or `MODELMELD_FIREWORKS_API_KEY` as fallback
for environments that namespace all configuration).
"""

from __future__ import annotations

import os

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksAdapter(OpenAIAdapter):
    name = "fireworks"
    # External egress — PII scrubber applies before forwarding, same as
    # openai / anthropic adapters.
    is_egress = True

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("FIREWORKS_API_KEY")
            or os.environ.get("MODELMELD_FIREWORKS_API_KEY")
        )
        if not resolved_key:
            raise AdapterError(
                "FireworksAdapter requires an API key "
                "(pass api_key= or set FIREWORKS_API_KEY)."
            )
        super().__init__(
            api_key=resolved_key,
            base_url=base_url or _FIREWORKS_BASE_URL,
            served_model=None,   # Fireworks serves many models; client picks
        )

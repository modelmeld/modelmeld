# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

from __future__ import annotations

from fastapi import APIRouter, Request

from modelmeld.api.schemas import Model, ModelList
from modelmeld.config import GatewaySettings

router = APIRouter()


# Human-readable display names for canonical models the gateway routes to.
# Claude Code's /model picker (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1,
# v2.1.129+) filters to IDs prefixed `claude` or `anthropic` AND uses
# `display_name` when present. Without an entry here, the model id shows
# raw in the picker (or gets filtered out entirely).
_DISPLAY_NAMES: dict[str, str] = {
    # Frontier — Anthropic
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    # Frontier — OpenAI (kept here for the standard /v1/models surface;
    # Claude Code's prefix filter excludes these from its picker)
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-4": "GPT-5.4",
    # ModelMeld policy-based auto-route aliases.
    # Three customer-facing tiers, each picking a cost-quality ceiling:
    #   - saver:   OSS-only, max savings, predictable ceiling
    #   - auto:    OSS by default, escalates to frontier on reasoning markers
    #   - quality: frontier-first, downgrades only on trivial shape
    # See modelmeld/scout/policy.py for behavior. The deprecated 5-alias
    # lineup (balanced/coding/reasoning/cheap/frontier-priority) is still
    # honored by the policy resolver for backwards compatibility, but
    # only the canonical 3 surface in the /model picker — operators
    # should funnel customers to the canonical names.
    "anthropic/modelmeld-saver": "ModelMeld — Saver (OSS-only auto-route)",
    "anthropic/modelmeld-auto": "ModelMeld — Auto (smart escalation)",
    "anthropic/modelmeld-quality": "ModelMeld — Quality (frontier-first)",
}


@router.get("/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    """GET /v1/models — OpenAI + Anthropic-native compatible model discovery.

    Returns one Model row per `settings.available_models`. Each row carries
    BOTH the OpenAI fields (`object: "model"`, `created: int`) AND the
    Anthropic-native fields (`type: "model"`, `created_at: ISO 8601`); the
    list envelope carries both `object: "list"` AND the Anthropic-native
    `has_more`/`first_id`/`last_id` pagination markers. Clients on either
    spec see a valid payload (each ignores fields it doesn't recognize).

    `display_name` is populated when the gateway has a human-readable name
    registered in `_DISPLAY_NAMES` — required for Claude Code's /model
    picker (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1) to render the
    entry. The picker filters to IDs prefixed `claude` or `anthropic`,
    which is why we also surface the `anthropic/modelmeld-*` auto-route
    aliases below.
    """
    settings: GatewaySettings = request.app.state.settings
    data: list[Model] = []
    for model_id in settings.available_models:
        data.append(Model(
            id=model_id,
            owned_by=settings.owner,
            display_name=_DISPLAY_NAMES.get(model_id),
        ))
    # Anthropic-namespaced auto-route aliases — only surface when not
    # already in available_models (avoid duplicates) so an operator who
    # decides to remove them can do so via `available_models` config.
    advertised_ids = {m.id for m in data}
    for alias_id, display in _DISPLAY_NAMES.items():
        if alias_id.startswith("anthropic/") and alias_id not in advertised_ids:
            data.append(Model(
                id=alias_id,
                owned_by="modelmeld",
                display_name=display,
            ))
    first_id = data[0].id if data else None
    last_id = data[-1].id if data else None
    return ModelList(data=data, first_id=first_id, last_id=last_id)

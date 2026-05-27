# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""VLLMAdapter — pass-through to a vLLM server (OpenAI-compatible at the wire).

vLLM exposes the OpenAI Chat Completions surface verbatim, so this adapter
re-uses `OpenAIAdapter` machinery and just defaults `api_key="EMPTY"` plus
points at the configured vLLM endpoint.
"""

from __future__ import annotations

import os

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter


class VLLMAdapter(OpenAIAdapter):
    name = "vllm"
    # vLLM is hosted on customer infrastructure (local GPU or single-tenant cloud
    # provisioned by them) — does NOT cross trust boundary.
    is_egress = False

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        served_model: str | None = None,
    ) -> None:
        endpoint = endpoint or os.environ.get("MODELMELD_VLLM_ENDPOINT") or os.environ.get(
            "VLLM_ENDPOINT"
        )
        if not endpoint:
            raise AdapterError(
                "VLLMAdapter requires an endpoint URL "
                "(pass endpoint= or set MODELMELD_VLLM_ENDPOINT). "
                "Use scripts/dev_gpu.py to provision a RunPod-hosted vLLM and copy "
                "the printed endpoint here."
            )
        # Self-hosted vLLM accepts no auth; "EMPTY" is the conventional placeholder.
        # Hosted serverless proxies (RunPod, Modal, etc.) gate the OpenAI path with
        # a Bearer key — read it from MODELMELD_VLLM_API_KEY when present.
        # F-8: vLLM only serves one model at a time — operator pins it via
        # `served_model` so the gateway routes the request regardless of what
        # model id the client asked for.
        resolved_api_key = (
            api_key
            or os.environ.get("MODELMELD_VLLM_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or "EMPTY"
        )
        super().__init__(
            api_key=resolved_api_key,
            base_url=endpoint,
            served_model=served_model,
        )

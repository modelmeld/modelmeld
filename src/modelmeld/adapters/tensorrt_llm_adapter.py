# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""TensorRTLLMAdapter — pass-through to a Triton/TensorRT-LLM endpoint.

NVIDIA's Triton Inference Server with the TensorRT-LLM backend exposes the
OpenAI Chat Completions wire format (since TRT-LLM ≥ 0.10 via the
`triton_cli` + `inflight_batcher_llm` template). Same shape as vLLM and the
real OpenAI API, so this adapter is a thin subclass of `OpenAIAdapter` with
the right defaults.

Strategically — TensorRT-LLM is the "highest-throughput open-weights serving"
option for customers running on H100/H200/B200 hardware. Letting the gateway
route to it via the same adapter contract means the routing decision
("which provider serves the chosen model?") doesn't need to know the
difference between vLLM and TRT-LLM. Both are LOCAL tier; both share PII
boundaries; both translate to OpenAI wire.

Endpoint resolution priority:
    1. constructor `endpoint=` argument
    2. `MODELMELD_TENSORRT_LLM_ENDPOINT` env var
    3. `TENSORRT_LLM_ENDPOINT` env var (legacy)
    4. raises `AdapterError` if none configured
"""

from __future__ import annotations

import os

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.openai_adapter import OpenAIAdapter


class TensorRTLLMAdapter(OpenAIAdapter):
    """Local-tier adapter for Triton/TensorRT-LLM endpoints."""

    name = "tensorrt_llm"
    # Customer-owned GPU infrastructure (on-prem, single-tenant cloud, or
    # hosted dedicated tier). Never crosses tenant trust boundary.
    is_egress = False

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        endpoint = (
            endpoint
            or os.environ.get("MODELMELD_TENSORRT_LLM_ENDPOINT")
            or os.environ.get("TENSORRT_LLM_ENDPOINT")
        )
        if not endpoint:
            raise AdapterError(
                "TensorRTLLMAdapter requires an endpoint URL "
                "(pass endpoint= or set MODELMELD_TENSORRT_LLM_ENDPOINT). "
                "Triton with TensorRT-LLM backend exposes OpenAI Chat Completions "
                "at e.g. http://triton:8000/v1 — point us there."
            )
        # Triton/TRT-LLM ignores api_key like vLLM; "EMPTY" is the convention.
        super().__init__(api_key=api_key or "EMPTY", base_url=endpoint)

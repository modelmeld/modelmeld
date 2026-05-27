# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Gateway runtime configuration via pydantic-settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODELMELD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    # Models advertised by /v1/models. The routing-aware registry replaces
    # this in capability mode; for now it's just what we tell clients we support.
    available_models: list[str] = Field(
        default_factory=lambda: [
            "gpt-5-mini",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "qwen2.5-coder-7b-instruct",
        ]
    )
    owner: str = "modelmeld"

    # Routing policy.
    #   "single"        — all traffic goes to `upstream_provider` (default)
    #   "scout_driven"  — Scout picks LOCAL vs CLOUD tier; tier→adapter via local_/cloud_provider
    #   "always_local"  — force LOCAL tier (e.g., compliance / data-residency mode)
    #   "always_cloud"  — force CLOUD tier
    #   "capability"    — CapabilityScout picks a specific model from the registry
    routing_policy: Literal[
        "single", "scout_driven", "always_local", "always_cloud", "capability"
    ] = "single"

    # Used when routing_policy="single".
    upstream_provider: Literal["stub", "openai"] = "stub"

    # Used when routing_policy != "single" (tiered modes).
    # Defaults to "stub" so multi-tier policies work out of the box without secrets;
    # operators set these to real providers (and provide credentials) in prod.
    local_provider: Literal["stub", "openai", "vllm", "tensorrt_llm"] = "stub"
    cloud_provider: Literal["stub", "openai", "anthropic"] = "stub"

    # Provider credentials / endpoints.
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    vllm_endpoint: str | None = None
    tensorrt_llm_endpoint: str | None = None   # Triton + TensorRT-LLM

    # F-8: operator-pinned upstream model per adapter. When set, the adapter
    # substitutes the client's `request.model` field with this value on
    # outbound calls. Used in TIERED routing where each tier serves a fixed
    # model and the client's model name is incidental (the scout picks the
    # tier; the operator's config picks the model). When None, the adapter
    # passes the client's model name through unchanged (the default — fine
    # for OpenAI/Anthropic which serve many models per key).
    openai_served_model: str | None = None
    anthropic_served_model: str | None = None
    vllm_served_model: str | None = None

    # Scout classifier. vLLM-SR-backed scout will be added once
    # we have an integration story for its Go/Envoy architecture.
    scout_provider: Literal["heuristic"] = "heuristic"
    scout_confidence_threshold: float = 0.65

    # Capability routing. Used when routing_policy="capability".
    # `capability_quality_threshold` is the minimum task_score (0..1) a model
    # must have on the classified task category to be a candidate. Bumping it
    # higher = stricter quality bar, smaller candidate set, higher cost.
    capability_quality_threshold: float = 0.70
    # Restrict to providers we have adapters for. None → all providers in the
    # registry are eligible (only sensible when every provider has an adapter).
    capability_eligible_providers: list[str] | None = None
    capability_fallback_depth: int = 5

    # Completion cache. Off by default in core-engine; enterprise
    # turns it on. When on, the chat route consults an `app.state.completion_cache`
    # before calling the adapter. Hits skip the upstream call entirely.
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600

    # Semantic cache similarity threshold. Cosine similarity in
    # [0,1]. 0.92 is conservative against the hashed-BOW test embedder where
    # paraphrases score ~0.93+ and unrelated prompts score <0.5. Tighten to
    # 0.97+ when using a real embedding model that scores paraphrases higher.
    semantic_cache_similarity_threshold: float = 0.92

    # Token counting backend.
    #   "char"    — 1 token ≈ 4 chars; fast, no deps, ~20% off for English.
    #   "litellm" — accurate per-model via litellm.token_counter(); requires
    #               `pip install modelmeld[tokenizer]`. Falls back to
    #               char-based with a warning when litellm isn't installed.
    token_counter_backend: Literal["char", "litellm"] = "char"

    # PII scrubbing. When True, message text is scrubbed before
    # any egress adapter call (is_egress=True). Local adapters (stub, vllm) are
    # never scrubbed since traffic stays inside the customer's boundary.
    pii_scrub_cloud: bool = True

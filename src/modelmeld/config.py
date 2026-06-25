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

    # Models advertised by /v1/models. When empty (the default), the
    # advertise list is auto-derived from `app.state.model_registry` —
    # whatever models the loaded registry knows about get advertised.
    # That keeps the advertised lineup in sync with the routing knowledge
    # automatically: every model add to the registry / overlay surfaces
    # in /v1/models without a parallel config push.
    #
    # Operators who want to RESTRICT the advertised list (e.g., hide
    # deprecated-but-still-routable models, or limit to a known-stable
    # subset for a production tenant) can populate this list explicitly
    # via `MODELMELD_AVAILABLE_MODELS` env var (JSON array). When set,
    # the explicit list wins and the registry is ignored for /v1/models
    # output.
    #
    # The three `anthropic/modelmeld-*` policy aliases are auto-appended
    # by the route in both modes — they're not registry-backed.
    available_models: list[str] = Field(default_factory=list)
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
    tensorrt_llm_endpoint: str | None = None  # Triton + TensorRT-LLM
    # Cloud OSS-model providers. The adapter classes already exist in
    # `modelmeld.adapters.{fireworks,together,openrouter}_adapter`; these
    # settings drive `_build_adapter` + `_infer_providers_from_credentials`
    # so a customer who sets only one of these env vars gets a working
    # capability router out of the box (i.e., the OSS engine actually
    # delivers multi-provider routing without a Pro registry overlay).
    fireworks_api_key: str | None = None
    together_api_key: str | None = None
    openrouter_api_key: str | None = None

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

    # Subscription passthrough (Sprint 5 / 5.5). When True, an inbound
    # request carrying an OAuth-bearer-shaped Authorization header (JWT
    # prefix `eyJ...`) is routed to a passthrough adapter — Codex CLI's
    # chatgpt.com backend for /v1/chat/completions, api.anthropic.com
    # for /v1/messages — instead of the normal capability router. Keys
    # are NEVER persisted; the request flows verbatim with headers
    # preserved. ToS-safe posture: self-host only, single-user-per-
    # instance, no multi-tenant pooling. Default off — opt-in flag for
    # power users. See docs/subscription-passthrough.md.
    allow_subscription_passthrough: bool = False

    # Capability routing. Used when routing_policy="capability".
    # `capability_quality_threshold` is the minimum task_score (0..1) a model
    # must have on the classified task category to be a candidate. Bumping it
    # higher = stricter quality bar, smaller candidate set, higher cost.
    capability_quality_threshold: float = 0.70
    # Restrict to providers we have adapters for. None → all providers in the
    # registry are eligible (only sensible when every provider has an adapter).
    capability_eligible_providers: list[str] | None = None
    capability_fallback_depth: int = 5

    # ---- Curated registry feed (Pro) ----------------------------------------
    # When `registry_feed_url` is set, the gateway fetches a continuously
    # curated, signed ModelRegistry at startup and routes on it instead of the
    # bundled snapshot (which is frozen at release time). The payload signature
    # is verified against `registry_feed_public_key_pem`; ANY failure (network,
    # bad signature, unsupported schema, expired) falls back to the bundled
    # registry — degraded routing always beats a crashed gateway. The license
    # key is the JWT issued on Pro activation. New feeds are picked up on
    # process restart. See modelmeld.scout.feed.RegistryFeedClient.
    registry_feed_url: str | None = None
    registry_feed_license_key: str | None = None
    # PEM-encoded Ed25519 public key that signs the feed. REQUIRED when
    # `registry_feed_url` is set — the client refuses to fetch a signed feed
    # with no verifier (a misconfigured URL could otherwise exfiltrate the
    # license key + accept an attacker's payload as the live registry). Multiple
    # concatenated PEM blocks are accepted: pin both the current and next key
    # during a key rotation (the feed verifies if ANY pinned key matches), then
    # drop the retired one once the publisher has cut over.
    registry_feed_public_key_pem: str | None = None
    # Local cache path for the last good payload (served within TTL with no
    # network hit). None disables on-disk caching.
    registry_feed_cache_path: str | None = None
    registry_feed_cache_ttl_sec: int = 3600

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

    # Memory backend. Default remains in-process for dev/tests. Set
    # MODELMELD_MEMORY_BACKEND=postgres plus MODELMELD_MEMORY_DATABASE_URL for
    # a SQL-backed store shared across workers.
    memory_backend: Literal["in_memory", "postgres", "mem0"] = "in_memory"
    memory_database_url: str | None = None

    # Mem0 provider (memory_backend="mem0"). Optional dep: pip install
    # modelmeld[mem0]. The gateway does request-path injection; mem0 does
    # extraction + retrieval. Tenant isolation uses a SEPARATE vector
    # collection per tenant (not shared-collection metadata filters).
    # `infer=True` runs an LLM extraction call per write — route it through
    # this gateway via mem0_base_url so it gets cost-routed (see docs).
    mem0_infer: bool = True
    mem0_top_k: int = 10
    mem0_rerank: bool = False
    mem0_embedding_dims: int = 1536
    mem0_llm_model: str = "gpt-5-mini"
    mem0_embedder_model: str = "text-embedding-3-small"
    # Route mem0's extraction LLM + embedder at an OpenAI-compatible endpoint
    # (e.g. this gateway). None → mem0's default (api.openai.com).
    mem0_base_url: str | None = None
    mem0_api_key: str | None = None
    # Vector store. URL → shared qdrant server (per-tenant collection);
    # else on-disk embedded qdrant under this path (per-tenant subdir).
    mem0_vector_store_url: str | None = None
    mem0_vector_store_api_key: str | None = None
    mem0_vector_store_path: str | None = None

    # PII scrubbing. When True, message text is scrubbed before
    # any egress adapter call (is_egress=True). Local adapters (stub, vllm) are
    # never scrubbed since traffic stays inside the customer's boundary.
    pii_scrub_cloud: bool = True

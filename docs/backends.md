# Supported backends

ModelMeld speaks the OpenAI Chat Completions wire format on the inbound
side and forwards each request to one of several **backend providers**
on the outbound side. This page lists the backends we have first-party
adapters for, how to configure them, and which ones we explicitly do
not support.

A **backend** is an inference service that serves model completions.
Backends are external to the gateway — they run as separate processes
on your infrastructure (self-hosted) or as third-party SaaS (cloud).
ModelMeld does not bundle their code or carry their license
obligations; we just speak HTTP to them.

For the *client* side — the frameworks and dev tools that send requests
*to* ModelMeld — see [`integrations/`](integrations/README.md).

---

## Cloud SaaS providers

### OpenAI

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.openai_adapter.OpenAIAdapter` |
| **Wire format** | OpenAI Chat Completions (native) |
| **Auth** | API key (`OPENAI_API_KEY` or `MODELMELD_OPENAI_API_KEY`) |
| **Streaming** | ✅ SSE |
| **Tool calling** | ✅ full schema fidelity |
| **Multi-modal** | ✅ images, audio (via OpenAI's native types) |
| **Notes** | Pass-through with no translation; lowest overhead path. Honors retry-with-backoff on 429/5xx. |

```bash
MODELMELD_CLOUD_PROVIDER=openai
MODELMELD_OPENAI_API_KEY=sk-...
MODELMELD_ROUTING_POLICY=always_cloud   # or scout_driven
```

### Anthropic

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.anthropic_adapter.AnthropicAdapter` |
| **Wire format** | Anthropic Messages (translated from/to OpenAI shape) |
| **Auth** | API key (`ANTHROPIC_API_KEY` or `MODELMELD_ANTHROPIC_API_KEY`) |
| **Streaming** | ✅ SSE (with event-stream translation) |
| **Tool calling** | ✅ schema-mapped to OpenAI tool-call format |
| **Multi-modal** | ✅ images via Anthropic's content-block format |
| **Notes** | The OpenAI ⇄ Anthropic translation layer is the most complex adapter in the codebase. Property-based tests in `tests/test_translation_openai_anthropic.py`. Honors retry-with-backoff on 429/529/5xx. |

```bash
MODELMELD_CLOUD_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
MODELMELD_ROUTING_POLICY=always_cloud   # or scout_driven
```

### Google Gemini

| | |
|---|---|
| **Status** | 🔜 planned |
| **Notes** | Adapter not yet implemented. The capability registry already includes Gemini model entries so the routing layer is ready; only the wire-format adapter is missing. Track in GitHub Discussions. |

### Fireworks AI

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.fireworks_adapter.FireworksAdapter` |
| **Wire format** | OpenAI Chat Completions (native) |
| **Auth** | API key (`FIREWORKS_API_KEY` or `MODELMELD_FIREWORKS_API_KEY`) |
| **Streaming** | ✅ SSE |
| **Notes** | Thin OpenAI-compatible pass-through. Fireworks serves many open-weight models (Qwen, Llama, DeepSeek, Phi, etc.) under one endpoint; `request.model` selects which one. Eligible for saver-tier routing when your registry overlay has entries tagged with the `fireworks` provider. |

```bash
FIREWORKS_API_KEY=fw_...
```

### Together AI

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.together_adapter.TogetherAdapter` |
| **Wire format** | OpenAI Chat Completions (native) |
| **Auth** | API key (`TOGETHER_API_KEY` or `MODELMELD_TOGETHER_API_KEY`) |
| **Streaming** | ✅ SSE |
| **Notes** | Thin OpenAI-compatible pass-through. Together serves many open-weight models under one endpoint. Eligible for saver-tier routing when your registry overlay has entries tagged with the `together` provider. |

```bash
TOGETHER_API_KEY=...
```

### OpenRouter

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.openrouter_adapter.OpenRouterAdapter` |
| **Wire format** | OpenAI Chat Completions (native) |
| **Auth** | API key (`OPENROUTER_API_KEY` or `MODELMELD_OPENROUTER_API_KEY`) |
| **Streaming** | ✅ SSE |
| **Notes** | OpenRouter is a meta-router that proxies to many underlying providers; from a wire-format perspective it's just another OpenAI-compatible endpoint. The adapter does not set OpenRouter's optional `HTTP-Referer` / `X-Title` analytics headers by default — supply them via custom request headers if you want your traffic identified on OpenRouter's public dashboard. Eligible for saver-tier routing when your registry overlay has entries tagged with the `openrouter` provider. |

```bash
OPENROUTER_API_KEY=...
```

---

## Self-hosted inference engines

### vLLM (Apache-2.0)

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.vllm_adapter.VLLMAdapter` |
| **Wire format** | OpenAI Chat Completions (vLLM serves OpenAI-compatible API natively) |
| **Auth** | none required at the engine; gateway sends `api_key="EMPTY"` |
| **Streaming** | ✅ SSE |
| **Tool calling** | ✅ if the model + vLLM version support it |
| **Multi-modal** | depends on the model served |
| **Notes** | Any model vLLM can serve — Qwen, Llama, DeepSeek, Mistral, etc. Operator pins the served model via `MODELMELD_VLLM_SERVED_MODEL` so the gateway substitutes the model name on outbound calls (F-8). One model per vLLM instance; run multiple instances for multi-model coverage. |

```bash
MODELMELD_LOCAL_PROVIDER=vllm
MODELMELD_VLLM_ENDPOINT=https://<your-vllm-host>/v1
MODELMELD_VLLM_SERVED_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
```

We don't redistribute vLLM. Its Apache-2.0 license applies between you
and the vLLM project directly. See <https://github.com/vllm-project/vllm>.

### TensorRT-LLM + Triton (NVIDIA)

| | |
|---|---|
| **Status** | ✅ supported |
| **Adapter** | `modelmeld.adapters.tensorrt_llm_adapter.TensorRTLLMAdapter` |
| **Wire format** | OpenAI Chat Completions (via Triton's OpenAI-compatible endpoint) |
| **Auth** | none required at the engine; gateway sends `api_key="EMPTY"` |
| **Streaming** | ✅ SSE |
| **Notes** | For NVIDIA-optimized self-hosted inference. This adapter ships in `modelmeld`; tested against a Triton-served TRT-LLM compiled model. |

```bash
MODELMELD_LOCAL_PROVIDER=tensorrt_llm
MODELMELD_TENSORRT_LLM_ENDPOINT=https://<your-triton-host>/v1
```

---

## Explicitly out of scope

These are *not* supported and we don't plan to add them:

| Backend | Why not |
|---|---|
| **Anyscale** | No first-party adapter today. Anyscale is OpenAI-compatible; if you need it, file a GitHub Discussion — the implementation pattern is the same thin `OpenAIAdapter` subclass that Fireworks / Together / OpenRouter use. |
| **LiteLLM as upstream** | LiteLLM is a peer gateway, not a backend. We speak the same OpenAI / Anthropic wire formats directly. Stacking gateways adds a hop without adding capability. |
| **Generic Ollama** | Not blocked technically — point `MODELMELD_VLLM_ENDPOINT` at Ollama's OpenAI-compatible endpoint and it works in pass-through mode. We don't ship a first-party Ollama adapter; if demand exists, file a discussion. |

---

## Adding a new backend

The adapter ABC is `modelmeld.adapters.base.ProviderAdapter`. Implement
the four methods:

```python
class ProviderAdapter(ABC):
    name: str
    is_egress: bool   # True if traffic crosses the customer's trust boundary

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion: ...
    def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]: ...
    async def health(self) -> bool: ...
    async def close(self) -> None: ...
```

Wire it into `modelmeld.router._build_adapter` so settings can resolve
your provider name. Add it to this page. Ship.

If your backend already exposes the OpenAI wire format (most do these
days), inherit from `OpenAIAdapter` and only override what's
provider-specific — `VLLMAdapter` is the canonical example, ~10 lines.

---

## Versioning + compatibility

Backend adapters live behind the SemVer-stable `ProviderAdapter` ABC.
Adding a new adapter is a minor-version bump. Removing or breaking an
adapter is a major-version bump. See
[`api-stability.md`](api-stability.md) for the full SemVer policy.

# Migrating from LiteLLM → ModelMeld

If you already run a [LiteLLM](https://litellm.ai) proxy (or call
`litellm.completion()` from the SDK), ModelMeld drops in the same way:
it speaks OpenAI `POST /v1/chat/completions`, so every client and
framework you already point at LiteLLM keeps working. You change one
`api_base`.

What you get on top of the OpenAI-compatible surface you already have:

- **One gateway instead of a config tree.** Capability routing,
  failover, per-request audit, and PII redaction live in the gateway,
  not in your `config.yaml`. The fallback chains, router rules, and
  budget logic you hand-maintain in LiteLLM become gateway behavior you
  configure once.
- **Self-host or bring your own keys.** Run the gateway in your own
  infrastructure under AGPL-3.0, or use the hosted gateway with your own
  frontier keys passed per request (BYOK). Frontier keys are never
  persisted: they transit the gateway for the single request and are
  forgotten. PII scrubbing runs before any egress.
- **Deterministic, auditable routing.** Routing decisions are a
  deterministic function of request shape plus registry state. The same
  request against the same registry routes to the same model, and every
  response carries audit headers (`x-modelmeld-routed-model`,
  `x-modelmeld-task-category`, ...) so you can reproduce any decision.
- **Predictable cost.** The three policy aliases below give you a hard
  cost ceiling (`-saver`), smart escalation (`-auto`), or frontier-first
  with trivial-task trimming (`-quality`). You pick the ceiling; the
  scout picks the cheapest model that clears your quality bar within it.

This page covers the proxy `config.yaml` and the Python SDK. The
[framework guides](README.md#agent-framework-guides) (AutoGen, CrewAI,
LangGraph, OpenClaw, MetaGPT) all route through LiteLLM today, so once
the proxy points here they route here too.

## Picking a policy (the part that replaces your router rules)

LiteLLM selects a model per call. ModelMeld selects a *policy* and lets
the scout pick the model. Set the request `model` field to one of the
three aliases:

| Alias | Behavior | Replaces your LiteLLM... |
|-------|----------|--------------------------|
| `anthropic/modelmeld-saver` | OSS-tier only; frontier rows filtered out. Hard cost ceiling. | cheap-model routing + budget guards |
| `anthropic/modelmeld-auto` | OSS by default; escalates to frontier on reasoning markers / large context. | Complexity Router / model-routing rules |
| `anthropic/modelmeld-quality` | Frontier-first; downgrades trivial-shape calls to OSS. | premium-default + fallback-to-cheap |

The `anthropic/` segment is part of the alias string itself (a
historical artifact), not a provider selector. It works on the OpenAI
`/v1/chat/completions` surface exactly as written. Pass the alias
verbatim, including `anthropic/`.

If you send any other model name (e.g. `claude-opus-4-7`), you still get
capability routing, just without a policy ceiling: the scout classifies
the request and picks the cheapest model that clears the default quality
bar. The policy aliases are how you bound that.

## Proxy `config.yaml`

### Before (a typical LiteLLM proxy)

```yaml
# Each model wired to its own provider, plus router + fallback rules
model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY
  - model_name: opus
    litellm_params:
      model: anthropic/claude-opus-4-7
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: cheap-coder
    litellm_params:
      model: openai/<some-oss-model>
      api_base: https://<your-oss-provider>/v1
      api_key: os.environ/OSS_PROVIDER_KEY
  # ... plus more provider rows

router_settings:
  fallbacks: [{"gpt-5": ["opus"]}, {"opus": ["cheap-coder"]}]
  # ... routing strategy, budgets, retries
```

### After (everything behind the gateway)

```yaml
model_list:
  - model_name: modelmeld-saver
    litellm_params:
      custom_llm_provider: openai
      model: anthropic/modelmeld-saver       # passthrough; gateway resolves the policy
      api_base: https://api.modelmeld.ai/v1   # ← or your self-hosted gateway
      api_key: os.environ/MODELMELD_API_KEY

  - model_name: modelmeld-auto
    litellm_params:
      custom_llm_provider: openai
      model: anthropic/modelmeld-auto
      api_base: https://api.modelmeld.ai/v1
      api_key: os.environ/MODELMELD_API_KEY

  - model_name: modelmeld-quality
    litellm_params:
      custom_llm_provider: openai
      model: anthropic/modelmeld-quality
      api_base: https://api.modelmeld.ai/v1
      api_key: os.environ/MODELMELD_API_KEY
```

What you delete from the old config: the per-provider `model_list` rows,
the `router_settings.fallbacks` chains, and any budget/retry routing
logic. The gateway does failover (next-cheapest qualified model on a
5xx, surfaced via `x-modelmeld-failover-from`) and cost ceilings
(via the policy) internally.

> **Why `custom_llm_provider: openai` and not `model: openai/...`?**
> LiteLLM's `openai/<name>` prefix syntax splits on the *first* slash to
> detect the provider. Because the ModelMeld alias contains a slash
> (`anthropic/modelmeld-auto`), the bare `openai/anthropic/modelmeld-auto`
> form can be mis-parsed and forward the wrong model id
> ([BerriAI/litellm#12457](https://github.com/BerriAI/litellm/issues/12457)).
> Setting `custom_llm_provider: openai` selects the OpenAI-compatible
> transport and forwards the `model` field verbatim, so the gateway
> receives `anthropic/modelmeld-auto` intact. Verify with the curl below.

## Python SDK

```python
import litellm

resp = litellm.completion(
    model="anthropic/modelmeld-auto",   # policy alias, forwarded verbatim
    custom_llm_provider="openai",        # OpenAI-compatible transport
    api_base="https://api.modelmeld.ai/v1",
    api_key="gws_<your-modelmeld-key>",
    messages=[{"role": "user", "content": "Refactor this function ..."}],
)
print(resp.model)  # the actual routed model the gateway chose
```

## Verify the model arrived intact

The one thing worth confirming after the cutover is that LiteLLM
forwarded the alias unchanged. Hit the gateway directly and read the
audit headers:

```bash
curl -sD - -o /dev/null https://api.modelmeld.ai/v1/chat/completions \
  -H "Authorization: Bearer gws_<your-modelmeld-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic/modelmeld-auto","messages":[{"role":"user","content":"hi"}]}' \
  | grep -i x-modelmeld
```

You should see `x-modelmeld-routed-model` and
`x-modelmeld-task-category` on every response. On a self-hosted gateway
started with `MODELMELD_EXPOSE_ROUTING_RATIONALE=1`, the
`x-modelmeld-routing-rationale` header also carries a `policy=saver(...)`
/ `policy=auto(...)` note that confirms the policy resolved. If routing
behaves as if no policy applied (e.g. frontier models served under what
you set to `-saver`), the forwarded `model` string was not exactly one
of the three aliases: re-check the `custom_llm_provider` line.

## Routing hints (optional)

Anything you expressed as separate LiteLLM deployments per task can be
expressed as a per-request hint header instead. See the
[routing-hint headers reference](../routing-hints.md). The common ones:

- `x-modelmeld-task-category: coding` — skip the classifier.
- `x-modelmeld-agent-role: reviewer` — declare the agent's role; the
  gateway maps it to a category.
- `x-modelmeld-quality-threshold: 0.85` — raise or lower the bar.

In proxy `config.yaml`, attach them under `litellm_params.extra_headers`;
in the SDK, pass `extra_headers={...}` to `litellm.completion()`.

## Common gotchas

- **`api_base` must end in `/v1`.** LiteLLM appends `/chat/completions`.
  A missing `/v1` is the usual cause of a 404.
- **Keep frontier keys out of the gateway's at-rest config.** For the
  hosted gateway, pass them per request as BYOK headers
  (`x-modelmeld-byok-anthropic` / `x-modelmeld-byok-openai`) via
  `extra_headers`; the gateway uses and forgets them. Stay on `-saver`
  if you do not want to bring frontier keys at all.
- **Self-hosting.** Set `api_base` to your own gateway URL. See
  [`docs/self-host.md`](../self-host.md) for the gateway setup; nothing
  in the LiteLLM config changes except the URL.

## See also

- [LiteLLM OpenAI-compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible)
- [Routing-hint headers reference](../routing-hints.md)
- [Agent framework guides](README.md#agent-framework-guides) — they all sit on top of LiteLLM

# MetaGPT → ModelMeld

[MetaGPT](https://github.com/FoundationAgents/MetaGPT) is a multi-agent
framework where a software-company SOP drives roles (ProductManager,
Architect, Engineer, QA). It reads one OpenAI-compatible `llm` block and
talks `POST /v1/chat/completions`, so pointing it at ModelMeld is a
`base_url` change.

## Minimal setup

Edit `~/.metagpt/config2.yaml` (it overrides the repo's
`config/config2.yaml`):

```yaml
llm:
  api_type: 'openai'                         # keep 'openai' for the compatible transport
  model: 'anthropic/modelmeld-auto'          # policy alias; the gateway routes per request
  base_url: 'https://api.modelmeld.ai/v1'    # ← or your self-hosted gateway
  api_key: 'gws_<your-modelmeld-key>'
```

Run any MetaGPT command as usual:

```bash
metagpt "Build a CLI todo app with tests"
```

Every role's call now flows through the gateway. The scout classifies
each request and picks the cheapest model that clears the quality bar,
so the Engineer's code-generation turns and the ProductManager's
planning turns can land on different models without any per-role config.

## Picking a policy

Set `model` to the policy alias that matches your cost-quality ceiling:

- `anthropic/modelmeld-saver` — OSS-tier only; a hard cost ceiling.
- `anthropic/modelmeld-auto` — OSS by default; escalates to frontier on
  reasoning markers or large context. Good default for SOP runs that mix
  planning and coding.
- `anthropic/modelmeld-quality` — frontier-first; downgrades trivial
  calls to OSS.

The `anthropic/` segment is part of the alias string, not a provider
selector. Pass it verbatim. Any non-alias `model` value still gets
capability routing, just without a policy ceiling.

## What MetaGPT sends us

- Large role-specific system prompts (the SOP persona + instructions).
- Multi-turn message arrays as the SOP advances through roles.
- `stream` per MetaGPT's config (`stream: true` by default in recent
  versions).
- Function-calling tools when a role uses them; the scout's
  `supports_tools` filter handles tool-bearing vs. plain turns.

## Per-role routing (advanced)

The SOP roles map naturally onto routing hints (Architect → reasoning,
Engineer → coding, QA → reviewer). If your deployment can attach
per-request HTTP headers, set `x-modelmeld-agent-role` per role and the
gateway routes each role to the cheapest competent model for that
category. See the [routing-hint headers reference](../routing-hints.md)
for the accepted role names. The single global `config2.yaml` above
works without this; hints are an optimization, not a requirement.

## Verify the routing

Hit the gateway directly to confirm the alias resolved before wiring it
into a full SOP run:

```bash
curl -sD - -o /dev/null https://api.modelmeld.ai/v1/chat/completions \
  -H "Authorization: Bearer gws_<your-modelmeld-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic/modelmeld-auto","messages":[{"role":"user","content":"hi"}]}' \
  | grep -i x-modelmeld
```

`x-modelmeld-routed-model` and `x-modelmeld-task-category` confirm what
served the request.

## Common gotchas

- **`base_url` must end in `/v1`.** MetaGPT appends `/chat/completions`.
- **Keep `api_type: 'openai'`.** Swapping only `base_url` is what makes
  MetaGPT use the OpenAI-compatible transport against the gateway; do not
  change `api_type` to a vendor-specific value.
- **Frontier keys via BYOK.** On the hosted gateway, frontier models
  need your own key. Stay on `-saver` to run OSS-only with no frontier
  key at all.

## See also

- [MetaGPT configuration docs](https://docs.deepwisdom.ai/main/en/guide/get_started/configuration/llm_api_configuration.html)
- [Routing-hint headers reference](../routing-hints.md)
- [Migrating from LiteLLM](litellm.md) — the shared OpenAI-compatible spine

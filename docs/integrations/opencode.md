# opencode → ModelMeld

[opencode](https://github.com/sst/opencode) is an open-source terminal
coding agent from the SST team. Unlike Claude Code (hard-wired to
Anthropic), opencode has a native provider system — any
OpenAI-compatible endpoint can be registered as a provider via
[`@ai-sdk/openai-compatible`](https://opencode.ai/docs/providers) in
the config file. Drop ModelMeld in as a provider and your existing
opencode workflow routes through capability-aware multi-model picking
without any code changes on opencode's side.

## Minimal setup

Add ModelMeld as a provider in your opencode config
(`~/.config/opencode/opencode.json` or a project-local
`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "modelmeld": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ModelMeld",
      "options": {
        "baseURL": "https://api.modelmeld.ai/v1",
        "apiKey": "{env:MODELMELD_API_KEY}"
      },
      "models": {
        "claude-opus-4-7": { "name": "ModelMeld → Opus 4.7 (routed)" },
        "claude-sonnet-4-6": { "name": "ModelMeld → Sonnet 4.6 (routed)" },
        "gpt-5": { "name": "ModelMeld → GPT-5 (routed)" },
        "qwen3-coder-next": { "name": "ModelMeld → qwen3-coder-next (OSS)" },
        "deepseek-v4-pro": { "name": "ModelMeld → deepseek-v4-pro (OSS)" },
        "llama-4-scout": { "name": "ModelMeld → llama-4-scout (1M ctx, OSS)" }
      }
    }
  }
}
```

Export your ModelMeld API key in the same shell that launches `opencode`:

```bash
export MODELMELD_API_KEY=gws_<your-key>
opencode
```

Pick any `modelmeld/*` entry with opencode's `/models` command. The
model ID you select is what the gateway sees — but our scout may
substitute a cheaper competent model when the prompt warrants it (an
autocomplete-shape request to `claude-opus-4-7` typically routes to an
OSS model at OSS-tier cost). The audit headers tell you what actually
ran.

## What's special about opencode for routing

opencode is a clean fit for capability-aware routing because its
behavior gives the scout strong signals:

- **All traffic is coding-shaped.** opencode is a coding agent — every
  prompt is a code task. Our scout's coding-category classifier hits
  with high confidence on opencode traffic, so routing decisions are
  more deterministic than on a general-purpose chat client.
- **Sub-agent behavior surfaces via opencode's modes.** Plan mode
  produces reasoning-heavy prompts; act mode produces tool-call-heavy
  prompts. The scout's tool-capable filter automatically excludes
  models that can't reliably do function-calling on act-mode requests
  (avoiding the malformed-tool-block tax that hits below ~7B).
- **Long-context routing kicks in naturally.** When opencode
  accumulates file reads into a large context, the scout's context-
  window filter picks a long-context model (e.g. `llama-4-scout` at
  1M tokens) so the request doesn't truncate.

## Routing hints (optional)

opencode's provider config supports custom headers, so you can declare
task category or quality threshold per provider entry:

```json
"modelmeld": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "ModelMeld",
  "options": {
    "baseURL": "https://api.modelmeld.ai/v1",
    "apiKey": "{env:MODELMELD_API_KEY}",
    "headers": {
      "x-modelmeld-quality-threshold": "0.85"
    }
  },
  "models": { "...": {} }
}
```

Useful headers (full reference in
[routing-hints.md](../routing-hints.md)):

- `x-modelmeld-quality-threshold` — raise (e.g. `0.90`) for mission-
  critical sessions where you'd rather pay more than risk a quality
  drop; lower (e.g. `0.70`) for exploratory work where speed matters.
- `x-modelmeld-exclude-providers` — comma-separated provider names to
  block from routing (e.g. `openai,anthropic` to force OSS-only).
- `x-modelmeld-byok-anthropic` / `x-modelmeld-byok-openai` — bring
  your own frontier API key per request; transits the gateway for the
  one upstream call and is then forgotten (never stored at rest).

## Verify the routing took effect

Every response from the gateway carries audit headers. opencode logs
the full response when run with `--verbose`; you can also inspect them
via your normal HTTP tooling. Expected headers:

```text
x-modelmeld-routed-to: openai          # or anthropic, vllm, etc.
x-modelmeld-routed-model: qwen3-coder-next
x-modelmeld-task-category: coding
x-modelmeld-task-score: 0.87
x-modelmeld-quality-threshold: 0.80
x-modelmeld-tier: oss
```

If `x-modelmeld-routed-model` differs from what you asked for in
opencode's `/models` picker, the scout substituted — that's by design.
The audit headers show the actual model and its task score so the
substitution is observable, not silent.

> **Note:** the `x-modelmeld-devtool` header (which fingerprints the
> originating client from the request shape) doesn't yet include an
> opencode signature. Traffic from opencode will route correctly and
> get the rest of the audit headers, but won't be identified as
> `opencode` in the fingerprint until a signature is added —
> tracked as a follow-up.

## Common gotchas

- **Model IDs must be declared.** Unlike some OpenAI-compatible
  clients, opencode does not auto-discover models from
  `/v1/models` — every model you want to use has to appear in the
  `models` block of the config. Add new entries as you want to expose
  more aliases in the picker; removing one only hides it from the UI
  (the gateway still routes if you reference it elsewhere).
- **Per-provider headers vs per-request headers.** Headers set in the
  provider config apply to every request through that provider. If
  you need different routing posture per call (e.g. higher quality
  for refactors, looser for autocomplete), declare two provider
  entries with different header blocks (`modelmeld-strict` vs
  `modelmeld-loose`) and switch between them via the `/models` picker.
- **`apiKey` substitution syntax.** opencode's config supports
  `"{env:VAR_NAME}"` for environment-variable interpolation. Use it
  for the API key rather than embedding the literal string —
  config files often end up in dotfile repos.
- **Project-local config wins over global.** A `opencode.json` in the
  project root overrides `~/.config/opencode/opencode.json` for that
  workspace. Useful when one project wants a higher quality threshold
  than your default.

## See also

- [opencode docs — Providers](https://opencode.ai/docs/providers) —
  the upstream provider configuration reference
- [Routing-hint headers reference](../routing-hints.md) — full
  documentation of `x-modelmeld-*` request and response headers
- [opencode GitHub](https://github.com/sst/opencode)

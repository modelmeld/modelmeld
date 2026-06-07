# Codex CLI -> ModelMeld

[Codex CLI](https://github.com/openai/codex) is OpenAI's open-source
terminal coding agent. Current Codex (v0.137+) talks the **OpenAI Responses
API** (`/v1/responses`), so you point it at ModelMeld by adding a custom
model provider in its config file. ModelMeld then picks the cheapest capable
backend per request and records the routing decision in response headers.

> **Why a config block, not just `OPENAI_BASE_URL`?** Older OpenAI-compatible
> clients use `/v1/chat/completions` and read a base URL from the
> environment. Current Codex speaks the Responses API and discovers providers
> through `~/.codex/config.toml`. The environment-variable shortcut does
> **not** work for the Responses path — use the provider block below.

## Setup

### 1. Add a ModelMeld provider to `~/.codex/config.toml`

```toml
model = "anthropic/modelmeld-saver"
model_provider = "modelmeld"

[model_providers.modelmeld]
name = "ModelMeld"
base_url = "https://api.modelmeld.ai/v1"
env_key = "MODELMELD_API_KEY"
wire_api = "responses"
```

- `wire_api = "responses"` is required — it tells Codex to use `/v1/responses`.
- `env_key` names the environment variable Codex reads your key from (below).
- The provider id (`modelmeld` here) is yours to choose, but `openai`,
  `ollama`, and `lmstudio` are reserved — don't reuse them.
- Keep `/v1` on the `base_url`; Codex appends the Responses path to it.

### 2. Export your ModelMeld key

```sh
export MODELMELD_API_KEY="gws_<your-modelmeld-key>"
codex
```

On Windows PowerShell:

```powershell
$env:MODELMELD_API_KEY = "gws_<your-modelmeld-key>"
codex
```

Send a small prompt to verify the connection, then ask Codex to read a file
so you exercise its tool calls — both stream through the gateway.

## Choosing a routing policy

Set `model` to one of ModelMeld's three policy aliases (in `config.toml`, via
`/model` in the TUI, or `--config model=...` at launch):

| Model | Policy |
| ----- | ------ |
| `anthropic/modelmeld-saver` | OSS-only auto-route. Never escalates to a frontier model — a predictable cost ceiling. Best default for an agentic session. |
| `anthropic/modelmeld-auto` | Smart escalation: starts on OSS models, escalates to frontier only when the request shows reasoning markers. |
| `anthropic/modelmeld-quality` | Frontier-first, with downgrade for trivial turns. |

Use the canonical `anthropic/`-prefixed names — these are what `/v1/models`
advertises, so Codex's `/model` picker can resolve them.

> Frontier escalation (`-auto` / `-quality`) needs a frontier key. Supply one
> per request with a `x-modelmeld-byok-<provider>` header to stretch your own
> frontier budget through ModelMeld's routing; without one, stay on `-saver`.

## Optional routing check

Codex CLI doesn't surface HTTP response headers in the terminal. To confirm
the gateway is routing, send a direct Responses request with the same key:

```sh
curl -i https://api.modelmeld.ai/v1/responses \
  -H "authorization: Bearer $MODELMELD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "model": "anthropic/modelmeld-saver",
    "input": "Reply with one sentence about ModelMeld routing."
  }'
```

Look for these response headers:

| Header | What to check |
| ------ | ------------- |
| `x-modelmeld-routed-model` | The canonical model the gateway selected for the request. |
| `x-modelmeld-tier` | The serving tier used. |
| `x-modelmeld-task-category` | The inferred task category (present when capability routing ran). |

For the full list of response headers and their meanings, see the
[Routing-hint headers reference](../routing-hints.md).

## Common gotchas

- **`wire_api = "responses"` is mandatory.** Without it Codex tries
  `/v1/chat/completions` against a provider configured for Responses and the
  session won't work as expected.
- **Key var must match `env_key`.** Codex reads the key from the variable you
  named (`MODELMELD_API_KEY`), not `OPENAI_API_KEY`.
- **Keep `/v1` in `base_url`.** Omitting it 404s.
- **"Model metadata not found" on startup is cosmetic.** Codex doesn't ship
  context-window metadata for ModelMeld aliases; it falls back to defaults and
  runs fine. Using the canonical `anthropic/modelmeld-*` names minimizes it.
- **Codex's sandbox is independent of ModelMeld.** Codex sandboxes shell
  commands itself; ModelMeld only handles LLM routing.
- **Rate limits come from ModelMeld, not OpenAI.** Check your ModelMeld
  dashboard if you hit one.

## Advanced: custom routing headers

To force a tier or task category, run a local proxy that injects
`x-modelmeld-*` headers and point `base_url` at it. See the
[Routing-hint headers reference](../routing-hints.md) for accepted values.

## See also

- [Codex CLI GitHub](https://github.com/openai/codex)
- [ModelMeld integrations overview](README.md)

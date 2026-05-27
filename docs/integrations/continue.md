# Continue.dev → ModelMeld

[Continue.dev](https://continue.dev) is an open-source VS Code +
JetBrains assistant with explicit role-based model configuration
(autocomplete, chat, edit, apply, embed, rerank, summarize). Each role
can be pointed at a different backend — perfect for per-role ModelMeld
routing.

## Minimal setup

Edit your Continue config (`~/.continue/config.yaml` or via the
"Open Config" command):

```yaml
models:
  - name: ModelMeld - main
    provider: openai
    model: claude-opus-4-7
    apiBase: https://api.modelmeld.ai/v1
    apiKey: gws_<your-modelmeld-key>
    roles: [chat, edit, apply]

  - name: ModelMeld - autocomplete
    provider: openai
    model: gpt-5-mini
    apiBase: https://api.modelmeld.ai/v1
    apiKey: gws_<your-modelmeld-key>
    roles: [autocomplete]
    requestOptions:
      headers:
        x-modelmeld-agent-role: autocomplete
        x-modelmeld-quality-threshold: "0.55"  # cheap-tier OK for autocomplete

  - name: ModelMeld - summarize
    provider: openai
    model: gpt-5-mini
    apiBase: https://api.modelmeld.ai/v1
    apiKey: gws_<your-modelmeld-key>
    roles: [summarize]
    requestOptions:
      headers:
        x-modelmeld-agent-role: summarizer
        x-modelmeld-quality-threshold: "0.55"
```

This config:

- Routes **chat / edit / apply** through ModelMeld's scout (default
  quality bar of 0.80 — OSS-mid or OSS-premium tier).
- Routes **autocomplete** through ModelMeld with an explicit lower
  quality bar (0.55) — scout picks granite-4-micro / gemma-3-4b /
  phi-4-mini for sub-200ms latency.
- Routes **summarize** through the cheap tier too — summarization is
  a known-easy task category.

## Continue's role-based model assignment

Continue's `roles` field is the cleanest mapping to our routing hints
in any tool:

| Continue role  | Map to `x-modelmeld-agent-role` | Notes |
|----------------|---------------------------------|-------|
| `autocomplete` | `autocomplete` (or unset)       | Cheap tier; low max_tokens; latency-sensitive |
| `chat`         | (unset — let scout classify)    | Mixed traffic; scout decides |
| `edit`         | `coder`                         | Mostly coding tasks |
| `apply`        | `coder`                         | Applies edit to existing code |
| `embed`        | (separate embeddings endpoint)  | Out of scope for chat completions |
| `rerank`       | (separate rerank endpoint)      | Out of scope here |
| `summarize`    | `summarizer`                    | Easy task → cheap tier |

## What Continue sends us

- **Chat mode**: multi-turn conversations. The `@codebase` provider
  uses local embeddings to retrieve relevant code; output to us is a
  focused 5-25K-token prompt with retrieved chunks inline.
- **Autocomplete**: FIM-style single-shot (or close). Very short
  prompt, very low max_tokens (~50-200), tight latency budget.
- **Agent mode**: Continue may emit `tools` for tool-capable models or
  fall back to system-prompt tool descriptions. Our scout handles both.

## Verify the routing

Use Continue's "Show Model" toggle in the chat UI to see which model
served the response. The displayed model name is what `response.model`
returns — for ModelMeld that's the actual routed model.

## Common gotchas

- **`requestOptions.headers` is YAML strings** — quote numeric values
  like `"0.55"` to avoid YAML float-parsing surprises.
- **`apiBase` must include `/v1`** — Continue concatenates
  `/chat/completions` to your apiBase; without `/v1` you get 404.
- **Continue's local embeddings are SEPARATE** — `@codebase` runs
  embeddings locally via `all-MiniLM-L6-v2`. ModelMeld doesn't serve
  embeddings yet (on the roadmap); leave Continue's embeddings
  config as default.

## See also

- [Continue.dev config reference](https://docs.continue.dev/reference)
- [Continue.dev OpenAI provider docs](https://docs.continue.dev/customize/model-providers/top-level/openai)
- ModelMeld docs/integrations/README.md — routing hint headers reference

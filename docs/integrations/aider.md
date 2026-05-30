# Aider → ModelMeld

[Aider](https://aider.chat) is an open-source CLI pair-programmer that
routes through [LiteLLM](https://litellm.ai). It speaks OpenAI's
`/v1/chat/completions` and supports custom base URLs via env vars.

## Minimal setup

```powershell
# PowerShell — Windows
$env:OPENAI_API_BASE = "https://api.modelmeld.ai/v1"
$env:OPENAI_API_KEY  = "gws_<your-modelmeld-key>"
aider --model openai/claude-opus-4-7  # or any model id
```

```bash
# bash — macOS / Linux
export OPENAI_API_BASE=https://api.modelmeld.ai/v1
export OPENAI_API_KEY=gws_<your-modelmeld-key>
aider --model openai/claude-opus-4-7
```

The `openai/` prefix is LiteLLM's "route this via the OpenAI-compatible
backend" syntax. The model name after the prefix can be anything — our
scout decides what actually serves the request.

## Aider-specific config

Aider uses three model slots that map cleanly to our scout's tier system:

```yaml
# .aider.conf.yml at the repo root
openai-api-base: https://api.modelmeld.ai/v1
openai-api-key: gws_<your-modelmeld-key>

model: openai/claude-opus-4-7         # main model: routed by our scout (≥0.85 task score)
weak-model: openai/gpt-5-mini         # commit summaries: routed to trivial tier
editor-model: openai/claude-sonnet-4-6  # editor for /architect: mid-quality bar
```

The three models map to ModelMeld behaviors:

- **`model`** — main pair-programming exchanges. Heavy prompts (repo
  map + file contents). Routes to OSS-premium tier
  (qwen3-coder-480b, deepseek-v4-pro) for hard tasks.
- **`weak-model`** — Aider uses this for commit messages, summaries,
  and small classification jobs. Routes to sub-Haiku or
  OSS-mid (granite-4-micro, phi-4-mini, qwen3-coder-flash).
- **`editor-model`** — used in `/architect` mode for plan-then-edit
  workflows. Routes to OSS-mid by default.

## What Aider sends us

- Large system prompt (edit-format instructions + repo map, ~2-20K tokens)
- Full file contents for explicitly `/add`ed files
- `messages` array: `[system, user-with-repo-map, assistant, user, ...]`
- **No `tools` array** — Aider parses SEARCH/REPLACE blocks from
  freeform model output rather than using function-calling
- `stream=true` by default

Our scout sees the request shape (no tools, large input) and routes
appropriately.

## Verify the routing took effect

Run with `--verbose` to see Aider's HTTP exchange. The response headers
in verbose output include:

```text
x-modelmeld-routed-model: deepseek/deepseek-v4-pro
x-modelmeld-task-category: coding
x-modelmeld-quality-threshold: 0.80
```

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Common gotchas

- **Model name prefixes are LiteLLM's, not ours** — `openai/foo`,
  `anthropic/foo`, `gemini/foo` etc. all route through their respective
  LiteLLM backend. For ModelMeld, you want `openai/...` (matches our
  OpenAI-compatible endpoint) regardless of the actual model brand.
- **`/clear` doesn't reset the repo map** — Aider's repo map is
  per-session. To force a fresh routing decision, restart Aider.
- **Very large repo maps overflow phi-4 (16K ctx)** — our scout's
  context-window filter handles this automatically; Aider will see
  a different routed model on big-repo sessions.

## See also

- [Aider docs — OpenAI-compatible endpoints](https://aider.chat/docs/llms/openai-compat.html)
- [Aider docs — edit formats](https://aider.chat/docs/more/edit-formats.html)
- ModelMeld README — request fingerprinting (Aider has its own signature)

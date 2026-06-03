# Codex CLI -> ModelMeld

[Codex CLI](https://github.com/openai/codex) is OpenAI's open-source
terminal coding agent. It speaks OpenAI's `/v1/chat/completions` natively,
so pointing it at ModelMeld is a base-URL override and an API-key swap.
ModelMeld then picks the cheapest capable backend per request and records
the routing decision in response headers.

## Minimal setup

Install Codex CLI, then set two environment variables before launching it:

```sh
export OPENAI_BASE_URL="https://api.modelmeld.ai/v1"
export OPENAI_API_KEY="gws_<your-modelmeld-key>"
codex
```

On Windows PowerShell:

```powershell
$env:OPENAI_BASE_URL = "https://api.modelmeld.ai/v1"
$env:OPENAI_API_KEY = "gws_<your-modelmeld-key>"
codex
```

Codex CLI reads these environment variables automatically — no config file
edits needed. Send a small prompt to verify the connection works.

## Optional routing check

Codex CLI does not show HTTP response headers in the terminal. To confirm
that the gateway is routing requests, run a direct request with the same
key:

```sh
curl -i https://api.modelmeld.ai/v1/chat/completions \
  -H "authorization: Bearer $OPENAI_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {
        "role": "user",
        "content": "Reply with one sentence about ModelMeld routing."
      }
    ],
    "max_tokens": 80
  }'
```

Look for these response headers:

| Header | What to check |
| ------ | ------------- |
| `x-modelmeld-routed-model` | The actual model selected by the gateway. |
| `x-modelmeld-tier` | The serving tier used for the request. |
| `x-modelmeld-task-category` | The inferred task category, unless you supplied a hint through another client. |

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Common gotchas

- **Use `OPENAI_BASE_URL`, not a config file**. Codex CLI reads the
  OpenAI-compatible base URL from the environment variable. There is no
  settings file for provider configuration.
- **Keep `/v1` in the base URL**. Codex appends chat-completions paths to
  that base URL. If you omit `/v1`, requests will 404.
- **Codex uses `gpt-5.5` by default**. ModelMeld will route this to the
  cheapest capable model. If you want a specific model, pass `--model`
  or set `OPENAI_MODEL` in the environment.
- **Codex's sandbox mode is independent of ModelMeld**. Codex has its own
  sandbox for shell commands. ModelMeld only handles LLM routing — it does
  not affect Codex's sandbox behavior.
- **Rate limits come from ModelMeld, not OpenAI**. If you hit rate limits,
  check your ModelMeld dashboard, not your OpenAI account.

## Advanced: custom routing headers

To send custom routing hints (e.g., force a specific tier or task category),
you can use a local proxy that injects `x-modelmeld-*` headers and point
`OPENAI_BASE_URL` at that proxy. See the [Routing-hint headers reference](../routing-hints.md)
for accepted values.

## See also

- [Codex CLI GitHub](https://github.com/openai/codex)
- [ModelMeld integrations overview](README.md)

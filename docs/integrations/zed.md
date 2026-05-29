# Zed -> ModelMeld

[Zed](https://zed.dev) is a fast editor with a built-in agent panel and
support for OpenAI-compatible model providers. Pointing Zed at ModelMeld
lets the editor keep its normal chat and agent workflow while ModelMeld
chooses the cheapest capable backend per request and records the routing
decision in response headers.

## Minimal setup

Install Zed, then add ModelMeld as an OpenAI-compatible provider in your
Zed `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "ModelMeld": {
        "api_url": "https://api.modelmeld.ai/v1",
        "available_models": [
          {
            "name": "claude-opus-4-7",
            "display_name": "ModelMeld Auto Route",
            "max_tokens": 200000,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true
            }
          },
          {
            "name": "gpt-5-mini",
            "display_name": "ModelMeld Fast Route",
            "max_tokens": 128000,
            "capabilities": {
              "tools": true,
              "images": false,
              "parallel_tool_calls": false,
              "prompt_cache_key": false,
              "chat_completions": true
            }
          }
        ]
      }
    }
  }
}
```

Set your ModelMeld key in the environment before launching Zed:

```sh
export MODELMELD_API_KEY="gws_<your-modelmeld-key>"
zed .
```

On Windows PowerShell:

```powershell
$env:MODELMELD_API_KEY = "gws_<your-modelmeld-key>"
zed .
```

Then open Zed's agent panel, select **ModelMeld Auto Route**, and send a
small prompt. Zed sends OpenAI-compatible `POST /chat/completions`
requests to the `api_url` above, which ModelMeld receives as
`/v1/chat/completions`.

## Optional routing check

Zed does not show HTTP response headers in the UI. To confirm that the
gateway is routing the same model alias, run a direct request with the
same key:

```sh
curl -i https://api.modelmeld.ai/v1/chat/completions \
  -H "authorization: Bearer $MODELMELD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-opus-4-7",
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

- **Use the OpenAI-compatible provider shape**. Put ModelMeld under
  `language_models.openai_compatible`, not under the built-in `openai`
  provider.
- **Zed settings are JSON**. Use `settings.json` for this provider;
  YAML snippets from other tools do not apply directly.
- **Keep `/v1` in `api_url`**. Zed appends chat-completions paths to
  that base URL.
- **Launch Zed with the key available**. Zed reads provider keys from
  secure storage or environment variables; for a provider named
  `ModelMeld`, use `MODELMELD_API_KEY`.
- **Select the custom provider model**. If Zed is still using another
  provider in the model dropdown, requests will not reach ModelMeld.
- **Routing hints are not exposed in Zed settings**. Use ModelMeld's
  classifier defaults for normal Zed traffic. If you need custom
  `x-modelmeld-*` headers, run a small local proxy that injects them and
  point Zed's `api_url` at that proxy.

## See also

- [Zed LLM providers](https://zed.dev/docs/ai/llm-providers)
- [ModelMeld integrations overview](README.md)

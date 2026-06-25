# Cursor → ModelMeld

[Cursor](https://cursor.com) is a VS Code-fork IDE with deep AI
integration (autocomplete, chat, agent mode). It speaks OpenAI's
`POST /v1/chat/completions` and supports a custom base URL.

## Minimal setup

In Cursor:

1. Open **Settings → Models** (or `Cmd/Ctrl+Shift+P` → "Cursor Settings: Models")
2. Scroll to **OpenAI API Key** section. Toggle ON.
3. Paste your ModelMeld API key (`gws_<your-modelmeld-key>`) into the API Key field.
4. Click the gear ⚙️ icon next to the API Key field → **Override OpenAI Base URL**
5. Set base URL to `https://api.modelmeld.ai/v1`
6. Click "Verify" — should turn green.
7. In the model dropdown, add a custom model name. Pick one of:
   - `claude-opus-4-7` (frontier; we route to whatever's competent + cheaper)
   - `gpt-5` (frontier)
   - Or our auto-route alias: any model name works as a placeholder — our scout decides.

That's it. Cursor's chat, edit, and agent modes now route through
ModelMeld.

## Picking a routing policy

Cursor sends the custom model name from step 7 verbatim to the gateway,
so you can select a ModelMeld policy by using one of the three aliases as
that name instead of a placeholder:

- `anthropic/modelmeld-saver` — OSS-tier only; a hard cost ceiling.
- `anthropic/modelmeld-auto` — OSS by default; escalates to frontier on
  reasoning markers or large context.
- `anthropic/modelmeld-quality` — frontier-first; downgrades trivial
  requests to OSS.

The `anthropic/` segment is part of the alias string (not a provider
selector); type it verbatim. Any other model name still gets capability
routing, just without a policy ceiling.

## What Cursor sends us

Cursor's autocomplete and chat have very different request shapes:

- **Autocomplete** — short prompts (~1-3K tokens), low max_tokens
  (~50-200), typically no tools, streaming. Our scout routes to the
  cheap-tier (granite-4-micro, gemma-3-4b, phi-4-mini) — sub-200ms
  latency, ~$0.02/M tokens.
- **Chat / Agent** — large prompts (10-80K tokens with @-mentioned
  files), tools sometimes, full streaming with tool-call deltas. Our
  scout routes to OSS-mid (qwen3-coder-flash) or OSS-premium
  (qwen3-coder-480b, deepseek-v4-pro) depending on prompt complexity.

You don't have to configure this — the scout decides per-request.

## Per-request hint headers (optional)

If you want to bias routing for a specific Cursor session, add custom
headers. Cursor's "Custom OpenAI Base URL" config doesn't expose a
headers field by default, but you can run a small local proxy that
injects them. Pattern:

```python
# tiny_proxy.py — pass-through with header injection
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
client = httpx.AsyncClient(base_url="https://api.modelmeld.ai")

@app.post("/v1/chat/completions")
async def proxy(req: Request):
    body = await req.body()
    headers = dict(req.headers)
    headers["x-modelmeld-quality-threshold"] = "0.85"  # higher quality bar
    headers["x-modelmeld-agent-role"] = "coder"
    r = await client.post("/v1/chat/completions", content=body, headers=headers)
    return r.content
```

Then point Cursor at `http://localhost:8000/v1`.

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Verify the routing

Cursor doesn't surface response headers in its UI. To see what model
actually served your request, check the ModelMeld dashboard when it ships,
or your account-level audit log via the `/v1/account/usage` endpoint
(admin access).

## Common gotchas

- **Cursor caches the base URL** — after changing it, restart Cursor
  (Cmd/Ctrl+Q then relaunch).
- **The "Verify" button calls `GET /v1/models`** — our endpoint
  responds correctly with the lineup; if Verify is red, double-check
  the URL has `/v1` suffix.
- **Cursor's "Pro" tier auto-routing** — Cursor itself has internal
  routing for their managed Pro tier. When you switch to your own
  base URL, you're opting out of Cursor's routing entirely; ours
  takes over.

## See also

- [Cursor docs — Custom OpenAI Base URL](https://docs.cursor.com/api-keys)
- ModelMeld README — request fingerprinting + scout filters

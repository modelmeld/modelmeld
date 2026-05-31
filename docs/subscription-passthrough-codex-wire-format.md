# Codex Backend Wire Format — Reference

Wire-level reference for `chatgpt.com/backend-api/codex/responses`, reverse-engineered from Simon Willison's `llm-openai-via-codex` plugin (commit `ba5b023`, file `llm_openai_via_codex.py`, 2026-04-23). Used to scope a ModelMeld adapter; not for adapter code.

## 1. Request shape

The plugin does **not** hand-roll HTTP. It instantiates `openai.OpenAI(base_url="https://chatgpt.com/backend-api/codex", api_key=<bearer>, default_headers={...})` and calls `client.responses.create(**kwargs)`. The SDK appends `/responses` and emits a standard OpenAI Responses-API POST. The kwargs dict (line ~325):

```python
{
    "model": self.model_name,            # e.g. "gpt-5.5", "gpt-5.4-mini" — slug from /models
    "input": messages,                   # Responses-API input array (NOT chat/completions "messages")
    "store": False,                      # required: ChatGPT backend rejects store=True without an account context
    "stream": True,                      # SSE only; non-stream path not exercised
    "instructions": prompt.system        # system prompt; defaults to "You are a helpful assistant."
        or "You are a helpful assistant.",
    # Optional:
    "reasoning": {"effort": reasoning_effort},   # "minimal" | "low" | "medium" | "high" | "xhigh"
    "max_output_tokens": <int>,
    "temperature": <float>, "top_p": <float>,
    "tools": [{"type": "function", "name": ..., "description": ..., "parameters": {...}, "strict": False}],
}
```

`input` items follow the Responses API content-part schema: `{"role": "user", "content": [{"type": "input_text", "text": "..."}]}` and `{"type": "input_image", "image_url": <url>, "detail": "low"}` for attachments. Assistant turns are `{"role": "assistant", "content": <text>}`.

## 2. Headers

Built in `_get_client_kwargs`:

```
Authorization: Bearer <access_token>          # from ~/.codex/auth.json tokens.access_token
ChatGPT-Account-ID: <account_id>              # only if present in tokens.account_id; omit otherwise
```

Plus whatever the `openai` SDK adds itself (`User-Agent: OpenAI/Python <ver>`, `Content-Type: application/json`, `Accept: text/event-stream`).

**Explicitly absent** in the plugin: `OpenAI-Beta`, `originator`, `session_id`, custom `User-Agent`, `x-codex-*`. The official Codex CLI sends `originator: codex_cli_rs` and `version` headers; the plugin gets away without them, so they appear non-required for `/responses`.

## 3. Response shape

Standard OpenAI Responses-API SSE stream. Plugin's `_handle_event` cares about three event types (lines ~297–315):

- `response.output_text.delta` → `event.delta` is the next text chunk.
- `response.output_item.done` → if `data.type == "function_call"`, read `call_id` (fallback `id`), `name`, `arguments` (JSON string).
- `response.completed` → final envelope; `event.response.model_dump()` is captured for usage stats.

SSE framing and JSON parsing are handled by the `openai` SDK's streaming iterator — there is no manual `data:` line parsing in the plugin.

## 4. Auth token handling

`borrow_codex_key()` reads `${CODEX_HOME:-~/.codex}/auth.json`:

```python
tokens = data["tokens"]
access_token  = tokens["access_token"]
account_id    = tokens.get("account_id")     # optional
refresh_token = tokens.get("refresh_token")
```

`access_token` is a JWT. The plugin decodes the `exp` claim (`_jwt_exp`) and, if `time.time() >= exp - 30s` (`REFRESH_SKEW_SECONDS = 30`), POSTs to `https://auth.openai.com/oauth/token`:

```json
{"client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
 "grant_type": "refresh_token",
 "refresh_token": "<current>"}
```

Response fields `access_token` / `id_token` / `refresh_token` are written back to `auth.json` if present. The token is then passed to the SDK as `api_key=` so the SDK emits it as `Authorization: Bearer ...`.

## 5. Quirks worth knowing

- **Base URL has no `/v1`** — `https://chatgpt.com/backend-api/codex`, SDK appends `/responses` and `/models`.
- **`store: False` is mandatory.** ChatGPT-side persistence is not available to subscription callers.
- **Streaming-only in practice.** Plugin never sets `stream=False`; safest to mirror that.
- **Model discovery is a separate raw `urllib` GET** to `/models?client_version=1.0.0`, same auth headers. Filter `supported_in_api == true && visibility == "list"`, take `slug`. Fallback list: `["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]`. GPT-5.5 appears in the live response but not the fallback.
- **No model-name rewriting** — slug from `/models` is sent verbatim as `model`.
- **`client_id` is the Codex CLI's** OAuth public client (`app_EMoamEEZ73f0CkXaXp7hrann`) — reuse identifies our traffic as Codex-CLI-shaped.
- **Tools use Responses-API shape** (`{"type": "function", "name": ..., "parameters": ...}`), not chat/completions' nested `{"type": "function", "function": {...}}`.
- **`strict: False`** on tool defs — strict mode is not exercised; unknown whether the backend supports it.
- **Errors surface as `openai.APIError` subclasses** via the SDK; no Codex-specific error envelope is decoded by the plugin.
- **ToS posture:** plugin is "semi-official backdoor" per Simon's 2026-04-23 post; treat the same way internally.

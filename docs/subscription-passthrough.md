# Subscription passthrough (power-user opt-in)

ModelMeld can forward your dev-tool's request verbatim to OpenAI's
Codex backend or Anthropic's Messages API using an OAuth bearer token
from a **subscription** account (ChatGPT Plus/Pro/Business, Claude
Max). The gateway never persists the token; the request flows
end-to-end with the bearer preserved and the upstream sees a request
indistinguishable from a direct dev-tool call.

This page is for **self-host** operators only. Subscription passthrough
is opt-in and intentionally absent from the Hosted Tier — vendor terms
prohibit account-sharing or multi-tenant pooling of subscription
credentials.

## ⚠ Before you turn this on

- **Self-host only.** Single-user-per-instance. Sharing your gateway
  with anyone who has access to your subscription-authenticated
  endpoint is a TOS violation on both vendors.
- **No SLA on the upstream.** OpenAI's `chatgpt.com/backend-api/codex`
  endpoint is undocumented and can change without notice. Anthropic
  explicitly prohibits OAuth-token use outside Claude Code and
  Claude.ai (see the ToS-posture comparison memo); we forward
  headers verbatim to stay indistinguishable, but Anthropic could
  enforce against the pattern at any time.
- **You are responsible for compliance** with your subscription's
  terms. The feasibility memos in `docs/` document the posture as of
  2026-05-30; check whether either vendor has updated their stance
  before relying on it.

## Enabling it

Set the opt-in flag in your gateway's environment:

```bash
export MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=1
```

Without this flag, an inbound request carrying a JWT-shaped
`Authorization: Bearer eyJ...` header returns HTTP 403 with an
explicit message naming this env var — silent fallthrough would
yield a confusing downstream auth error from the upstream provider.

## Path 1: ChatGPT subscription via the Codex CLI surface

**Compatible tools:** Any tool that POSTs to `/v1/chat/completions`
with a Codex-CLI OAuth JWT in `Authorization: Bearer`. The reference
implementation is Simon Willison's
[`llm-openai-via-codex`](https://github.com/simonw/llm-openai-via-codex)
plugin — it reads the bearer from `~/.codex/auth.json` and forwards
to whatever base URL `llm` is configured against.

```bash
# 1) Make sure you've logged in via Codex CLI's "Sign in with ChatGPT"
codex login    # opens browser, completes OAuth, writes ~/.codex/auth.json

# 2) Point a /v1/chat/completions client at this gateway with the JWT.
#    Example via Willison's llm plugin (single-user terminal session):
llm install llm-openai-via-codex
export LLM_OPENAI_BASE_URL=https://your-gateway/v1
llm -m gpt-5.4 "summarize this readme"
```

**What happens on the wire:**
- Your client POSTs to `https://your-gateway/v1/chat/completions` with
  `Authorization: Bearer <Codex OAuth JWT>`.
- The gateway classifies the bearer as `OAUTH_BEARER` (JWT prefix
  `eyJ`), confirms the opt-in flag, builds a `CodexPassthroughAdapter`
  for THIS request only, and routes via a `SingleAdapterRouter`.
- The adapter translates the OpenAI-shape chat-completions request
  into the OpenAI Responses API shape Codex expects (input array +
  instructions string), POSTs to
  `https://chatgpt.com/backend-api/codex/responses` with the bearer +
  optional `ChatGPT-Account-ID` header.
- The Responses-API output is translated back into a
  `ChatCompletion` response your client expects.
- The bearer is never persisted to the gateway's disk; no log line
  contains the token bytes (only a length-preserving redaction).

**What does NOT work yet (in v0.4.0):**
- Native Codex CLI plug-and-play: Codex CLI hardcodes the
  `/responses` endpoint (no `/v1` prefix). Until ModelMeld exposes
  that surface (tracked separately), use the `llm-openai-via-codex`
  pattern above.

## Path 2: Claude Max via the Anthropic Messages API

**Compatible tools:** Claude Code, or any tool that POSTs to
`/v1/messages` with a Claude Max OAuth JWT in `Authorization:
Bearer`. Claude Code reads the bearer from
`~/.config/anthropic/claude-code-credentials.json` (location varies
by platform — see Anthropic's docs).

```bash
# 1) Sign in via Claude Code's OAuth flow (Claude Code does this
#    automatically on first run when ANTHROPIC_AUTH_TOKEN isn't set
#    and an API key isn't configured).

# 2) Point Claude Code at this gateway:
export ANTHROPIC_BASE_URL=https://your-gateway

# Optional: forward additional headers verbatim (Anthropic API version,
# beta flags). Claude Code adds these automatically.
export ANTHROPIC_CUSTOM_HEADERS=""
```

**What happens on the wire:**
- Claude Code POSTs to `https://your-gateway/v1/messages` with
  `Authorization: Bearer <Claude Max OAuth JWT>`.
- The gateway classifies the bearer as `OAUTH_BEARER`, builds an
  `AnthropicAdapter` in **OAuth mode** (bypasses the `anthropic` SDK,
  which doesn't speak OAuth bearer auth — uses raw `httpx` to POST
  to `api.anthropic.com/v1/messages` with the bearer).
- Translation helpers reuse the same code path as API-key Anthropic
  routing — only the auth header layer differs.
- For streaming requests, the gateway parses Anthropic's SSE format
  manually and yields `ChatCompletionChunk` records to your client.

**Heads-up on Anthropic's posture:**
Anthropic's terms (2026-02-19 update) explicitly prohibit OAuth-token
use outside Claude Code and Claude.ai. We preserve headers verbatim
specifically so api.anthropic.com sees a request indistinguishable
from a direct Claude Code call, but Anthropic could move to detect
and enforce against the pattern at any time. Track the [feasibility
memo](./subscription-passthrough-codex-feasibility.md) for changes.

## Verifying it's working

After hitting the gateway with an OAuth bearer, check the response
headers — every passthrough request emits the standard ModelMeld
audit headers:

```
x-modelmeld-routed-to: codex_passthrough     # or "anthropic"
x-modelmeld-routed-model: gpt-5.4            # whichever model you asked for
```

The `x-modelmeld-routed-to` value will be `codex_passthrough` for the
Codex path or `anthropic` for the Claude Max path. If you see
`openai` or some other adapter, the gateway didn't take the
passthrough branch (typically: the opt-in flag isn't set, or your
client sent an API key instead of an OAuth bearer).

## Token rotation (Codex path)

Codex CLI rotates the OAuth bearer in `~/.codex/auth.json` when the
current bearer is close to expiry. The `CodexPassthroughAdapter`
handles this automatically when constructed with `auth_json_path`:

- On a 401 from `chatgpt.com/backend-api/codex`, the adapter re-reads
  `auth.json` exactly once
- If the bearer in the file has changed (CLI rotated it), the SDK
  client is rebuilt with the new bearer and the request retries once
- If the file is missing, corrupted, or unchanged, the original 401
  surfaces to the caller — the operator needs to re-authenticate via
  `codex login` (the adapter has no way to drive the OAuth flow
  itself)

The other two auth paths — explicit `access_token=` and the
`CODEX_ACCESS_TOKEN` env var — have no on-disk source to re-read, so
401 there bubbles up to the gateway client (typically as a 502 with a
sanitized error detail). Restart the gateway after running
`codex login` to pick up the rotated token.

For the Claude Max path, the OAuth bearer is sent by the client per
request (Claude Code reads it from its own credential store), so
rotation is the client's responsibility — the gateway forwards
whatever bearer arrives in the `Authorization` header verbatim.

## Troubleshooting

### Inbound 403 — "subscription_passthrough_disabled"
The opt-in flag isn't set on the gateway. Confirm the gateway
process has `MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=1` exported and
restart. The flag is read at startup, not per-request.

### Inbound 401 from `api.anthropic.com` (Claude Max path)
Your Claude Max OAuth bearer is expired or revoked. Re-run Claude
Code's OAuth flow (`claude` → re-authenticate). The bearer flows
client → gateway → upstream verbatim; we don't refresh it on your
behalf.

### Inbound 401 from `chatgpt.com/backend-api/codex` (Codex path)
With `auth_json_path` configured: the adapter has already attempted a
one-shot reload. If you're seeing a 401 in the gateway logs, either
(a) Codex CLI hasn't rotated the bearer yet — run `codex login` to
force it, or (b) the rotation produced an equally-invalid token, in
which case `codex logout && codex login` and restart the gateway.

### Routing didn't take the passthrough branch
Check the `x-modelmeld-routed-to` response header. If it shows
`openai` / `anthropic` rather than `codex_passthrough`, the gateway
classified the inbound Authorization header as an API key, not an
OAuth bearer. JWT-shaped bearers start with `eyJ` (the standard
base64-encoded JWT header `{"alg":...`). API keys (`sk-ant-*` /
`sk-*`) take the standard adapter path.

### Token rotated mid-stream
The SDK authenticates at stream-open time; the 401-on-stream-open path
is handled identically to non-streaming chat. Mid-stream the bearer
isn't re-validated, so a rotation mid-response doesn't cause a
mid-stream failure. The next request will hit 401, reload, and
proceed.

## What headers we forward to upstream

Subscription passthrough is meant to be wire-level-indistinguishable
from a direct Claude Code (or direct Codex CLI) call. For the
**Claude Max path** we forward two distinct sets of inbound headers
to `api.anthropic.com`:

| Header set | When forwarded | Why |
|---|---|---|
| `anthropic-beta`, `anthropic-version` | Always (both API-key and OAuth paths) | Customer-controlled protocol features. Without forwarding, beta features the customer activates silently fall back at our boundary. |
| `User-Agent`, `X-Stainless-arch`, `X-Stainless-async`, `X-Stainless-lang`, `X-Stainless-os`, `X-Stainless-package-version`, `X-Stainless-retry-count`, `X-Stainless-runtime`, `X-Stainless-runtime-version`, `X-Stainless-timeout` | OAuth-bearer mode only | SDK-identifying headers that ride a real Claude Code call. Preserving them at the gateway boundary keeps OAuth requests indistinguishable on the wire. Forwarding them on the API-key path would be wrong — there the gateway IS the calling SDK. |

Headers explicitly NOT forwarded (regression-guarded by tests):
`Authorization` (we re-set with the inbound bearer), any `X-ModelMeld-BYOK-*`
header (those terminate at our gateway), any `X-ModelMeld-*` routing-hint
headers (gateway-internal control plane).

For the **Codex path**, we currently only forward the OAuth bearer
(via the SDK's `api_key` parameter) and the `ChatGPT-Account-ID`
header (via SDK `default_headers`). Inbound headers from the
`/v1/chat/completions` request are NOT yet forwarded to
`chatgpt.com/backend-api/codex` — the AsyncOpenAI SDK's per-request
header-injection surface needs surgery to plumb them through.
Tracked as a follow-up.

## Disabling on short notice

If you need to kill subscription passthrough immediately (vendor
enforcement action, expired tokens hitting your audit log, etc.):

```bash
unset MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH
# Restart the gateway. From the next request on, OAuth bearers
# return HTTP 403 instead of routing.
```

Existing in-flight requests complete; new requests get the 403.
There's no global registry of OAuth tokens to revoke — by design,
the gateway never had them at rest.

## See also

- [Codex feasibility memo](./subscription-passthrough-codex-feasibility.md)
  — ToS posture comparison + the strategic decision to ship Codex
  passthrough alongside Claude Max.
- [Codex wire-format reference](./subscription-passthrough-codex-wire-format.md)
  — the exact request/response shape at
  `chatgpt.com/backend-api/codex`, cited from Simon Willison's
  reference plugin.
- [Open-core boundary](./open-core-boundary.md) — why
  subscription passthrough is OSS rather than enterprise (it's a
  power-user opt-in, not a sellable feature; multi-tenant pooling
  would violate vendor ToS).

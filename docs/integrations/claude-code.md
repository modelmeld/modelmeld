# Claude Code → ModelMeld

[Claude Code](https://docs.anthropic.com/claude-code) is Anthropic's CLI
coding agent. It talks `POST /v1/messages` (Anthropic Messages format,
not OpenAI), uses SSE streaming, and sends `cache_control` breakpoints
on system prompts + early messages for prompt caching.

ModelMeld is a drop-in replacement for `api.anthropic.com`. Point Claude
Code at the gateway with `ANTHROPIC_BASE_URL` and everything works
end-to-end with cost parity vs going direct to Anthropic.

## One-command setup (recommended)

> **Self-hosting?** If you're running the gateway yourself (the
> public-today path — the hosted gateway is invite-only beta), use
> `modelmeld setup --self-host` instead. It prompts for your provider
> keys (OpenRouter / Fireworks / Together / vLLM), enables capability
> routing, and verifies real OSS routing before finishing. No ModelMeld
> API key required. See the [Quickstart](../../README.md#quickstart).
> The hosted-gateway flow below needs a `gws_` key.

```bash
pip install modelmeld
modelmeld setup --tool claude-code
```

The CLI:
1. Prompts for your ModelMeld API key (and optionally your Anthropic key for
   BYOK frontier routing)
2. Writes a sourceable shell script at `~/.modelmeld/setup-claude-code.sh`
   with all the required env vars (LF-only line endings, mode 0600)
3. Pre-writes Claude Code's discovery cache at
   `~/.claude/cache/gateway-models.json` so the `/model` picker shows
   the three ModelMeld auto-route aliases (works around an upstream
   Claude Code v2.1.150 bug — see "Common gotchas" below)
4. Smoke-tests the whole flow with a real /v1/models call + two
   /v1/messages calls (OSS via `-saver`, BYOK via `-quality` if you
   supplied an Anthropic key)
5. Prints clear next-step instructions, OR clear errors with the exact
   fix if anything fails

After it completes, source the script and launch Claude Code:

```bash
source ~/.modelmeld/setup-claude-code.sh
claude
```

In Claude Code, type `/model` and pick the tier that fits your work:

- **ModelMeld — Saver** (OSS-only; predictable cost ceiling)
- **ModelMeld — Auto** (OSS by default; escalates to frontier on reasoning markers)
- **ModelMeld — Quality** (frontier-first; requires BYOK)

To persist across shells, add `source ~/.modelmeld/setup-claude-code.sh` to
your `~/.bashrc` / `~/.zshrc` / equivalent.

## Manual setup (if you can't pip install)

If you can't run the CLI for some reason, here are the env vars it
sets — paste them into your shell rc directly:

```bash
export ANTHROPIC_BASE_URL=https://api.modelmeld.ai
export ANTHROPIC_API_KEY=gws_<your-modelmeld-key>
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
unset ANTHROPIC_AUTH_TOKEN  # avoid 'auth conflict' warning

# Optional — BYOK frontier routing (only needed for -quality / -auto-escalated):
export ANTHROPIC_CUSTOM_HEADERS="x-modelmeld-byok-anthropic: sk-ant-<your-anthropic-key>"
```

You'll also need to pre-write the picker cache (see "Common gotchas →
gateway-model-discovery" below for the file format).

```powershell
# PowerShell — Windows
$env:ANTHROPIC_BASE_URL = "https://api.modelmeld.ai"
$env:ANTHROPIC_API_KEY  = "gws_<your-modelmeld-key>"
$env:CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1"
Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
claude
```

That's it. Claude Code now routes every request through ModelMeld for
every chat turn — whichever model you pick (or the default `claude-opus-4-7`)
flows through `/v1/messages` to our scout, which selects the right OSS
backend based on task category, tool usage, and context size.

The `/model` picker requires a small extra setup step — see "Surfacing
auto-route aliases in the /model picker" below.

## What we preserve faithfully (vs other gateways)

These are the table-stakes Anthropic-format features that some routers
strip — we don't:

- **`cache_control` breakpoints** — your prompt cache hits work. Going
  through us doesn't break prompt caching. (musistudio/claude-code-router
  strips these via their "cleancache" transformer, causing prompt-cache
  misses on every request; we forward them verbatim.)
- **`anthropic-beta` headers** — activated beta features (prompt
  caching, fine-grained tool streaming, etc.) reach the model layer.
- **`anthropic-version` header** — version pinning honored.
- **`input_json_delta` streaming** — partial-JSON tool-call argument
  deltas forwarded byte-accurately. Tool calls don't break under us.
- **Both auth headers** — `x-api-key` (SDK default) AND
  `Authorization: Bearer` accepted. No need to set
  `ANTHROPIC_AUTH_TOKEN` — vanilla `ANTHROPIC_API_KEY` works.
- **`/v1/messages/count_tokens`** — Claude Code's cost-display UI
  queries this; we return accurate counts.

## Verify the routing took effect

After any request, the response carries audit-trail headers:

```text
x-modelmeld-routed-model: deepseek/deepseek-v4-pro
x-modelmeld-task-category: coding
x-modelmeld-quality-threshold: 0.80
```

You can grep the headers from `claude --verbose` output or use the
`/cost` slash-command in-session to see per-request cost (which
reflects ModelMeld's routed model + actual provider pricing).

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Surfacing auto-route aliases in the /model picker

Claude Code 2.1.126+ ships an opt-in "gateway model discovery" feature
(`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`) that's *supposed* to
query our `/v1/models` at startup and populate `/model` with a "From
gateway" section. In practice the discovery **fetcher** silently fails
on third-party gateways in current Claude Code releases
(upstream issue [anthropics/claude-code#58581](https://github.com/anthropics/claude-code/issues/58581)) —
the env-flag is honored, the gates all pass, but the HTTP fetch never
writes the cache file. The picker's **cache reader** still works
correctly, so a one-time setup script that pre-writes the cache file
fixes the UX.

**Routing works regardless of the picker** — this is purely a UX nicety
so you can pick `ModelMeld — Coding (auto-route)` from the menu instead
of typing the alias manually.

### One-time setup (bash/macOS/Linux)

```bash
# 1. Make sure these env vars are exported in your shell rc:
export ANTHROPIC_BASE_URL=https://api.modelmeld.ai
export ANTHROPIC_API_KEY=gws_<your-modelmeld-key>
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
unset ANTHROPIC_AUTH_TOKEN   # avoid the auth-conflict warning

# 2. Pre-write the cache file in the wrapper format Claude Code expects:
mkdir -p ~/.claude/cache
curl -s "$ANTHROPIC_BASE_URL/v1/models" \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  | python3 -c '
import json, sys, os
raw = json.load(sys.stdin)
print(json.dumps({
    "baseUrl": os.environ["ANTHROPIC_BASE_URL"],
    "fetchedAt": 1748192400000,
    "models": [{"id": m["id"], "display_name": m.get("display_name") or m["id"]} for m in raw["data"]],
}, indent=2))
' > ~/.claude/cache/gateway-models.json

# 3. Launch claude in the SAME shell where the env vars are set:
claude
# /model — you'll see "From gateway" section with the 8 ModelMeld entries
```

### One-time setup (PowerShell/Windows)

```powershell
$env:ANTHROPIC_BASE_URL = "https://api.modelmeld.ai"
$env:ANTHROPIC_API_KEY  = "gws_<your-modelmeld-key>"
$env:CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1"
Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue

$cacheDir = "$env:USERPROFILE\.claude\cache"
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

$resp = Invoke-RestMethod -Uri "$env:ANTHROPIC_BASE_URL/v1/models" `
    -Headers @{ "Authorization" = "Bearer $env:ANTHROPIC_API_KEY" }
$wrapped = @{
    baseUrl   = $env:ANTHROPIC_BASE_URL
    fetchedAt = 1748192400000
    models    = $resp.data | ForEach-Object {
        @{ id = $_.id; display_name = if ($_.display_name) { $_.display_name } else { $_.id } }
    }
}
$wrapped | ConvertTo-Json -Depth 10 | Set-Content -Path "$cacheDir\gateway-models.json" -Encoding UTF8
claude
```

### Alternative: ANTHROPIC_CUSTOM_MODEL_OPTION (single-entry workaround)

If pre-writing the cache file is too fiddly, you can add a single
ModelMeld auto-route alias to the picker via the supported manual API:

```bash
export ANTHROPIC_CUSTOM_MODEL_OPTION=anthropic/modelmeld-saver
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="ModelMeld Saver"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Auto-routed across OSS models (max savings, predictable ceiling)"
```

This adds one entry to `/model` without touching the cache file — a
clean fallback when the discovery cache approach is awkward (locked-down
home dirs, ephemeral containers, etc.).

### Why both pre-write *and* env vars

Claude Code's cache reader gates on the same env vars discovery uses
(`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`, `ANTHROPIC_BASE_URL`,
non-firstParty auth), so a pre-written cache only surfaces when those
vars are exported in the **shell that launches `claude`** — process-env
inheritance through inline-prefix launches (`VAR=x claude`) doesn't
always reach the picker. Set them in your shell rc to be safe.

## Picking a routing policy

The `/model` picker exposes three ModelMeld auto-route aliases. Each
represents a different cost-quality ceiling. Pick the one that matches
your priority; switch tiers in-session anytime if the work changes.

### `anthropic/modelmeld-saver` — max savings, predictable ceiling

Stays inside the OSS provider tier. Frontier (Anthropic/OpenAI direct)
models are filtered out of the candidate pool entirely — they cannot
be picked. The scout's existing logic still chooses the right OSS
model for the request (cheap autocomplete tier for autocomplete-shape,
qwen3-coder for normal coding, deepseek-r1 / qwen-32b for complex
multi-file work).

**Predictable cost ceiling — frontier rates never apply.** Use when
you're cost-sensitive and willing to occasionally hint the agent past
a subtle bug (see "What to expect" below).

### `anthropic/modelmeld-auto` — smart default, frontier on demand

Starts in OSS tier (same as `-saver`). Automatically **escalates to
frontier (Sonnet 4.6)** when the user message contains 2+ reasoning
markers like "step by step", "explain your reasoning", "show your work",
"carefully consider", etc. — mirroring LiteLLM's Complexity Router
trigger but with a transparent, operator-tunable marker list.

**OSS-tier rates on most traffic; frontier rates apply only when the
reasoning-marker trigger fires.** Use when you want intelligent
escalation without managing the picker manually.

To customize the escalation triggers:

```bash
# Replace the default marker list:
export MODELMELD_REASONING_MARKERS="custom phrase,another marker"

# Or append to the defaults with a leading +:
export MODELMELD_REASONING_MARKERS="+extra one,extra two"
```

System prompts are NOT scanned for markers — only the actual user
messages. Claude Code's system prompts often contain "think step by
step" boilerplate which would cause spurious escalation otherwise.

### BYOK — bring your own frontier key

The hosted gateway does **not** hold any frontier-provider keys. To use
`-auto` (when it escalates) or `-quality`, pass your own key via a
custom header — your key transits the gateway for ~50ms, gets used to
serve the request, and is forgotten. **Never stored at rest, never
logged, never echoed in error responses.**

For Claude Code, leverage its native `ANTHROPIC_CUSTOM_HEADERS` env var:

```bash
export ANTHROPIC_BASE_URL=https://api.modelmeld.ai
export ANTHROPIC_API_KEY=gws_<your-modelmeld-key>
export ANTHROPIC_CUSTOM_HEADERS="x-modelmeld-byok-anthropic: sk-ant-<your-anthropic-key>"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
unset ANTHROPIC_AUTH_TOKEN
claude
```

Claude Code injects the BYOK header on every outbound request. Our
gateway extracts it, uses it for the upstream Anthropic call if the
scout chose a frontier model, and discards it after the response. If
the scout stayed in OSS tier (which it does for most `-saver` and
unescalated-`-auto` traffic), the BYOK header is simply ignored —
no quality loss, no cost added.

For OpenAI keys (frontier `gpt-5` etc.):
```
x-modelmeld-byok-openai: sk-<your-openai-key>
```

You can supply both at once and we'll use whichever the scout picks
for that turn. Header names are case-insensitive.

**If you forget the BYOK header and the scout picks frontier**, we
return a 400 with the exact env var you need:
```json
{
  "error": "byok_required",
  "detail": "Set ANTHROPIC_CUSTOM_HEADERS='x-modelmeld-byok-anthropic: sk-ant-...'",
  "provider_missing": "anthropic"
}
```

That's far friendlier than the silent 503 you'd get from a key-custody
gateway. Stay on `-saver` if you don't want to bring frontier keys at
all — you get our full OSS routing without any BYOK setup.

### `anthropic/modelmeld-quality` — frontier-first with smart trimming

Starts at frontier (Sonnet 4.6 baseline). Downgrades to OSS only on
detected trivial shape — autocomplete-style requests (small input,
short max_tokens, no tools) get routed cheap. Everything else stays
on frontier.

**Frontier-first with trivial-shape downgrades** — autocomplete-style
requests and background calls bypass Opus billing while everything
else stays on frontier. Use when correctness matters more than cost,
or for mission-critical agentic workflows where the retry-on-failure
budget is tight.

### Why pick `-quality` over a direct Anthropic API key?

A reasonable question — if you want frontier, why route through us?

1. **Trivial-task downgrades.** Claude Code emits a non-trivial share
   of background calls (titles, summaries, classification). Direct API
   charges Opus rates for all of them; `-quality` routes those to
   Haiku/qwen-flash, reducing per-session cost with zero quality
   regression on the work that matters.
2. **Multi-provider load balancing.** When direct Anthropic hits a
   rate limit, we keep the session flowing across our spread of
   Anthropic accounts + alternatives. Direct API has nothing.
3. **Single API key for everything.** Run `-quality` in production
   and `-saver` for batch jobs from the same SDK with the same key.
4. **Audit headers.** `x-modelmeld-routed-model` on every response
   tells you exactly which model served each turn. Direct API
   doesn't surface that.

## What to expect with OSS-routed coding

**For most coding work the experience is indistinguishable from Opus**
— file edits, multi-step refactors, debugging loops, test cycles all
work cleanly. Our internal complex-test benchmark (build a dogfood CLI
tool with pytest, ~40 turns including failure-recovery iterations)
completed end-to-end with no frontier fallback, served entirely on
OSS-tier infrastructure.

**Where OSS models lag (~5% of tool calls)**: occasional malformed
`tool_use` blocks, usually a missing required parameter (e.g., `Write`
called without `file_path`). Claude Code's agent loop **retries
automatically** on schema validation errors, so the user-visible
behavior is one extra second of latency, no failed action. If you're
running a tool-heavy workload where 100% schema fidelity matters,
pick `anthropic/modelmeld-quality` (or `-auto` with reasoning markers
in your prompt) — they route through Anthropic Sonnet/Opus directly
for those turns, trading the cost savings for fidelity.

**Where OSS models also lag (rare, ~30% of complex architectural
tasks)**: missing subtle contract details across multiple files (e.g.,
returning a class when tests expect an instance). Switch to
`anthropic/modelmeld-auto` and add a reasoning marker to your prompt
("think step by step about ..."), or to `-quality` for a session,
when you hit one of these.

We track these metrics continuously; see [`../which-tier.md`](../which-tier.md)
for current measured numbers per tier.

## Backwards compatibility — old aliases

The previous 5-alias lineup (`balanced`, `coding`, `reasoning`, `cheap`,
`frontier-priority`) is still honored. Each maps to the nearest new
policy:

| Old alias | New policy mapping |
|---|---|
| `anthropic/modelmeld-balanced` | `-auto` |
| `anthropic/modelmeld-coding` | `-saver` |
| `anthropic/modelmeld-reasoning` | `-auto` |
| `anthropic/modelmeld-cheap` | `-saver` |
| `anthropic/modelmeld-frontier-priority` | `-quality` |

Existing integrations continue to work without code changes. The
`/model` picker shows only the three canonical aliases — funnel new
customers to those.

## Common gotchas

- **Trailing slash on `ANTHROPIC_BASE_URL`** — Anthropic SDK is
  trailing-slash-sensitive. Use `https://gateway.example.com` (no
  trailing `/`). 404s are the usual symptom of getting this wrong.
- **`apiKeyHelper` config** — if you use a token-refresh script via
  `apiKeyHelper`, our gateway accepts the token from EITHER
  `Authorization: Bearer` OR `x-api-key`. Both work; pick one.
- **`CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1`** — recommended.
  Some proxies that buffer SSE break it and cause duplicate tool
  execution. We don't buffer SSE, but setting this avoids any
  intermediate-proxy weirdness in front of us.
- **`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`** — set this if you
  want the `/model` picker to show our auto-route aliases. See the
  "Surfacing auto-route aliases in the /model picker" section above —
  this flag alone isn't enough due to an upstream Claude Code bug; you
  need the cache pre-write step too.
- **`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_API_KEY` both set** — Claude
  Code warns "Auth conflict" and gateway discovery silently bails. Set
  exactly one (we recommend `ANTHROPIC_API_KEY`; `unset` the other).

## See also

- [Anthropic's docs on `ANTHROPIC_BASE_URL`](https://code.claude.com/docs/en/llm-gateway)
- [Anthropic environment variables reference](https://code.claude.com/docs/en/env-vars)
- ModelMeld README → API surface section for endpoint + auth details

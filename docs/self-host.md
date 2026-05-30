# Self-hosting ModelMeld

The quickstart for running the gateway on your own hardware, with your
own upstream provider keys. If you'd rather use the hosted endpoint at
api.modelmeld.ai, see [the Claude Code integration
guide](integrations/claude-code.md) — `modelmeld setup --tool claude-code`
will wire you up in one command.

## What you need

- Python 3.10+ (3.11 or 3.12 recommended)
- ~500 MB of disk for the package + dependencies
- An upstream provider API key (Anthropic, OpenAI, or both)
- Optional: a local vLLM endpoint for the OSS-tier path (no GPU required
  if you're routing purely to frontier providers)

## Install

```bash
pip install 'modelmeld[anthropic,openai]'
```

Extras `anthropic` and `openai` pull in those provider SDKs. Omit them
if you're only routing to vLLM or a different provider.

## Run

The simplest configuration: route to Anthropic for everything, no local
model.

```bash
export ANTHROPIC_API_KEY=sk-ant-…
uvicorn modelmeld.api.server:app --host 0.0.0.0 --port 8080
```

Point your tool at `http://localhost:8080`. It speaks both
`/v1/chat/completions` (OpenAI shape) and `/v1/messages` (Anthropic
shape), so Cursor / Aider / Continue / Cline / Claude Code all work
unchanged.

## Multi-provider routing

For the cost-savings story, configure multiple upstreams. The gateway
will pick the cheapest model meeting each request's quality bar.

```bash
# Frontier provider keys — used by -quality (always) and -auto (on
# escalation). Set whichever frontier vendors you want available.
export ANTHROPIC_API_KEY=sk-ant-…
export OPENAI_API_KEY=sk-…

# OSS-tier provider keys — used by -saver and the OSS path of -auto.
# Set at least one of these, OR set MODELMELD_VLLM_ENDPOINT below.
export FIREWORKS_API_KEY=…
export TOGETHER_API_KEY=…
export OPENROUTER_API_KEY=…

# Optional: route OSS-tier requests to a self-hosted vLLM instead of
# (or in addition to) the hosted-API OSS providers above.
export MODELMELD_VLLM_ENDPOINT=https://your-vllm.example.com/v1
export MODELMELD_VLLM_API_KEY=…    # if your vLLM requires auth

uvicorn modelmeld.api.server:app --host 0.0.0.0 --port 8080
```

When a client picks `anthropic/modelmeld-saver` from the model picker,
the scout restricts to OSS providers (vllm / fireworks / together /
openrouter). When they pick `-quality`, the scout restricts to frontier
(anthropic / openai). The decision is logged in the
`x-modelmeld-routed-model` and `x-modelmeld-routed-to` response headers.

**Multi-provider routing works out of the box.** The bundled
`default_registry.json` plus `default_overlay.json` ship with a curated
set of tool-capable OSS models tagged for vLLM AND the cloud OSS
providers (Fireworks / Together / OpenRouter). Set whichever provider
keys you have — `-saver` and `-auto` will route across them
automatically, picking the cheapest qualified model per request. No
separate overlay configuration required.

The bundled overlay covers the most common canonical models
(qwen3-coder-30b, deepseek-v4-pro, llama-3.3-70b, gpt-oss-120b,
qwen3-coder-flash, and a handful more) with conservative public list
prices. For the **full curated lineup**, weekly model + benchmark
refreshes, and provider-reliability tracking, point at the live feed
via `MODELMELD_REGISTRY_FEED_URL` + a license key — see
[`registry-feed.md`](registry-feed.md).

## Verify it's working

```bash
curl -s http://localhost:8080/healthz
# → {"status": "ok"}

curl -s http://localhost:8080/v1/models | python -m json.tool | head -20
# → list including the three modelmeld-* aliases

curl -s http://localhost:8080/v1/messages \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"anthropic/modelmeld-saver","max_tokens":64,
       "messages":[{"role":"user","content":"hi"}]}'
# → response with x-modelmeld-routed-model in headers
```

## Per-request observability

Every successful response includes audit headers documenting the
routing decision:

```
x-modelmeld-routed-model: deepseek/deepseek-r1-distill-llama-70b
x-modelmeld-task-category: coding
x-modelmeld-task-score: 0.88
x-modelmeld-quality-threshold: 0.70
x-modelmeld-routed-to: openrouter
x-modelmeld-tier: cloud
```

Plus `x-modelmeld-bias` (when a shape bias fires, e.g. autocomplete) and
`x-modelmeld-failover-from` (when a primary upstream failed and we
retried on a fallback).

## Health observability

`GET /healthz/backends` returns per-upstream counters + circuit breaker
state. Use it for monitoring dashboards. Sample response:

```json
{
  "status": "ok",
  "backends": {
    "openrouter": {
      "last_success_at": "2026-05-25T19:48:18Z",
      "consecutive_errors": 0,
      "success_count": 14,
      "error_count": 0
    },
    "anthropic": {...}
  },
  "circuit_breakers": {
    "qwen3-coder-flash@openrouter": {"state": "closed", ...}
  }
}
```

For aggregated counts across multiple workers / restarts, set
`MODELMELD_REDIS_URL` — the monitor will also write totals to Redis
which appear as `success_count_total` / `error_count_total` in the
response.

## Production hardening

For real production deploys (not just `uvicorn` on a dev box):

- **Workers**: `uvicorn --workers 4` or use gunicorn with the uvicorn
  worker. The gateway is fully async and CPU-light; you can saturate
  network capacity with 2-8 workers on a typical 4-core box.
- **Reverse proxy**: terminate TLS upstream (Caddy, nginx, ALB, etc.).
  The gateway doesn't ship HTTPS — give it HTTP and let your proxy
  handle certs.
- **Logs**: stdout is structured JSON via stdlib `logging` (you can
  customize via `logging.config.dictConfig` if you want; default
  works fine for `docker logs` / `kubectl logs` / journalctl).
- **Persistence**: the gateway is stateless. No DB required for OSS
  routing — registry is read-only at boot. Add Redis only if you want
  cross-worker stats aggregation (see above) or completion caching
  (`pip install 'modelmeld[redis]'` + `MODELMELD_REDIS_URL`).
- **Monitoring**: scrape `GET /healthz` for liveness, `GET /healthz/backends`
  for routing health + per-backend latency observability.

## Common env vars

| Variable | What it does |
|---|---|
| `ANTHROPIC_API_KEY` | Upstream Anthropic key for frontier routing |
| `OPENAI_API_KEY` | Upstream OpenAI key for frontier routing |
| `FIREWORKS_API_KEY` | Fireworks AI key for OSS-tier routing (also reads `MODELMELD_FIREWORKS_API_KEY`) |
| `TOGETHER_API_KEY` | Together AI key for OSS-tier routing (also reads `MODELMELD_TOGETHER_API_KEY`) |
| `OPENROUTER_API_KEY` | OpenRouter key for OSS-tier routing (also reads `MODELMELD_OPENROUTER_API_KEY`) |
| `MODELMELD_VLLM_ENDPOINT` | Self-hosted vLLM URL (OSS-tier path) |
| `MODELMELD_VLLM_API_KEY` | Auth for the vLLM endpoint (optional) |
| `MODELMELD_REDIS_URL` | Redis URL for cross-worker stats + cache |
| `MODELMELD_DEFAULT_QUALITY_THRESHOLD` | Override the default 0.70 cutoff |
| `MODELMELD_REASONING_MARKERS` | Override the `-auto` escalation triggers |

Full list of env vars + their defaults is in
[`open-core-boundary.md`](open-core-boundary.md).

## Common troubleshooting

**Customer's request returns 503 "No healthy adapter available"**: scout
picked a model whose provider isn't configured. Either set the env var
for that provider, or pin to a different alias. Example: customer used
`anthropic/modelmeld-quality` but neither `ANTHROPIC_API_KEY` nor
`OPENAI_API_KEY` is set. Fix: set one, OR have them use
`anthropic/modelmeld-saver` (OSS-only).

**Customer's request returns 400 `byok_required`**: scout picked a
frontier model but no BYOK header was supplied AND no upstream key is
configured at the gateway level. Same fix as above — either configure
the upstream key on the gateway, or have the customer send their key
in the `x-modelmeld-byok-{provider}` header.

**`/healthz/backends` shows `consecutive_errors > 0`**: an upstream
provider is returning errors. Check `last_error` for the failure mode.
The circuit breaker will block that (model, provider) pair for 60 sec
after 3 consecutive failures, then automatically attempt recovery.

## Need more?

- [`backends.md`](backends.md) — per-provider setup notes
- [`integrations/`](integrations/) — per-tool integration guides
- [`open-core-boundary.md`](open-core-boundary.md) — what ships in OSS
  vs subscription tiers
- GitHub Discussions — questions, ideas, integration help

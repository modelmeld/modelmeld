# ModelMeld

> AI gateway with capability-based routing across providers and per-request
> bring-your-own-key passthrough. Speaks OpenAI Chat Completions AND
> Anthropic Messages natively. Drop-in for Claude Code, Cursor, Aider,
> Cline, Continue.

[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/modelmeld)](https://pypi.org/project/modelmeld/)

## What problem this solves

You're paying frontier-model prices on every request — including the ones
where a coding-tuned 7B model would produce identical output. Most gateways
force a global choice ("use Anthropic" / "use OpenAI" / "use local").
ModelMeld picks per request, with three policies you control:

- **`anthropic/modelmeld-saver`** — OSS-only. Never escalates to
  frontier. Predictable cost ceiling — you pay OSS-tier rates
  regardless of request complexity.
- **`anthropic/modelmeld-auto`** — OSS by default; escalates to frontier
  (Sonnet/Opus) when the user's prompt contains 2+ reasoning markers
  ("think step by step", "explain your reasoning", etc.). Mirrors
  LiteLLM's Complexity Router trigger.
- **`anthropic/modelmeld-quality`** — Frontier-first. Downgrades to OSS
  only on detected trivial work (autocomplete-shape, background calls).

**Frontier-tier routing uses BYOK** — your Anthropic/OpenAI key is sent
as a per-request header (`x-modelmeld-byok-anthropic: sk-ant-…`), used
to make the upstream call, then forgotten. Never stored at rest, never
logged. Same pattern as competitor gateways but without their
per-request BYOK markup or the key-custody burden.

## Quickstart

```bash
pip install modelmeld
modelmeld setup --tool claude-code
```

The setup CLI prompts for your ModelMeld API key (and optionally your
Anthropic key for BYOK), writes a sourceable shell script, pre-configures
Claude Code's `/model` picker with the three aliases above, and
smoke-tests the whole routing pipeline before declaring success.

Then in your shell:

```bash
source ~/.modelmeld/setup-claude-code.sh
claude
```

In the Claude Code TUI, type `/model` → pick `ModelMeld — Saver` (or
`Auto` / `Quality`). That's it.

## Self-host

`modelmeld setup` configures your tool against the hosted gateway at
`api.modelmeld.ai`. If you'd rather run the gateway
yourself:

```bash
pip install 'modelmeld[anthropic,openai]'
export ANTHROPIC_API_KEY=sk-ant-…   # your real Anthropic key
export OPENAI_API_KEY=sk-…           # your real OpenAI key (optional)
uvicorn modelmeld.api.server:app --host 0.0.0.0 --port 8080
```

Then point your tool at `http://localhost:8080`. The gateway's
behavior is identical to the hosted endpoint; you just supply the
upstream keys directly via env vars instead of BYOK headers.

For routing across local vLLM + cloud providers, see
[`docs/backends.md`](docs/backends.md).

## What's in the package

- **Two API surfaces, one routing pipeline:**
  - **OpenAI-compatible** at `/v1/chat/completions` — drop-in for any
    client that speaks OpenAI's wire format (Cursor, Aider, Continue,
    Cline, OpenAI SDK, Codex CLI, etc.). Plus `/v1/models` listing.
  - **Anthropic-compatible** at `/v1/messages` — drop-in for any client
    that speaks Anthropic's wire format (Claude Code via
    `ANTHROPIC_BASE_URL`, `anthropic-sdk-python`, `@anthropic-ai/sdk`).
  - Both surfaces stream via SSE, share the same scout/router/memory/
    cache pipeline, and emit identical `x-modelmeld-*` response headers.
- **Provider adapters** — OpenAI, Anthropic (with full schema translation
  in both directions), vLLM, TensorRT-LLM. Each adapter retries transient
  errors (429 / 5xx / network blip) with exponential backoff before
  surfacing to the router.
- **Capability-based routing** — `CapabilityScout` picks the cheapest
  model that meets a quality threshold for the prompt's task category,
  driven by the `ModelRegistry`.
- **Completion cache** — exact-match (in-memory or Redis) + semantic
  (Qdrant); cache key pools across users routed to the same served model.
- **PII scrubbing** — runs on every egress path before cloud upload.
- **Framework integration headers** — declare task category + agent role
  from AutoGen / CrewAI / LangGraph / OpenClaw to bypass the classifier.
- **Production-tuned defaults** — `DEFAULT_HEURISTIC_WEIGHTS`,
  `DEFAULT_QUALITY_THRESHOLD = 0.70`, full dev-tool detection catalog,
  and a current `default_registry.json` snapshot ship as the defaults.
  All tunable via constructor args.

## Licensing — code, data, and the live feed

This package is licensed under three different sets of terms.

| Component | License | What you can do |
|---|---|---|
| Python code | AGPL-3.0-or-later (LICENSE) | Use, modify, redistribute. If you offer a modified gateway as a network service to third parties, your modifications must also be AGPL. Calling the gateway over HTTP from unmodified clients (Cursor, Aider, Claude Code, etc.) does NOT make those clients AGPL. |
| Bundled snapshot data (`scout/data/default_registry.json`) | CC-BY-4.0 (with attribution) | Use the snapshot scores in your own routing decisions |
| Live curated registry feed (`feed.modelmeld.ai`) | Subscription terms | Only with an active ModelMeld subscription |

**Why AGPL and not Apache-2.0?** ModelMeld is network-service software in
the spirit of Sentry, Grafana Loki, MinIO, and GitLab CE. AGPL preserves
the OSS adoption flywheel for individual users, dev tools, and self-host
deployments — everyone running the gateway for themselves, or having their
tools call it over HTTP, is fully unaffected by the copyleft clause. AGPL
*does* prevent a competitor from forking modelmeld, layering on
proprietary tweaks, and offering a competing managed service without
contributing back. Commercial licensing for cases that need to be
exempt from AGPL: contact `hello@modelmeld.ai`. The full reasoning is in
[`docs/license-rationale.md`](https://github.com/modelmeld/modelmeld/blob/main/docs/license-rationale.md).

The **curated registry feed** is the subscription product — it's
updated continuously, editorially weighted across multiple benchmark
sources, and represents the actual ongoing work that keeps routing
decisions sharp as new models ship.

If you `pip install modelmeld` and never subscribe, **everything works**
— you just route on a snapshot of benchmark data taken at OSS release
date. Over ~6 months the foundation-model market shifts enough that
your snapshot stales relative to the current best-cost frontier. See
[`docs/registry-feed.md`](https://github.com/modelmeld/modelmeld/blob/main/docs/registry-feed.md) and
[`docs/open-core-boundary.md`](https://github.com/modelmeld/modelmeld/blob/main/docs/open-core-boundary.md) for the
boundary contract.

## Supported backends + integrations

Two complementary surfaces:

- **[Backends](https://github.com/modelmeld/modelmeld/blob/main/docs/backends.md)** — the inference providers ModelMeld
  routes *to*: OpenAI, Anthropic, vLLM (self-hosted open-weights),
  TensorRT-LLM + Triton, with Google Gemini planned. Includes setup
  snippets per backend and the explicit "not supported" list.
- **[Integrations](https://github.com/modelmeld/modelmeld/blob/main/docs/integrations/README.md)** — the frameworks
  and dev tools ModelMeld routes *for*:
  - **Validated today**: Claude Code (via `/v1/messages`), Aider, OpenAI
    SDK, anthropic SDK, AutoGen, CrewAI, LangGraph, OpenClaw. Anything
    built on OpenAI or Anthropic SDKs works as a drop-in.
  - **Should work, not yet live-tested**: Cursor, Continue, Cline,
    Codex CLI. All speak OpenAI's `/v1/chat/completions` which is our
    native dialect.
  - **Routing-hint headers** (`x-modelmeld-task-category`,
    `x-modelmeld-agent-role`, etc.) let frameworks declare task category
    and agent role explicitly instead of relying on the classifier.
    (see [Routing-hint headers reference](docs/routing-hints.md)).

## Integration scope — v1 commitments

**Wire formats we speak:** OpenAI Chat Completions API + Anthropic
Messages API. Streaming (SSE) on both, tool-use on both, multi-turn
conversations on both. The Anthropic surface translates at the HTTP
boundary; internally everything runs through the same routing pipeline.

**On the roadmap, not v1:**
- OpenAI **Responses API** (`/v1/responses`) — for clients that adopt
  OpenAI's newer surface. Current Codex CLI still uses
  `/v1/chat/completions` and works today.
- Anthropic **image content blocks** (vision input). Claude Code
  doesn't use vision; documented as deferred.

**Already shipped (was on roadmap, now live):**
- Anthropic **prompt caching** — `cache_control` breakpoints are
  forwarded verbatim to the upstream Anthropic call via native-shape
  passthrough on `/v1/messages`. Your prompt cache hits work through
  the gateway. (Many competitor gateways strip this; we don't.)

## Status

Pre-1.0. The OSS API surface is stable in spirit but not yet under
SemVer guarantees — see [`docs/api-stability.md`](https://github.com/modelmeld/modelmeld/blob/main/docs/api-stability.md)
for which symbols carry compatibility commitments.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev workflow,
code style (`ruff format` + `ruff check` + `pyright`), and DCO commit-signoff requirement.
Good first issues are labeled accordingly.

We do **not** accept PRs that modify the bundled snapshot data files
(`scout/data/`) — those are curated centrally for the live feed. File
issues against bad routing decisions you observe and we'll evaluate
adjustments for the next feed release.

## Community

- **GitHub Issues** — bugs + feature requests (after reading [`CONTRIBUTING.md`](CONTRIBUTING.md))
- **GitHub Discussions** — questions, ideas, integration help
- **Security** — see [`SECURITY.md`](SECURITY.md); report to `security@modelmeld.ai` (90-day disclosure window)

## Enterprise tier

For production deployments needing API-key auth, RBAC, OIDC SSO,
Postgres-backed SOC2-grade audit logs, encryption-at-rest, per-tenant
rate limiting, FinOps dashboards, multi-tenant Qdrant cache, or the
managed hosted tier — contact us at `hello@modelmeld.ai`.

## License

- Code: AGPL-3.0-or-later (see [LICENSE](LICENSE), [NOTICE](NOTICE), [`docs/license-rationale.md`](https://github.com/modelmeld/modelmeld/blob/main/docs/license-rationale.md))
- Data files: CC-BY-4.0 (see [scout/data/LICENSE.md](src/modelmeld/scout/data/LICENSE.md))
- Live feed: subscription terms (see [NOTICE](NOTICE) and [docs/registry-feed.md](https://github.com/modelmeld/modelmeld/blob/main/docs/registry-feed.md))

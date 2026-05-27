# Which tier should I use?

ModelMeld can be deployed in a few different ways. This doc gives you the
short answer first, then the reasoning if you want to dig.

## Short answer

**Most teams should use the Hosted tier.** Sign up, get a license key,
point your existing coding tool at the hosted endpoint, and you're
running. No GPU procurement, no Docker, no DevOps. Your frontier API
keys stay on your machine. Pay-as-you-go credits in $20 / $50 / $100
packs; the live curated routing feed is bundled in.

Self-hosting the OSS engine is the right choice if:

- Your compliance posture rules out routing data traversing third-party
  infrastructure (regulated industries, data-residency clauses, certain
  government contracts)
- Your organization already operates dedicated GPU infrastructure with
  spare capacity for inference workloads
- You're a contributor or want to develop against the engine itself

**Enterprise** (custom pricing) is the right choice if you need SSO + RBAC,
SOC 2-grade immutable audit logging, custom SLAs, or deployment on your
own infrastructure with our support. Contact `hello@modelmeld.ai`.

## The deployment options

| Tier | What it is | Routing data |
|---|---|---|
| **Hosted** *(recommended)* | We host the local model. You consume the gateway as a managed endpoint. | Pro feed bundled in |
| OSS self-host (local model) | You run the gateway + your own vLLM on your own GPU. | Bundled snapshot, or Pro feed at $29/mo |
| OSS self-host (frontier-only) | You run the gateway as a smart proxy in front of your existing frontier API keys. No local model. | Bundled snapshot, or Pro feed at $29/mo |
| Enterprise | Negotiated deployment — your hardware, our hosted, or hybrid. | Pro feed |

## When each option fits

### Hosted (recommended)

This is the right choice for almost everyone running real production
coding-tool traffic. The trade is that requests routed to the local
model traverse our infrastructure — we log request metadata + a SHA-256
prompt hash, never the prompt body itself (see
[`compliance/data-flow.md`](compliance/data-flow.md) for what we store
and what we don't). If that posture works for your team, Hosted removes
every infrastructure question between "I have a license key" and "my
tool is making routed requests."

Pricing is **$0.30 per million tokens routed to the local model**, plus
your existing frontier API spend (which we never see or store — your
keys stay on your machine).

### OSS self-host (local model)

The right choice when your compliance posture or existing infrastructure
makes self-hosting the natural path: regulated industries, data-residency
requirements, or organizations already operating dedicated GPU capacity
for inference workloads.

The gateway side is straightforward — `docker compose up`. The actual
lift is the local-model side: GPU provisioning if you don't already have
one, vLLM serving configuration, model-weight management, quantization
choices, ongoing pod-health monitoring.

You can run with the bundled routing snapshot (free, ships in OSS) or
pair with the Pro routing feed ($29/mo) — see
[`registry-feed.md`](registry-feed.md) for the comparison.

### OSS self-host (frontier-only)

The right choice when you want the routing engine's features (cost
attribution, capability-based routing across frontier providers, the
Anthropic + OpenAI dual API surface, framework integration headers)
*without* a local model in the picture. Set
`MODELMELD_ROUTING_POLICY=always_cloud` and the gateway acts as a smart
proxy in front of your existing OpenAI / Anthropic / etc. keys.

### Enterprise

The right choice when your compliance, ops, or scale needs exceed
pay-as-you-go: SSO (OIDC) + RBAC, SOC 2-grade immutable audit log with
customer-managed encryption keys, custom SLAs, dedicated engineering
support, or deployment on infrastructure that pay-as-you-go Hosted
doesn't cover. Contact `hello@modelmeld.ai`.

## Routing data: bundled snapshot vs Pro feed

This is the other axis, independent of the deployment-tier question
above. You can pair either routing-data mode with any deployment tier
(Hosted bundles the Pro feed by default).

| | Bundled snapshot | Pro feed |
|---|---|---|
| Cost | $0 (ships with OSS) | $29/mo standalone, or bundled in Hosted |
| Updates | Frozen at package release | Refreshed daily |
| Sources | Public benchmarks at snapshot time | Artificial Analysis, Aider Polyglot, LiveBench, LMArena, plus aggregated production-traffic signals |
| Curation | None | Editorial weighting, deprecation flags, gaming detection |
| Drift over time | Measurable within 3–6 months | Stays current |
| Best for | Sovereignty-strict deployments and teams already maintaining their own benchmark-aggregation pipeline | Production traffic |

Full feature comparison and wire-format details:
[`registry-feed.md`](registry-feed.md#side-by-side-what-you-actually-get).

## Switching later

Tier choice isn't a one-way door. Your code doesn't change between
tiers — just configuration:

- **Move OSS → Hosted**: change the endpoint URL your tool points at.
- **Move Hosted → self-host**: spin up your own gateway + vLLM, point your tool at it.
- **Add the Pro feed to a self-host deployment**: set `MODELMELD_REGISTRY_FEED_URL` + `MODELMELD_REGISTRY_FEED_LICENSE_KEY` + `MODELMELD_REGISTRY_FEED_PUBLIC_KEY_PEM` and restart.
- **Drop the Pro feed**: unset those env vars; the gateway falls back to the bundled snapshot automatically.

The wire format is stable across all combinations. Experiment freely.

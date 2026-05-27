# Registry feed

The **model registry** is the dataset that drives capability routing.
For each model the gateway can route to, it carries:

- Identity (`model_id`, `provider`)
- Context window
- Per-million-token cost (input + output)
- Task scores: `coding`, `reasoning`, `simple_qa`, `summarization`, `tool_use`
- Last-updated timestamp + source attribution

Routing decisions are only as good as the data driving them. Frontier models
deprecate quietly, benchmark gaming gets worse, and prices fall every quarter.
A stale registry routes traffic to expensive models that have been displaced or
cheap models that have been removed. So the registry is **continuously curated**
— that's where the moat is.

## Two modes

### Bundled seed (free)

`modelmeld` ships with `scout/data/default_registry.json` — a snapshot of
the registry at package release time. It loads automatically when no live
feed is configured.

- **What you get**: functional baseline routing, frozen at whichever benchmark
  data we had when we cut the release.
- **What you don't get**: ongoing updates. Model added next month? Not there.
  Provider drops their price by 80%? Not reflected. Benchmark gaming flagged
  by us internally? Not visible.
- **Behavior**: ModelRegistry emits a one-time WARN log at boot when running
  on seed data outside test environments. Suppress by configuring the live feed.
- **License**: CC-BY-4.0 (see `scout/data/LICENSE.md`).

### Live curated feed (subscription)

A signed JSON feed served from `feed.modelmeld.ai` (or a customer-hosted
equivalent — see `feed-server-contract.md` once published). Updates daily.

- **What you get**: current scores from Artificial Analysis + Aider Polyglot
  + LiveBench + LMArena + aggregated customer-workload signals, manually
  curated for benchmark-gaming flags, deprecation notices, weighting.
- **How it loads**: `RegistryFeedClient` fetches over HTTPS with a license
  key in `Authorization: Bearer`. Server signs the response body with
  Ed25519; client verifies against a pinned public key. Cached locally
  with a configurable TTL (default 1 hour).
- **Failure modes**: every failure path falls back to the bundled seed.
  Network unreachable, license invalid, signature mismatch, schema bump
  we don't recognize — all silently degrade to seed without crashing the
  gateway. Loud WARN logs identify which failure mode hit.
- **License**: subscription terms (separate agreement from the AGPL-3.0
  code license). Subscribe at `hello@modelmeld.ai`.

## Side-by-side: what you actually get

| Dimension | Bundled snapshot (free, in OSS) | Pro feed ($29/mo) |
|---|---|---|
| **Update cadence** | Frozen at package release | Refreshed daily |
| **Benchmark sources aggregated** | Single snapshot of public benchmarks at release time | Artificial Analysis, Aider Polyglot, LiveBench, LMArena, plus aggregated production-traffic signals |
| **Editorial curation** | None — raw snapshot | Manual weighting per source, suspected-gaming flags, deprecation tracking |
| **New-model coverage** | None until you upgrade the package | Same-week coverage once we've benchmarked a release |
| **Price-change tracking** | Static at snapshot time | Reflected on the next refresh after providers move |
| **Routing-quality drift over time** | Degrades — measurable within ~3–6 months as the foundation-model market shifts | Stays current |
| **Signed payload + tamper detection** | N/A (local file) | Ed25519-signed; client verifies before consuming |
| **Cached fallback if network breaks** | Always available (the snapshot) | Falls back to bundled snapshot automatically |
| **Best for** | Tinkering, sovereignty-strict deployments, teams who maintain their own benchmark-aggregation pipeline | Anyone running real production traffic who doesn't want to build a benchmark pipeline ($29/mo < 1 engineer-day of labor cost per year to do it yourself) |

If you're trying to decide which one fits your situation, see
[`which-tier.md`](which-tier.md) for fuller decision guidance.

## Configuration

`GatewaySettings` (env-prefix `MODELMELD_`):

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `registry_feed_url` | str / None | None | Empty → bundled seed; set to enable live fetch |
| `registry_feed_license_key` | str / None | None | `Bearer` token sent on every feed request |
| `registry_feed_public_key_pem` | str / None | None | Pinned Ed25519 verify key (PEM format). Required for signature verification — leaving it None disables verification with a startup warning |
| `registry_feed_cache_path` | str / None | None | Local cache file path. Skip caching if None |
| `registry_feed_cache_ttl_seconds` | int | 3600 | How long a cached fetch is reused |
| `registry_feed_refresh_interval_seconds` | int | 86400 | Background-refresh cadence |

Minimum production config:

```bash
MODELMELD_REGISTRY_FEED_URL="https://feed.modelmeld.ai/v1/registry"
MODELMELD_REGISTRY_FEED_LICENSE_KEY="gws-lk-..."   # from admin dashboard
MODELMELD_REGISTRY_FEED_PUBLIC_KEY_PEM="-----BEGIN PUBLIC KEY-----\n..."
MODELMELD_REGISTRY_FEED_CACHE_PATH="/var/cache/gateway/registry.json"
```

## Wire format

`GET <feed_url>`

```http
Authorization: Bearer <license_key>
Accept: application/json
```

Response:

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Feed-Signature: <base64-encoded Ed25519 signature over raw body>

{
  "schema_version": 1,
  "feed_version": 42,
  "issued_at": "2026-05-20T00:00:00Z",
  "valid_until": "2026-05-21T00:00:00Z",
  "registry": {
    "version": 1,
    "last_updated": "2026-05-20T00:00:00Z",
    "models": [
      {
        "model_id": "claude-opus-4-7",
        "provider": "anthropic",
        "context_window": 200000,
        "cost_per_m_input": 5.00,
        "cost_per_m_output": 25.00,
        "task_scores": {"coding": 0.81, "reasoning": 0.90, ...},
        "last_updated": "2026-05-20T00:00:00Z",
        "source": "feed:v42"
      },
      ...
    ]
  }
}
```

The signature covers the **raw response body bytes** — verify before parsing.

## Failure semantics

The client NEVER raises on fetch failure. It returns a `FeedFetchResult` with
a `source` field telling you what it loaded:

| `source` | Meaning |
|---|---|
| `"feed"` | Freshly fetched + verified |
| `"cached"` | Loaded from local cache within TTL |
| `"seed"` | Bundled stale data (fetch failed, signature invalid, etc.) |

Production code can read `result.source` to log warnings, expose metrics,
or fail closed (refuse to route until a real feed is available) if that's
your operational preference. The default behavior is "degrade quietly to
seed" because routing on stale data is usually less bad than refusing to
serve traffic.

## Self-hosting the feed server

The actual feed-server implementation is commercial infrastructure outside
this repo. If you're an enterprise customer building an internal mirror,
the wire contract is the only thing you need to implement — see
`feed-server-contract.md` for the full spec
including signing, license key validation, and rate-limit headers.

The contract is intentionally minimal so customers running fully air-gapped
deployments can stand up their own mirror with whatever data sources they
trust, signed with their own key, without depending on `feed.modelmeld.ai`.

---

_Last reviewed: 2026-05-20._

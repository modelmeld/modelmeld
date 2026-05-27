# Open-Core Boundary Contract

This document defines the boundary between the two packages in this repository and the rules that govern what may cross it. The boundary is **enforced in CI** by `scripts/verify_boundary.py`. A violation is a build break, not a code-review nit.

If you are about to add an import, edit a `pyproject.toml`, or restructure a module, read this first.

---

## The three distribution tiers

This is actually a **three-tier** distribution model, not two. The curated
model registry was split off the package into a separate subscription feed.

| Tier | Path | License | Distribution |
|---|---|---|---|
| `modelmeld` (code) | `src/modelmeld/` | AGPL-3.0-or-later | Public OSS |
| Bundled seed data | `src/modelmeld/scout/data/` | CC-BY-4.0 (see `LICENSE.md` in that dir) | Public OSS, stale by design |
| Live curated registry feed | `feed.modelmeld.ai` | Subscription terms (separate agreement) | Paid, signed JSON, daily refresh |
| `modelmeld-enterprise` | `enterprise-control/` | Proprietary | Always private |

`modelmeld` is the future open-source release. It must be lift-and-shippable
into a fresh public repository without dragging any enterprise code or hidden
enterprise assumptions with it. `modelmeld-enterprise` is the closed-source
control plane that wraps and extends `modelmeld` for paying customers.

The **bundled seed** under `scout/data/` is functional but explicitly stale —
it ships with the package and emits a one-time WARN log when loaded in non-test
environments. Operators who want correct production routing decisions fetch the
**live feed** via `RegistryFeedClient` with a per-tenant license
key issued from an active subscription.

This split is intentional: code is permissively licensed to maximize adoption;
the curated registry is a subscription product so a fork can use the code
freely but doesn't automatically get the live data. See
`NOTICE` and `src/modelmeld/scout/data/LICENSE.md`
for the legal details.

---

## The one rule

**`modelmeld` may not import from `modelmeld_enterprise`. Ever. Under any circumstance.**

The reverse direction — `modelmeld_enterprise` importing from `modelmeld` — is the entire point. Enterprise wraps and extends core. Core never reaches back.

Enforcement: `scripts/verify_boundary.py` walks the AST of every `.py` file under `src/modelmeld/` and fails CI with exit code 1 if any `import modelmeld_enterprise` or `from modelmeld_enterprise...` is found. The script runs on every PR and on `main`.

---

## Worked examples

### Allowed: `modelmeld_enterprise` imports `modelmeld`

```python
# enterprise-control/src/modelmeld_enterprise/audit/middleware.py
from modelmeld.api.server import build_app  # ✓ allowed
from modelmeld.hooks import on_request_complete  # ✓ allowed

def wrap_with_audit(app):
    ...
```

This is the normal direction. Enterprise depends on core; core defines the extension surface; enterprise plugs into it.

### Allowed: `modelmeld` imports third-party OSS libraries

```python
# src/modelmeld/api/server.py
from fastapi import FastAPI  # ✓ allowed (MIT)
import httpx                  # ✓ allowed (BSD)
from pydantic import BaseModel  # ✓ allowed (MIT)
```

Permissive OSS dependencies (Apache-2, MIT, BSD, ISC) are fine. AGPL-compatible
copyleft dependencies (GPL-3, AGPL-3, LGPL-3) are also fine for `modelmeld`
itself, but **never** introduce them into `modelmeld_enterprise`, which must stay
free of AGPL contamination to be distributable as a proprietary control plane.

### Forbidden: `modelmeld` imports `modelmeld_enterprise`

```python
# src/modelmeld/api/server.py
from modelmeld_enterprise.audit import log  # ✗ FORBIDDEN
import modelmeld_enterprise.telemetry        # ✗ FORBIDDEN
from modelmeld_enterprise.auth.rbac import check  # ✗ FORBIDDEN
```

If you need enterprise functionality to happen during request handling, the correct pattern is:

1. **Define an extension hook** in `modelmeld` (e.g. `hooks.py` exposes `on_request_complete: Callable | None = None`).
2. **Implement the hook** in `modelmeld_enterprise` and register it at app startup.
3. **Core invokes the hook** if registered, treats it as a no-op if not.

This keeps the core engine fully functional standalone for OSS users while letting the enterprise plane attach behavior cleanly.

### Forbidden by spirit, not (yet) by the linter

These don't trigger the linter but violate the boundary contract — they will be caught in code review:

- Adding a hard dependency on PostgreSQL, Redis, Qdrant, SSO providers, or any enterprise infrastructure into `modelmeld`'s `dependencies` list. Such dependencies must be optional extras (`[scout]`, etc.) or live in `modelmeld_enterprise` only.
- Wiring `modelmeld` configuration to read from enterprise-specific environment variables or secret stores.
- Adding `modelmeld_enterprise` as a build-time test dependency to `modelmeld` to "make tests easier."

---

## Why the rule exists

1. **Relicensing freedom.** When we open-source `modelmeld`, we choose the license. If enterprise code ever touched core, we'd have to relicense the entanglement, possibly losing freedom to dual-license later.
2. **OSS adoption flywheel.** Individual developers must be able to `pip install modelmeld` and use it standalone with zero exposure to enterprise infrastructure (no Postgres, no audit log, no RBAC, no SSO). If they can't, they don't adopt, and the open-core strategy fails.
3. **Acquisition / diligence cleanliness.** Acquirers will run static analysis on the public OSS package. If they find proprietary symbols or shimmed dependencies, the diligence question becomes "what else is mixed in here?" Clean boundary = clean answer.
4. **Operational simplicity.** When core changes, enterprise must adapt. When enterprise changes, core is untouched. One-way dependencies make this predictable.
5. **Moat protection via the data feed, not the code.** The AGPL-3.0 license on the code maximizes adoption for individuals + dev tools (calling the gateway over HTTP from unmodified clients doesn't trigger copyleft), while deterring hyperscaler forks that would otherwise relaunch us as a closed managed service. The actual moat — the continuously-curated model registry — lives in the subscription feed, not in the package. A fork has the code but stale data; getting to parity requires rebuilding the curation operation, which is ongoing IP (editorial weighting, benchmark-gaming flags, model deprecation tracking) rather than a one-time engineering effort. See `NOTICE` for the explicit code-vs-data licensing terms.

---

## How to run the linter locally

```bash
# Lint the real modelmeld tree (default)
python scripts/verify_boundary.py

# Lint a specific path
python scripts/verify_boundary.py --root some/other/path

# Add an extra forbidden prefix
python scripts/verify_boundary.py --forbidden some_other_internal_pkg

# Run the linter's own test suite
python -m pytest scripts/tests
```

Exit codes:

- `0` — clean.
- `1` — one or more boundary violations found. Output lists `file:line: BOUNDARY VIOLATION: <import statement>`.
- `2` — root path does not exist (usage error).

---

## What to do if you hit a violation

1. **Don't `# noqa` it.** The linter has no suppression mechanism, by design.
2. Ask: what enterprise functionality does this core code actually need?
3. Add a hook (or extend an existing one) in `modelmeld/hooks.py` that exposes the extension point.
4. Move the enterprise-specific logic into `modelmeld_enterprise/` and register it through the hook.
5. The core code calls the hook (or a no-op default), unaware of the enterprise implementation.

If the answer is "core just needs to *know* something about enterprise state," that's almost always a sign the design needs work — core should not know about enterprise. Reach for an inversion-of-control pattern instead.

---

## Routing weights are open; the tuning pipeline is paid

This section is the open-core boundary applied to the *routing IP*. It
answers the recurring question: "if the code is AGPL-3.0, what's the
moat?"

### What ships in OSS

All routing **framework** lives in `modelmeld`:

- `Scout` ABC, `HeuristicScout`, scoring pipeline
- `Fingerprinter` + `PatternProvider` ABC + `DefaultPatternProvider`
- `CapabilityScout`, `ModelRegistry` lookup logic
- `TaskCategoryClassifier`
- All benchmark adapters (`AiderPolyglotFetcher`, `ArtificialAnalysisFetcher`,
  `LiveBenchFetcher`, `LMArenaFetcher`) — these read public benchmark data

All current **production values** also ship in OSS, as the defaults:

- `DEFAULT_HEURISTIC_WEIGHTS` — the tuned scoring constants we run in
  production (short-prompt boost, complex-keyword penalty, etc.)
- `DEFAULT_TASK_CATEGORY_WEIGHTS` — tuned task classifier weights
- `DEFAULT_QUALITY_THRESHOLD = 0.70` — production capability threshold
- The full dev-tool regex catalog in `DefaultPatternProvider`
- `default_registry.json` with current per-model task scores, frozen at
  OSS release date (marked via `snapshot_release_date`)

**Why ship the production values?** A neutral "demo" config would
sandbag OSS adopters — they'd see degraded performance, conclude the
project doesn't work, and never become customers. Shipping the real
values means OSS works well out of the box; the moat is in the
*continuous curation*, not in withholding what we know today.

### What stays in `modelmeld_enterprise.routing_tuning`

The *methodology* that derives those values, plus the publishing
pipeline:

- `FeedPublisher` — signs registry payloads with the operator's Ed25519
  private key (the same one issuing license JWTs) so the
  paid feed's `X-Feed-Signature` header verifies against the published
  public key
- `WeightTuningHarness` — A/B framework for validating proposed weight
  changes against historical traffic before promotion to the live feed
- (future) the multi-source aggregation logic that combines public
  benchmark data into the production task_scores via weighted source-
  credibility scoring + outlier rejection

### Why this split holds water

The Maxmind GeoIP precedent: the *code* is free (Maxmind's GeoIP2
library is open source), the *daily-curated data* is paid. Customers
pay for the freshness, not the algorithm.

For us:

- A snapshot ages ~50% in 12 months as the foundation-model market
  deflates and new models ship. A fork built against today's snapshot
  is meaningfully behind within 6 months.
- The curation pipeline (which sources to trust at what weight, how to
  normalize across heterogeneous benchmarks, when to promote a new
  weight set vs sit on the existing one) is the actual recurring work.
- Customers paying for the feed are paying for the *ongoing curation*
  done by our team — the code that does the curation belongs where the
  work happens.

### The conversion path

OSS adopter installs the package → ships with current production
defaults → works well immediately → over 6+ months the snapshot ages
relative to the market → they notice "newer cheaper models my gateway
doesn't know about" → upgrade conversation is natural: "subscribe to
the feed, your gateway stays current."

This is structurally the same as the seed-vs-paid-feed split: operators
can override defaults for organization-specific needs (e.g. sovereignty-
biased self-hosters who want a stronger short-prompt boost) without
forking the package.

---

## Related rules (not enforced by this linter but binding)

- **No direct code copying from other repositories.** Third-party patterns must be re-implemented in our own code. Dependencies via `pyproject.toml` are fine; source-level copying is not. Preserves IP cleanliness for relicensing.
- **`modelmeld` namespace for code identifiers.** Package name, imports, env vars (`MODELMELD_*`), HTTP headers (`x-modelmeld-*`), Docker images, and Helm charts all use the brand-aligned name. Brand and code namespaces are unified (decided 2026-05-23).
- **No hard dependencies on enterprise infrastructure in `modelmeld`.** PostgreSQL, Redis, Qdrant, SSO providers, etc. belong in `modelmeld_enterprise` or as optional extras (`[scout]`, etc.).

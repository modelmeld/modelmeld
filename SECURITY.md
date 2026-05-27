# Security policy

## Reporting a vulnerability

**Do not file a public GitHub issue for security-relevant findings.**
Instead, email `security@modelmeld.ai` with:

- A description of the issue
- Steps to reproduce (or a proof-of-concept)
- Your assessment of impact
- Any suggested mitigation

We acknowledge receipt within **3 business days** and aim to provide an
initial assessment within **7 business days**. We follow a **90-day
responsible-disclosure window** by default: from receipt of a credible
report, we commit to a fix or mitigation within 90 days before any
public disclosure. Reporters can extend the window by mutual agreement
if a fix requires substantial coordination (e.g. a dependency upstream).

We do not currently run a paid bounty program. We do credit reporters
in release notes (with their consent) and are happy to provide a
disclosure timeline letter for CVE submission or employer reporting.

## Supported versions

Pre-1.0: security fixes ship on the latest `0.y` minor release only.
After 1.0: we'll backport security fixes to the two most recent minor
versions for at least 6 months; specifics will land in this file at
1.0 cut.

## Supply-chain stance

The March 2026 LiteLLM supply-chain incident is the kind of failure
mode we design against. Concrete commitments:

- **Signed releases** — every PyPI artifact is signed with Sigstore
  (`sigstore-python`) and the signature is published as a GitHub
  release asset. Verification instructions are in the release notes.
- **SBOMs** — each release ships a CycloneDX SBOM as a release asset,
  enumerating every transitive dependency with its version + license.
  Useful for downstream procurement / security review.
- **Hashed lockfile** — the dependency lockfile (`uv.lock`) pins every
  transitive dependency by hash. CI installs from the lockfile, not
  from `pyproject.toml` ranges, so a malicious version published to a
  matching range can't sneak into a build.
- **Pinned CI deps** — GitHub Actions are pinned to commit SHAs, not
  version tags. A compromised tag pointer can't redirect our CI.
- **PyPI Trusted Publishing** — no long-lived API tokens; releases
  authenticate to PyPI via OIDC from our release workflow.
- **Reproducible builds** — the build is deterministic given the
  lockfile + a fixed Python minor version. CI publishes a hash of
  the resulting wheel so independent verifiers can rebuild and
  compare.

## Current security posture

Honest transparency about what we have and don't have today (pre-1.0):

### What we have implemented

- **Static analysis** — `ruff` (lint + format) and `pyright` (type
  checking) enforced in CI on every PR.
- **Dependency scanning** — `pip-audit` runs on push, PR, and weekly
  cron via `.github/workflows/security.yml`.
- **Secret scanning** — `gitleaks` runs on push + PR; blocks merges
  on any credential-pattern match.
- **Internal security review** — multi-agent code review of the
  pre-launch surface (auth, tenant isolation, PII flow, endpoint
  authorization, supply chain, config/ops hardening) completed
  2026-05-25. Findings tracked + remediated before public release.
- **No-custody architecture for frontier keys** — BYOK keys transit
  per-request in `x-modelmeld-byok-*` headers; never persisted at
  rest, never logged. See the Claude Code integration guide for the
  threat model.
- **PII scrubbing** — runs on every cloud-egress path before upstream
  provider calls. Configurable via `MODELMELD_PII_SCRUB_ENABLED`.

### What we do NOT yet have

- **No SOC 2 audit** — Type 1 audit planned post-1.0; Type 2 follows
  the 6-month observation window after that.
- **No ISO 27001** — not on the near-term roadmap.
- **No HIPAA / PCI-DSS / FedRAMP** — different compliance + sales
  motion; not a focus area.
- **No 3rd-party penetration test** — we've done internal security
  review but have not yet engaged an external firm. Planned post-1.0
  as part of SOC 2 prep.
- **No formal bug bounty** — see "Reporting a vulnerability" above;
  we credit reporters in release notes with their consent.
- **No signed SBOMs yet** — SBOMs ship with releases as CycloneDX,
  but the signed-SBOM commitment in "Supply-chain stance" is a
  target, not yet fully wired.

### Trust through transparency

We'd rather tell you what we don't have than imply we have it. If you
need compliance attestations we don't yet hold for a deployment
decision, contact `hello@modelmeld.ai` — we can share our current
security posture documentation, SOC 2 roadmap, and customer-facing
DPA/SCC templates under NDA.

## What we consider in-scope

- Authentication / authorization bypass in `modelmeld`
- Credential leakage through logs, error messages, or response headers
- PII leakage past the scrubber
- Cross-tenant data exposure (cache, memory, registry, audit logs)
- Cryptographic mistakes in the licensing / feed-signature code
- Supply-chain anomalies (unexpected dependencies, modified upstream)
- Resource-exhaustion attacks that bring down a single-tenant deployment

## What we consider out-of-scope

- Issues that require attacker-controlled physical access to the host
- Bugs in third-party services we proxy to (report those to OpenAI /
  Anthropic / etc. directly; we coordinate if it affects our
  abstraction layer)
- Findings in `enterprise-control` reported through this channel are
  accepted but will be addressed under the commercial agreement
  governing your access to that code

## PGP

PGP keys are not currently published; security mail to the address
above is read by the maintainer team. If your disclosure requires
encrypted transit, request a key in your initial email and we'll
respond with a public key for follow-up.

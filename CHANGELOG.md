# Changelog

All notable changes to `modelmeld` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/) per the policy in
[`docs/api-stability.md`](../docs/api-stability.md).

This file is **mostly auto-generated** going forward via
[release-please](https://github.com/googleapis/release-please) from our
[Conventional Commits](https://www.conventionalcommits.org/). The
release workflow opens a PR that updates this file + the version in
`pyproject.toml`; merging the PR tags + publishes the release. Manual
edits to this file land via PR like anything else and survive
release-please regenerations as long as they sit above the
auto-generated section markers.

## [Unreleased]

Changes destined for the next release land here. See open PRs against
`main` for what's brewing.

## [0.1.0] — initial public release

First public release of `modelmeld`. The package has been developed
internally through Phases 0–10 with substantial test coverage but this
is the first version published to PyPI.

### Highlights

- **OpenAI-compatible API surface** — `/v1/chat/completions` (streaming
  + non-streaming), `/v1/models`, `/healthz`. Drop-in for any client
  that already speaks OpenAI's wire format.
- **Capability-based routing** — `CapabilityScout` picks the cheapest
  model meeting a quality threshold for the prompt's task category,
  driven by the bundled `ModelRegistry` snapshot. Falls over to
  alternate models when the chosen provider is unhealthy.
- **Provider adapters** — OpenAI, Anthropic (with full schema
  translation), vLLM (self-hosted open-weights), TensorRT-LLM/Triton.
  All adapters retry transient errors (429 / 5xx / 529 / network blip)
  with exponential backoff before surfacing to the router.
- **Completion cache** — exact-match (in-memory or Redis) + semantic
  (Qdrant); cache key pools across users routed to the same served
  model.
- **PII scrubbing** — runs on every egress path before cloud upload.
- **Framework integration headers** — declare task category + agent
  role from AutoGen / CrewAI / LangGraph / OpenClaw to bypass the
  classifier.
- **Production-tuned defaults** — `DEFAULT_HEURISTIC_WEIGHTS`,
  `DEFAULT_QUALITY_THRESHOLD = 0.70`, full dev-tool detection
  catalog (Cursor / Claude Code / Aider / Cline / GitHub Copilot),
  and a current `default_registry.json` snapshot ship as the defaults.
- **License-key JWT verification** — `modelmeld.licensing` ships
  the OSS-side Ed25519 verifier for the registry feed subscription
  flow. Issuance lives in the commercial control plane.

### Open-core boundary

- **Code:** AGPL-3.0-or-later (this package)
- **Bundled snapshot data:** CC-BY-4.0 (see
  `src/modelmeld/scout/data/LICENSE.md`)
- **Live curated registry feed:** subscription (see
  `docs/registry-feed.md`)

`modelmeld` is fully functional on its own. The live feed is the
ongoing curation that keeps routing decisions current as the
foundation-model market shifts; the bundled snapshot stales over ~6
months as new models ship and prices drop.

### Known limitations at 0.1.0

- API surface is "stable in spirit but not under SemVer guarantees" —
  see `docs/api-stability.md`. Pre-1.0 we may break things; we'll call
  it out in the changelog when we do.
- No Google Gemini adapter yet (planned).
- No persistent memory backend out of the box — `MemoryStore` ABC
  ships; concrete Postgres-backed implementation lives in the
  enterprise control plane. OSS users get the in-process backend.
- Dashboard / admin UI is in the enterprise tier only.

[Unreleased]: https://github.com/ModelMeld/modelmeld/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ModelMeld/modelmeld/releases/tag/v0.1.0

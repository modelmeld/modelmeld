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

## [0.7.0](https://github.com/modelmeld/modelmeld/compare/v0.6.3...v0.7.0) (2026-06-01)


### Features

* **api:** /version endpoint for deploy verification ([d55869e](https://github.com/modelmeld/modelmeld/commit/d55869ec650a2bff689259b6c954f9baec871d94))

## [0.6.3](https://github.com/modelmeld/modelmeld/compare/v0.6.2...v0.6.3) (2026-06-01)


### Bug Fixes

* **build:** pin hatchling &lt;1.28 for PyPI publish-action compatibility ([a86abc9](https://github.com/modelmeld/modelmeld/commit/a86abc9d47135a3793bda41a576bd3b616f0141f))

## [0.6.2](https://github.com/modelmeld/modelmeld/compare/v0.6.1...v0.6.2) (2026-05-31)


### Bug Fixes

* **api:** audit header emits canonical model_id from the registry ([c9837f6](https://github.com/modelmeld/modelmeld/commit/c9837f62af5be740f628b1faaf3193c877293020))

## [0.6.1](https://github.com/modelmeld/modelmeld/compare/v0.6.0...v0.6.1) (2026-05-31)


### Bug Fixes

* **api:** /v1/models auto-derives lineup from the model registry ([0296eb1](https://github.com/modelmeld/modelmeld/commit/0296eb1c9f45d2ee86146e25e46b5621b5fd61ed))

## [0.6.0](https://github.com/modelmeld/modelmeld/compare/v0.5.0...v0.6.0) (2026-05-31)


### Features

* **adapter:** Codex passthrough auto-reloads bearer on 401 (token rotation) ([2925184](https://github.com/modelmeld/modelmeld/commit/2925184ca73e404d04088724ad00f8c0c1365e19))
* **api:** forward SDK camouflage headers on Anthropic OAuth passthrough ([1a13000](https://github.com/modelmeld/modelmeld/commit/1a13000770974a5a7a747fe7fc5cadac1d13095c))
* **scout:** fingerprint opencode from self-identifying system prompts ([11455cb](https://github.com/modelmeld/modelmeld/commit/11455cb46ac3aea217894c8d5fd1fd867eb99f46))

## [0.5.0](https://github.com/modelmeld/modelmeld/compare/v0.4.0...v0.5.0) (2026-05-31)


### Features

* **api:** echo x-modelmeld-agent-role in response headers ([d024a02](https://github.com/modelmeld/modelmeld/commit/d024a0227968bb98094613ac1d27af6373127cf0))


### Documentation

* **integrations:** add opencode integration guide ([74b16c9](https://github.com/modelmeld/modelmeld/commit/74b16c9ec2ff9c670eda331702ea31dce383b6c5))

## [0.4.0](https://github.com/modelmeld/modelmeld/compare/v0.3.1...v0.4.0) (2026-05-31)


### Features

  - **subscription passthrough**: Self-hosters can now point Claude Code
    (Claude Max) and OpenAI Codex CLI / `llm-openai-via-codex` (ChatGPT
    Plus/Pro/Business) at this gateway and route via OAuth bearer to the
    vendor backend. Gated by `MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=1`.
    Self-host only — see `docs/subscription-passthrough.md` for ToS
    guardrails and setup.
  - **scout**: `-auto` policy now falls back to a reasoning-capable OSS
    model when no frontier adapter is available, instead of returning 503.
  - **audit**: `RequestCompletedEvent` carries `quality_threshold` and
    `requires_tool_use` so downstream audit consumers can diagnose
    unexpected routing decisions post-hoc.


### Documentation

  - New: `docs/subscription-passthrough.md` — setup guide for the OAuth
    passthrough flows.
  - New: `docs/subscription-passthrough-codex-feasibility.md` and
    `-wire-format.md` — ToS posture and technical reference.


### Bug Fixes

* **release:** pass tag_name explicitly to gh-release action ([cf9596c](https://github.com/modelmeld/modelmeld/commit/cf9596c887fd6b34b6608561d7ac37c27e17a8ba))
* **release:** publish to PyPI before Sigstore signing; isolate SBOM path ([fb3f4c3](https://github.com/modelmeld/modelmeld/commit/fb3f4c37c59a49e4a767a3d130eb42c3896b6ed5))

## [0.3.0](https://github.com/modelmeld/modelmeld/compare/v0.2.0...v0.3.0) (2026-05-31)


### Features

* **scout:** relocate benchmark fetcher infrastructure to Pro feed service ([e618da7](https://github.com/modelmeld/modelmeld/commit/e618da7c599ca812b78d13c38418853d0ab4a0d3))

## [0.2.0](https://github.com/modelmeld/modelmeld/compare/v0.1.3...v0.2.0) (2026-05-30)


### Features

* **scout:** multi-provider routing for OSS self-hosters ([b76d985](https://github.com/modelmeld/modelmeld/commit/b76d98507440e6eb1fc2c01451186bda7f0e6c12))


### Documentation

* fix conftest.py path + smooth claude-code routing wording ([185d7e9](https://github.com/modelmeld/modelmeld/commit/185d7e9e9e88710a3e173c670020f3f2bcd2e768))
* update self-host + backends for out-of-box multi-provider ([755aace](https://github.com/modelmeld/modelmeld/commit/755aace7fb22ad32403f94a4a8e02a74fd18b616))

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

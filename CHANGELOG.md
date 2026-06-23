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

## [0.18.0](https://github.com/modelmeld/modelmeld/compare/v0.17.0...v0.18.0) (2026-06-23)


### Features

* **adapters:** deterministic upstream ([#110](https://github.com/modelmeld/modelmeld/issues/110)) ([5b3f412](https://github.com/modelmeld/modelmeld/commit/5b3f412c045613345850a6a82cde1394860f393d))
* **cli:** add `modelmeld setup --self-host` onboarding wizard ([#102](https://github.com/modelmeld/modelmeld/issues/102)) ([d8dc908](https://github.com/modelmeld/modelmeld/commit/d8dc908231dd99bb7690380238fc1da86d08cd81))
* **registry:** add claude-opus-4-8 to base registry ([#106](https://github.com/modelmeld/modelmeld/issues/106)) ([17a7c06](https://github.com/modelmeld/modelmeld/commit/17a7c06c06db6b651e37c4944fc5df0673258ab3))
* **registry:** add cost-efficient leader coders (qwen3.7-plus, mimo-v2.5, hy3-preview) ([#105](https://github.com/modelmeld/modelmeld/issues/105)) ([232445b](https://github.com/modelmeld/modelmeld/commit/232445bb37ca308c40dda254fc9af231158b605a))
* **registry:** add current leader coders to the multi-provider overlay ([#103](https://github.com/modelmeld/modelmeld/issues/103)) ([68c8162](https://github.com/modelmeld/modelmeld/commit/68c8162b51bc707f00bb3fd12e72e800ff2d8081))
* **registry:** encode measured agentic_coding on overlay leader rows ([#113](https://github.com/modelmeld/modelmeld/issues/113)) ([4c6077c](https://github.com/modelmeld/modelmeld/commit/4c6077c893c3e792618928121fbfa7d5bebb22e7))
* **scout:** -auto prefers the cheapest RELIABLE model on agentic routes ([#114](https://github.com/modelmeld/modelmeld/issues/114)) ([7f93244](https://github.com/modelmeld/modelmeld/commit/7f93244f51e501e29b2cdad02cf854ef1906a7f4))
* **scout:** add per-model `enabled` flag to gate routing ([#116](https://github.com/modelmeld/modelmeld/issues/116)) ([b8dad1c](https://github.com/modelmeld/modelmeld/commit/b8dad1c63547e36f56de79bccbdbb8d93d581a7f))
* **scout:** de-prefer un-probed models on -auto agentic routes ([#115](https://github.com/modelmeld/modelmeld/issues/115)) ([7b7bd25](https://github.com/modelmeld/modelmeld/commit/7b7bd25344aa3d30d6ce83959cc36afeda67d2d5))
* **scout:** large-context escalation prior for -auto ([#111](https://github.com/modelmeld/modelmeld/issues/111)) ([2a66cb7](https://github.com/modelmeld/modelmeld/commit/2a66cb758d4561fd57b298376ace32ffed81bffb))
* **scout:** observe-only stall detection for reactive escalation (shadow, default off) ([#112](https://github.com/modelmeld/modelmeld/issues/112)) ([c5c4d8e](https://github.com/modelmeld/modelmeld/commit/c5c4d8ed80ea1436b9a2c0039a5987fbfe0d5d57))
* **scout:** structural escalate-detector for -auto (flag-gated, default off) ([#109](https://github.com/modelmeld/modelmeld/issues/109)) ([d86788c](https://github.com/modelmeld/modelmeld/commit/d86788c8051727a71554eac702280bc75c25c998))


### Bug Fixes

* **adapters:** Together health check must tolerate its bare-list /models ([#104](https://github.com/modelmeld/modelmeld/issues/104)) ([a7ca2cd](https://github.com/modelmeld/modelmeld/commit/a7ca2cd9410b90a928686a984829444c8b19a86a))
* **routing:** apply provider_model_id on capability egress, not the canonical id ([#99](https://github.com/modelmeld/modelmeld/issues/99)) ([8580bd7](https://github.com/modelmeld/modelmeld/commit/8580bd75e84eabac98635a66fd851b31febb4a0c))
* **translation:** forward cache_control on Anthropic-&gt;OpenAI so providers can cache ([#108](https://github.com/modelmeld/modelmeld/issues/108)) ([4a68775](https://github.com/modelmeld/modelmeld/commit/4a68775684e941b7c23d30b2f7671c771433850c))
* **translation:** forward Codex reasoning.effort on the Responses surface ([#101](https://github.com/modelmeld/modelmeld/issues/101)) ([ed03cce](https://github.com/modelmeld/modelmeld/commit/ed03cce9062da28dbcb524cc2549337707e3f4da))


### Refactors

* **registry:** use a consistent dashed convention for model ids ([#118](https://github.com/modelmeld/modelmeld/issues/118)) ([3915a3f](https://github.com/modelmeld/modelmeld/commit/3915a3f4a7b7f0b140fa1c2b23c7c8b718dc9bb6))

## [0.17.0](https://github.com/modelmeld/modelmeld/compare/v0.16.1...v0.17.0) (2026-06-12)


### Features

* **routing:** -quality ranks on the agentic_coding prior when available ([#96](https://github.com/modelmeld/modelmeld/issues/96)) ([ce97658](https://github.com/modelmeld/modelmeld/commit/ce976580efc25bcd3e5fa4e0dfe9200bb49fe988))


### Bug Fixes

* **routing:** D1 imputes median latency for unmeasured models, not zero ([#98](https://github.com/modelmeld/modelmeld/issues/98)) ([f5972aa](https://github.com/modelmeld/modelmeld/commit/f5972aa754bb7ad9082eaf916f55897f873531c8))

## [0.16.1](https://github.com/modelmeld/modelmeld/compare/v0.16.0...v0.16.1) (2026-06-12)


### Bug Fixes

* **routing:** -quality picks strongest model on agentic work, not cheapest frontier ([#92](https://github.com/modelmeld/modelmeld/issues/92)) ([1621b28](https://github.com/modelmeld/modelmeld/commit/1621b2856d237068af203d2dddca0784bed019c6))

## [0.16.0](https://github.com/modelmeld/modelmeld/compare/v0.15.0...v0.16.0) (2026-06-12)


### Features

* **metrics:** expose uptime_seconds on GET /metrics ([#88](https://github.com/modelmeld/modelmeld/issues/88)) ([c65b6fc](https://github.com/modelmeld/modelmeld/commit/c65b6fc8dc9b9a79fe0ea24d7adba699ffc91ef9))
* **routing:** env-gated model pin for end-to-end model benchmarking ([#89](https://github.com/modelmeld/modelmeld/issues/89)) ([cd21895](https://github.com/modelmeld/modelmeld/commit/cd2189505ddbfa546baefea0a1a6332347776b67))
* **routing:** latency-aware ranking for auto-policy agentic requests ([#90](https://github.com/modelmeld/modelmeld/issues/90)) ([05f54e9](https://github.com/modelmeld/modelmeld/commit/05f54e9c07f70471ca5c4aff40815d9a61d29d48))


### Bug Fixes

* **api:** return canonical routed model in /v1/messages response body ([#86](https://github.com/modelmeld/modelmeld/issues/86)) ([22a56a4](https://github.com/modelmeld/modelmeld/commit/22a56a48b42b012769c06cea2d56838ecea15a20))

## [0.15.0](https://github.com/modelmeld/modelmeld/compare/v0.14.0...v0.15.0) (2026-06-10)


### Features

* **registry:** B-3 Phase 1 — capability-aware substitution reconciliation ([#83](https://github.com/modelmeld/modelmeld/issues/83)) ([21a42a4](https://github.com/modelmeld/modelmeld/commit/21a42a4cc57cb0748f5e74ed48246bb03a3e4933))
* **translation:** B-3 Phase 2 — forward reasoning intent to OSS backends ([#85](https://github.com/modelmeld/modelmeld/issues/85)) ([b9b0dbc](https://github.com/modelmeld/modelmeld/commit/b9b0dbca51cddd83eee4b220f8adfa2bbde4d039))

## [0.14.0](https://github.com/modelmeld/modelmeld/compare/v0.13.0...v0.14.0) (2026-06-10)


### Features

* **registry:** add GLM-4.6, Qwen3-Coder-480B, MiniMax-M2, Devstral-2512 to the overlay ([#80](https://github.com/modelmeld/modelmeld/issues/80)) ([8ceac3d](https://github.com/modelmeld/modelmeld/commit/8ceac3d899c705ae9e4c2e87c569a52a0c6b3975))
* **registry:** replace provisional scores with measured tool_use for the four new models ([#82](https://github.com/modelmeld/modelmeld/issues/82)) ([4eaca91](https://github.com/modelmeld/modelmeld/commit/4eaca91ec90aa7c5e6caa54fb556b4e67a233293))

## [0.13.0](https://github.com/modelmeld/modelmeld/compare/v0.12.0...v0.13.0) (2026-06-09)


### Features

* **api:** add GET /metrics local request/spend observability surface ([#79](https://github.com/modelmeld/modelmeld/issues/79)) ([625f957](https://github.com/modelmeld/modelmeld/commit/625f957a3bb6e3405ca7a34be93c62aac75bc8d6))


### Bug Fixes

* **adapters:** wrap stream-iteration errors as AdapterError so failover engages ([#77](https://github.com/modelmeld/modelmeld/issues/77)) ([d5ec3ea](https://github.com/modelmeld/modelmeld/commit/d5ec3ea004158847405a4b5c16d3df53ab821790))
* **api:** log full credential-redacted upstream error on 5xx failover ([#78](https://github.com/modelmeld/modelmeld/issues/78)) ([4f52559](https://github.com/modelmeld/modelmeld/commit/4f525596cc2617c5a982bf31fe6bf13449761669))
* **routing:** demote llama-3.3-70b agentic tool_use below the default floor ([#74](https://github.com/modelmeld/modelmeld/issues/74)) ([f88e4d0](https://github.com/modelmeld/modelmeld/commit/f88e4d02ebd161fb1eb67ac9124070f0ddba9ea8))
* **translation:** keep system at front and tool replies adjacent for strict OpenAI-compatible backends ([#76](https://github.com/modelmeld/modelmeld/issues/76)) ([3a70acb](https://github.com/modelmeld/modelmeld/commit/3a70acb6f2a45d36e891d67164db6d4fee970a48))

## [0.12.0](https://github.com/modelmeld/modelmeld/compare/v0.11.0...v0.12.0) (2026-06-08)


### Features

* **observability:** opt-in routing-rationale debug header ([#63](https://github.com/modelmeld/modelmeld/issues/63)) ([e5b7e56](https://github.com/modelmeld/modelmeld/commit/e5b7e567a8c53bab56fbadc0073f85f676b14143))


### Bug Fixes

* **anthropic:** drop client thinking config on model substitution ([#66](https://github.com/modelmeld/modelmeld/issues/66)) ([f80e4de](https://github.com/modelmeld/modelmeld/commit/f80e4dec006390b5786531e57650a6c26ddcbc83))
* **anthropic:** hoist system-role messages to top-level system in passthrough ([#70](https://github.com/modelmeld/modelmeld/issues/70)) ([a9ad579](https://github.com/modelmeld/modelmeld/commit/a9ad57930337423f10bec2e37237989fef061959))
* **anthropic:** route unknown native fields via extra_body, not kwargs ([#65](https://github.com/modelmeld/modelmeld/issues/65)) ([5d79f8c](https://github.com/modelmeld/modelmeld/commit/5d79f8c3b3008867b04d09e4bd6565c9e6130dc5))
* **api:** coerce routing-header values to latin-1 to avoid 500 ([#72](https://github.com/modelmeld/modelmeld/issues/72)) ([ea73016](https://github.com/modelmeld/modelmeld/commit/ea7301642f365a9f13577ece77c7e63b0d215bdd))
* **messages:** accept system-role messages in the array, hoist to system ([#64](https://github.com/modelmeld/modelmeld/issues/64)) ([8e99e9e](https://github.com/modelmeld/modelmeld/commit/8e99e9ec4a31acb4408367bbd7df23e39fa079fe))
* **messages:** drop client 'effort' too on model substitution ([#68](https://github.com/modelmeld/modelmeld/issues/68)) ([a221a93](https://github.com/modelmeld/modelmeld/commit/a221a93cae7449cf52f419ce6fb4bed6621812e1))
* **messages:** drop client thinking at the route on model substitution ([#67](https://github.com/modelmeld/modelmeld/issues/67)) ([c0dd6b3](https://github.com/modelmeld/modelmeld/commit/c0dd6b39d080161fa29b2b20c902056806a66ea4))
* **messages:** drop context_management with the thinking cluster on substitution ([#71](https://github.com/modelmeld/modelmeld/issues/71)) ([312b005](https://github.com/modelmeld/modelmeld/commit/312b0054d16dbf132ddea506c5b240389bc89f41))
* **messages:** strip output_config (nested effort) on model substitution ([#69](https://github.com/modelmeld/modelmeld/issues/69)) ([ee51d24](https://github.com/modelmeld/modelmeld/commit/ee51d24c207e695c34b4d2bdb2ae19603754d313))
* **routing:** load multi-provider overlay so cloud OSS providers are routable ([#73](https://github.com/modelmeld/modelmeld/issues/73)) ([eaefb1b](https://github.com/modelmeld/modelmeld/commit/eaefb1b8b1daf08a809e64c14ef993128187f43e))


### Documentation

* add llms.txt and AGENTS.md for AI-agent readers ([#61](https://github.com/modelmeld/modelmeld/issues/61)) ([481fecf](https://github.com/modelmeld/modelmeld/commit/481fecf91c684e06cad0ae941058d5a26415813e))

## [0.11.0](https://github.com/modelmeld/modelmeld/compare/v0.10.3...v0.11.0) (2026-06-07)


### Features

* **cli:** add Codex CLI to `modelmeld setup --tool` ([#60](https://github.com/modelmeld/modelmeld/issues/60)) ([2fbdedd](https://github.com/modelmeld/modelmeld/commit/2fbdedd9303243cb4d98d0926e40147db605e14f))
* **scout:** fingerprint Codex CLI in the DevTool detector ([#58](https://github.com/modelmeld/modelmeld/issues/58)) ([99e615b](https://github.com/modelmeld/modelmeld/commit/99e615b87b1905fee5d6034792195a114303f3b8))

## [0.10.3](https://github.com/modelmeld/modelmeld/compare/v0.10.2...v0.10.3) (2026-06-07)


### Documentation

* correct Codex setup and refresh stale claims after 0.8-0.10 ([#56](https://github.com/modelmeld/modelmeld/issues/56)) ([2775cbc](https://github.com/modelmeld/modelmeld/commit/2775cbc3accb24751f0e7dd2046aa3d2b128da63))

## [0.10.2](https://github.com/modelmeld/modelmeld/compare/v0.10.1...v0.10.2) (2026-06-07)


### Bug Fixes

* **responses:** estimate token usage when streaming upstream omits it ([#54](https://github.com/modelmeld/modelmeld/issues/54)) ([14dc246](https://github.com/modelmeld/modelmeld/commit/14dc246d231e9de051d625688a765950b824f7b1))

## [0.10.1](https://github.com/modelmeld/modelmeld/compare/v0.10.0...v0.10.1) (2026-06-07)


### Bug Fixes

* **responses:** accept multi-turn input with tool calls and results ([#52](https://github.com/modelmeld/modelmeld/issues/52)) ([2b4e020](https://github.com/modelmeld/modelmeld/commit/2b4e02058beca548670ee096bc5bb46bf0d0a1ca))

## [0.10.0](https://github.com/modelmeld/modelmeld/compare/v0.9.0...v0.10.0) (2026-06-07)


### Features

* **responses:** stream tool calls as function_call items ([#50](https://github.com/modelmeld/modelmeld/issues/50)) ([bff6d75](https://github.com/modelmeld/modelmeld/commit/bff6d759aa249cfcf4d024b66e288576b4697629))

## [0.9.0](https://github.com/modelmeld/modelmeld/compare/v0.8.2...v0.9.0) (2026-06-06)


### Features

* **responses:** /v1/responses endpoint with streaming (Codex CLI surface) ([#48](https://github.com/modelmeld/modelmeld/issues/48)) ([d95d3af](https://github.com/modelmeld/modelmeld/commit/d95d3af2a54415ea6ca0769faafac41522ce3c4c))

## [0.8.2](https://github.com/modelmeld/modelmeld/compare/v0.8.1...v0.8.2) (2026-06-06)


### Bug Fixes

* **messages:** surface cache_control stats on the streaming path ([#46](https://github.com/modelmeld/modelmeld/issues/46)) ([f5aacc5](https://github.com/modelmeld/modelmeld/commit/f5aacc59d49c7c582eddd7a2efc739377eded4a2))

## [0.8.1](https://github.com/modelmeld/modelmeld/compare/v0.8.0...v0.8.1) (2026-06-06)


### Bug Fixes

* **api:** pin chat response model field to the canonical routed model ([#43](https://github.com/modelmeld/modelmeld/issues/43)) ([61949e7](https://github.com/modelmeld/modelmeld/commit/61949e77437fe618ec143db3045131d3ae33b611))

## [0.8.0](https://github.com/modelmeld/modelmeld/compare/v0.7.5...v0.8.0) (2026-06-06)


### Features

* **memory:** gateway-native memory via Mem0 provider (modelmeld[mem0]) ([#41](https://github.com/modelmeld/modelmeld/issues/41)) ([b4ed0d5](https://github.com/modelmeld/modelmeld/commit/b4ed0d55e9c58d6f53b8fadebc92f861031fbb1c))

## [0.7.5](https://github.com/modelmeld/modelmeld/compare/v0.7.4...v0.7.5) (2026-06-06)


### Refactors

* **memory:** introduce MemoryProvider seam over the tiered store ([#39](https://github.com/modelmeld/modelmeld/issues/39)) ([960eae8](https://github.com/modelmeld/modelmeld/commit/960eae8697624aff26304753285621ca1ee111b5))

## [0.7.4](https://github.com/modelmeld/modelmeld/compare/v0.7.3...v0.7.4) (2026-06-03)


### Documentation

* add Codex CLI integration guide ([#32](https://github.com/modelmeld/modelmeld/issues/32)) ([6ad4351](https://github.com/modelmeld/modelmeld/commit/6ad4351b80bab1265dbdb6938f2549b41815e861)), closes [#11](https://github.com/modelmeld/modelmeld/issues/11)

## [0.7.3](https://github.com/modelmeld/modelmeld/compare/v0.7.2...v0.7.3) (2026-06-02)


### Documentation

* rewrite README for top-of-fold conversion ([d0d33ac](https://github.com/modelmeld/modelmeld/commit/d0d33ac1f78674e7658b4d337ff53365aac24d4d))

## [0.7.2](https://github.com/modelmeld/modelmeld/compare/v0.7.1...v0.7.2) (2026-06-02)


### Bug Fixes

* **api:** count_tokens no longer requires max_tokens; scout-fail returns 400 with friendly detail ([1f21646](https://github.com/modelmeld/modelmeld/commit/1f216463255e725a282abfebf6264350729e63b1))

## [0.7.1](https://github.com/modelmeld/modelmeld/compare/v0.7.0...v0.7.1) (2026-06-02)


### Bug Fixes

* **api:** surface Anthropic cache stats in /v1/messages responses ([02fa31b](https://github.com/modelmeld/modelmeld/commit/02fa31b45352aa8c7a08661974e667ab6306eb52))

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

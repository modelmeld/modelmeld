# API stability + SemVer policy

This document defines what counts as a **public API** of `modelmeld`
and the compatibility commitments around each surface. Pre-1.0 we may
break things; this document is honest about what's stable enough to
depend on today vs what may shift.

The short version: **the HTTP routes are the most stable thing in the
package**. Python symbols that you import from `modelmeld.*` are
mostly stable but may shift in pre-1.0 if we find a better factoring.
Anything under `modelmeld._internal.*` or named with a leading
underscore is fair game.

## What we mean by "API"

Three distinct surfaces, with three different stability levels:

| Surface | Audience | Stability |
|---|---|---|
| HTTP routes (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/models`, `/healthz`) | API consumers (Cursor, Claude Code, Codex CLI, framework integrations, etc.) | SemVer-stable from 0.1.0; breaking changes are major-version bumps |
| Top-level Python imports (`from modelmeld.X import Y`) for non-underscore names listed in this doc | Python integrators embedding the gateway in their own apps | SemVer-stable from 1.0.0; pre-1.0 may break with clear changelog entries |
| Internal symbols (anything with `_` prefix, anything in `_internal.*`, plus per-module helpers not listed here) | Project maintainers only | No stability guarantees, change without notice |

## What's covered by SemVer from 0.1.0

These are stable from the first public release. Breaking changes
require a major-version bump.

### HTTP routes

- `POST /v1/chat/completions` — OpenAI-compatible request and response
  schema. Streaming via SSE. Adding **new optional** request fields or
  response headers is non-breaking. Removing a documented field, or
  changing its semantics, is breaking.
- `POST /v1/messages` — Anthropic Messages API request and response
  schema. Streaming via SSE in Anthropic's event format
  (`message_start` / `content_block_*` / `message_delta` /
  `message_stop`). Supports tool definitions, tool_use blocks,
  tool_result blocks, and Anthropic prompt caching — `cache_control`
  breakpoints are forwarded verbatim to the upstream call via
  native-shape passthrough. Out of scope for v1: image content blocks.
  Same non-breaking / breaking rules as `/v1/chat/completions`.
- `POST /v1/responses` — OpenAI Responses API shape (the Codex CLI
  surface). Streaming via SSE as typed Responses events (`response.*`),
  including `function_call` output items. Accepts multi-turn input
  (`message` / `function_call` / `function_call_output` / `reasoning`
  items). Same non-breaking / breaking rules as `/v1/chat/completions`.
- `GET /v1/models` — returns the configured `available_models` list in
  OpenAI shape.
- `GET /healthz` — liveness probe; returns 200 on a running gateway.
- Response headers prefixed `x-modelmeld-*`.
- Request headers prefixed `x-modelmeld-*` accepted from clients (see
  `docs/integrations/README.md` for the routing-hint header set).

### Configuration

- The set of `MODELMELD_*` environment variable names.
- The `pyproject.toml` package name (`modelmeld`) and the
  top-level import path (`modelmeld`).
- The shape of `default_registry.json` — at the JSON-schema level.
  Adding fields is non-breaking; removing or repurposing them is.

## What's covered by SemVer from 1.0.0 (not yet)

These are stable in spirit but may shift between 0.x releases. Pin to
a specific version if breakage would hurt you.

### Python imports

These names + signatures are intended to be stable but pre-1.0
revisions may rename or relocate them. Material changes will be in
the changelog.

- `from modelmeld import build_app` — FastAPI factory
- `from modelmeld.config import GatewaySettings` — Pydantic settings model
- `from modelmeld.adapters.base import ProviderAdapter, AdapterError,`
  `TransientAdapterError, PermanentAdapterError`
- `from modelmeld.adapters.retry import RetryConfig, retry_async,`
  `is_transient_error, wrap_as_adapter_error`
- `from modelmeld.adapters.openai_adapter import OpenAIAdapter`
- `from modelmeld.adapters.anthropic_adapter import AnthropicAdapter`
- `from modelmeld.adapters.vllm_adapter import VLLMAdapter`
- `from modelmeld.adapters.tensorrt_llm_adapter import TensorRTLLMAdapter`
- `from modelmeld.router import build_router, Router, RoutingPolicy,`
  `RoutingDecision, RouterError, SingleAdapterRouter, TieredRouter`
- `from modelmeld.scout.base import Scout, ScoutDecision, Tier`
- `from modelmeld.scout.heuristics import HeuristicScout,`
  `HeuristicWeights, DEFAULT_HEURISTIC_WEIGHTS`
- `from modelmeld.scout.devtool import DevTool, Fingerprint,`
  `Fingerprinter, PatternProvider, DefaultPatternProvider`
- `from modelmeld.scout.capability import CapabilityScout,`
  `CapabilityDecision, DEFAULT_QUALITY_THRESHOLD, NoEligibleModelError`
- `from modelmeld.scout.task_category import TaskCategoryClassifier,`
  `TaskCategoryDecision, TaskCategoryWeights,`
  `DEFAULT_TASK_CATEGORY_WEIGHTS, TASK_CATEGORIES`
- `from modelmeld.scout.registry import ModelRegistry, ModelEntry,`
  `default_registry`
- `from modelmeld.licensing import verify_license_jwt,`
  `LicenseClaims, LicenseKeyError, LicenseKeyExpiredError,`
  `LicenseKeySignatureError, LicenseKeyMalformedError,`
  `LicenseKeyKidMismatchError, load_public_key_pem,`
  `public_key_fingerprint, peek_unverified`
- `from modelmeld.hooks import HookRegistry, RequestCompletedEvent`

### Behaviors

- **Routing decision algorithm** — the heuristic + capability scout
  output shape, the failover-on-transient-error policy, the
  `HeuristicWeights` arithmetic. The defaults themselves may shift
  between releases as we tune; changes will land in the changelog.
- **PII scrubbing** — runs on every `is_egress=True` adapter call.
  The exact set of patterns recognized may grow; we won't shrink it
  without a major-version bump.
- **Cache key derivation** — opaque hash; treat as a black box. Cache
  cold-misses on upgrade are acceptable; explicit cache invalidation
  is not promised across minor versions.

## What's explicitly NOT covered

These are **internal** and may change without notice:

- Anything with a leading underscore (`_extract_text`, `_PATTERNS`, etc.)
- Modules under `modelmeld._internal.*` (no such module exists today;
  this is the reservation)
- The exact wire format of provider-specific outbound calls (e.g. the
  Anthropic Messages payload we construct). Cosmetic differences in
  how we structure the upstream request are not API-breaking, as long
  as the upstream's response is equivalent.
- Logger names, log message strings, log levels for non-failure events.
- Hook event payload schemas during pre-1.0. We try to keep these
  stable, but if we need to add a field or restructure, we will.
- Internal test fixtures (`conftest.py` etc.)
- Database schema for the optional Postgres-backed memory store (when
  it lands).

## Deprecation policy

When we need to break something:

1. Add a `DeprecationWarning` in the affected import or HTTP route.
2. Document it in the next release's `CHANGELOG.md` under a
   `### Deprecated` heading.
3. Keep the deprecated path working for at least **one minor version**
   (pre-1.0) or **one major version** (post-1.0).
4. Remove it in the next major version with a changelog entry under
   `### Removed`.

If a deprecation has been issued and you have a hard reason it can't be
removed on the planned timeline, file an issue. We're not in the
business of arbitrary breakage.

## Why pre-1.0 is honestly pre-1.0

Some routing primitives — particularly the heuristic scout's scoring
formula and the dev-tool fingerprint regex set — will keep evolving as
real dogfooding traffic informs them. We'd rather make those changes
visibly during 0.x than freeze decisions before validating them.

If you need a hard stability commitment for a production deployment,
pin to a specific patch version (e.g. `modelmeld==0.1.3`, not
`modelmeld~=0.1`) and read the changelog before bumping.

## Reporting compatibility breaks we missed

If you observe a change between releases that broke your usage and
wasn't in the changelog, file an issue. We treat unannounced breakage
as a bug, not a policy choice. We may or may not revert (the change
might have been necessary for a security or correctness reason), but
we will improve the changelog discipline going forward.

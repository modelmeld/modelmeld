# Integrations

ModelMeld serves two wire formats so it's a drop-in backend for both
camps of coding tools:

- **OpenAI Chat Completions** at `POST /v1/chat/completions` — for
  Cursor, Aider, Cline, Continue.dev, and most agent frameworks
  (AutoGen, CrewAI, LangGraph, OpenClaw, etc.)
- **Anthropic Messages** at `POST /v1/messages` — for Claude Code and
  any Anthropic-SDK-native client (full spec compliance: `cache_control`
  preservation, `count_tokens`, `anthropic-beta` header forwarding,
  `display_name` on `/v1/models`)

Pick the tool you use; the integration guides below show the minimum
config patches needed.

## What the gateway does for you

Without any framework code changes:

- **Capability routing** — the gateway picks the cheapest model that meets
  your quality bar for whatever the prompt actually needs (coding,
  reasoning, summarization, simple_qa, tool_use). The framework writes
  `model="claude-opus-4-7"`; the gateway might serve `qwen3-coder-next`
  if it meets the quality bar for that prompt at OSS-tier rates.
- **Sovereignty + visibility** — every request lands in your audit trail
  with prompt hash, routed model, tier, cost, devtool fingerprint, and
  PII redactions. PII scrubbing runs before any cloud egress.
- **Failover** — if the chosen provider 5xx's mid-request, the gateway
  rolls over to the next-cheapest qualified model and adds an
  `x-modelmeld-failover-from` response header so you can see what
  happened.

## Routing-hint headers

For a complete, canonical reference of all `x-modelmeld-*` request and response headers — including accepted values, overrides, examples, and which integrations set them — see [Routing-hint headers reference](../routing-hints.md).

## Coding tool guides

Drop-in setup for the coding agents customers actually run day to day:

- [**Claude Code**](claude-code.md) — Anthropic CLI; speaks `/v1/messages`
- [**Cursor**](cursor.md) — IDE; speaks OpenAI `/v1/chat/completions`
- [**Aider**](aider.md) — pair-programming CLI via LiteLLM
- [**Cline**](cline.md) — VS Code agent with XML tool calls
- [**Continue.dev**](continue.md) — VS Code/JetBrains with per-role models
- [**opencode**](opencode.md) — SST's terminal coding agent with a native provider system
- [**Zed**](zed.md) — editor agent panel via OpenAI-compatible provider
- [**Codex CLI**](codex-cli.md) — OpenAI terminal agent via the Responses API (`/v1/responses`)

## Agent framework guides

For multi-agent orchestration frameworks where each agent has a
declared role:

- [AutoGen (Microsoft Research)](autogen.md)
- [CrewAI](crewai.md)
- [LangGraph](langgraph.md)
- [OpenClaw](openclaw.md)
- [MetaGPT](metagpt.md)

These frameworks route LLM calls through [LiteLLM](https://litellm.ai)
by default, so the [LiteLLM migration guide](litellm.md) is the shared
spine: point the proxy (or SDK) at the gateway once and every framework
on top of it routes here.

## Migrating from LiteLLM

Already running a LiteLLM proxy or SDK? ModelMeld is the same
OpenAI-compatible drop-in plus capability routing, deterministic audit
headers, and the three policy aliases. You change one `api_base` and
delete your fallback/router rules.

- [**LiteLLM → ModelMeld**](litellm.md) — before/after `config.yaml`
  diff, the Python SDK, and how to verify the cutover.

## Gateway-managed memory (experimental)

For agents/workflows that call the API directly and want the gateway to
remember the conversation — no SDK, just a session-id header:

- [**Conversation memory at the gateway**](memory.md) — opt-in mem0-backed
  memory via `MODELMELD_MEMORY_BACKEND=mem0`. (Coding tools manage their own
  history and don't need this.)

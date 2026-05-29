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

When the framework already knows what each agent does, it can tell us:

| Header                                 | Effect |
| -------------------------------------- | ------ |
| `x-modelmeld-task-category`          | One of `coding`, `reasoning`, `simple_qa`, `summarization`, `tool_use`. Bypasses the heuristic classifier. |
| `x-modelmeld-agent-role`             | A role name (e.g. `coder`, `researcher`, `reviewer`, `summarizer`, `executor`). Mapped to a task category server-side. |
| `x-modelmeld-quality-threshold`      | Float in `[0, 1]`. Raises or lowers the minimum task score the chosen model must have. |
| `x-modelmeld-exclude-providers`      | Comma-separated provider names to forbid (e.g. `openai,anthropic` for residency / cost ceilings). |

If both `task-category` and `agent-role` are present, `task-category` wins.

## Response headers you can read

Every chat-completions response carries audit-trail headers so you can
reproduce any routing decision client-side. Decisions are deterministic
functions of request shape + registry state — same request, same
registry, same model. Compare with OpenRouter's explicitly
non-deterministic Auto Router or Martian's closed-source "Model
Mapping": ours is auditable.

| Header                                 | Meaning |
| -------------------------------------- | ------- |
| `x-modelmeld-routed-to`              | Adapter that served the request (`openai`, `anthropic`, `vllm`, `fireworks`, `together`, `openrouter`). |
| `x-modelmeld-routed-model`           | The actual model the gateway sent upstream. May differ from `model` in your request body. |
| `x-modelmeld-task-category`          | Final category used for routing (`coding`, `reasoning`, `simple_qa`, `summarization`, `tool_use`). |
| `x-modelmeld-category-source`        | One of `classifier`, `hint:task_category`, `hint:agent_role`. Tells you whether your hint took effect. |
| `x-modelmeld-task-score`             | Chosen model's score on that category. |
| `x-modelmeld-quality-threshold`      | Threshold that was applied (post-bias if any). |
| `x-modelmeld-tier`                   | `local` (vLLM) or `cloud`. |
| `x-modelmeld-failover-from`          | Set when failover happened — value is the original tier. |
| `x-modelmeld-devtool`                | Detected client tool + confidence, e.g. `cursor:0.80` or `claude_code:0.95`. Absent for unknown clients. |
| `x-modelmeld-bias`                   | Set when a shape-based routing bias was applied, e.g. `autocomplete_shape`. Absent when no bias. |
| `x-modelmeld-redactions`             | PII categories that were scrubbed before egress (e.g. `EMAIL:2,SSN:1`). |
| `x-modelmeld-cache`                  | One of `hit`, `hit-semantic`, `miss`, `bypass`. Tells you whether the response came from cache. |

## Coding tool guides

Drop-in setup for the coding agents customers actually run day to day:

- [**Claude Code**](claude-code.md) — Anthropic CLI; speaks `/v1/messages`
- [**Cursor**](cursor.md) — IDE; speaks OpenAI `/v1/chat/completions`
- [**Aider**](aider.md) — pair-programming CLI via LiteLLM
- [**Cline**](cline.md) — VS Code agent with XML tool calls
- [**Continue.dev**](continue.md) — VS Code/JetBrains with per-role models
- [**Zed**](zed.md) — editor agent panel via OpenAI-compatible provider

## Agent framework guides

For multi-agent orchestration frameworks where each agent has a
declared role:

- [AutoGen (Microsoft Research)](autogen.md)
- [CrewAI](crewai.md)
- [LangGraph](langgraph.md)
- [OpenClaw](openclaw.md)

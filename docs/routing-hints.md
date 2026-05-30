# Routing-hint headers reference

Canonical reference for `x-modelmeld-*` headers used to control routing and audit trail. Each integration guide links here.

## Request headers

### x-modelmeld-task-category
- **Accepted values:** `coding`, `reasoning`, `simple_qa`, `summarization`, `tool_use`
- **Overrides:** Bypasses the heuristic classifier
- **Example:** `x-modelmeld-task-category: coding`
- **Integrations commonly setting this:** Agent frameworks (AutoGen, CrewAI, LangGraph, OpenClaw); any client adding custom headers

### x-modelmeld-agent-role
- **Accepted values:** Role name (e.g. `coder`, `researcher`, `reviewer`, `summarizer`, `executor`) — mapped to a task category server-side
- **Overrides:** Provides an alternative way to set task category; if both `task-category` and `agent-role` are present, `task-category` wins
- **Example:** `x-modelmeld-agent-role: coder`
- **Integrations commonly setting this:** Agent frameworks with per-agent role declarations

### x-modelmeld-quality-threshold
- **Accepted values:** Float in `[0, 1]`
- **Overrides:** Raises or lowers the minimum task score the chosen model must have
- **Example:** `x-modelmeld-quality-threshold: 0.85`
- **Integrations commonly setting this:** Any integration that exposes a quality slider

### x-modelmeld-exclude-providers
- **Accepted values:** Comma-separated provider names (e.g. `openai,anthropic`)
- **Overrides:** Forbids the listed providers from being chosen for this request
- **Example:** `x-modelmeld-exclude-providers: openai,anthropic`
- **Integrations commonly setting this:** Users with data-residency or cost constraints; any client that allows custom headers

### x-modelmeld-byok-anthropic / x-modelmeld-byok-openai (BYOK headers)
- **Accepted values:** Your own Anthropic or OpenAI API key
- **Overrides:** Uses your key instead of the gateway's configured key; key is never stored or logged
- **Example:** `x-modelmeld-byok-anthropic: sk-ant-...`
- **Integrations commonly setting this:** Any tool configured via `modelmeld setup` or that allows setting custom HTTP headers (Claude Code, Aider, etc.)

## Response headers

All response headers are emitted by the gateway and can be read by any integration. They provide audit-trail data.

### x-modelmeld-routed-to
- **Meaning:** Adapter that served the request (`openai`, `anthropic`, `vllm`, `fireworks`, `together`, `openrouter`)

### x-modelmeld-routed-model
- **Meaning:** The actual model the gateway sent upstream; may differ from `model` in the request

### x-modelmeld-task-category
- **Meaning:** Final category used for routing (`coding`, `reasoning`, `simple_qa`, `summarization`, `tool_use`)

### x-modelmeld-category-source
- **Meaning:** One of `classifier`, `hint:task_category`, `hint:agent_role` — indicates whether a hint took effect

### x-modelmeld-task-score
- **Meaning:** Chosen model's score on that category (float)

### x-modelmeld-quality-threshold
- **Meaning:** Threshold applied (post-bias if any)

### x-modelmeld-tier
- **Meaning:** `local` (vLLM) or `cloud`

### x-modelmeld-failover-from
- **Meaning:** Set when failover happened — value is the original tier

### x-modelmeld-devtool
- **Meaning:** Detected client tool + confidence, e.g. `cursor:0.80` or `claude_code:0.95`; absent for unknown clients

### x-modelmeld-bias
- **Meaning:** Set when a shape-based routing bias was applied, e.g. `autocomplete_shape`; absent when no bias

### x-modelmeld-redactions
- **Meaning:** PII categories scrubbed before egress, e.g. `EMAIL:2,SSN:1`

### x-modelmeld-cache
- **Meaning:** One of `hit`, `hit-semantic`, `miss`, `bypass`

# OpenClaw → ModelMeld

[OpenClaw](https://github.com/openclaw/openclaw) is an open-source
Claude-Code-style coding agent. It already supports OpenAI-compatible
endpoints and inherits headers from the configured base client.

## Minimal config

```yaml
# ~/.openclaw/config.yml
provider: openai
openai:
  base_url: http://gateway.internal:8080/v1
  api_key: gws_<your-key>
  default_headers:
    x-modelmeld-agent-role: coder
    x-modelmeld-quality-threshold: "0.85"
```

That's it. OpenClaw's existing model selector still works (you can pick
e.g. `claude-opus-4-7`) but the gateway may substitute a cheaper
competent model when appropriate.

## Why this works well for OpenClaw specifically

OpenClaw's prompts are uniformly coding tasks — exactly the category
the gateway is best at routing cheaply. In practice, coding-shape
prompts route to Qwen and DeepSeek coder models that are close to
frontier on real-world coding tasks at OSS-tier rates, so per-token
cost lands well below going direct to Anthropic for the same quality.

## Sub-agent routing

OpenClaw spawns sub-agents (planner, edit-applier, reviewer). If you
want different routing per sub-agent, set the role per
sub-agent class:

```yaml
sub_agents:
  planner:
    default_headers:
      x-modelmeld-agent-role: planner
      x-modelmeld-quality-threshold: "0.85"
  editor:
    default_headers:
      x-modelmeld-agent-role: coder
      x-modelmeld-quality-threshold: "0.80"
  reviewer:
    default_headers:
      x-modelmeld-agent-role: reviewer
      x-modelmeld-quality-threshold: "0.90"
```

## Verifying the cost win

Run a 100-task benchmark with and without the gateway pointed at:

```bash
openclaw benchmark --suite swe-bench-verified-mini
```

Then compare `total_cost_usd` from the two runs. The gateway exposes
per-request cost attribution via the `x-modelmeld-task-score` +
`routed-model` headers — pair with the registry's cost data to compute
exact savings.

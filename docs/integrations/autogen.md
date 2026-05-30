# AutoGen → ModelMeld

[AutoGen](https://github.com/microsoft/autogen) sends OpenAI-compatible
requests via the `model_client` abstraction. Point it at the gateway and
add a per-agent hint header.

## Minimal patch

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

gateway_client = OpenAIChatCompletionClient(
    model="claude-opus-4-7",                      # gateway will override based on hint
    base_url="http://gateway.internal:8080/v1",   # ← ModelMeld
    api_key="gws_<your-key>",
    # Per-agent hint via OpenAI's extra_headers passthrough
    default_headers={
        "x-modelmeld-agent-role": "coder",      # this agent is a coder
    },
)
```

## Per-agent routing — one client per role

The killer use case. Spin up one client per role; the gateway picks the
cheapest competent model for each.

```python
def make_client(role: str, quality_threshold: float = 0.80):
    return OpenAIChatCompletionClient(
        model="claude-opus-4-7",
        base_url="http://gateway.internal:8080/v1",
        api_key="gws_...",
        default_headers={
            "x-modelmeld-agent-role": role,
            "x-modelmeld-quality-threshold": str(quality_threshold),
        },
    )

coder    = AssistantAgent("Coder",    model_client=make_client("coder"))
reviewer = AssistantAgent("Reviewer", model_client=make_client("reviewer",  0.90))
planner  = AssistantAgent("Planner",  model_client=make_client("planner",   0.85))
```

The `Coder` agent gets routed to a cheap coding-strong model (e.g.
`qwen3-coder-next`). The `Reviewer` gets a stronger model at threshold
0.90. The `Planner` gets a reasoning-strong model.

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## GroupChat — different role per agent

```python
from autogen import GroupChat, GroupChatManager

agents = [coder, reviewer, planner]
chat = GroupChat(agents=agents, messages=[], max_round=12)
manager = GroupChatManager(groupchat=chat, llm_config=manager_config)
```

Each agent's calls carry its own hint header → cost-optimized routing per turn.

## Caveats

- AutoGen sometimes spawns sub-clients for tool execution. Make sure the
  `default_headers` are inherited (they are, in the official extensions).
- If you set both `task-category` and `agent-role`, the explicit category
  wins. Use `task-category` when the agent's role doesn't map cleanly
  (e.g. a single agent doing multiple things — set the category per call).

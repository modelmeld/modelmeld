# CrewAI → ModelMeld

[CrewAI](https://github.com/crewAIInc/crewAI) routes all calls through
LiteLLM. Two integration paths: configure LiteLLM's OpenAI provider to
point at the gateway, or use CrewAI's per-agent `llm` override.

## Recommended: per-agent LLM with hint header

```python
from crewai import Agent
from langchain_openai import ChatOpenAI   # CrewAI accepts any LangChain LLM

def gateway_llm(role: str, quality: float = 0.80) -> ChatOpenAI:
    return ChatOpenAI(
        model="claude-opus-4-7",
        openai_api_base="http://gateway.internal:8080/v1",
        openai_api_key="gws_<your-key>",
        default_headers={
            "x-modelmeld-agent-role": role,
            "x-modelmeld-quality-threshold": str(quality),
        },
    )

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="...",
    llm=gateway_llm("researcher", quality=0.85),
)

writer = Agent(
    role="Tech Content Writer",
    goal="Craft compelling stories",
    backstory="...",
    llm=gateway_llm("writer", quality=0.80),
)

coder = Agent(
    role="Python Developer",
    goal="Build the prototype",
    backstory="...",
    llm=gateway_llm("coder", quality=0.85),
)
```

CrewAI passes `default_headers` through to LiteLLM → through to the
gateway. Each agent gets a cost-optimized model for its role.

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Alternative: per-task task_category override

If a single agent handles different categories of work in different
tasks, set the category per Task instead of per Agent:

```python
from crewai import Task

# Override at the Task level by passing a callable that recomputes headers
research_task = Task(
    description="Summarize the latest Anthropic research",
    agent=researcher,
    # CrewAI doesn't have a first-class header override; the cleanest path
    # is to instantiate a per-task ChatOpenAI with different default_headers.
)
```

## Sovereignty / residency

For EU-only routing:

```python
gateway_llm("researcher", quality=0.85).default_headers["x-modelmeld-exclude-providers"] = "openai,anthropic"
```

That falls the gateway through to your on-prem vLLM models. If no
qualifying model exists, you get a 503 (fail-closed, by design).

# LangGraph → ModelMeld

[LangGraph](https://github.com/langchain-ai/langgraph) builds stateful,
multi-actor graphs on top of LangChain. Each node typically has its own
LLM; point each one at the gateway with a node-specific hint.

## Pattern: one LLM per node

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

def llm_for(role: str, quality: float = 0.80) -> ChatOpenAI:
    return ChatOpenAI(
        model="claude-opus-4-7",
        openai_api_base="http://gateway.internal:8080/v1",
        openai_api_key="gws_<your-key>",
        default_headers={
            "x-modelmeld-agent-role": role,
            "x-modelmeld-quality-threshold": str(quality),
        },
    )

planner_llm  = llm_for("planner",  quality=0.85)
coder_llm    = llm_for("coder",    quality=0.85)
reviewer_llm = llm_for("reviewer", quality=0.90)

def planner_node(state):
    return {"plan": planner_llm.invoke(state["request"]).content}

def coder_node(state):
    return {"code": coder_llm.invoke(state["plan"]).content}

def reviewer_node(state):
    return {"review": reviewer_llm.invoke(state["code"]).content}

graph = StateGraph(...)
graph.add_node("planner",  planner_node)
graph.add_node("coder",    coder_node)
graph.add_node("reviewer", reviewer_node)
```

The gateway sees each node's request with its own header and routes
accordingly. Tracing through the response headers tells you which model
served each node:

```python
response = coder_llm.invoke(messages)
# response.response_metadata["headers"]["x-modelmeld-routed-model"]  # e.g. "qwen3-coder-next"
```

For the full list of response headers and their meanings, see the [Routing-hint headers reference](../routing-hints.md).

## Pattern: dynamic category per state

If your graph branches on the task type, override the category per call:

```python
from langchain_core.messages import HumanMessage

def adaptive_node(state):
    category = state.get("category", "coding")  # set upstream
    response = coder_llm.invoke(
        [HumanMessage(content=state["prompt"])],
        config={"configurable": {
            "extra_headers": {"x-modelmeld-task-category": category}
        }},
    )
    return {"result": response.content}
```

## Streaming

LangGraph's streaming mode works with the gateway's SSE stream as-is.
You'll see the routing headers in the first response chunk via
`response_metadata`.

## Tool calls

Set `x-modelmeld-agent-role: executor` (or `tool_caller`) on nodes
that primarily invoke tools — that maps to the `tool_use` category and
picks a model strong at tool calling.

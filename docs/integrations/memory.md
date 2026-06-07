# Conversation memory at the gateway (experimental)

> **Status: experimental.** The API and defaults may change. Off by default —
> the gateway does no memory work unless you opt in (see below).

Most coding tools (Claude Code, Cursor, Aider, Cline) manage their own
conversation history and resend it every turn — they don't need this. This is
for the other case: **an agent or workflow calling the API directly that sends
only the latest message and expects the gateway to remember the rest.**

The pitch: **memory with zero client code — just send a session-id header.** No
SDK to wire in, no vector store for you to operate. The gateway records each
exchange and injects the relevant prior context on the next request.

It's backed by [mem0](https://github.com/mem0ai/mem0) (Apache-2.0) running
behind the gateway's request path. mem0 does the extraction + retrieval; the
gateway does the transparent injection.

## Enable it

```bash
pip install 'modelmeld[mem0]'
```

```bash
export MODELMELD_MEMORY_BACKEND=mem0
# Route mem0's own LLM + embedder calls through THIS gateway (see "Cost" below):
export MODELMELD_MEM0_BASE_URL=http://localhost:8080/v1
export MODELMELD_MEM0_API_KEY=<a-modelmeld-key>
# Vector store: a qdrant server (shared, per-tenant collection)...
export MODELMELD_MEM0_VECTOR_STORE_URL=http://localhost:6333
# ...or omit the URL and set a path for an embedded on-disk store (per-tenant subdir):
# export MODELMELD_MEM0_VECTOR_STORE_PATH=/var/lib/modelmeld/mem0
```

The default `in_memory` / `postgres` tiered backends are unaffected — this only
applies when `MODELMELD_MEMORY_BACKEND=mem0`.

## Use it

Memory activates when the request carries a session id. Send the same id across
turns and the gateway threads context for you:

```bash
curl https://your-gateway/v1/chat/completions \
  -H "Authorization: Bearer gws_<key>" \
  -H "x-modelmeld-session-id: agent-run-42" \
  -d '{"model":"anthropic/modelmeld-auto",
       "messages":[{"role":"user","content":"Remember: my name is Alice."}]}'

# next turn — send ONLY the new message; the gateway recalls the rest:
curl https://your-gateway/v1/chat/completions \
  -H "Authorization: Bearer gws_<key>" \
  -H "x-modelmeld-session-id: agent-run-42" \
  -d '{"model":"anthropic/modelmeld-auto",
       "messages":[{"role":"user","content":"What is my name?"}]}'
```

Headers:

| Header | Meaning |
|---|---|
| `x-modelmeld-session-id` | **Required to activate memory.** Stable per conversation. |
| `x-modelmeld-memory-mode` | `augment` (default) / `full` / `off`. |
| `x-modelmeld-user-id` | Optional caller-supplied user id (agent frameworks). |

No session id → no memory work happens; the request is a plain pass-through.

## Cost — read this

Memory is **not free.** With the default `infer=True`, every write runs an
**LLM extraction call** (plus embedding calls) so mem0 can distill durable facts
from the turn. That cost is real and you should plan for it.

Two levers keep it cheap:

1. **Route mem0's extraction through this gateway** (`MODELMELD_MEM0_BASE_URL`
   pointed at your own `/v1`). Then the extraction call is itself cost-routed by
   the gateway's normal routing — sent to a cheap OSS model, or to a local model
   you host (in which case it's effectively free).
2. **Set `MODELMELD_MEM0_INFER=false`** to skip extraction entirely and store
   raw turns. No per-write LLM call; lower retrieval quality.

There is no "free memory." Anyone claiming otherwise is hiding the extraction
cost. Measure it for your workload before relying on it.

## Tenant isolation

Each tenant gets its **own vector collection** (not a shared collection with
metadata filters) — see [multi-tenant-isolation.md](../multi-tenant-isolation.md).
Within a tenant, memories are scoped to the session. A request's tenant comes
from gateway auth; anonymous traffic shares one sentinel namespace, so run with
auth enabled in production.

## Settings reference

| Env var | Default | Notes |
|---|---|---|
| `MODELMELD_MEMORY_BACKEND` | `in_memory` | Set to `mem0` to enable. |
| `MODELMELD_MEM0_INFER` | `true` | `false` = raw store, no per-write LLM. |
| `MODELMELD_MEM0_TOP_K` | `10` | Memories retrieved per request. |
| `MODELMELD_MEM0_RERANK` | `false` | mem0 reranking on retrieval. |
| `MODELMELD_MEM0_BASE_URL` | _(mem0 default)_ | Route mem0's LLM + embedder here. |
| `MODELMELD_MEM0_API_KEY` | — | Key for the above. |
| `MODELMELD_MEM0_LLM_MODEL` | `gpt-5-mini` | Extraction model. |
| `MODELMELD_MEM0_EMBEDDER_MODEL` | `text-embedding-3-small` | Embedding model. |
| `MODELMELD_MEM0_VECTOR_STORE_URL` | — | qdrant server URL (shared, per-tenant collection). |
| `MODELMELD_MEM0_VECTOR_STORE_API_KEY` | — | API key for a remote qdrant endpoint (optional). |
| `MODELMELD_MEM0_VECTOR_STORE_PATH` | — | Embedded on-disk qdrant (per-tenant subdir). |
| `MODELMELD_MEM0_EMBEDDING_DIMS` | `1536` | Embedding vector dimensions (match your embedder model). |

## Limitations

- Experimental; defaults and surface may change.
- `infer=True` adds latency + cost per write (the extraction call).
- mem0 telemetry is disabled by default (`MEM0_TELEMETRY=False`); set it
  explicitly to re-enable.
- Retrieval quality depends on the extraction + embedding models you point it at.

# Design — `/v1/messages` (Anthropic-compatible API surface)

**Status:** Implemented
**Owner:** (TBD)
**Branch (proposed):** `feat/anthropic-messages-api`

---

## Goal

Expose an **Anthropic-format HTTP endpoint** on modelmeld so Claude Code (and anything else built on the `anthropic` SDK) can use the gateway transparently via `ANTHROPIC_BASE_URL=http://localhost:8000`.

The endpoint must support enough of Anthropic's Messages API surface that Claude Code's normal workflow — including tool-use for file edits and bash commands — works without modification.

## Non-goals (v1)

- Anthropic prompt caching (`cache_control` blocks) — accept the field, ignore it; document as known limitation
- Anthropic Beta features (computer-use, multi-document, code-execution) — not in scope
- Anthropic's `count_tokens` endpoint — separate work item
- Anthropic's `models` listing endpoint — already covered by existing `/v1/models` (OpenAI shape); cross-API model discovery is a separate concern
- Image content blocks — defer (Claude Code doesn't use vision; we can add later if a user needs it)

---

## Architecture

### The key insight

The gateway uses **OpenAI `ChatCompletionRequest` / `ChatCompletion` as its internal lingua franca**. All routing, memory injection, caching, scrubbing, and adapter dispatch operate on these types. Adapters translate at the edge: e.g. `AnthropicAdapter.chat()` calls `to_anthropic_params(request)` before invoking the Anthropic SDK, and wraps the SDK response with `from_anthropic_response(...)` to return a `ChatCompletion`.

For `/v1/messages` we add the **reverse-direction translation** at the HTTP boundary:

```
┌────────────────────────────────────────────────────────────────────┐
│  Client (Claude Code, anthropic SDK)                               │
│    ▼  Anthropic JSON wire format                                   │
│  POST /v1/messages                                                  │
│    │                                                               │
│    │  ┌── from_anthropic_request() ──────────────────┐  NEW         │
│    │  │  Anthropic JSON → ChatCompletionRequest      │              │
│    │  └──────────────────────────────────────────────┘              │
│    ▼                                                               │
│  Existing pipeline (UNCHANGED):                                    │
│    routing hints, memory inject, scout, scrubber,                  │
│    router.route(), adapter.chat() or stream_chat(),                │
│    memory write-back, hooks                                        │
│    ▼  internal ChatCompletion / ChatCompletionChunk[]              │
│    │  ┌── to_anthropic_response() ───────────────────┐  NEW         │
│    │  │  ChatCompletion → Anthropic Message dict     │              │
│    │  └──────────────────────────────────────────────┘              │
│    │  ┌── to_anthropic_stream_events() ─────────────┐  NEW          │
│    │  │  ChatCompletionChunk[] → Anthropic SSE      │              │
│    │  └──────────────────────────────────────────────┘              │
│    ▼  Anthropic JSON or SSE wire format                            │
│  Response                                                           │
└────────────────────────────────────────────────────────────────────┘
```

**Nothing in the internal pipeline changes.** The shape conversion happens entirely at the route's edges. This means:

- Scout still classifies in OpenAI shape (its existing heuristics, regexes, fingerprints all keep working)
- Memory injection still operates on `ChatCompletionRequest` (no work needed in `memory/context.py`)
- The same `AnthropicAdapter` egress path is reused — when a `/v1/messages` request gets routed to Anthropic, we *receive* Anthropic format, *translate to OpenAI internally*, *translate back to Anthropic to call the upstream API*. That's two translations on the cloud path. Acceptable cost; the architecture stays simple.

### Why not skip the internal OpenAI representation when both endpoints + upstream are Anthropic?

Tempting: when a `/v1/messages` request gets routed to `AnthropicAdapter`, why not pass the Anthropic dict through directly? Two reasons:

1. **Memory injection** mutates the request shape by prepending synthetic system messages. The injector operates on `ChatCompletionRequest`. Bypassing the internal type means re-implementing the injector for Anthropic shape — duplicated invariant, drift risk.

2. **Scout fingerprinting + scrubber** also operate on the internal shape. Same drift-risk argument.

The price of double translation on the Anthropic-in → Anthropic-out path is ~milliseconds of CPU per request. Negligible vs. network latency.

---

## File layout

```
src/modelmeld/
  api/
    routes/
      messages.py                    NEW  — Anthropic /v1/messages route
      chat.py                              (unchanged)
      models.py                            (unchanged)
      healthz.py                           (unchanged)
    schemas_anthropic.py             NEW  — Pydantic models for Anthropic wire shape
  translation/
    openai_anthropic.py              MODIFIED — add reverse-direction functions
    __init__.py                      MODIFIED — re-export new functions

tests/
  test_translation_anthropic_to_openai.py   NEW  — translation unit tests
  test_route_messages.py                    NEW  — route-level integration tests
  test_route_messages_streaming.py          NEW  — streaming SSE format tests
  test_route_messages_toolcalls.py          NEW  — tool-use roundtrip tests
```

### What goes in `messages.py`

Mirror the structure of `chat.py`. Same auth-identity extraction, same memory-identity extraction, same routing-hint extraction, same scout call, same adapter dispatch, same hook firing, same memory write-back. The only differences:

- Parse `AnthropicMessagesRequest` instead of `ChatCompletionRequest` from the body
- Translate to `ChatCompletionRequest` via `from_anthropic_request()` before any pipeline logic runs
- Translate the resulting `ChatCompletion` back to Anthropic shape before returning
- Emit Anthropic-shape SSE events on the streaming path

Memory write-back, hooks, headers — all reused as-is. The hook event still records OpenAI-shape internals; that's correct (the wire shape doesn't affect the audit trail).

### What goes in `schemas_anthropic.py`

Pydantic models for Anthropic's request/response/stream-event wire shape. Modeled from the public Anthropic Messages API spec — not source-copied from the `anthropic` SDK. Key types:

- `AnthropicMessagesRequest` — `model`, `messages`, `system` (str | list[block]), `max_tokens` (required), `tools`, `tool_choice`, `temperature`, `top_p`, `stop_sequences`, `stream`, `metadata`
- `AnthropicMessage` — `role: "user"|"assistant"`, `content: str | list[ContentBlock]`
- Content blocks: `AnthropicTextBlock`, `AnthropicImageBlock`, `AnthropicToolUseBlock`, `AnthropicToolResultBlock`
- `AnthropicMessageResponse` — `id`, `type: "message"`, `role`, `content: list[block]`, `model`, `stop_reason`, `stop_sequence`, `usage`
- Stream events: `MessageStartEvent`, `ContentBlockStartEvent`, `ContentBlockDeltaEvent`, `ContentBlockStopEvent`, `MessageDeltaEvent`, `MessageStopEvent`, `PingEvent`

### What goes in `openai_anthropic.py` (additions)

Three new functions that mirror the existing patterns:

```python
def from_anthropic_request(req: AnthropicMessagesRequest) -> ChatCompletionRequest:
    """Translate Anthropic Messages request → OpenAI ChatCompletionRequest."""
    # - Combine `system` (top-level) into a SystemMessage at the front of messages
    # - Convert each Anthropic message: text blocks → TextPart; tool_use → tool_calls
    #   on AssistantMessage; tool_result → ToolMessage
    # - Map `max_tokens` → `max_tokens`; `stop_sequences` → `stop`
    # - Convert tool definitions: Anthropic {name, description, input_schema} → OpenAI Tool(function=...)
    # - Convert tool_choice: {"type": "auto"|"any"|"tool"} → OpenAI shape


def to_anthropic_response(completion: ChatCompletion, request_model: str) -> dict[str, Any]:
    """Translate internal ChatCompletion → Anthropic Messages response dict."""
    # - choice.message.content → text block(s)
    # - choice.message.tool_calls → tool_use block(s)
    # - finish_reason → stop_reason (reverse of _STOP_REASON_MAP)
    # - Usage: prompt_tokens → input_tokens, completion_tokens → output_tokens


class OpenAIToAnthropicStreamTranslator:
    """Accumulate OpenAI chunks; emit Anthropic SSE events in correct order.

    Anthropic stream protocol:
        event: message_start         { "message": {id, model, role, content: [], stop_reason: null, usage: {input_tokens, output_tokens: 0}} }
        event: content_block_start   { "index": 0, "content_block": {type:"text", text:""} }
        event: content_block_delta   { "index": 0, "delta": {type:"text_delta", text:"..."} }
        ... more deltas ...
        event: content_block_stop    { "index": 0 }
        (if tool_use: another block_start/deltas/stop with type:"tool_use")
        event: message_delta         { "delta": {stop_reason, stop_sequence}, "usage": {output_tokens} }
        event: message_stop          {}
    """
    def translate_chunk(self, chunk: ChatCompletionChunk) -> list[dict]: ...
    def finalize(self) -> list[dict]: ...   # emit message_delta + message_stop at end
```

---

## Translation tables

### Request: Anthropic → OpenAI

| Anthropic field | OpenAI field | Notes |
|---|---|---|
| `model` | `model` | Pass through unchanged |
| `messages[].role` ("user"/"assistant") | `messages[].role` | Direct mapping |
| `messages[].content` (str) | `messages[].content` (str) | Direct mapping |
| `messages[].content[]` (text block) | `messages[].content[]` (TextPart) | `{"type":"text","text":"..."}` → TextPart |
| `messages[].content[]` (tool_use block) | `AssistantMessage.tool_calls[]` | `{type:"tool_use",id,name,input}` → ToolCall (arguments=json.dumps(input)) |
| `messages[].content[]` (tool_result block) | new `ToolMessage` | `{type:"tool_result",tool_use_id,content}` → ToolMessage(tool_call_id=tool_use_id, content=content) |
| `messages[].content[]` (image block) | `messages[].content[]` (ImagePart) | base64 source → data URL; url source → url; **deferred to v2** |
| `system` (str) | leading `SystemMessage` | Insert at front of messages |
| `system` (list[block]) | leading `SystemMessage` | Concat text blocks |
| `max_tokens` (required) | `max_tokens` | Direct mapping |
| `temperature` | `temperature` | Direct mapping |
| `top_p` | `top_p` | Direct mapping |
| `stop_sequences` (list) | `stop` (list) | Direct mapping |
| `stream` | `stream` | Direct mapping |
| `tools[]` | `tools[]` | `{name,description,input_schema}` → `Tool(function=FunctionDef(name,description,parameters=input_schema))` |
| `tool_choice` `{type:"auto"}` | `"auto"` | Map literal |
| `tool_choice` `{type:"any"}` | `"required"` | Map literal |
| `tool_choice` `{type:"tool",name}` | `{type:"function",function:{name}}` | Specific tool |
| `metadata` | (ignored, accepted) | Anthropic carries `user_id`; not used by our routing yet |
| `cache_control` blocks | (ignored, accepted) | Document as known limitation |

### Response: OpenAI → Anthropic

| OpenAI field | Anthropic field | Notes |
|---|---|---|
| `id` | `id` | Direct (or generate new `msg_*` if missing) |
| (literal) | `type` | Always `"message"` |
| `choices[0].message.role` | `role` | Always `"assistant"` for responses |
| `choices[0].message.content` (str) | `content[]` (one text block) | `[{type:"text",text:content}]` |
| `choices[0].message.tool_calls[]` | `content[]` (tool_use blocks) | Each ToolCall → `{type:"tool_use",id,name,input:json.loads(arguments)}` |
| `model` | `model` | Direct |
| `choices[0].finish_reason` | `stop_reason` | "stop"→"end_turn", "length"→"max_tokens", "tool_calls"→"tool_use", "content_filter"→"refusal" |
| (none) | `stop_sequence` | Null unless we tracked the stop sequence (we don't currently) |
| `usage.prompt_tokens` | `usage.input_tokens` | Direct |
| `usage.completion_tokens` | `usage.output_tokens` | Direct |

### Streaming: OpenAI chunks → Anthropic events

OpenAI streams text chunk by chunk inside `choices[0].delta.content`. Tool calls stream as `delta.tool_calls[].function.arguments` accumulating JSON. Anthropic streams as **discrete content blocks**: a `content_block_start` event opens block N, then `content_block_delta` events emit deltas of that block, then `content_block_stop` closes it. Switching from text to tool-use is a new block.

The translator must:

1. On first chunk with `delta.role` or first non-empty delta: emit `message_start`
2. Track current block type. First text delta → emit `content_block_start{index:0, content_block:{type:"text"}}` then `content_block_delta{type:"text_delta", text:<delta>}`
3. Subsequent text deltas → another `content_block_delta`
4. When a `tool_calls` delta arrives with a new `index`: close previous block with `content_block_stop`, open new tool_use block with `content_block_start{content_block:{type:"tool_use",id,name,input:{}}}`
5. Tool argument JSON pieces → `content_block_delta{type:"input_json_delta",partial_json:<piece>}`
6. On final chunk (`finish_reason` set): close last block, emit `message_delta{delta:{stop_reason,...},usage:{output_tokens}}`, emit `message_stop`
7. SSE format: every event prefixed with `event: <type>\n` and `data: <json>\n\n`

There's existing precedent: the AnthropicAdapter's `AnthropicStreamTranslator` does the *reverse* (Anthropic events → OpenAI chunks). The new `OpenAIToAnthropicStreamTranslator` mirrors its state-machine pattern.

---

## Route implementation sketch

```python
# src/modelmeld/api/routes/messages.py

@router.post("/messages", response_model=None)
async def anthropic_messages(
    body: AnthropicMessagesRequest,
    fastapi_request: Request,
    response: Response,
) -> dict | StreamingResponse:
    # Same boilerplate as chat.py: extract auth, memory identity, routing hints,
    # check api_key model allowlist, get scrubber/hooks/memory/etc. from app.state.
    ...

    # NEW: translate Anthropic shape → internal OpenAI shape
    try:
        internal_request = from_anthropic_request(body)
    except TranslationError as e:
        raise HTTPException(status_code=400, detail=f"translation_error: {e}")

    # Existing pipeline, identical to chat.py
    decision = await rt.route(internal_request, hints=hints)
    mem_context = await assemble_context(memory, mem_identity)
    internal_request = inject_into_request(internal_request, mem_context)
    outgoing, redactions = _maybe_scrub(internal_request, decision, scrubber)

    if outgoing.stream:
        return await _stream_messages_with_failover(
            rt, decision, outgoing, redactions, hooks, request_id, started,
            identity, memory, mem_identity, token_counter,
        )

    # Reuse caching, hooks, memory writeback identically to chat.py
    # (Refactor opportunity: extract a shared "execute internal request" helper.
    #  For v1 implementation, copy-and-adapt is acceptable; refactor in a follow-up
    #  once both routes are stable.)
    ...

    # NEW: translate internal completion → Anthropic shape
    anthropic_response_dict = to_anthropic_response(completion, body.model)
    return anthropic_response_dict
```

### Streaming response shape

```python
async def _sse_anthropic_stream(...) -> AsyncIterator[str]:
    translator = OpenAIToAnthropicStreamTranslator()
    try:
        if first_chunk is not None:
            for event in translator.translate_chunk(first_chunk):
                yield _format_sse(event)
        async for chunk in aiter:
            for event in translator.translate_chunk(chunk):
                yield _format_sse(event)
        for event in translator.finalize():
            yield _format_sse(event)
    except Exception as e:
        # Anthropic doesn't define a standard error-event format. Closest is
        # emitting `message_delta` with stop_reason=null + closing the stream.
        # Document the chosen behavior.
        ...
```

---

## Decisions for review

These I want your input on before writing code:

### D-1: `max_tokens` handling

Anthropic requires `max_tokens` in every request. OpenAI treats it as optional. The Anthropic SDK clients (including Claude Code) will always include it, so in practice this is a non-issue for the dogfood test.

**Proposal:** Require `max_tokens` in `AnthropicMessagesRequest`. Pydantic returns 400 if missing. Matches Anthropic's behavior.

### D-2: System prompt format

Anthropic accepts `system` as either a string OR a list of content blocks (e.g., for prompt caching). OpenAI's SystemMessage takes string OR list of TextPart.

**Proposal:** Accept both. List-of-blocks → join text blocks with `\n\n` into a single SystemMessage. Ignore (but accept) any `cache_control` field on blocks.

### D-3: Tool result content type

Anthropic's `tool_result` block has a `content` field that can be a string OR a list of content blocks (e.g., for tool returning structured output). OpenAI's ToolMessage is string-only.

**Proposal:** If tool_result.content is a string, pass through. If list of text blocks, join with `\n`. If list contains image blocks, raise TranslationError (defer image-bearing tool results to v2).

### D-4: Header naming consistency

Existing chat route emits headers like `x-modelmeld-routed-to`. The /v1/messages route should emit the same headers (consistent across both surfaces) — but per task #93 these will be renamed to `x-modelmeld-*` in a separate rebrand sweep. Don't preempt the rebrand here.

**Proposal:** Emit identical `x-modelmeld-*` headers from `/v1/messages` as `/v1/chat/completions`. Rebrand sweeps both at once.

### D-5: Shared helpers — refactor or copy-and-adapt for v1?

The chat.py route has substantial helpers (`_completion_with_failover`, `_stream_with_failover`, memory write-back, hook firing, cache lookup). The messages.py route needs the same logic on the internal-shape request. Two options:

- **(a) Refactor** `_completion_with_failover` etc. into shape-agnostic helpers shared between routes. Cleaner long-term.
- **(b) Copy-and-adapt** for v1, refactor in a follow-up after both routes prove stable.

**Proposal:** (b) for v1. Refactoring touches two routes simultaneously; if anything goes wrong it's harder to isolate. Get /v1/messages working as a standalone route first, then DRY-refactor with confidence and proper tests on both sides.

### D-6: Streaming response headers

OpenAI SSE responses set `content-type: text/event-stream`. Anthropic does the same. The existing `_stream_with_failover` sets routing headers on the StreamingResponse. We do the same here.

**Proposal:** No change from existing pattern.

### D-7: `ping` events

Anthropic sometimes emits `event: ping\n\n` events during long streams to keep connections alive. The translator could emit periodic pings, but it's not strictly required for Claude Code to work.

**Proposal:** Don't emit ping events in v1. If we see disconnect issues during dogfood testing, add a periodic ping.

### D-8: Tool-use input streaming partial JSON

When Anthropic streams tool calls, it emits the tool's `input` field as `partial_json` deltas — strings that, concatenated, form valid JSON. OpenAI streams tool-call arguments the same way (in `function.arguments` deltas). Translation is direct.

**Edge case:** if OpenAI doesn't emit partial chunks but instead drops the full JSON in one delta (some non-streaming adapters might), we'll emit one big `input_json_delta`. Claude Code's parser handles this fine because it just concatenates and parses at block_stop.

**Proposal:** Direct mapping, no special handling.

---

## Test plan

### Unit tests (translation)

`test_translation_anthropic_to_openai.py`:
- Simple user message round-trips
- System prompt as string vs list
- Multi-turn with assistant text
- Tool definitions translate cleanly
- `tool_use` block on assistant → tool_calls on AssistantMessage
- `tool_result` block on user → ToolMessage
- `max_tokens`/`temperature`/`top_p`/`stop_sequences` pass through
- `tool_choice` variants
- Missing `max_tokens` → ValidationError
- Image block → TranslationError (with clear message)

`test_translation_openai_to_anthropic.py` (extends existing test file):
- ChatCompletion with text-only → text block
- ChatCompletion with tool_calls → tool_use blocks
- finish_reason mappings (stop, length, tool_calls, content_filter)
- Usage tokens translate

### Streaming unit tests

`test_translation_stream_openai_to_anthropic.py`:
- Text-only stream: chunks → `message_start` + `content_block_start(text)` + N×`content_block_delta(text_delta)` + `content_block_stop` + `message_delta` + `message_stop`
- Tool-call stream: text chunks + tool_call chunks → text block + tool_use block in correct order
- Empty stream: still emit message_start + message_delta(stop_reason="end_turn") + message_stop
- Finish reason propagates correctly

### Route integration tests

`test_route_messages.py` (using FastAPI TestClient or httpx.ASGITransport — same pattern as bench scripts):
- POST a real Anthropic-shaped request → assert 200 + correct response shape
- POST with missing `max_tokens` → 400
- POST with multi-turn + system prompt → assert response message correct
- POST with `stream=true` → assert event sequence correct
- POST a request that triggers routing to local (mock vLLM adapter) → assert routing happens
- POST a request that triggers routing to Anthropic (mock the adapter) → assert no double-translation bugs
- Verify memory write-back happens with correct session/turn semantics
- Verify hooks fire on success and failure
- Verify routing headers (`x-modelmeld-*`) appear on the response

`test_route_messages_toolcalls.py`:
- POST with tool definitions → response includes tool_use blocks when LLM chooses to call
- Multi-turn with tool_result block → request is correctly translated for routing
- Streaming tool calls → SSE events emit tool_use block correctly

`test_route_messages_anthropic_passthrough.py`:
- POST Anthropic-shape, route to Anthropic backend → end-to-end roundtrip works (double-translation doesn't corrupt anything). Use a mock AnthropicAdapter that just echoes the request shape back.

### Hand-verification (post-merge, before dogfood)

- `curl` an Anthropic-shape request at `/v1/messages` directly, see Anthropic-shape response
- `curl` with `Accept: text/event-stream` and `stream:true`, see correct SSE event sequence

### Dogfood (separate session, post-merge)

- Set `ANTHROPIC_BASE_URL=http://localhost:8000` in a new shell
- Start gateway with always_cloud routing
- Run a Claude Code session in that shell — ask it to do a real coding task (read a file, edit a file, run a test). All operations should work transparently. Watch gateway logs to see request flow.

---

## Effort estimate

| Phase | Estimate |
|---|---|
| Anthropic schemas (Pydantic models) | 2 hours |
| `from_anthropic_request` (no streaming) | 3 hours |
| `to_anthropic_response` (no streaming) | 2 hours |
| `OpenAIToAnthropicStreamTranslator` | 4 hours |
| Route file `messages.py` (non-streaming) | 3 hours |
| Route streaming path | 3 hours |
| Unit tests for translation | 3 hours |
| Integration tests for route | 3 hours |
| Streaming tests | 2 hours |
| Tool-use roundtrip tests | 2 hours |
| Documentation update | 1 hour |
| **Total** | **~28 hours** (~3-4 working days at focused pace) |

This is longer than my earlier "~1-2 days" estimate — I underestimated tool-use tests and streaming translation. Realistic budget: **3-4 days**.

If we cut tool-use to v2 (Claude Code without tool-use is non-functional, so this would gate the dogfood test): -8 hours = ~2 days. **Not recommended** — tool-use is the whole point for Claude Code.

If we cut streaming to v2 (Claude Code can work non-streaming, though UX is meaningfully worse): -9 hours = ~2.5 days. **Defensible cut** if we need to ship faster.

---

## Open questions for you

1. **Approve scope as written?** Or trim (no streaming v1, ship a working non-streaming `/v1/messages` first)?
2. **Decisions D-1 through D-8 above** — any pushback?
3. **Feature branch name** `feat/anthropic-messages-api` — fine?
4. **Test commitment** — should we set a target like "every code change accompanied by a test"? (Per existing project rhythm.)
5. **Anything else you want covered in the design before coding starts?**

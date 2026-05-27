# Cline → ModelMeld

[Cline](https://github.com/cline/cline) (formerly Claude Dev) is a
VS Code agent that executes tools (read_file, search_files,
write_to_file, execute_command, etc.) via XML-tagged outputs the model
emits. It speaks OpenAI's `/v1/chat/completions` and has a dedicated
"OpenAI Compatible" provider in its settings UI.

## Minimal setup

In VS Code:

1. Open Cline's sidebar
2. Click the settings ⚙️ icon
3. **API Provider** dropdown → "OpenAI Compatible"
4. Fill in:
   - **Base URL**: `https://api.modelmeld.ai/v1`
   - **API Key**: `gws_<your-modelmeld-key>`
   - **Model ID**: any value (e.g. `claude-opus-4-7`) — our scout decides

Cline will immediately use the gateway. The next message in the chat
panel routes through ModelMeld.

## What Cline sends us

- Very large system prompt (~11K chars) defining Cline's XML tool syntax
- Messages alternate user/assistant; user messages contain tool-result
  blocks from the prior assistant's tool calls
- **No `tools` field** — Cline uses prompt-based XML tool calls
  (`<read_file><path>...</path></read_file>`), not OpenAI's
  function-calling protocol
- `stream=true` mandatory — Cline iterates the chunk stream and parses
  XML out of the output text as it arrives

Our scout sees no tools field, so the `supports_tools` filter doesn't
fire. Routing is purely on prompt complexity + size.

## Cline's plan/act modes

Cline supports two operating modes that map to different routing tiers:

- **Plan mode** — generates a step-by-step plan before executing.
  Higher reasoning bar; scout routes to OSS-premium or frontier.
- **Act mode** — executes tools step by step. Routing varies per
  step; many steps are simple "now read this file" which route
  cheap.

You can set different models per mode in Cline's settings if you want
finer control, though our scout handles the variation automatically.

## Verify the routing

Cline displays the model name in each message's footer:

```
qwen3-coder-flash · 14.3K tokens · $0.0042
```

The model name comes from `response.model` which we set to our
routed model. To see the full audit trail, check the gateway's
account dashboard.

## Common gotchas

- **`Custom Headers` field** — Cline's OpenAI Compatible provider has
  an optional Custom Headers field. Use it to pass routing hints:
  ```
  x-modelmeld-quality-threshold: 0.85
  x-modelmeld-agent-role: coder
  ```
- **Huge `read_file` results blow context** — when Cline reads a big
  file, the result lands in the next user message. Our scout's
  context-window filter catches this and routes to a long-context
  model automatically. If you're hitting the wall regularly, lower
  Cline's max-file-size limit.
- **Cline retries on context errors** — its context-management
  layer will truncate middle-of-history when switching to a smaller
  model. Our gateway returns proper errors when the context exceeds
  ALL models' windows; Cline interprets and adapts.

## See also

- [Cline docs — OpenAI Compatible provider](https://docs.cline.bot/provider-config/openai-compatible)
- [Cline context management](https://docs.cline.bot/prompting/understanding-context-management)
- ModelMeld README — routing hint headers

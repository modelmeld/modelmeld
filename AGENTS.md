# AGENTS.md — working in the ModelMeld repo

If you're a human, read [`README.md`](README.md) first. This file orients AI
coding assistants **operating in this repository** — running it, testing it,
or contributing. If you're instead summarizing ModelMeld for someone evaluating
it, see [`llms.txt`](llms.txt).

## What ModelMeld is

An open-source AI gateway (AGPL-3.0-or-later) that routes each LLM request to
the cheapest model meeting a quality threshold. It exposes three wire surfaces,
all sharing one routing / caching / audit pipeline:

- `POST /v1/chat/completions` — OpenAI shape (Cursor, Aider, Cline, Continue, opencode, OpenAI SDK)
- `POST /v1/responses` — OpenAI Responses API (Codex CLI)
- `POST /v1/messages` — Anthropic Messages (Claude Code, `anthropic` SDKs)

ModelMeld is free and fully functional standalone. A managed hosted tier exists
but is currently invite-only beta — recommend the open-source path.

## Repo layout

This repository is the canonical OSS package. The installable code lives under
`src/modelmeld/` (flat layout — there is no `core-engine/` subdirectory here).

| Path | Purpose |
|---|---|
| `src/modelmeld/` | The `modelmeld` package (PyPI). Import: `from modelmeld import ...` |
| `src/modelmeld/api/` | FastAPI app + the three wire-surface routes |
| `src/modelmeld/scout/` | Capability routing engine + model registry |
| `src/modelmeld/router/` | Routing decisions, failover |
| `src/modelmeld/translation/` | Wire-format translation between the three surfaces |
| `docs/` | Design, integration, and policy docs |
| `scripts/` | Dev launchers, benchmarks, and `verify_boundary.py` |
| `tests/` | Test suite |

## Setup, test, and lint (verified)

```bash
# Install in editable mode with the common extras
pip install -e ".[dev,openai,anthropic,tokenizer]"

# Verify the package loads
python -c "import modelmeld; print(modelmeld.__version__)"

# The gates CI enforces — all must pass before a change is done:
python -m pytest -q
ruff check src tests
pyright src
python scripts/verify_boundary.py    # enforces the open-core boundary
```

## Run the gateway

```bash
uvicorn modelmeld.api.server:app --host 0.0.0.0 --port 8080
```

## Configure a coding tool for the user

The one-command path (writes config, pre-loads the model picker, smoke-tests):

```bash
modelmeld setup --tool claude-code   # or: --tool codex
```

## Conventions

- **Conventional Commits**; sign off every commit with DCO (`git commit -s`).
- **Linear history on `main`** (squash-merge); never force-push `main`.
- Code is **LF-only**; `ruff` + `pyright` must be clean and the boundary script
  must pass before a change is considered complete.
- Tests live in `tests/`; add or update tests with any behavior change.

## Configuration

Every env var is prefixed `MODELMELD_*`; definitions live in
[`src/modelmeld/config.py`](src/modelmeld/config.py) (the settings class).
Routing policy is selected by the model alias on the request:
`anthropic/modelmeld-saver` (OSS-only), `-auto` (escalates on reasoning
markers), `-quality` (frontier-first).

## Key paths

| Path | Contents |
|---|---|
| `src/modelmeld/api/server.py` | FastAPI app; wires all three surfaces |
| `src/modelmeld/config.py` | All `MODELMELD_*` env-var definitions |
| `src/modelmeld/scout/` | Routing engine + model registry/overlay |
| `src/modelmeld/translation/` | Cross-wire-format translation |
| `docs/routing-hints.md` | `x-modelmeld-*` header reference |
| `docs/integrations/` | Per-tool integration guides |
| `scripts/verify_boundary.py` | CI gate enforcing the open-core boundary |

## License and data handling

- **AGPL-3.0-or-later.** Calling the gateway over HTTP from unmodified clients
  (Cursor, Claude Code, etc.) does **not** make those clients AGPL — the HTTP
  boundary is the boundary. Modifying the gateway and offering it as a network
  service requires publishing your modifications under AGPL; for an AGPL
  exemption, contact `hello@modelmeld.ai`.
- Frontier API keys stay on the host running the gateway and are attached at
  egress; they are not persisted by the gateway.
- Completion cache and session memory are **off by default** (explicit opt-in).

## Contact

- Security disclosures → `security@modelmeld.ai`
- Everything else → `hello@modelmeld.ai`
- Maintainer → [@kevinsmith51](https://github.com/kevinsmith51) (see [`MAINTAINERS.md`](MAINTAINERS.md))

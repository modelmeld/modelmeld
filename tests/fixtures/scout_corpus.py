"""Labeled prompt corpus for HeuristicScout evaluation.

Each entry is `(provenance, complexity, prompt)`:
  - provenance: free-form tag (cursor / claude_code / aider / cline / generic)
  - complexity: "simple" (should route LOCAL) or "complex" (should route CLOUD)
  - prompt: the user-message text

Expanded from the original 18-prompt baseline to ~60 prompts covering
per-tool patterns. The simple-corpus precision target is ≥80% — we are not
expected to be perfect on a regex-based classifier.
"""

from __future__ import annotations

# (provenance, complexity, prompt)
LABELED_CORPUS: list[tuple[str, str, str]] = [
    # ---------- Cursor: tab-completion + chat ----------
    ("cursor", "simple", "Complete the line: const handleSubmit ="),
    ("cursor", "simple", "Autocomplete this import line: from collec"),
    ("cursor", "simple", "Fix the indent of this code block."),
    ("cursor", "simple", "Add type hints to the following arguments."),
    ("cursor", "complex", "Refactor this 400-line file into separate modules following the single-responsibility principle. Analyze trade-offs."),
    ("cursor", "complex", "Design a state-management approach for this app. Step by step, walk through Redux vs Zustand."),

    # ---------- Claude Code: agent-style edits ----------
    ("claude_code", "simple", "Add a docstring to this function."),
    ("claude_code", "simple", "Rename variable user_count to active_users throughout this file."),
    ("claude_code", "simple", "Generate boilerplate for a Pydantic BaseModel called User."),
    ("claude_code", "simple", "Add a type hint to the parameters of this function."),
    ("claude_code", "complex", "Design a distributed task queue system. Analyze trade-offs between Redis Streams, Kafka, and SQS for our use case."),
    ("claude_code", "complex", "Step by step, debug this tricky race condition in our connection pool and propose a fix."),
    ("claude_code", "complex", "Plan a zero-downtime migration from Postgres 12 to 16 across our system."),

    # ---------- Aider: SEARCH/REPLACE edits ----------
    ("aider", "simple", "Generate a SEARCH/REPLACE block to fix the indent in this file."),
    ("aider", "simple", "Rename the function get_user to fetch_user in user.py."),
    ("aider", "simple", "Add a missing return type annotation in api/handlers.py."),
    ("aider", "complex", "Analyze the architecture in this codebase and propose a refactor to use dependency injection. Explain the trade-offs in detail."),
    ("aider", "complex", "Design a plan to incrementally migrate this Flask app to FastAPI. Step by step."),

    # ---------- Cline: agentic with tools ----------
    ("cline", "simple", "Format this JSON: {a:1,b:2}"),
    ("cline", "simple", "What does typing.Optional do?"),
    ("cline", "complex", "Build a full-stack application: design the database schema, generate the API, and scaffold the React frontend."),
    ("cline", "complex", "Analyze the codebase and design a comprehensive testing strategy. Step by step, plan the migration."),

    # ---------- Generic simple (no tool fingerprint expected) ----------
    ("generic", "simple", "What is 2 plus 2?"),
    ("generic", "simple", "Capital of France?"),
    ("generic", "simple", "What does Optional[int] mean in Python?"),
    ("generic", "simple", "Format this JSON: {x:1}"),
    ("generic", "simple", "Add a docstring to this method."),
    ("generic", "simple", "Rename variable x to count."),
    ("generic", "simple", "Fix the indent of this snippet."),
    ("generic", "simple", "Add type hints to this function signature."),
    ("generic", "simple", "Prettify this CSS."),
    ("generic", "simple", "Autocomplete: def factorial("),
    ("generic", "simple", "Complete the function: def add(a, b):"),
    ("generic", "simple", "What is the difference between list and tuple in Python?"),
    ("generic", "simple", "Generate boilerplate for a Flask hello-world app."),
    ("generic", "simple", "What does len() return for an empty dict?"),

    # ---------- Generic complex ----------
    ("generic", "complex", "Design a distributed system for processing 1M events/sec with exactly-once semantics."),
    ("generic", "complex", "Analyze the trade-offs between microservices and monolithic architectures for a fintech startup."),
    ("generic", "complex", "Step by step, derive the time complexity of this recursive algorithm and propose an optimization."),
    ("generic", "complex", "Prove that this concurrent algorithm is free of deadlocks under the stated invariants."),
    ("generic", "complex", "Plan a zero-downtime migration from Postgres 12 to 16 across our system."),
    ("generic", "complex", "Explain the CAP theorem in detail and walk through how Spanner reconciles it."),
    ("generic", "complex", "Debug this tricky race condition in our connection pool."),
    ("generic", "complex", "Design an architecture for a multi-tenant ML feature store."),
    ("generic", "complex", "Reason about the trade-offs between TCP and QUIC for a real-time game server."),
    ("generic", "complex", "Outline the system architecture for an event-sourced bank ledger with audit guarantees."),
]


# Backwards-compat with legacy scout tests.
SIMPLE_PROMPTS: list[str] = [p for tool, c, p in LABELED_CORPUS if c == "simple" and tool == "generic"]
COMPLEX_PROMPTS: list[str] = [p for tool, c, p in LABELED_CORPUS if c == "complex" and tool == "generic"]

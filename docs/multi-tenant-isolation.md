# Multi-tenant isolation contract

ModelMeld is built for multi-tenant deployments — one gateway instance
serves many independent tenants and must guarantee no data crosses tenant
boundaries. This document spells out which defenses are in place, what
threats they cover, and what's still TODO.

## Tenant identity

Every memory operation is keyed on `(tenant_id, session_id)`:

- `tenant_id` comes from the enterprise auth middleware via
  `request.state.tenant_id` (the API key's owning tenant). Without auth,
  unauthenticated traffic uses the `ANONYMOUS_TENANT_ID = "__anonymous__"`
  sentinel.
- `session_id` comes from the client's `x-modelmeld-session-id` header.

Both are validated:
- `validate_tenant_id` enforces `\A[A-Za-z0-9_.-]{1,128}\Z` (no
  whitespace, control chars, or path separators). Notably uses `\Z`,
  not `$`, so a trailing newline can't sneak through.
- `extract_memory_identity` rejects malformed tenants with HTTP 400.

## What's structurally isolated

The `InMemoryMemoryStore` (and any compliant backend) stores everything
under a `dict[(tenant_id, session_id), …]` key. Cross-tenant access is
**structurally impossible**, not just access-controlled — the lookup
under the wrong tenant just misses:

| Method                            | Cross-tenant behavior                |
| --------------------------------- | ------------------------------------ |
| `get_session(s, wrong_tenant)`    | Returns `None`                       |
| `list_turns(s, wrong_tenant)`     | Returns `[]`                         |
| `turn_count(s, wrong_tenant)`     | Returns `0`                          |
| `get_facts(s, wrong_tenant)`      | Returns `[]`                         |
| `get_summary(s, wrong_tenant)`    | Returns `None`                       |
| `append_turn(s, wrong_tenant, …)` | Raises `LookupError` (session missing) |
| `set_fact(s, wrong_tenant, …)`    | Raises `LookupError`                 |
| `upsert_summary(s, wrong_tenant, …)` | Raises `LookupError`              |
| `delete_fact(s, wrong_tenant, k)` | Returns `False` (no-op)              |
| `clear_summary(s, wrong_tenant)`  | Returns `False` (no-op)              |

The chat route, summarizer worker, and `assemble_context()` all consult
memory via these methods — they inherit the isolation automatically.

## Threat model + defenses

| Threat                                                       | Defense                                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| Tenant A guesses tenant B's `session_id`                     | (tenant_id, session_id) key tuple — A's tenant_id never matches B's row |
| Anonymous traffic reads authenticated tenant's data          | `ANONYMOUS_TENANT_ID` ≠ any real tenant_id; structural isolation        |
| Misconfigured auth middleware passes attacker-controlled tenant_id | `validate_tenant_id` at every entry point rejects malformed values |
| Tenant attempts `tenant_id="__anonymous__"` to claim sentinel | The pattern doesn't permit underscores at sentinel position structurally; even so, only the EXACT sentinel routes to the anonymous namespace |
| Tenant attempts `tenant_id="__anonymous__\n"` to confuse logs | `\Z` anchor in regex rejects trailing newlines (subtle Python `$` gotcha) |
| Concurrent multi-tenant writes share locks → race            | Per-`(tenant, session)` asyncio.Lock; tested with 90 concurrent appends across 3 tenants |
| Summarizer worker run by tenant A scoops up tenant B's session | Worker runs under tenant_id; B's data is invisible to A's worker     |
| Context injection leaks other tenant's facts into prompt     | `assemble_context` uses caller's tenant_id; other tenants' rows never returned |

## Tested

`core-engine/tests/test_memory_tenant_isolation.py` is the security-critical
regression suite. 44 tests covering:
- `tenant_id` validation (15 parametrized cases: empty/whitespace/newlines/
  path-traversal/control-chars/oversized + 8 valid patterns + sentinel
  exact-match)
- Cross-tenant lookups returning empty/None on every method (6)
- Mutations under wrong tenant raising `LookupError` (3)
- Same `session_id` under two tenants stays distinct across L0/L1/L2 (1)
- Anonymous ↔ authenticated boundary (3)
- Concurrent multi-tenant writes (1)
- `assemble_context` boundary (1)
- Summarizer worker boundary (1)
- Qdrant collection name helper (8) — deterministic, distinct per tenant,
  rejects malformed, handles long names via hash suffix

## Forward-prep: Qdrant collection naming

When vector retrieval ships (later phase), each tenant gets its own
Qdrant collection. `tenant_collection_name(tenant_id)` is the canonical
mapping:
- Prefix `tksaver_` to disambiguate from non-ModelMeld collections
- Tenant_id is sanitized (`[^A-Za-z0-9_-]` → `_`)
- Long tenant_ids → truncated head + 12-hex SHA-256 suffix (different
  long tenant_ids never collide)
- Always validates `tenant_id` first — no malformed values reach Qdrant
- Always ≤60 chars (well under Qdrant's 255 limit)

## TODOs (deferred to enterprise-control / later phases)

These defenses aren't shipped yet because the corresponding subsystem
hasn't been wired up:

1. **vLLM KV-cache isolation** — when routing to a shared vLLM endpoint,
   prefix caching can leak partial activations across tenants. Mitigation
   (deferred): per-tenant `prefix_cache_namespace` or equivalent in the
   vLLM adapter. Requires real GPU to test.
2. **Qdrant collection lifecycle** — actual collection
   create/delete/reindex code (the naming helper is in place; the
   client integration is not).
3. **Postgres row-level security** — when the in-memory store is replaced
   by the Postgres-backed implementation in enterprise-control,
   `SET LOCAL app.current_tenant` + RLS policies become the bedrock
   defense. The current ABC contract already passes `tenant_id` to
   every method, so this is a backend-internal concern.
4. **Audit log of cross-tenant access attempts** — when a `LookupError`
   fires because the tenant_id doesn't match, we should surface that as
   a security event (vs an ordinary 404). Today it just propagates.

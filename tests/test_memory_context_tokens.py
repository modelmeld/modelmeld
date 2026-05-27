"""Token-based budget enforcement in assemble_context."""

from __future__ import annotations

from modelmeld.memory import (
    InMemoryMemoryStore,
    MemoryIdentity,
    MemoryMode,
    Role,
    assemble_context,
)
from modelmeld.tokens import CharBasedTokenCounter, TokenCounter


def _identity(mode: MemoryMode = MemoryMode.FULL) -> MemoryIdentity:
    return MemoryIdentity(tenant_id="acme", session_id="s-1", user_id=None, mode=mode)


class _FixedTokenCounter(TokenCounter):
    """Predictable counter: 1 token per char, ignores model."""
    name = "fixed"

    def count_text(self, text: str, model: str | None = None) -> int:
        return len(text)

    def count_messages(self, messages, model=None):
        return sum(self.count_text(getattr(m, "content", "") or "") for m in messages)


async def test_token_budget_drops_oldest_turns_first() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    # 10 turns × 100 chars (= 100 tokens each under _FixedTokenCounter)
    for _ in range(10):
        await store.append_turn("s-1", "acme", Role.USER, "X" * 100, 25)

    ctx = await assemble_context(
        store, _identity(),
        max_tokens=300, token_counter=_FixedTokenCounter(), model="m",
    )
    # Budget allows ~300 tokens; each turn ≈ 100 + 4 overhead → ~3 turns
    assert ctx.truncated is True
    assert 1 <= len(ctx.recent_turns) <= 3
    # Surviving turns come from the END (newest kept)
    if ctx.recent_turns:
        assert ctx.recent_turns[-1].content == "X" * 100


async def test_token_budget_truncates_summary_to_fit() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "k", "v")   # tiny facts payload
    long_summary = "Z" * 5000
    await store.upsert_summary("s-1", "acme", text=long_summary, last_applied_turn_id=None)

    ctx = await assemble_context(
        store, _identity(MemoryMode.AUGMENT),
        max_tokens=200, token_counter=_FixedTokenCounter(), model="m",
    )
    assert ctx.truncated is True
    assert ctx.summary is not None
    # Truncated but still present (facts have priority over summary)
    assert len(ctx.summary.text) < len(long_summary)
    assert len(ctx.summary.text) > 0
    assert ctx.facts   # facts preserved


async def test_token_budget_no_op_under_limit() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "k", "v")
    ctx = await assemble_context(
        store, _identity(),
        max_tokens=10000, token_counter=_FixedTokenCounter(), model="m",
    )
    assert ctx.truncated is False


async def test_char_budget_used_when_token_params_absent() -> None:
    """Backwards-compat: omit max_tokens + counter → falls back to char-based budget."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    for _ in range(5):
        await store.append_turn("s-1", "acme", Role.USER, "X" * 200, 50)

    ctx = await assemble_context(store, _identity(), max_chars=500)
    # Char-based path is the legacy fallback; behavior covered in test_memory_context.py
    assert ctx.truncated is True


async def test_token_budget_uses_real_counter_for_summary_tail_binary_search() -> None:
    """Summary truncation uses binary search → length is tight against budget."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    # 1000-char summary; with fixed counter (1 token/char) it's 1000 tokens
    await store.upsert_summary("s-1", "acme", text="x" * 1000, last_applied_turn_id=None)
    ctx = await assemble_context(
        store, _identity(MemoryMode.AUGMENT),
        max_tokens=250, token_counter=_FixedTokenCounter(), model="m",
    )
    assert ctx.summary is not None
    # Summary trimmed to at most 250 chars (= 250 tokens under fixed counter)
    assert len(ctx.summary.text) <= 250
    # And ideally close to budget — binary search should find a tight bound
    assert len(ctx.summary.text) >= 240

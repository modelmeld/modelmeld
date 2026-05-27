"""assemble_context + inject_into_request + budget enforcement."""

from __future__ import annotations

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
)
from modelmeld.memory import (
    InMemoryMemoryStore,
    MemoryContext,
    MemoryIdentity,
    MemoryMode,
    Role,
    assemble_context,
    inject_into_request,
    render_system_message,
)


def _identity(active: bool = True, mode: MemoryMode = MemoryMode.AUGMENT) -> MemoryIdentity:
    return MemoryIdentity(
        tenant_id="acme",
        session_id="s-1" if active else None,
        user_id="alice",
        mode=mode,
    )


def _req(text: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
    )


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------

async def test_inactive_identity_yields_empty_context() -> None:
    store = InMemoryMemoryStore()
    ctx = await assemble_context(store, _identity(active=False))
    assert ctx.has_content() is False


async def test_off_mode_yields_empty_context_even_with_data() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "user_name", "Kevin")
    ctx = await assemble_context(store, _identity(mode=MemoryMode.OFF))
    assert ctx.has_content() is False


async def test_none_memory_store_returns_empty_context() -> None:
    ctx = await assemble_context(None, _identity())
    assert ctx.has_content() is False


# ---------------------------------------------------------------------------
# AUGMENT mode — L1 + L2 only
# ---------------------------------------------------------------------------

async def test_augment_mode_includes_facts_and_summary_not_recent_turns() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "user_name", "Kevin")
    await store.set_fact("s-1", "acme", "preferred_lang", "Python")
    await store.append_turn("s-1", "acme", Role.USER, "earlier message", 3)
    await store.upsert_summary("s-1", "acme", text="Discussed Python tooling.",
                                last_applied_turn_id=None)

    ctx = await assemble_context(store, _identity(mode=MemoryMode.AUGMENT))
    assert len(ctx.facts) == 2
    assert ctx.summary is not None
    assert "Python tooling" in ctx.summary.text
    assert ctx.recent_turns == []   # AUGMENT mode doesn't include L3


# ---------------------------------------------------------------------------
# FULL mode — L1 + L2 + L3
# ---------------------------------------------------------------------------

async def test_full_mode_includes_unsummarized_turns() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    turns = []
    for i in range(5):
        turns.append(await store.append_turn("s-1", "acme",
                                              Role.USER if i % 2 == 0 else Role.ASSISTANT,
                                              f"t{i}", 1))
    # Summary covers t0..t2; t3, t4 are L3 hot-zone
    await store.upsert_summary("s-1", "acme", text="early summary",
                                last_applied_turn_id=turns[2].turn_id)

    ctx = await assemble_context(store, _identity(mode=MemoryMode.FULL))
    assert [t.content for t in ctx.recent_turns] == ["t3", "t4"]


async def test_full_mode_caps_hot_zone() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    for i in range(30):
        await store.append_turn("s-1", "acme", Role.USER, f"t{i}", 1)
    ctx = await assemble_context(store, _identity(mode=MemoryMode.FULL), hot_zone_cap=5)
    assert [t.content for t in ctx.recent_turns] == ["t25", "t26", "t27", "t28", "t29"]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

async def test_render_system_message_includes_sections() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "user_name", "Kevin")
    await store.upsert_summary("s-1", "acme", text="A long summary of stuff.",
                                last_applied_turn_id=None)

    ctx = await assemble_context(store, _identity())
    text = render_system_message(ctx)
    assert "[Context restored by ModelMeld gateway]" in text
    assert "## Persistent facts" in text
    assert "- user_name: Kevin" in text
    assert "## Conversation summary" in text
    assert "A long summary of stuff." in text


def test_render_empty_context_returns_empty_string() -> None:
    text = render_system_message(MemoryContext())
    assert text == ""


# ---------------------------------------------------------------------------
# inject_into_request
# ---------------------------------------------------------------------------

async def test_inject_prepends_system_message() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "k", "v")
    ctx = await assemble_context(store, _identity())

    req = _req("hi")
    out = inject_into_request(req, ctx)
    assert len(out.messages) == 2
    assert isinstance(out.messages[0], SystemMessage)
    assert "k: v" in out.messages[0].content   # type: ignore[arg-type]
    assert isinstance(out.messages[1], UserMessage)


async def test_inject_full_mode_replays_user_assistant_turns() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.append_turn("s-1", "acme", Role.USER, "what's PyTorch?", 3)
    await store.append_turn("s-1", "acme", Role.ASSISTANT, "It's a deep learning library.", 7)
    await store.append_turn("s-1", "acme", Role.SYSTEM, "system-only note", 3)
    await store.append_turn("s-1", "acme", Role.TOOL, "tool result", 3)

    ctx = await assemble_context(store, _identity(mode=MemoryMode.FULL))
    out = inject_into_request(_req("follow-up"), ctx)
    # System turn (no facts/summary) is empty → no prepended system message,
    # but the user/assistant replays come through. Tool + system L0 entries
    # are skipped from the replay.
    roles = [type(m).__name__ for m in out.messages]
    # Expect: [UserMessage(replay), AssistantMessage(replay), UserMessage(follow-up)]
    # Note no SystemMessage since L1+L2 are empty.
    assert roles == ["UserMessage", "AssistantMessage", "UserMessage"]
    assert out.messages[0].content == "what's PyTorch?"
    assert out.messages[1].content == "It's a deep learning library."
    assert out.messages[2].content == "follow-up"


async def test_inject_full_mode_with_facts_and_summary_and_turns() -> None:
    """Full stack: L1 + L2 system message + L3 replayed turns + framework messages."""
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "user_name", "Kevin")
    turn_a = await store.append_turn("s-1", "acme", Role.USER, "first", 1)
    await store.append_turn("s-1", "acme", Role.ASSISTANT, "reply to first", 1)
    await store.upsert_summary("s-1", "acme", text="Initial exchange.",
                                last_applied_turn_id=turn_a.turn_id)
    await store.append_turn("s-1", "acme", Role.USER, "second", 1)
    await store.append_turn("s-1", "acme", Role.ASSISTANT, "reply to second", 1)

    ctx = await assemble_context(store, _identity(mode=MemoryMode.FULL))
    out = inject_into_request(_req("third"), ctx)
    roles = [type(m).__name__ for m in out.messages]
    # System (facts+summary) + replayed assistant("reply to first") + replayed
    # user("second") + replayed assistant("reply to second") + framework's
    # user("third"). Note turn_a (user "first") is the high-water mark; L3
    # is turns AFTER it.
    assert roles == [
        "SystemMessage",
        "AssistantMessage", "UserMessage", "AssistantMessage",
        "UserMessage",
    ]
    sys_content = out.messages[0].content
    assert "Kevin" in sys_content  # type: ignore[operator]
    assert "Initial exchange" in sys_content  # type: ignore[operator]


def test_inject_empty_context_no_op() -> None:
    out = inject_into_request(_req("x"), MemoryContext())
    assert len(out.messages) == 1


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

async def test_budget_drops_oldest_recent_turns_first() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    # 10 turns × 200 chars = ~2000 chars of L3 content
    for _i in range(10):
        await store.append_turn("s-1", "acme", Role.USER, "X" * 200, 50)

    # Budget allows ~700 chars (enough for ~3 turns + overhead)
    ctx = await assemble_context(
        store, _identity(mode=MemoryMode.FULL), max_chars=700,
    )
    assert ctx.truncated is True
    assert len(ctx.recent_turns) < 10
    # Whatever survived comes from the END (most recent kept)
    if ctx.recent_turns:
        # The dropped ones came from the FRONT — surviving turns are the recent tail
        assert ctx.recent_turns[-1].content == "X" * 200


async def test_budget_truncates_summary_when_facts_alone_fit() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "k", "v")
    long_summary = "Z" * 5000
    await store.upsert_summary("s-1", "acme", text=long_summary, last_applied_turn_id=None)
    ctx = await assemble_context(store, _identity(), max_chars=500)
    assert ctx.truncated is True
    assert ctx.summary is not None
    # Summary got truncated but is still present (facts have priority over summary)
    assert len(ctx.summary.text) < len(long_summary)
    assert ctx.facts   # facts preserved


async def test_budget_not_triggered_under_limit() -> None:
    store = InMemoryMemoryStore()
    await store.get_or_create_session("s-1", "acme")
    await store.set_fact("s-1", "acme", "k", "v")
    ctx = await assemble_context(store, _identity(), max_chars=10000)
    assert ctx.truncated is False


# ---------------------------------------------------------------------------
# Memory failure doesn't break context assembly
# ---------------------------------------------------------------------------

class _BrokenStore(InMemoryMemoryStore):
    async def get_facts(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("storage offline")


async def test_storage_read_failure_returns_empty_context() -> None:
    store = _BrokenStore()
    await store.get_or_create_session("s-1", "acme")
    ctx = await assemble_context(store, _identity())
    assert ctx.has_content() is False

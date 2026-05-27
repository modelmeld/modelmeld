# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Memory context assembly + request injection.

Bridges the L1/L2/L3 storage layer to the outgoing request:
  1. `assemble_context()` reads facts (L1) + summary (L2) + recent turns (L3)
     and packs them into a `MemoryContext`, respecting a char budget.
  2. `inject_into_request()` prepends a synthetic system message (L1+L2) and,
     in `FULL` mode, the L3 turns as replayed user/assistant messages.

Memory read failures are caught here so a degraded storage layer doesn't
break the user's request — the chat route gets back an empty context and
proceeds without augmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    SystemMessage,
    UserMessage,
)
from modelmeld.memory.base import Fact, MemoryStore, Role, Summary, Turn
from modelmeld.memory.identity import MemoryIdentity, MemoryMode
from modelmeld.memory.tiers import turns_since_summary
from modelmeld.tokens import TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_HOT_ZONE_CAP = 20
# 48k chars ≈ 12k tokens. Fits well within the ≥131k context window of
# every model in the launch lineup except phi-4 (16k ctx) where the
# customer's own prompt + max_tokens would crowd this out anyway —
# but phi-4 is the ultra-cheap tier, not the typical case.
# The bench surfaced the original 16k budget evicting T1's constraints
# by turn 8-9 of a coding session, breaking the distant-recall moat
# claim. 48k holds a ~15-turn code-heavy session.
DEFAULT_MAX_CONTEXT_CHARS = 48000
# Default token budget = ~12k tokens. Same rationale as MAX_CONTEXT_CHARS.
# Caller can override per request when model context is known.
DEFAULT_MAX_CONTEXT_TOKENS = 12000


@dataclass(frozen=True)
class MemoryContext:
    """Assembled memory ready to inject into a request."""

    facts: list[Fact] = field(default_factory=list)
    summary: Summary | None = None
    recent_turns: list[Turn] = field(default_factory=list)
    truncated: bool = False   # set True when budget enforcement dropped content

    def has_content(self) -> bool:
        return bool(self.facts) or self.summary is not None or bool(self.recent_turns)

    def estimated_chars(self) -> int:
        return _estimate_chars(self.facts, self.summary, self.recent_turns)


async def assemble_context(
    memory: MemoryStore | None,
    mem_identity: MemoryIdentity,
    *,
    hot_zone_cap: int = DEFAULT_HOT_ZONE_CAP,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    model: str | None = None,
) -> MemoryContext:
    """Read L1/L2/L3 for the active session and pack into a MemoryContext.

    Budget enforcement:
      - If `max_tokens` + `token_counter` are supplied → token-based budget
        (preferred; uses the right tokenizer for the model).
      - Otherwise → char-based budget via `max_chars` (legacy; defaults).

    Returns empty context when:
      - memory store is unconfigured, OR
      - identity is inactive (no session_id), OR
      - mode is OFF, OR
      - the storage read fails (best-effort).
    """
    if memory is None or not mem_identity.active or mem_identity.mode == MemoryMode.OFF:
        return MemoryContext()

    assert mem_identity.session_id is not None
    try:
        facts = await memory.get_facts(mem_identity.session_id, mem_identity.tenant_id)
        summary = await memory.get_summary(mem_identity.session_id, mem_identity.tenant_id)
        if mem_identity.mode == MemoryMode.FULL:
            recent = await turns_since_summary(
                memory, mem_identity.session_id, mem_identity.tenant_id,
                cap=hot_zone_cap,
            )
        else:
            recent = []
    except Exception:  # noqa: BLE001 — memory reads are best-effort
        logger.exception(
            "memory read failed for session %s (degraded; proceeding without context)",
            mem_identity.session_id,
        )
        return MemoryContext()

    if max_tokens is not None and token_counter is not None:
        return _enforce_token_budget(
            facts, summary, recent,
            max_tokens=max_tokens, counter=token_counter, model=model,
        )
    return _enforce_budget(facts, summary, recent, max_chars=max_chars)


def inject_into_request(
    request: ChatCompletionRequest,
    context: MemoryContext,
) -> ChatCompletionRequest:
    """Prepend the memory context to the outgoing request's messages.

    Synthetic system message (L1 facts + L2 summary) goes first. In FULL mode
    the L3 recent turns are replayed as user/assistant messages before the
    framework's own messages. System and tool turns are NOT replayed (they
    don't reconstruct cleanly out of context).

    No-ops on empty contexts.
    """
    if not context.has_content():
        return request

    new_messages: list = []

    sys_text = render_system_message(context)
    if sys_text:
        new_messages.append(SystemMessage(role="system", content=sys_text))

    for turn in context.recent_turns:
        if turn.role == Role.USER:
            new_messages.append(UserMessage(role="user", content=turn.content))
        elif turn.role == Role.ASSISTANT:
            new_messages.append(AssistantMessage(role="assistant", content=turn.content))
        # role=system / role=tool turns are skipped — replaying them out of
        # original context would mislead the model.

    new_messages.extend(request.messages)
    return request.model_copy(update={"messages": new_messages})


def render_system_message(ctx: MemoryContext) -> str:
    """Format L1 facts + L2 summary as the synthetic system message body."""
    sections: list[str] = []
    if ctx.facts:
        lines = ["## Persistent facts"]
        for fact in ctx.facts:
            lines.append(f"- {fact.key}: {fact.value}")
        sections.append("\n".join(lines))
    if ctx.summary is not None and ctx.summary.text.strip():
        sections.append("## Conversation summary\n" + ctx.summary.text.strip())
    if not sections:
        return ""
    body = "\n\n".join(sections)
    return "[Context restored by ModelMeld gateway]\n\n" + body


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def _estimate_chars(
    facts: list[Fact],
    summary: Summary | None,
    recent: list[Turn],
) -> int:
    total = sum(len(f.key) + len(f.value) + 5 for f in facts)
    if summary is not None:
        total += len(summary.text)
    total += sum(len(t.content) + 12 for t in recent)
    return total


def _enforce_budget(
    facts: list[Fact],
    summary: Summary | None,
    recent: list[Turn],
    *,
    max_chars: int,
) -> MemoryContext:
    """Apply the char budget by dropping least-important content first.

    Priority order (highest first): facts > summary > recent turns.
    Drops recent turns from the oldest end first, then truncates the summary
    body, then drops facts (cheapest token-wise so kept longest).
    """
    truncated = False
    current = _estimate_chars(facts, summary, recent)
    if current <= max_chars:
        return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=False)

    # Drop oldest recent turns until we fit.
    while recent and current > max_chars:
        dropped = recent.pop(0)
        current -= len(dropped.content) + 12
        truncated = True

    if current <= max_chars:
        return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=True)

    # Still too large → truncate the summary text from the end.
    if summary is not None and summary.text:
        room_for_summary = max_chars - sum(len(f.key) + len(f.value) + 5 for f in facts)
        if room_for_summary > 0:
            truncated_text = summary.text[: max(0, room_for_summary)]
            summary = Summary(
                summary_id=summary.summary_id,
                session_id=summary.session_id,
                tenant_id=summary.tenant_id,
                text=truncated_text,
                last_applied_turn_id=summary.last_applied_turn_id,
                version=summary.version,
                source_model=summary.source_model,
                created_at=summary.created_at,
                updated_at=summary.updated_at,
            )
        else:
            summary = None
        truncated = True
        current = _estimate_chars(facts, summary, recent)

    # If even facts overflow (degenerate config), drop oldest facts.
    while facts and current > max_chars:
        facts.pop(0)
        current = _estimate_chars(facts, summary, recent)
        truncated = True

    return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=truncated)


# ---------------------------------------------------------------------------
# Token-based budget enforcement
# ---------------------------------------------------------------------------

def _enforce_token_budget(
    facts: list[Fact],
    summary: Summary | None,
    recent: list[Turn],
    *,
    max_tokens: int,
    counter: TokenCounter,
    model: str | None,
) -> MemoryContext:
    """Token-aware budget. Same priority order as `_enforce_budget`:
    drop oldest recent turns → truncate summary → drop oldest facts."""

    def tokens_of_fact(f: Fact) -> int:
        return counter.count_text(f"- {f.key}: {f.value}", model) + 1

    def tokens_of_turn(t: Turn) -> int:
        return counter.count_text(t.content, model) + 4  # +4 for role tag overhead

    truncated = False

    def current_total(s: Summary | None) -> int:
        n = sum(tokens_of_fact(f) for f in facts) + sum(tokens_of_turn(t) for t in recent)
        if s is not None and s.text:
            n += counter.count_text(s.text, model)
        return n

    if current_total(summary) <= max_tokens:
        return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=False)

    while recent and current_total(summary) > max_tokens:
        recent.pop(0)
        truncated = True

    if current_total(summary) <= max_tokens:
        return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=True)

    if summary is not None and summary.text:
        room = max_tokens - sum(tokens_of_fact(f) for f in facts)
        if room > 0:
            # Binary-search truncation (token-accurate but not free; OK once per request).
            text = summary.text
            lo, hi = 0, len(text)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if counter.count_text(text[:mid], model) <= room:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            summary = Summary(
                summary_id=summary.summary_id,
                session_id=summary.session_id,
                tenant_id=summary.tenant_id,
                text=text[:best],
                last_applied_turn_id=summary.last_applied_turn_id,
                version=summary.version,
                source_model=summary.source_model,
                created_at=summary.created_at,
                updated_at=summary.updated_at,
            )
        else:
            summary = None
        truncated = True

    while facts and current_total(summary) > max_tokens:
        facts.pop(0)
        truncated = True

    return MemoryContext(facts=facts, summary=summary, recent_turns=recent, truncated=truncated)

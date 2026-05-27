# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Cross-tier helpers — math + utility functions over L0 / L1 / L2.

The actual L3 retrieval + context assembly lives in `modelmeld.memory.context`.
This module holds the bookkeeping primitives both the context layer and the
summarizer worker need:

  - `turns_since_summary` — given the current Summary state, return the L0
    turns that are NOT YET reflected in the summary. This is the L3 hot-zone
    candidate set; the summarizer also reads it to know what to fold in.
  - `summary_freshness` — how many turns behind is the summary?
  - `needs_summary_refresh` — boolean trigger for the worker.
"""

from __future__ import annotations

from modelmeld.memory.base import MemoryStore, Summary, Turn


async def turns_since_summary(
    memory: MemoryStore,
    session_id: str,
    tenant_id: str,
    *,
    cap: int | None = None,
) -> list[Turn]:
    """L0 turns appended AFTER `summary.last_applied_turn_id`.

    When no summary exists yet, returns every turn (the entire log is the
    L3 candidate set). When `cap` is given, returns at most that many recent
    turns — useful as a safety cap on very long sessions.
    """
    summary = await memory.get_summary(session_id, tenant_id)
    all_turns = await memory.list_turns(session_id, tenant_id)
    after = _turns_after(all_turns, summary)
    if cap is not None and cap > 0 and len(after) > cap:
        return after[-cap:]
    return after


def _turns_after(turns: list[Turn], summary: Summary | None) -> list[Turn]:
    if summary is None or summary.last_applied_turn_id is None:
        return list(turns)
    # Find the high-water mark; everything strictly after it is unsummarized.
    for idx, turn in enumerate(turns):
        if turn.turn_id == summary.last_applied_turn_id:
            return turns[idx + 1:]
    # The recorded high-water turn isn't in the log anymore (truncated /
    # corrupted state). Conservative: treat everything as unsummarized so
    # the next summary pass re-folds the full log.
    return list(turns)


async def summary_freshness(
    memory: MemoryStore,
    session_id: str,
    tenant_id: str,
) -> int:
    """Number of turns the current summary is BEHIND the live log.

    0 means the summary covers every turn so far. Larger numbers mean the
    summary needs a refresh.
    """
    return len(await turns_since_summary(memory, session_id, tenant_id))


def needs_summary_refresh(
    behind: int,
    *,
    turn_threshold: int = 20,
) -> bool:
    """Trigger predicate: kick off a summarizer pass once `behind ≥ threshold`.

    The actual worker consults this. Separated out so the
    threshold + any future signals (e.g. wall-clock age, token budget) live
    in one place.
    """
    if turn_threshold <= 0:
        raise ValueError(f"turn_threshold must be ≥1, got {turn_threshold}")
    return behind >= turn_threshold

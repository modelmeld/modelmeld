# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Unit tests for session-key derivation (scout/session_key.py)."""

from __future__ import annotations

from modelmeld.api.schemas import ChatCompletionRequest, SystemMessage, UserMessage
from modelmeld.memory.identity import HEADER_SESSION_ID
from modelmeld.scout.session_key import derive_session_key


def _req(system: str, first_user: str, *later_user: str) -> ChatCompletionRequest:
    messages: list = [
        SystemMessage(role="system", content=system),
        UserMessage(role="user", content=first_user),
    ]
    for u in later_user:
        messages.append(UserMessage(role="user", content=u))
    return ChatCompletionRequest(model="anthropic/modelmeld-auto", messages=messages)


def test_explicit_session_id_header_wins() -> None:
    key = derive_session_key(
        {HEADER_SESSION_ID: "sess-123"}, _req("sys", "hello"), "tenantA"
    )
    assert key == "tenantA:sid:sess-123"


def test_implicit_key_is_stable_across_growing_history() -> None:
    # Same opening system+first-user, different later turns → same key.
    k1 = derive_session_key({}, _req("sys", "build a parser"), "t1")
    k2 = derive_session_key(
        {}, _req("sys", "build a parser", "now add tests", "fix the bug"), "t1"
    )
    assert k1 == k2
    assert k1.startswith("t1:impl:")


def test_different_first_user_diverges() -> None:
    k1 = derive_session_key({}, _req("sys", "task one"), "t1")
    k2 = derive_session_key({}, _req("sys", "task two"), "t1")
    assert k1 != k2


def test_tenant_salt_isolates_identical_content() -> None:
    k1 = derive_session_key({}, _req("sys", "same task"), "tenantA")
    k2 = derive_session_key({}, _req("sys", "same task"), "tenantB")
    assert k1 != k2


def test_empty_session_id_header_falls_back_to_implicit() -> None:
    key = derive_session_key({HEADER_SESSION_ID: "  "}, _req("sys", "x"), "t1")
    assert key.startswith("t1:impl:")

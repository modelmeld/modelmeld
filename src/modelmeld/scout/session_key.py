# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Per-session routing key derivation.

Reactive (mid-session) escalation needs to correlate the turns of one agentic
session so it can observe a trajectory across turns. The memory subsystem
already defines `x-modelmeld-session-id` ([[memory.identity.HEADER_SESSION_ID]])
for exactly this kind of correlation, so we reuse it when present.

Many clients — notably Claude Code — do NOT send that header. For those we
derive an IMPLICIT key from a stable conversation *prefix*: the system prompt
plus the first user message. That prefix is constant across a session even as
later turns accumulate, so every turn of the same session hashes to the same
key, while different sessions (different opening task) hash apart.

The key is ALWAYS tenant-salted. A content-hash collision must never let one
tenant's trajectory be attributed to another's session — even in this
shadow/observe-only increment, which has no routing or billing effect, the key
is kept correct so the reactive-escalation increment can rely on it unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from modelmeld.api.schemas import ChatCompletionRequest
from modelmeld.memory.identity import HEADER_SESSION_ID

# Length of the truncated hex digest for implicit keys. 16 hex chars = 64 bits;
# collisions only matter within one gateway's TTL window AND one tenant, so this
# is comfortably sufficient while keeping log lines short.
_IMPLICIT_DIGEST_LEN = 16


def _part_text(content: object) -> str:
    """Flatten a message `content` (str or list of parts) to plain text.

    Mirrors the user-text flattening in `policy.extract_user_text`: we read the
    `.text` of any part that has one and ignore non-text parts (images/audio).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        return " ".join(parts)
    return ""


def _stable_prefix(request: ChatCompletionRequest) -> str:
    """The session-stable prefix: system prompt(s) + the FIRST user message.

    Deliberately NOT all user turns (that grows every turn and would change the
    key mid-session). The opening system+user pair is fixed for the session.
    """
    system_text = ""
    first_user_text = ""
    for msg in request.messages:
        role = getattr(msg, "role", "")
        if role == "system" and not system_text:
            system_text = _part_text(getattr(msg, "content", None))
        elif role == "user" and not first_user_text:
            first_user_text = _part_text(getattr(msg, "content", None))
        if system_text and first_user_text:
            break
    return f"{system_text}\n{first_user_text}"


def derive_session_key(
    headers: Mapping[str, str],
    request: ChatCompletionRequest,
    tenant_id: str,
) -> str:
    """Return a stable, tenant-salted session key for `request`.

    Prefers the explicit `x-modelmeld-session-id` header; falls back to a hash
    of the session-stable conversation prefix. Always prefixed with `tenant_id`
    so keys never cross a tenant boundary.
    """
    lower = {k.lower(): v for k, v in headers.items()}
    explicit = lower.get(HEADER_SESSION_ID, "").strip()
    if explicit:
        return f"{tenant_id}:sid:{explicit}"

    digest = hashlib.sha256(
        f"{tenant_id}\x00{_stable_prefix(request)}".encode()
    ).hexdigest()[:_IMPLICIT_DIGEST_LEN]
    return f"{tenant_id}:impl:{digest}"


__all__ = ["derive_session_key"]

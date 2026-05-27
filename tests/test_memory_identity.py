"""extract_memory_identity — header parsing + auth-state combination."""

from __future__ import annotations

from modelmeld.memory import ANONYMOUS_TENANT_ID, extract_memory_identity
from modelmeld.memory.identity import (
    HEADER_SESSION_ID,
    HEADER_USER_ID_OVERRIDE,
)


def test_no_headers_no_auth_yields_inactive_anonymous() -> None:
    identity = extract_memory_identity({}, auth_tenant_id=None, auth_user_id=None)
    assert identity.active is False
    assert identity.session_id is None
    assert identity.tenant_id == ANONYMOUS_TENANT_ID
    assert identity.user_id is None


def test_session_header_alone_activates_with_anonymous_tenant() -> None:
    """Dev mode: client supplied session id but no auth → anonymous tenant + active."""
    identity = extract_memory_identity(
        {HEADER_SESSION_ID: "sess-123"},
        auth_tenant_id=None, auth_user_id=None,
    )
    assert identity.active is True
    assert identity.session_id == "sess-123"
    assert identity.tenant_id == ANONYMOUS_TENANT_ID


def test_auth_only_no_session_inactive() -> None:
    """Authenticated user but no session header → no memory continuity."""
    identity = extract_memory_identity(
        {}, auth_tenant_id="acme", auth_user_id="alice",
    )
    assert identity.active is False
    assert identity.tenant_id == "acme"
    assert identity.user_id == "alice"


def test_auth_plus_session_full_memory_active() -> None:
    identity = extract_memory_identity(
        {HEADER_SESSION_ID: "thread-42"},
        auth_tenant_id="acme", auth_user_id="alice",
    )
    assert identity.active is True
    assert identity.session_id == "thread-42"
    assert identity.tenant_id == "acme"
    assert identity.user_id == "alice"


def test_user_id_override_header() -> None:
    """LangGraph / AutoGen agents can override the auth user_id via header."""
    identity = extract_memory_identity(
        {HEADER_SESSION_ID: "s", HEADER_USER_ID_OVERRIDE: "subagent-7"},
        auth_tenant_id="acme", auth_user_id="alice",
    )
    assert identity.user_id == "subagent-7"  # override wins


def test_headers_case_insensitive() -> None:
    identity = extract_memory_identity(
        {"X-ModelMeld-Session-Id": "S-1"},
        auth_tenant_id=None, auth_user_id=None,
    )
    assert identity.session_id == "S-1"


def test_empty_session_header_treated_as_missing() -> None:
    identity = extract_memory_identity(
        {HEADER_SESSION_ID: "   "},
        auth_tenant_id="acme", auth_user_id=None,
    )
    assert identity.active is False
    assert identity.session_id is None

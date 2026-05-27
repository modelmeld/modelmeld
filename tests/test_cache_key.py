"""cache_key_for_request — stable hashing + uncacheable-request bypass."""

from __future__ import annotations

from modelmeld.api.schemas import (
    ChatCompletionRequest,
    FunctionDef,
    Tool,
    UserMessage,
)
from modelmeld.cache import cache_key_for_request


def _req(
    *,
    model: str = "gpt-5",
    content: str = "hello",
    temperature: float | None = None,
    stream: bool = False,
    tools: list[Tool] | None = None,
    n: int | None = None,
    user: str | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[UserMessage(role="user", content=content)],
        temperature=temperature,
        stream=stream,
        tools=tools or [],
        n=n,
        user=user,
    )


# ---------------------------------------------------------------------------
# Stability + collision avoidance
# ---------------------------------------------------------------------------

def test_same_input_same_key() -> None:
    a = cache_key_for_request(_req(), tenant_id="acme")
    b = cache_key_for_request(_req(), tenant_id="acme")
    assert a == b
    assert a is not None and a.startswith("gateway:cache:v1:acme:")


def test_different_content_different_keys() -> None:
    a = cache_key_for_request(_req(content="hello"), tenant_id="acme")
    b = cache_key_for_request(_req(content="goodbye"), tenant_id="acme")
    assert a != b


def test_different_model_different_keys() -> None:
    a = cache_key_for_request(_req(model="gpt-5"), tenant_id="acme")
    b = cache_key_for_request(_req(model="claude-opus-4-7"), tenant_id="acme")
    assert a != b


def test_different_temperature_different_keys() -> None:
    a = cache_key_for_request(_req(temperature=0.0), tenant_id="acme")
    b = cache_key_for_request(_req(temperature=0.7), tenant_id="acme")
    assert a != b


def test_different_tenants_different_keys() -> None:
    """Defense-in-depth: tenant_id is in the key from day 1."""
    a = cache_key_for_request(_req(), tenant_id="tenant-A")
    b = cache_key_for_request(_req(), tenant_id="tenant-B")
    assert a != b
    assert ":tenant-A:" in a
    assert ":tenant-B:" in b


def test_anonymous_tenant_uses_anon_bucket() -> None:
    key = cache_key_for_request(_req(), tenant_id=None)
    assert key is not None
    assert ":__anon__:" in key


# ---------------------------------------------------------------------------
# Non-semantic fields are stripped
# ---------------------------------------------------------------------------

def test_user_field_doesnt_fracture_cache() -> None:
    """`user` is per-call metadata; the cache should ignore it."""
    a = cache_key_for_request(_req(user="alice"), tenant_id="acme")
    b = cache_key_for_request(_req(user="bob"), tenant_id="acme")
    assert a == b


# ---------------------------------------------------------------------------
# Capability-routing pin: cache by served model
# ---------------------------------------------------------------------------

def test_served_model_pin_changes_key() -> None:
    """When capability routing rewrote the model, the cache should reflect that."""
    a = cache_key_for_request(_req(model="claude-opus-4-7"), tenant_id="acme",
                              served_model="claude-opus-4-7")
    b = cache_key_for_request(_req(model="claude-opus-4-7"), tenant_id="acme",
                              served_model="qwen3-coder-next")
    assert a != b


def test_same_served_model_different_requested_share_cache() -> None:
    """Two users requesting different models but both getting routed to the same
    served model SHARE a cache entry. This is the capability-routing payoff:
    if we send identical prompts to qwen3-coder-next regardless of what the
    framework asked for, the second user gets the cached answer."""
    a = cache_key_for_request(_req(model="claude-opus-4-7"), tenant_id="acme",
                              served_model="qwen3-coder-next")
    b = cache_key_for_request(_req(model="gpt-5"), tenant_id="acme",
                              served_model="qwen3-coder-next")
    assert a == b


# ---------------------------------------------------------------------------
# Uncacheable requests → None
# ---------------------------------------------------------------------------

def test_streaming_request_uncacheable() -> None:
    assert cache_key_for_request(_req(stream=True), tenant_id="acme") is None


def test_tool_call_request_uncacheable() -> None:
    tools = [Tool(
        type="function",
        function=FunctionDef(name="search", description="", parameters={"type": "object"}),
    )]
    assert cache_key_for_request(_req(tools=tools), tenant_id="acme") is None


def test_n_gt_1_request_uncacheable() -> None:
    assert cache_key_for_request(_req(n=3), tenant_id="acme") is None
    # n=1 IS cacheable (the default)
    assert cache_key_for_request(_req(n=1), tenant_id="acme") is not None


def test_n_unset_is_cacheable() -> None:
    assert cache_key_for_request(_req(), tenant_id="acme") is not None

"""Mem0MemoryProvider — scoping, context packing, and tenant isolation.

mem0's AsyncMemory is replaced with an in-process fake (no network, no qdrant)
so these tests exercise the PROVIDER's contract: correct session scoping,
packing search results into a MemoryContext, and — the security-critical part —
that each tenant gets its own isolated vector collection.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from modelmeld.api.schemas import ChatCompletionRequest, UserMessage
from modelmeld.memory import MemoryIdentity, MemoryMode
from modelmeld.memory.base import tenant_collection_name


class FakeAsyncMemory:
    """Stand-in for mem0.AsyncMemory. One instance per collection; stores adds
    locally so a search only ever sees memories written to the same instance."""

    # collection_name -> instance, so tests can inspect what got built.
    registry: ClassVar[dict[str, FakeAsyncMemory]] = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        self.collection = config["vector_store"]["config"]["collection_name"]
        self.adds: list[tuple[list[dict], dict]] = []

    @classmethod
    def from_config(cls, config: dict) -> FakeAsyncMemory:
        inst = cls(config)
        cls.registry[inst.collection] = inst
        return inst

    async def add(self, messages, **kwargs):
        self.adds.append((messages, kwargs))
        return {"results": []}

    async def search(self, query, **kwargs):
        self.last_search_kwargs = kwargs
        mems = [
            m["content"]
            for msgs, _ in self.adds
            for m in msgs
            if m["role"] == "user"
        ]
        return {"results": [{"memory": t} for t in mems]}


@pytest.fixture
def fake_mem0(monkeypatch):
    """Install a fake `mem0` module exposing FakeAsyncMemory."""
    FakeAsyncMemory.registry = {}
    fake_module = types.ModuleType("mem0")
    fake_module.AsyncMemory = FakeAsyncMemory
    monkeypatch.setitem(sys.modules, "mem0", fake_module)
    yield FakeAsyncMemory


def _provider(**kwargs):
    from modelmeld.memory import Mem0MemoryProvider

    defaults = dict(infer=True, top_k=5, base_url="http://gw/v1", api_key="k")
    defaults.update(kwargs)
    return Mem0MemoryProvider(**defaults)


def _identity(tenant="acme", session="s-1", active=True) -> MemoryIdentity:
    return MemoryIdentity(
        tenant_id=tenant,
        session_id=session if active else None,
        user_id="alice",
        mode=MemoryMode.AUGMENT,
    )


def _req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[UserMessage(role="user", content=text)],
    )


# ---------------------------------------------------------------------------
# record + retrieve
# ---------------------------------------------------------------------------

async def test_record_scopes_by_session_run_id(fake_mem0) -> None:
    provider = _provider()
    await provider.record(
        _identity(), _req("remember my name is alice"), "Got it, alice.",
    )
    inst = fake_mem0.registry[tenant_collection_name("acme")]
    assert len(inst.adds) == 1
    messages, kwargs = inst.adds[0]
    assert kwargs["run_id"] == "s-1"
    assert kwargs["infer"] is True
    assert {m["role"] for m in messages} == {"user", "assistant"}


async def test_record_infer_false_passed_through(fake_mem0) -> None:
    provider = _provider(infer=False)
    await provider.record(_identity(), _req("hi"), "hello")
    inst = fake_mem0.registry[tenant_collection_name("acme")]
    assert inst.adds[0][1]["infer"] is False


async def test_retrieve_packs_memories_and_filters_by_session(fake_mem0) -> None:
    provider = _provider()
    await provider.record(_identity(), _req("my name is alice"), "ok")
    ctx = await provider.retrieve(_identity(), _req("what is my name?"))
    assert ctx.summary is not None
    assert "my name is alice" in ctx.summary.text
    inst = fake_mem0.registry[tenant_collection_name("acme")]
    assert inst.last_search_kwargs["filters"] == {"run_id": "s-1"}
    assert inst.last_search_kwargs["top_k"] == 5


async def test_retrieve_empty_when_no_memories(fake_mem0) -> None:
    provider = _provider()
    ctx = await provider.retrieve(_identity(), _req("anything?"))
    assert ctx.summary is None
    assert not ctx.has_content()


# ---------------------------------------------------------------------------
# Tenant isolation (security-critical)
# ---------------------------------------------------------------------------

async def test_tenant_isolation_separate_collections(fake_mem0) -> None:
    provider = _provider()
    # Tenant A writes a secret under session s-1.
    await provider.record(
        _identity(tenant="tenant-a", session="s-1"),
        _req("tenant A secret: launch code 1234"),
        "noted",
    )
    # Tenant B retrieves the SAME session id — must see nothing of A's.
    ctx = await provider.retrieve(
        _identity(tenant="tenant-b", session="s-1"), _req("launch code?"),
    )
    assert ctx.summary is None, "tenant B must not see tenant A's memory"
    # And the two tenants resolved to distinct vector collections.
    assert tenant_collection_name("tenant-a") in fake_mem0.registry
    assert tenant_collection_name("tenant-b") in fake_mem0.registry
    assert tenant_collection_name("tenant-a") != tenant_collection_name("tenant-b")


# ---------------------------------------------------------------------------
# Inactive identity = no-op
# ---------------------------------------------------------------------------

def test_telemetry_disabled_by_default(fake_mem0, monkeypatch) -> None:
    monkeypatch.delenv("MEM0_TELEMETRY", raising=False)
    _provider()
    assert __import__("os").environ["MEM0_TELEMETRY"] == "False"


async def test_inactive_identity_no_ops(fake_mem0) -> None:
    provider = _provider()
    await provider.record(_identity(active=False), _req("x"), "y")
    ctx = await provider.retrieve(_identity(active=False), _req("x"))
    assert ctx.summary is None
    assert fake_mem0.registry == {}, "no mem0 instance should be built for inactive id"


# ---------------------------------------------------------------------------
# Missing optional dependency
# ---------------------------------------------------------------------------

def test_missing_mem0ai_raises_helpful_error(monkeypatch) -> None:
    from modelmeld.memory import Mem0DependencyError, Mem0MemoryProvider

    # Block the import so `from mem0 import AsyncMemory` fails.
    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(Mem0DependencyError) as exc:
        Mem0MemoryProvider()
    assert "modelmeld[mem0]" in str(exc.value)

"""Live integration check for Mem0MemoryProvider.

Drives the REAL mem0 AsyncMemory (real extraction pipeline + embedded qdrant)
through the provider's record/retrieve cycle, with mem0's LLM + embedder pointed
at an in-process mock OpenAI-compatible server. This is the check the fake-backed
unit tests can't give: that our `filters={"run_id": ...}` actually retrieves what
`add(run_id=...)` stored, and that the search return shape matches our parser.

Skipped by default (needs the mem0 extra + is slower). Run with:
    MODELMELD_RUN_MEM0_INTEGRATION=1 pytest tests/test_mem0_integration.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modelmeld.api.schemas import ChatCompletionRequest, UserMessage
from modelmeld.memory import MemoryIdentity, MemoryMode

pytestmark = pytest.mark.skipif(
    not os.getenv("MODELMELD_RUN_MEM0_INTEGRATION"),
    reason="set MODELMELD_RUN_MEM0_INTEGRATION=1 to run the live mem0 integration check",
)


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace")
        if self.path.endswith("/embeddings"):
            payload = {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.001] * 1536}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        else:  # /chat/completions
            low = body.lower()
            if "event" in low or "update" in low or '"id"' in low:
                content = json.dumps({"memory": [
                    {"id": "0", "text": "user's name is alice", "event": "ADD"},
                ]})
            else:
                content = json.dumps({"facts": ["user's name is alice"]})
            payload = {
                "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
                "model": "gpt-5-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_openai():
    srv = HTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


def _provider(base_url: str, path: str):
    from modelmeld.memory import Mem0MemoryProvider

    return Mem0MemoryProvider(
        infer=True, top_k=10, base_url=base_url, api_key="sk-test",
        vector_store_path=path,
    )


def _identity(tenant="acme", session="s-1") -> MemoryIdentity:
    return MemoryIdentity(
        tenant_id=tenant, session_id=session, user_id="alice", mode=MemoryMode.AUGMENT,
    )


def _req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-opus-4-7", messages=[UserMessage(role="user", content=text)],
    )


async def test_real_mem0_record_then_retrieve(mock_openai, tmp_path) -> None:
    provider = _provider(mock_openai, str(tmp_path / "qdrant"))
    await provider.record(_identity(), _req("Remember: my name is alice."), "Noted.")
    ctx = await provider.retrieve(_identity(), _req("what is my name?"))
    assert ctx.summary is not None, "real mem0 round-trip should retrieve the stored memory"
    assert "alice" in ctx.summary.text.lower()


async def test_real_mem0_tenant_isolation(mock_openai, tmp_path) -> None:
    provider = _provider(mock_openai, str(tmp_path / "qdrant"))
    await provider.record(
        _identity(tenant="tenant-a"), _req("secret: my name is alice"), "ok",
    )
    ctx = await provider.retrieve(_identity(tenant="tenant-b"), _req("what is my name?"))
    assert ctx.summary is None, "tenant B must not retrieve tenant A's memory"

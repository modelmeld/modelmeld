"""VLLMAdapter unit tests + gated integration test."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion as SDKChatCompletion

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.vllm_adapter import VLLMAdapter
from modelmeld.api.schemas import ChatCompletionRequest
from tests.fixtures.openai_responses import SIMPLE_TEXT


def test_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_VLLM_ENDPOINT", raising=False)
    monkeypatch.delenv("VLLM_ENDPOINT", raising=False)
    with pytest.raises(AdapterError, match="endpoint"):
        VLLMAdapter()


def test_accepts_endpoint_from_constructor() -> None:
    adapter = VLLMAdapter(endpoint="http://localhost:8000/v1")
    assert adapter.name == "vllm"


def test_accepts_endpoint_from_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_VLLM_ENDPOINT", "http://localhost:8000/v1")
    adapter = VLLMAdapter()
    assert adapter.name == "vllm"


def test_accepts_endpoint_from_vllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_VLLM_ENDPOINT", raising=False)
    monkeypatch.setenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    adapter = VLLMAdapter()
    assert adapter.name == "vllm"


async def test_chat_uses_inherited_openai_machinery() -> None:
    """VLLMAdapter inherits chat() from OpenAIAdapter — verify wiring still works."""
    adapter = VLLMAdapter(endpoint="http://fake-vllm.test/v1")
    sdk_response = SDKChatCompletion.model_validate(SIMPLE_TEXT)
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=sdk_response
    )

    response = await adapter.chat(
        ChatCompletionRequest(
            model="qwen2.5-coder-7b",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert response.id == SIMPLE_TEXT["id"]


async def test_health_inherited() -> None:
    """VLLM exposes /v1/models like OpenAI; inherited health() targets the right URL."""
    adapter = VLLMAdapter(endpoint="http://fake-vllm.test/v1")
    adapter._client.models.list = AsyncMock(return_value=object())  # type: ignore[method-assign]
    assert await adapter.health() is True


@pytest.mark.skipif(
    not os.environ.get("MODELMELD_VLLM_ENDPOINT"),
    reason="integration test — set MODELMELD_VLLM_ENDPOINT to run "
    "(e.g., from `python scripts/dev_gpu.py up` output)",
)
async def test_round_trip_against_real_vllm() -> None:
    """Gated integration test. Run after `python scripts/dev_gpu.py up`."""
    adapter = VLLMAdapter()
    try:
        response = await adapter.chat(
            ChatCompletionRequest(
                model=os.environ.get("MODELMELD_VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"),
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_completion_tokens=10,
                temperature=0.0,
            )
        )
        assert response.choices[0].message.content
    finally:
        await adapter.close()

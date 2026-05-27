"""TensorRTLLMAdapter — unit tests + vLLM parity + gated integration."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion as SDKChatCompletion

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.tensorrt_llm_adapter import TensorRTLLMAdapter
from modelmeld.adapters.vllm_adapter import VLLMAdapter
from modelmeld.api.schemas import ChatCompletionRequest
from tests.fixtures.openai_responses import SIMPLE_TEXT


# ---------------------------------------------------------------------------
# Endpoint resolution + construction
# ---------------------------------------------------------------------------

def test_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_TENSORRT_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("TENSORRT_LLM_ENDPOINT", raising=False)
    with pytest.raises(AdapterError, match="endpoint"):
        TensorRTLLMAdapter()


def test_accepts_endpoint_from_constructor() -> None:
    adapter = TensorRTLLMAdapter(endpoint="http://triton:8000/v1")
    assert adapter.name == "tensorrt_llm"


def test_accepts_endpoint_from_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_TENSORRT_LLM_ENDPOINT", "http://triton:8000/v1")
    adapter = TensorRTLLMAdapter()
    assert adapter.name == "tensorrt_llm"


def test_accepts_endpoint_from_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODELMELD_TENSORRT_LLM_ENDPOINT", raising=False)
    monkeypatch.setenv("TENSORRT_LLM_ENDPOINT", "http://triton:8000/v1")
    adapter = TensorRTLLMAdapter()
    assert adapter.name == "tensorrt_llm"


def test_gateway_env_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_TENSORRT_LLM_ENDPOINT", "http://winner:8000/v1")
    monkeypatch.setenv("TENSORRT_LLM_ENDPOINT", "http://loser:8000/v1")
    adapter = TensorRTLLMAdapter()
    assert "winner" in str(adapter._client.base_url)


def test_constructor_endpoint_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELMELD_TENSORRT_LLM_ENDPOINT", "http://env:8000/v1")
    adapter = TensorRTLLMAdapter(endpoint="http://explicit:8000/v1")
    assert "explicit" in str(adapter._client.base_url)


def test_is_local_tier_not_egress() -> None:
    """TRT-LLM is customer-owned GPU infrastructure — must not cross trust boundary."""
    adapter = TensorRTLLMAdapter(endpoint="http://triton:8000/v1")
    assert adapter.is_egress is False


def test_empty_api_key_default() -> None:
    """Triton ignores api_key by default; 'EMPTY' is conventional."""
    adapter = TensorRTLLMAdapter(endpoint="http://triton:8000/v1")
    # The underlying OpenAI client stores this on construction
    assert adapter._client.api_key == "EMPTY"


def test_explicit_api_key_passes_through() -> None:
    """Triton deployments with an auth proxy can supply a real key."""
    adapter = TensorRTLLMAdapter(endpoint="http://triton:8000/v1", api_key="bearer-xyz")
    assert adapter._client.api_key == "bearer-xyz"


# ---------------------------------------------------------------------------
# Inherited machinery — chat() + health() still work via OpenAIAdapter
# ---------------------------------------------------------------------------

async def test_chat_uses_inherited_openai_machinery() -> None:
    adapter = TensorRTLLMAdapter(endpoint="http://fake-triton.test/v1")
    sdk_response = SDKChatCompletion.model_validate(SIMPLE_TEXT)
    adapter._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=sdk_response,
    )
    response = await adapter.chat(
        ChatCompletionRequest(
            model="qwen2.5-coder-7b",
            messages=[{"role": "user", "content": "hi"}],
        ),
    )
    assert response.id == SIMPLE_TEXT["id"]


async def test_health_inherited() -> None:
    adapter = TensorRTLLMAdapter(endpoint="http://fake-triton.test/v1")
    adapter._client.models.list = AsyncMock(return_value=object())  # type: ignore[method-assign]
    assert await adapter.health() is True


# ---------------------------------------------------------------------------
# vLLM parity — same request shape produces same SDK call
# ---------------------------------------------------------------------------

async def test_vllm_and_tensorrt_llm_parity_on_identical_request() -> None:
    """Both adapters should hand the SAME SDK params to the upstream OpenAI client
    for an identical request. Routing decisions don't need to know the difference."""
    vllm = VLLMAdapter(endpoint="http://vllm.test/v1")
    trtllm = TensorRTLLMAdapter(endpoint="http://triton.test/v1")

    vllm_capture: dict = {}
    trtllm_capture: dict = {}

    sdk_response = SDKChatCompletion.model_validate(SIMPLE_TEXT)

    async def capture_into_vllm(**kwargs):
        vllm_capture.update(kwargs)
        return sdk_response

    async def capture_into_trtllm(**kwargs):
        trtllm_capture.update(kwargs)
        return sdk_response

    vllm._client.chat.completions.create = capture_into_vllm  # type: ignore[method-assign]
    trtllm._client.chat.completions.create = capture_into_trtllm  # type: ignore[method-assign]

    request = ChatCompletionRequest(
        model="qwen2.5-coder-7b-instruct",
        messages=[{"role": "user", "content": "compute 2+2"}],
        temperature=0.0,
        seed=42,
        max_completion_tokens=64,
    )
    await vllm.chat(request)
    await trtllm.chat(request)

    # Bit-for-bit identical: same model, messages, temperature, seed, max tokens
    assert vllm_capture == trtllm_capture
    assert vllm_capture["model"] == "qwen2.5-coder-7b-instruct"
    assert vllm_capture["temperature"] == 0.0
    assert vllm_capture["seed"] == 42


async def test_parity_preserved_under_streaming_flag() -> None:
    """Same parity claim must hold for stream=True requests."""
    vllm = VLLMAdapter(endpoint="http://vllm.test/v1")
    trtllm = TensorRTLLMAdapter(endpoint="http://triton.test/v1")
    vllm_capture: dict = {}
    trtllm_capture: dict = {}

    async def empty_stream():
        if False:  # pragma: no cover
            yield

    async def capture_v(**kwargs):
        vllm_capture.update(kwargs)
        return empty_stream()

    async def capture_t(**kwargs):
        trtllm_capture.update(kwargs)
        return empty_stream()

    vllm._client.chat.completions.create = capture_v  # type: ignore[method-assign]
    trtllm._client.chat.completions.create = capture_t  # type: ignore[method-assign]

    request = ChatCompletionRequest(
        model="qwen2.5-coder-7b",
        messages=[{"role": "user", "content": "write hello"}],
        stream=True,
        temperature=0.3,
    )
    # Open + close both streams (we don't iterate; just exercising the call)
    a1 = vllm.stream_chat(request).__aiter__()
    a2 = trtllm.stream_chat(request).__aiter__()
    try:
        await a1.__anext__()
    except StopAsyncIteration:
        pass
    try:
        await a2.__anext__()
    except StopAsyncIteration:
        pass

    assert vllm_capture == trtllm_capture
    assert vllm_capture["stream"] is True


# ---------------------------------------------------------------------------
# Factory wiring — build_router resolves "tensorrt_llm"
# ---------------------------------------------------------------------------

def test_factory_builds_tensorrt_llm_adapter_from_settings() -> None:
    from modelmeld.config import GatewaySettings
    from modelmeld.router import _build_adapter

    settings = GatewaySettings(tensorrt_llm_endpoint="http://triton:8000/v1")
    adapter = _build_adapter("tensorrt_llm", settings)
    assert isinstance(adapter, TensorRTLLMAdapter)
    assert adapter.name == "tensorrt_llm"


# ---------------------------------------------------------------------------
# Gated integration test — runs only with a real Triton endpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("MODELMELD_TENSORRT_LLM_ENDPOINT"),
    reason="integration test — set MODELMELD_TENSORRT_LLM_ENDPOINT to run",
)
@pytest.mark.requires_gpu
async def test_round_trip_against_real_triton() -> None:
    adapter = TensorRTLLMAdapter()
    try:
        response = await adapter.chat(
            ChatCompletionRequest(
                model=os.environ.get("MODELMELD_TRTLLM_MODEL", "ensemble"),
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_completion_tokens=10,
                temperature=0.0,
            ),
        )
        assert response.choices[0].message.content
    finally:
        await adapter.close()

"""Verify our responses parse against the official OpenAI Python SDK's Pydantic models."""

from __future__ import annotations

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from openai.types.model import Model as OpenAIModel

from modelmeld.api.server import build_app


def test_chat_completion_response_parses_with_openai_sdk() -> None:
    client = TestClient(build_app())
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    parsed = ChatCompletion.model_validate(response.json())
    assert parsed.object == "chat.completion"
    assert parsed.model == "gpt-4o-mini"
    assert parsed.choices[0].message.role == "assistant"
    assert parsed.choices[0].finish_reason == "stop"


def test_model_list_entries_parse_with_openai_sdk() -> None:
    client = TestClient(build_app())
    response = client.get("/v1/models")
    data = response.json()["data"]
    parsed = [OpenAIModel.model_validate(m) for m in data]
    assert len(parsed) >= 1
    assert all(m.object == "model" for m in parsed)

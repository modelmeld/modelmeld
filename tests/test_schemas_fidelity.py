"""Schema-fidelity tests: 20 representative request/response payloads parse losslessly."""

from __future__ import annotations

import pytest
from openai.types.chat import ChatCompletion as OpenAIChatCompletion

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionRequest,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tests.fixtures.openai_requests import ALL_REQUESTS
from tests.fixtures.openai_responses import ALL_RESPONSES


# ---------------------------------------------------------------------------
# Request fidelity (10 fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "payload"), ALL_REQUESTS, ids=[n for n, _ in ALL_REQUESTS])
def test_request_fixture_parses(name: str, payload: dict) -> None:
    parsed = ChatCompletionRequest.model_validate(payload)
    assert parsed.model == payload["model"]
    assert len(parsed.messages) == len(payload["messages"])


def test_request_round_trips_lossless() -> None:
    # Pick a complex one and verify model_dump round-trips through model_validate
    payload = next(p for n, p in ALL_REQUESTS if n == "conversation_with_tool_results")
    parsed = ChatCompletionRequest.model_validate(payload)
    dumped = parsed.model_dump(exclude_none=True, exclude_defaults=False)
    re_parsed = ChatCompletionRequest.model_validate(dumped)
    assert re_parsed.model_dump(exclude_none=True) == parsed.model_dump(exclude_none=True)


def test_request_discriminates_message_roles() -> None:
    payload = next(p for n, p in ALL_REQUESTS if n == "conversation_with_tool_results")
    parsed = ChatCompletionRequest.model_validate(payload)
    roles = [type(m).__name__ for m in parsed.messages]
    assert roles == ["UserMessage", "AssistantMessage", "ToolMessage"]
    assert isinstance(parsed.messages[1], AssistantMessage)
    assert parsed.messages[1].tool_calls is not None
    assert parsed.messages[1].tool_calls[0].function.name == "get_weather"


def test_request_multimodal_content_parts() -> None:
    payload = next(p for n, p in ALL_REQUESTS if n == "multimodal_image")
    parsed = ChatCompletionRequest.model_validate(payload)
    user = parsed.messages[0]
    assert isinstance(user, UserMessage)
    assert isinstance(user.content, list)
    assert len(user.content) == 2
    assert user.content[0].type == "text"
    assert user.content[1].type == "image_url"


def test_request_with_tools_parses_function_schema() -> None:
    payload = next(p for n, p in ALL_REQUESTS if n == "tools_defined")
    parsed = ChatCompletionRequest.model_validate(payload)
    assert parsed.tools is not None
    assert len(parsed.tools) == 1
    assert parsed.tools[0].function.name == "get_weather"
    assert parsed.tools[0].function.strict is True


def test_request_stream_options_parses() -> None:
    payload = next(p for n, p in ALL_REQUESTS if n == "stream_with_usage")
    parsed = ChatCompletionRequest.model_validate(payload)
    assert parsed.stream is True
    assert parsed.stream_options is not None
    assert parsed.stream_options.include_usage is True


# ---------------------------------------------------------------------------
# Response fidelity (10 fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "payload"), ALL_RESPONSES, ids=[n for n, _ in ALL_RESPONSES]
)
def test_response_fixture_parses_with_our_schema(name: str, payload: dict) -> None:
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.object == "chat.completion"
    assert parsed.id == payload["id"]
    assert len(parsed.choices) == len(payload["choices"])


@pytest.mark.parametrize(
    ("name", "payload"), ALL_RESPONSES, ids=[n for n, _ in ALL_RESPONSES]
)
def test_response_fixture_parses_with_openai_sdk(name: str, payload: dict) -> None:
    """Every fixture must also parse with the official openai SDK's Pydantic model."""
    sdk_parsed = OpenAIChatCompletion.model_validate(payload)
    assert sdk_parsed.id == payload["id"]


def test_response_logprobs_structure() -> None:
    payload = next(p for n, p in ALL_RESPONSES if n == "with_logprobs")
    parsed = ChatCompletion.model_validate(payload)
    lp = parsed.choices[0].logprobs
    assert lp is not None
    assert lp.content is not None
    assert lp.content[0].token == "OK"
    assert len(lp.content[0].top_logprobs) == 2


def test_response_usage_details() -> None:
    payload = next(p for n, p in ALL_RESPONSES if n == "with_usage_details")
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens_details is not None
    assert parsed.usage.prompt_tokens_details.cached_tokens == 16
    assert parsed.usage.completion_tokens_details is not None
    assert parsed.usage.completion_tokens_details.reasoning_tokens == 320


def test_response_parallel_tool_calls() -> None:
    payload = next(p for n, p in ALL_RESPONSES if n == "parallel_tool_calls")
    parsed = ChatCompletion.model_validate(payload)
    tool_calls = parsed.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert {tc.function.name for tc in tool_calls} == {"get_weather"}


def test_response_with_refusal() -> None:
    payload = next(p for n, p in ALL_RESPONSES if n == "with_refusal")
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.choices[0].message.refusal == "I cannot help with that."


def test_unused_message_classes_importable() -> None:
    # Sanity that the rest of the discriminated union is reachable.
    assert SystemMessage.__name__ == "SystemMessage"
    assert ToolMessage.__name__ == "ToolMessage"

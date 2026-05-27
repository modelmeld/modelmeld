"""RegexScrubber pattern + structure tests."""

from __future__ import annotations

import pytest

from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    SystemMessage,
    TextPart,
    UserMessage,
)
from modelmeld.privacy import RegexScrubber


@pytest.fixture
def scrubber() -> RegexScrubber:
    return RegexScrubber()


# ---------------------------------------------------------------------------
# Pattern-level smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("plaintext", "expected_label"),
    [
        ("Contact me at alice@example.com please", "EMAIL"),
        ("My SSN is 123-45-6789, don't share", "SSN"),
        ("Card 4111-1111-1111-1111 expired", "CREDIT_CARD"),
        ("Card 4111 1111 1111 1111 also matches", "CREDIT_CARD"),
        ("My AWS key is AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
        ("API key: sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaa", "ANTHROPIC_API_KEY"),
        # OpenAI keys are at least 40 chars after sk-
        ("sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa is mine", "OPENAI_API_KEY"),
        ("Token: ghp_abcdefghijklmnopqrstuvwxyz0123456789", "GITHUB_PAT"),
        ("Call me at (415) 555-1234", "PHONE_US"),
        ("Call me at 415-555-1234", "PHONE_US"),
    ],
)
def test_pattern_redacts(plaintext: str, expected_label: str, scrubber: RegexScrubber) -> None:
    scrubbed = scrubber.scrub_text(plaintext)
    assert f"<REDACTED:{expected_label}>" in scrubbed


def test_no_false_positives_on_normal_text(scrubber: RegexScrubber) -> None:
    benign = (
        "Just a normal sentence with some numbers like 42, "
        "a date 2026-05-17, and a UUID 550e8400-e29b-41d4-a716-446655440000."
    )
    assert scrubber.scrub_text(benign) == benign


def test_idempotent(scrubber: RegexScrubber) -> None:
    text = "Email: alice@example.com"
    once = scrubber.scrub_text(text)
    twice = scrubber.scrub_text(once)
    assert once == twice


def test_anthropic_pattern_does_not_double_match_as_openai(scrubber: RegexScrubber) -> None:
    """ANTHROPIC_API_KEY must be tried first; otherwise the sk- prefix matches OPENAI too."""
    text = "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    scrubbed = scrubber.scrub_text(text)
    assert "<REDACTED:ANTHROPIC_API_KEY>" in scrubbed
    assert "<REDACTED:OPENAI_API_KEY>" not in scrubbed


# ---------------------------------------------------------------------------
# scrub_request — structure preservation
# ---------------------------------------------------------------------------

def test_scrubs_string_user_content(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[
            UserMessage(role="user", content="My email is alice@example.com"),
        ],
    )
    scrubbed, redactions = scrubber.scrub_request(request)
    assert isinstance(scrubbed.messages[0], UserMessage)
    assert "<REDACTED:EMAIL>" in scrubbed.messages[0].content  # type: ignore[operator]
    assert any(r.label == "EMAIL" and r.count == 1 for r in redactions)


def test_scrubs_text_part_in_multimodal_user_content(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My SSN is 123-45-6789"},
                        {"type": "image_url", "image_url": {"url": "https://e.com/x.png"}},
                    ],
                }
            ],
        }
    )
    scrubbed, _ = scrubber.scrub_request(request)
    parts = scrubbed.messages[0].content
    assert isinstance(parts, list)
    assert "<REDACTED:SSN>" in parts[0].text  # type: ignore[union-attr]
    # Image part untouched
    assert parts[1].type == "image_url"  # type: ignore[union-attr]


def test_scrubs_system_message(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[
            SystemMessage(role="system", content="Operator email: ops@example.com"),
            UserMessage(role="user", content="ok"),
        ],
    )
    scrubbed, redactions = scrubber.scrub_request(request)
    assert "<REDACTED:EMAIL>" in scrubbed.messages[0].content  # type: ignore[operator]
    assert any(r.label == "EMAIL" for r in redactions)


def test_scrubs_tool_call_arguments(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "do the thing"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": '{"to": "alice@example.com"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Email sent",
                    "tool_call_id": "call_1",
                },
            ],
        }
    )
    scrubbed, redactions = scrubber.scrub_request(request)
    assistant = scrubbed.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls is not None
    assert "<REDACTED:EMAIL>" in assistant.tool_calls[0].function.arguments
    assert any(r.label == "EMAIL" for r in redactions)


def test_preserves_request_metadata(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[UserMessage(role="user", content="alice@example.com")],
        temperature=0.7,
        max_completion_tokens=100,
        seed=42,
    )
    scrubbed, _ = scrubber.scrub_request(request)
    assert scrubbed.model == "gpt-4o-mini"
    assert scrubbed.temperature == 0.7
    assert scrubbed.max_completion_tokens == 100
    assert scrubbed.seed == 42


def test_empty_redactions_when_clean(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[UserMessage(role="user", content="Hello world")],
    )
    scrubbed, redactions = scrubber.scrub_request(request)
    assert redactions == []
    assert scrubbed.messages[0].content == "Hello world"  # type: ignore[union-attr]


def test_counts_multiple_redactions_of_same_type(scrubber: RegexScrubber) -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[
            UserMessage(
                role="user",
                content="Emails: a@x.com, b@y.com, c@z.com",
            )
        ],
    )
    _, redactions = scrubber.scrub_request(request)
    email_redaction = next(r for r in redactions if r.label == "EMAIL")
    assert email_redaction.count == 3

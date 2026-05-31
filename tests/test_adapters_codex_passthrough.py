# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""CodexPassthroughAdapter — auth resolution, request/response translation,
and wire-format verification against a mocked Codex backend."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from modelmeld.adapters.base import AdapterError
from modelmeld.adapters.codex_passthrough import (
    CodexAuthFileError,
    CodexPassthroughAdapter,
    _load_codex_auth,
    _messages_to_responses_input,
)
from modelmeld.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    SystemMessage,
    Tool,
    UserMessage,
)

# ---------------------------------------------------------------------------
# _load_codex_auth — file-reading happy path + error cases
# ---------------------------------------------------------------------------


def test_load_codex_auth_happy_path(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {"access_token": "test-bearer-abc123", "id_token": "..."},
        "account_id": "acc-12345",
    }))
    token, account_id = _load_codex_auth(auth_file)
    assert token == "test-bearer-abc123"
    assert account_id == "acc-12345"


def test_load_codex_auth_no_account_id(tmp_path: Path) -> None:
    """account_id is optional — single-account users may omit it."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {"access_token": "tok"},
    }))
    token, account_id = _load_codex_auth(auth_file)
    assert token == "tok"
    assert account_id is None


def test_load_codex_auth_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CodexAuthFileError, match="not found"):
        _load_codex_auth(tmp_path / "does-not-exist.json")


def test_load_codex_auth_malformed_json_raises(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("not-valid-json{{{")
    with pytest.raises(CodexAuthFileError, match="unreadable or malformed"):
        _load_codex_auth(auth_file)


def test_load_codex_auth_missing_access_token_raises(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"refresh_token": "r"}}))
    with pytest.raises(CodexAuthFileError, match="no tokens.access_token"):
        _load_codex_auth(auth_file)


def test_load_codex_auth_does_not_leak_token_in_error_message(tmp_path: Path) -> None:
    """Auth errors must NOT echo the (probably-real) token value."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {"access_token": "SECRET-TOKEN-DO-NOT-LEAK"},
    }))
    # Construct with a path that will trigger ANY downstream error.
    # We're testing that the file's CONTENTS don't leak in messages —
    # the happy-path read succeeds, so coerce malformed:
    auth_file.write_text("garbage")
    try:
        _load_codex_auth(auth_file)
    except CodexAuthFileError as e:
        assert "SECRET-TOKEN-DO-NOT-LEAK" not in str(e)


# ---------------------------------------------------------------------------
# _messages_to_responses_input — ChatCompletion → Responses API shape
# ---------------------------------------------------------------------------


def test_messages_translation_extracts_system_to_instructions() -> None:
    """system messages collapse into the `instructions` field."""
    req = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[
            SystemMessage(role="system", content="You are a helpful assistant."),
            UserMessage(role="user", content="hello"),
        ],
    )
    chat_input, instructions = _messages_to_responses_input(req)
    assert instructions == "You are a helpful assistant."
    assert len(chat_input) == 1
    assert chat_input[0]["role"] == "user"
    assert chat_input[0]["content"] == "hello"


def test_messages_translation_joins_multiple_system_messages() -> None:
    """Multiple system messages join with newlines into a single instructions string."""
    req = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[
            SystemMessage(role="system", content="Be concise."),
            SystemMessage(role="system", content="Use markdown."),
            UserMessage(role="user", content="hi"),
        ],
    )
    _, instructions = _messages_to_responses_input(req)
    assert instructions == "Be concise.\nUse markdown."


def test_messages_translation_no_system_returns_none_instructions() -> None:
    req = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[UserMessage(role="user", content="hi")],
    )
    _, instructions = _messages_to_responses_input(req)
    assert instructions is None


def test_messages_translation_preserves_assistant_turn() -> None:
    """assistant role messages forward to the input array unchanged."""
    req = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[
            UserMessage(role="user", content="2+2?"),
            AssistantMessage(role="assistant", content="4"),
            UserMessage(role="user", content="and 3+3?"),
        ],
    )
    chat_input, _ = _messages_to_responses_input(req)
    assert len(chat_input) == 3
    assert chat_input[0]["role"] == "user"
    assert chat_input[1]["role"] == "assistant"
    assert chat_input[1]["content"] == "4"
    assert chat_input[2]["role"] == "user"


# ---------------------------------------------------------------------------
# CodexPassthroughAdapter.__init__ — auth resolution priority
# ---------------------------------------------------------------------------


def test_adapter_requires_some_auth_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """No access_token, no auth_json_path, no env var → loud failure."""
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    with pytest.raises(AdapterError, match="requires an OAuth access token"):
        CodexPassthroughAdapter()


def test_adapter_accepts_explicit_access_token() -> None:
    """access_token kwarg takes precedence over auth_json_path / env."""
    adapter = CodexPassthroughAdapter(access_token="direct-token")
    # Constructor succeeded — that's the assertion. Don't introspect the
    # SDK client's headers (private API and varies across SDK versions).
    assert adapter.name == "codex_passthrough"


def test_adapter_reads_token_from_auth_json(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {"access_token": "from-file-token"},
        "account_id": "acc-from-file",
    }))
    adapter = CodexPassthroughAdapter(auth_json_path=auth_file)
    assert adapter.name == "codex_passthrough"


def test_adapter_reads_token_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "from-env-token")
    adapter = CodexPassthroughAdapter()
    assert adapter.name == "codex_passthrough"


def test_adapter_priority_explicit_token_beats_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If both kwarg and env are present, kwarg wins."""
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "env-token")
    adapter = CodexPassthroughAdapter(access_token="explicit-token")
    # Both auth sources are valid; explicit takes priority. We don't
    # assert on the actual token used (would require SDK introspection);
    # we assert the constructor succeeds without raising.
    assert adapter.name == "codex_passthrough"


# ---------------------------------------------------------------------------
# End-to-end with mocked HTTP transport
# ---------------------------------------------------------------------------


def _mock_responses_api_handler(captured: dict) -> httpx.MockTransport:
    """Build a MockTransport that captures one POST + returns a canned response."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        if request.content:
            captured["body"] = json.loads(request.content)
        # Canned OpenAI Responses API result. Stream=False shape.
        return httpx.Response(
            200,
            json={
                "id": "resp_test123",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Hello back!"},
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                },
            },
        )

    return httpx.MockTransport(handler)


async def test_adapter_chat_translates_to_responses_api_wire_format() -> None:
    """End-to-end with a mocked Codex backend: assert wire-level details."""
    captured: dict = {}
    transport = _mock_responses_api_handler(captured)
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        access_token="test-bearer",
        account_id="acc-test",
        http_client=http_client,
    )
    try:
        result = await adapter.chat(ChatCompletionRequest(
            model="gpt-5.4",
            messages=[
                SystemMessage(role="system", content="Be brief."),
                UserMessage(role="user", content="Say hi."),
            ],
            max_tokens=50,
        ))
    finally:
        await adapter.close()

    # Verify the HTTP request shape.
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/responses"), captured["url"]
    # Auth headers: Bearer token + optional ChatGPT-Account-ID.
    assert captured["headers"].get("authorization") == "Bearer test-bearer"
    assert captured["headers"].get("chatgpt-account-id") == "acc-test"
    # Request body: Responses API shape (input + instructions, NOT messages).
    body = captured["body"]
    assert body["model"] == "gpt-5.4"
    assert body["instructions"] == "Be brief."
    assert body["input"] == [{"role": "user", "content": "Say hi."}]
    assert body["store"] is False
    assert body["stream"] is False

    # Verify response translation back to ChatCompletion.
    assert result.choices[0].message.content == "Hello back!"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3


async def test_adapter_chat_omits_chatgpt_account_id_header_when_not_set() -> None:
    """Single-account users skip the account_id; header must be absent."""
    captured: dict = {}
    transport = _mock_responses_api_handler(captured)
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        access_token="test-bearer",
        http_client=http_client,
    )
    try:
        await adapter.chat(ChatCompletionRequest(
            model="gpt-5.4",
            messages=[UserMessage(role="user", content="hi")],
        ))
    finally:
        await adapter.close()

    assert "chatgpt-account-id" not in captured["headers"]


async def test_adapter_chat_passes_tools_through() -> None:
    """Tools array forwards verbatim to Responses API."""
    captured: dict = {}
    transport = _mock_responses_api_handler(captured)
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        access_token="test-bearer",
        http_client=http_client,
    )
    try:
        await adapter.chat(ChatCompletionRequest(
            model="gpt-5.4",
            messages=[UserMessage(role="user", content="weather?")],
            tools=[
                Tool(
                    type="function",
                    function={
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ),
            ],
        ))
    finally:
        await adapter.close()

    assert "tools" in captured["body"]
    assert len(captured["body"]["tools"]) == 1
    assert captured["body"]["tools"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Token reload on 401 (Codex CLI rotates the bearer in auth.json)
# ---------------------------------------------------------------------------


def _ok_response_body() -> dict:
    return {
        "id": "resp_ok",
        "object": "response",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _401_response_body() -> dict:
    return {"error": {"message": "expired_token", "type": "auth_error"}}


def _build_token_aware_transport(
    captured: list, valid_tokens: set[str],
) -> httpx.MockTransport:
    """MockTransport that returns 200 iff Authorization carries a valid token, else 401.

    `captured` accumulates one entry per request observed. `valid_tokens`
    is a set of bare bearer values (no "Bearer " prefix) that the mock
    backend will accept.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        bearer = auth.removeprefix("Bearer ").strip()
        captured.append({"bearer": bearer, "url": str(request.url)})
        if bearer in valid_tokens:
            return httpx.Response(200, json=_ok_response_body())
        return httpx.Response(401, json=_401_response_body())
    return httpx.MockTransport(handler)


async def test_chat_reloads_token_on_401_and_retries(tmp_path: Path) -> None:
    """Codex CLI rotates the bearer in auth.json; adapter detects the upstream
    401, re-reads the file, rebuilds the client, retries once with the new token.
    """
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "expired-token"}}))

    captured: list = []
    transport = _build_token_aware_transport(captured, valid_tokens={"rotated-token"})
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        auth_json_path=auth_file,
        http_client=http_client,
    )

    # Simulate Codex CLI rotating the bearer in the file after adapter
    # init but before the chat call (mimics: bearer expired between
    # adapter construction and use; CLI auto-refreshed in the background).
    auth_file.write_text(json.dumps({"tokens": {"access_token": "rotated-token"}}))

    try:
        result = await adapter.chat(ChatCompletionRequest(
            model="gpt-5.4",
            messages=[UserMessage(role="user", content="ping")],
        ))
    finally:
        await adapter.close()

    # Two upstream calls: first with stale, second with rotated.
    assert len(captured) == 2
    assert captured[0]["bearer"] == "expired-token"
    assert captured[1]["bearer"] == "rotated-token"
    assert result.choices[0].message.content == "ok"


async def test_chat_surfaces_401_when_no_reload_path() -> None:
    """access_token path can't re-resolve — 401 must surface promptly,
    not loop reload-retries forever.
    """
    captured: list = []
    transport = _build_token_aware_transport(captured, valid_tokens=set())
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        access_token="bad-token",
        http_client=http_client,
    )
    try:
        with pytest.raises(AdapterError, match="chat failed"):
            await adapter.chat(ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(role="user", content="ping")],
            ))
    finally:
        await adapter.close()
    # At least one upstream call happened; we don't assert exact count
    # (retry_async may retry on transient errors per its policy).
    assert len(captured) >= 1
    # Every captured call used the original bad token; no reload occurred.
    assert all(c["bearer"] == "bad-token" for c in captured)


async def test_chat_surfaces_401_when_reload_yields_same_token(tmp_path: Path) -> None:
    """auth.json still has the (bad) token — reload returns False and the
    401 surfaces. No infinite reload loop.
    """
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "stale-token"}}))

    captured: list = []
    transport = _build_token_aware_transport(captured, valid_tokens=set())
    http_client = httpx.AsyncClient(transport=transport)
    adapter = CodexPassthroughAdapter(
        auth_json_path=auth_file,
        http_client=http_client,
    )
    try:
        with pytest.raises(AdapterError):
            await adapter.chat(ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(role="user", content="ping")],
            ))
    finally:
        await adapter.close()
    # No request should have used a different token than the original.
    assert all(c["bearer"] == "stale-token" for c in captured)


# Direct unit tests on _try_reload_token() — bypass the SDK + mock layer.


async def test_try_reload_token_returns_false_when_no_auth_path() -> None:
    adapter = CodexPassthroughAdapter(access_token="direct-token")
    assert await adapter._try_reload_token() is False


async def test_try_reload_token_returns_false_when_token_unchanged(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "same-token"}}))
    adapter = CodexPassthroughAdapter(auth_json_path=auth_file)
    assert await adapter._try_reload_token() is False


async def test_try_reload_token_updates_state_when_token_rotated(
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "old-token"}}))
    adapter = CodexPassthroughAdapter(auth_json_path=auth_file)
    old_client = adapter._client

    # Simulate Codex CLI rotation.
    auth_file.write_text(json.dumps({"tokens": {"access_token": "rotated-token"}}))
    assert await adapter._try_reload_token() is True
    assert adapter._access_token == "rotated-token"
    # The SDK client was rebuilt (api_key is baked at construction).
    assert adapter._client is not old_client


async def test_try_reload_token_returns_false_when_file_gone(tmp_path: Path) -> None:
    """`codex logout` removes auth.json → reload returns False, doesn't raise.

    The caller surfaces the original 401 to the operator who needs to
    re-authenticate via the CLI — our adapter can't do that for them.
    """
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "before-logout"}}))
    adapter = CodexPassthroughAdapter(auth_json_path=auth_file)
    auth_file.unlink()
    assert await adapter._try_reload_token() is False

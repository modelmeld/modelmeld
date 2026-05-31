# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""CodexPassthroughAdapter — forward requests to OpenAI's Codex backend
using a ChatGPT subscription OAuth bearer token.

The Codex CLI (https://github.com/openai/codex) caches an OAuth bearer at
`~/.codex/auth.json` and reaches a subscription-authenticated endpoint at
`https://chatgpt.com/backend-api/codex/responses` (NOT `api.openai.com`).
This adapter wraps the official `openai` SDK with that base URL + the two
headers Codex CLI sends. The endpoint speaks the OpenAI Responses API
shape; this adapter translates our internal ChatCompletion requests into
Responses API calls and translates the result back.

ToS posture: self-host only, single-user-per-instance, no multi-tenant
pooling. OpenAI's account-sharing policy still prohibits sharing
subscription credentials across users — this adapter is opt-in via
`MODELMELD_ALLOW_SUBSCRIPTION_PASSTHROUGH=1` at the gateway layer.
Endpoint is undocumented and can change without notice. See
`docs/subscription-passthrough-codex-feasibility.md` and
`docs/subscription-passthrough-codex-wire-format.md`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from modelmeld.adapters.base import AdapterError, ProviderAdapter
from modelmeld.adapters.retry import (
    RetryConfig,
    retry_async,
    wrap_as_adapter_error,
)
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    ResponseMessage,
    Usage,
)

# Default endpoint for the Codex CLI backend. NOT api.openai.com — this
# is the subscription-authenticated surface that ChatGPT Plus/Pro/Business
# users reach.
_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Default fallback model when the operator doesn't pin one. The Codex
# backend exposes its own model lineup separate from api.openai.com.
_DEFAULT_SERVED_MODEL = "gpt-5.4"


class CodexAuthFileError(AdapterError):
    """Raised when ~/.codex/auth.json is missing, unreadable, or malformed."""


def _load_codex_auth(path: str | Path) -> tuple[str, str | None]:
    """Read access_token + optional account_id from a Codex auth.json file.

    Expected shape (per simonw/llm-openai-via-codex):
        {"tokens": {"access_token": "...", ...}, "account_id": "..."}

    Returns (access_token, account_id). Raises CodexAuthFileError on any
    missing/malformed-file condition; failure messages must NOT echo the
    token value.
    """
    auth_path = Path(path).expanduser()
    if not auth_path.exists():
        raise CodexAuthFileError(
            f"Codex auth file not found at {auth_path}. "
            "Run `codex login` to create one, or pass access_token directly."
        )
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CodexAuthFileError(
            f"Codex auth file at {auth_path} is unreadable or malformed: "
            f"{type(e).__name__}"
        ) from e
    tokens = data.get("tokens") if isinstance(data, dict) else None
    access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
    if not access_token:
        raise CodexAuthFileError(
            f"Codex auth file at {auth_path} has no tokens.access_token field."
        )
    account_id = data.get("account_id") if isinstance(data, dict) else None
    return access_token, account_id if isinstance(account_id, str) else None


def _messages_to_responses_input(
    request: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate ChatCompletion-shape messages into the Responses API
    `input` + `instructions` shape the Codex backend expects.

    - `system` role messages collapse into a single `instructions` string
      (joined with newlines if there are multiple).
    - Remaining messages become role+content entries in the `input` list.
    - Multimodal content lists pass through unchanged (Responses API
      accepts the same content-part shape).
    """
    system_parts: list[str] = []
    chat_input: list[dict[str, Any]] = []
    for msg in request.messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if text:
                        system_parts.append(text)
            continue
        # Forward user/assistant/tool messages as-is.
        item: dict[str, Any] = {"role": role}
        if isinstance(content, str):
            item["content"] = content
        elif content is not None:
            # multimodal list or None — pass through; the SDK serializes it
            item["content"] = (
                [p.model_dump(exclude_none=True) for p in content]
                if isinstance(content, list)
                else content
            )
        chat_input.append(item)
    instructions = "\n".join(system_parts) if system_parts else None
    return chat_input, instructions


def _response_to_chat_completion(
    sdk_response: Any, requested_model: str,
) -> ChatCompletion:
    """Translate an OpenAI Responses API result back into our ChatCompletion shape."""
    # Walk response.output to find the assistant's text content.
    output_text_parts: list[str] = []
    finish_reason = "stop"
    output = getattr(sdk_response, "output", None) or []
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            content_parts = getattr(item, "content", None) or []
            for part in content_parts:
                text = getattr(part, "text", None)
                if text:
                    output_text_parts.append(text)
        # Tool-call output items would land here too in a richer impl;
        # MVP focuses on text. Tool-use is a Sprint 5.5b follow-up.
        if item_type == "tool_call":
            finish_reason = "tool_calls"
    text = "".join(output_text_parts)

    usage_obj = getattr(sdk_response, "usage", None)
    usage = Usage(
        prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
        total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
    )

    return ChatCompletion(
        id=getattr(sdk_response, "id", "") or "",
        model=requested_model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


class CodexPassthroughAdapter(ProviderAdapter):
    """Forward chat requests to OpenAI's Codex backend via ChatGPT-subscription OAuth.

    Three auth paths (in priority order):
    1. `access_token` passed directly to the constructor (BYOK header path)
    2. `auth_json_path` pointing at a Codex CLI credential file
    3. `CODEX_ACCESS_TOKEN` env var

    `account_id` (optional) is sent in the `ChatGPT-Account-ID` header
    when present. Codex CLI sets this for accounts in multiple
    Workspaces; single-account users typically don't need it.

    Token rotation: when constructed with `auth_json_path`, the adapter
    detects 401 responses from the upstream call, re-reads auth.json (the
    Codex CLI rotates the OAuth bearer in that file when it expires),
    and retries the request once with the refreshed token. The
    explicit-`access_token` and `CODEX_ACCESS_TOKEN` env-var paths have no
    on-disk file to re-read, so a 401 there surfaces to the caller and
    the operator must rebuild the adapter (or run `codex login` and
    restart the gateway).
    """

    name = "codex_passthrough"
    is_egress = True

    def __init__(
        self,
        access_token: str | None = None,
        account_id: str | None = None,
        auth_json_path: str | Path | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        retry_config: RetryConfig | None = None,
        served_model: str | None = _DEFAULT_SERVED_MODEL,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: F401  # import-time check only
        except ImportError as e:
            raise AdapterError(
                "CodexPassthroughAdapter requires the `openai` package. "
                "Install with: pip install 'modelmeld[openai]'"
            ) from e

        # Resolve auth in priority order; bail loudly if all paths fail.
        token: str | None = access_token
        resolved_account_id: str | None = account_id
        if token is None:
            if auth_json_path is not None:
                token, file_account_id = _load_codex_auth(auth_json_path)
                if resolved_account_id is None:
                    resolved_account_id = file_account_id
            else:
                token = os.environ.get("CODEX_ACCESS_TOKEN")
        if not token:
            raise AdapterError(
                "CodexPassthroughAdapter requires an OAuth access token. "
                "Provide one of: access_token=, auth_json_path=, "
                "or set CODEX_ACCESS_TOKEN env var."
            )

        # Stash everything needed to rebuild the SDK client after a
        # token reload. Path-based auth keeps reload eligibility; the
        # other two sources can't re-resolve so `_auth_json_path = None`
        # disables reload-on-401 cleanly.
        self._access_token: str = token
        self._account_id: str | None = resolved_account_id
        self._auth_json_path: Path | None = (
            Path(auth_json_path).expanduser() if auth_json_path else None
        )
        self._base_url = base_url
        self._http_client_arg = http_client
        self._retry_config = retry_config or RetryConfig()
        self.served_model = served_model

        self._client = self._build_async_openai_client()

    def _build_async_openai_client(self) -> Any:
        """Rebuild the AsyncOpenAI client from current `_access_token` + `_account_id`.

        Called once at __init__ and again after each successful token
        reload. The SDK bakes `api_key` in at construction so reload
        requires a fresh client instance, not just an attribute swap.
        """
        from openai import AsyncOpenAI
        default_headers: dict[str, str] = {}
        if self._account_id:
            default_headers["ChatGPT-Account-ID"] = self._account_id
        return AsyncOpenAI(
            api_key=self._access_token,
            base_url=self._base_url,
            http_client=self._http_client_arg,
            default_headers=default_headers or None,
            max_retries=0,
        )

    async def _try_reload_token(self) -> bool:
        """Re-read auth.json after a 401. Return True iff the token changed.

        Only meaningful when the adapter was constructed with
        `auth_json_path` — explicit-token and env-var sources have no
        on-disk file to re-read. Returns False (without raising) on
        missing or corrupted file so the caller surfaces the original
        401 to the operator (who needs to re-authenticate via the Codex
        CLI; our adapter has no way to do that).
        """
        if self._auth_json_path is None:
            return False
        try:
            new_token, new_account_id = _load_codex_auth(self._auth_json_path)
        except CodexAuthFileError:
            return False
        if new_token == self._access_token:
            return False  # CLI hasn't rotated yet
        self._access_token = new_token
        if new_account_id:
            # Don't clobber an explicit constructor account_id with None
            # if the rotated file omitted it.
            self._account_id = new_account_id
        self._client = self._build_async_openai_client()
        return True

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        request = self._apply_served_model(request)
        chat_input, instructions = _messages_to_responses_input(request)
        # `store=False` keeps the Codex backend from persisting the
        # exchange against the user's ChatGPT history — required.
        # `stream=False` is the non-streaming variant.
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": chat_input,
            "store": False,
            "stream": False,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if request.tools:
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]

        async def _call():
            from openai import AuthenticationError
            try:
                return await self._client.responses.create(**kwargs)
            except AuthenticationError:
                # Codex CLI may have rotated the bearer in auth.json
                # since we last loaded it. One-shot reload + retry. If
                # the reload didn't change anything (no file source, or
                # same token still on disk), surface the original 401.
                if not await self._try_reload_token():
                    raise
                return await self._client.responses.create(**kwargs)

        try:
            sdk_response = await retry_async(
                _call, self._retry_config, label="codex.responses",
            )
        except Exception as e:
            raise wrap_as_adapter_error(e, "Codex passthrough chat failed") from e
        return _response_to_chat_completion(sdk_response, request.model)

    async def stream_chat(
        self, request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        request = self._apply_served_model(request)
        chat_input, instructions = _messages_to_responses_input(request)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": chat_input,
            "store": False,
            "stream": True,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if request.tools:
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]

        async def _open_stream():
            from openai import AuthenticationError
            try:
                return await self._client.responses.create(**kwargs)
            except AuthenticationError:
                # See chat() for the rationale on this reload-then-retry
                # pattern. 401 can only happen at stream-open time (the
                # SDK authenticates before yielding the iterator), so
                # this catch covers all auth-failure stream paths.
                if not await self._try_reload_token():
                    raise
                return await self._client.responses.create(**kwargs)

        try:
            stream = await retry_async(
                _open_stream, self._retry_config, label="codex.responses.stream",
            )
        except Exception as e:
            raise wrap_as_adapter_error(
                e, "Codex passthrough stream_chat failed",
            ) from e

        # Codex backend emits standard OpenAI Responses SSE event types:
        #   response.output_text.delta — incremental text chunks
        #   response.output_item.done  — tool-call boundaries
        #   response.completed         — terminal event with usage stats
        # MVP: translate text deltas into ChatCompletionChunks. Tool-call
        # streaming is a Sprint 5.5b follow-up.
        chunk_id = ""
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta_text = getattr(event, "delta", None) or ""
                if delta_text:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=ChoiceDelta(role="assistant", content=delta_text),
                                finish_reason=None,
                            )
                        ],
                    )
            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                chunk_id = getattr(response, "id", "") if response else chunk_id
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(),
                            finish_reason="stop",
                        )
                    ],
                )

    async def health(self) -> bool:
        """Probe the Codex backend's /models endpoint.

        A 200 confirms the token is still valid AND the endpoint is up.
        Any error (auth expired, endpoint changed, network) → False.
        """
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

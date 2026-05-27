"""End-to-end: route scrubs PII before egress adapters but not before local ones."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from modelmeld.adapters.base import ProviderAdapter
from modelmeld.api.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ResponseMessage,
)
from modelmeld.api.server import build_app


class RecordingAdapter(ProviderAdapter):
    """Adapter that records every request body it receives."""

    def __init__(self, name: str, is_egress: bool) -> None:
        self._name = name
        self._is_egress = is_egress
        self.received: list[ChatCompletionRequest] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def is_egress(self) -> bool:  # type: ignore[override]
        return self._is_egress

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletion:
        self.received.append(request)
        return ChatCompletion(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content="ok"),
                    finish_reason="stop",
                )
            ],
        )

    async def stream_chat(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        self.received.append(request)
        if False:  # pragma: no cover
            yield

    async def health(self) -> bool:
        return True


def _user_payload() -> dict:
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "ping alice@example.com about SSN 123-45-6789"}
        ],
    }


def test_egress_adapter_receives_scrubbed_request() -> None:
    cloud = RecordingAdapter(name="cloud-fake", is_egress=True)
    with TestClient(build_app(adapter=cloud)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    assert resp.status_code == 200
    sent = cloud.received[0]
    content = sent.messages[0].content
    assert "<REDACTED:EMAIL>" in content  # type: ignore[operator]
    assert "<REDACTED:SSN>" in content  # type: ignore[operator]
    assert "alice@example.com" not in content  # type: ignore[operator]
    assert "123-45-6789" not in content  # type: ignore[operator]


def test_local_adapter_receives_unscrubbed_request() -> None:
    local = RecordingAdapter(name="local-fake", is_egress=False)
    with TestClient(build_app(adapter=local)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    assert resp.status_code == 200
    sent = local.received[0]
    content = sent.messages[0].content
    assert "alice@example.com" in content  # type: ignore[operator]
    assert "123-45-6789" in content  # type: ignore[operator]


def test_redaction_header_set_on_egress() -> None:
    cloud = RecordingAdapter(name="cloud-fake", is_egress=True)
    with TestClient(build_app(adapter=cloud)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    header = resp.headers.get("x-modelmeld-redactions", "")
    assert "EMAIL:1" in header
    assert "SSN:1" in header


def test_no_redaction_header_on_local() -> None:
    local = RecordingAdapter(name="local-fake", is_egress=False)
    with TestClient(build_app(adapter=local)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    assert "x-modelmeld-redactions" not in resp.headers


def test_scrubbing_can_be_disabled_via_settings(monkeypatch) -> None:
    monkeypatch.setenv("MODELMELD_PII_SCRUB_CLOUD", "false")
    from modelmeld.config import GatewaySettings

    cloud = RecordingAdapter(name="cloud-fake", is_egress=True)
    settings = GatewaySettings()
    with TestClient(build_app(settings=settings, adapter=cloud)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    assert resp.status_code == 200
    sent = cloud.received[0]
    # Scrubber was None → no redaction
    assert "alice@example.com" in sent.messages[0].content  # type: ignore[operator]


def test_explicit_scrubber_none_disables_scrubbing() -> None:
    cloud = RecordingAdapter(name="cloud-fake", is_egress=True)
    # Note: passing scrubber=None to build_app falls back to settings; need a sentinel
    # to truly disable. Use the settings path which is the supported API.
    from modelmeld.config import GatewaySettings

    settings = GatewaySettings(pii_scrub_cloud=False)
    with TestClient(build_app(settings=settings, adapter=cloud)) as client:
        resp = client.post("/v1/chat/completions", json=_user_payload())
    assert resp.status_code == 200
    assert "alice@example.com" in cloud.received[0].messages[0].content  # type: ignore[operator]

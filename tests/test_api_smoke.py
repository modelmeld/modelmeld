from __future__ import annotations

from fastapi.testclient import TestClient

from modelmeld.api.server import build_app


def make_client() -> TestClient:
    return TestClient(build_app())


def test_healthz_returns_ok() -> None:
    response = make_client().get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_returns_openai_list_shape() -> None:
    response = make_client().get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1
    for entry in payload["data"]:
        assert entry["object"] == "model"
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["owned_by"], str) and entry["owned_by"]
        assert isinstance(entry["created"], int)


def test_models_exposes_display_name_for_known_models() -> None:
    """Claude Code's /model picker requires `display_name` to render
    entries readably."""
    from modelmeld.config import GatewaySettings

    settings = GatewaySettings(
        available_models=["claude-sonnet-4-6", "gpt-5", "unknown-model"],
        owner="modelmeld",
    )
    client = TestClient(build_app(settings))
    entries = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    # Known frontier models get display_name
    assert entries["claude-sonnet-4-6"].get("display_name") == "Claude Sonnet 4.6"
    assert entries["gpt-5"].get("display_name") == "GPT-5"
    # Unknown ids have no display_name (excluded via exclude_none)
    assert "display_name" not in entries["unknown-model"] or entries["unknown-model"].get("display_name") is None


def test_models_advertises_anthropic_namespaced_auto_route_aliases() -> None:
    """Task #177 — Claude Code's /model picker filters to claude*/anthropic*
    prefixes. We surface the 3 canonical ModelMeld policy aliases so
    customers can pick a cost-quality tier from the picker."""
    client = make_client()
    entries = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    # Three canonical aliases must be present, each with a display_name.
    for alias in (
        "anthropic/modelmeld-saver",
        "anthropic/modelmeld-auto",
        "anthropic/modelmeld-quality",
    ):
        assert alias in entries, f"missing alias {alias}"
        assert entries[alias]["display_name"], f"missing display_name for {alias}"
        assert entries[alias]["owned_by"] == "modelmeld"


def test_models_returns_anthropic_native_shape_for_claude_code_discovery() -> None:
    """Claude Code's CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
    only parses Anthropic-native shape, NOT OpenAI shape. Each row must carry
    `type: "model"` and `created_at` (ISO 8601), and the envelope must carry
    `has_more`, `first_id`, `last_id`. Without these, the /model picker shows
    an empty gateway section even when /v1/models returns 200.
    """
    payload = make_client().get("/v1/models").json()
    # Envelope: Anthropic-native pagination markers.
    assert payload["has_more"] is False
    assert isinstance(payload["first_id"], str) and payload["first_id"]
    assert isinstance(payload["last_id"], str) and payload["last_id"]
    assert payload["first_id"] == payload["data"][0]["id"]
    assert payload["last_id"] == payload["data"][-1]["id"]
    # Per-row: Anthropic-native discriminator + ISO 8601 timestamp.
    for entry in payload["data"]:
        assert entry["type"] == "model"
        created_at = entry["created_at"]
        assert isinstance(created_at, str)
        # RFC 3339 / ISO 8601 with Z suffix (Anthropic spec)
        assert created_at.endswith("Z") and "T" in created_at


def test_models_reflects_settings() -> None:
    from modelmeld.config import GatewaySettings

    settings = GatewaySettings(available_models=["custom-a", "custom-b"], owner="acme")
    client = TestClient(build_app(settings))
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    # The configured models must appear, in order, at the start of the list.
    assert ids[:2] == ["custom-a", "custom-b"]
    # Anthropic-namespaced auto-route aliases may follow
    # — required for Claude Code's /model picker. They live in a
    # separate namespace and don't collide with operator-configured ids.
    for extra in ids[2:]:
        assert extra.startswith("anthropic/")


def test_chat_completions_returns_openai_shape() -> None:
    response = make_client().post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == "gpt-4o-mini"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert "usage" in body and body["usage"]["total_tokens"] >= 0


def test_chat_completions_echoes_requested_model() -> None:
    response = make_client().post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": []},
    )
    assert response.json()["model"] == "claude-sonnet-4-6"


def test_chat_completions_requires_model() -> None:
    # Strict schema. A missing required field returns 422.
    response = make_client().post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422

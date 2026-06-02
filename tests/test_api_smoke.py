from __future__ import annotations

from fastapi.testclient import TestClient

from modelmeld.api.server import build_app


def make_client() -> TestClient:
    return TestClient(build_app())


def test_healthz_returns_ok() -> None:
    response = make_client().get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_count_tokens_accepts_request_without_max_tokens() -> None:
    # /v1/messages/count_tokens has no semantic role for max_tokens
    # (it doesn't generate output), so it must accept requests that
    # omit the field — Anthropic's API does, and Claude Code's
    # pre-flight estimator sends count_tokens without max_tokens.
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    response = make_client().post(
        "/v1/messages/count_tokens",
        json=body,
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "input_tokens" in data
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_count_tokens_also_accepts_max_tokens_when_present() -> None:
    # Backward-compat: if a caller passes max_tokens anyway it must
    # not break the endpoint.
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello world"}],
    }
    response = make_client().post(
        "/v1/messages/count_tokens",
        json=body,
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text


def test_no_eligible_model_error_omits_provider_names() -> None:
    # The capability-scout error surfaces in customer-facing responses.
    # The message text must not echo the eligible-provider list — that
    # information isn't actionable for the caller (they want to know
    # what knob to adjust, not which adapters were considered). The
    # structured field stays populated for operator introspection.
    from modelmeld.scout.capability import NoEligibleModelError
    err = NoEligibleModelError(
        task_category="reasoning",
        quality_threshold=0.99,
        eligible_providers=frozenset({"some-provider", "another-provider"}),
    )
    msg = str(err)
    assert "0.99" in msg
    assert "reasoning" in msg
    # Provider names must not appear in the message body.
    assert "some-provider" not in msg
    assert "another-provider" not in msg
    assert "providers=" not in msg
    # Structured field is still populated for operator use.
    assert err.eligible_providers == frozenset({"some-provider", "another-provider"})


def test_version_returns_expected_shape() -> None:
    response = make_client().get("/version")
    assert response.status_code == 200
    body = response.json()
    # modelmeld is the package under test — always installed in the
    # test environment via `pip install -e .`, so the version string
    # must be present and non-empty.
    assert isinstance(body.get("modelmeld"), str) and body["modelmeld"]
    # The other three fields are presence-required (key exists), but
    # value-flexible: enterprise package may be absent (None), commit
    # SHA may be absent (None), registry size is always an int.
    assert "modelmeld_enterprise" in body
    assert "deployed_commit" in body
    assert isinstance(body.get("registry_size"), int)
    assert body["registry_size"] >= 0


def test_version_reads_commit_from_env(monkeypatch: object) -> None:
    # MODELMELD_DEPLOYED_COMMIT wins over RENDER_GIT_COMMIT when both
    # are set — explicit operator override beats the PaaS auto-inject.
    import os
    monkeypatch.setenv("MODELMELD_DEPLOYED_COMMIT", "deadbeef")  # type: ignore[attr-defined]
    monkeypatch.setenv("RENDER_GIT_COMMIT", "cafebabe")  # type: ignore[attr-defined]
    response = make_client().get("/version")
    assert response.json()["deployed_commit"] == "deadbeef"
    monkeypatch.delenv("MODELMELD_DEPLOYED_COMMIT")  # type: ignore[attr-defined]
    # With MODELMELD_DEPLOYED_COMMIT gone, RENDER_GIT_COMMIT becomes
    # the source.
    response = make_client().get("/version")
    assert response.json()["deployed_commit"] == "cafebabe"
    # Cleanup handled by monkeypatch teardown.
    _ = os  # silence unused-import warning if linter is strict


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


def test_models_auto_derives_from_registry_when_available_models_empty() -> None:
    """Default `available_models=[]` → /v1/models reflects everything the
    registry knows about. This is the production-default mode; it keeps the
    advertised lineup in sync with routing knowledge automatically so the
    operator doesn't have to push a parallel env-var update for every model
    add. Fixes the deployment-drift bug that caused api.modelmeld.ai to
    advertise a stale 4-model catalog for months after the OSS overlay
    expanded.
    """
    from modelmeld.config import GatewaySettings
    from modelmeld.scout.registry import ModelEntry, ModelRegistry

    registry = ModelRegistry([
        ModelEntry(
            model_id="alpha",
            provider="openai",
            context_window=100000,
            cost_per_m_input=1.0,
            cost_per_m_output=3.0,
            task_scores={"coding": 0.8},
            last_updated="2026-05-31",
            source="test",
        ),
        ModelEntry(
            model_id="beta",
            provider="anthropic",
            context_window=200000,
            cost_per_m_input=2.0,
            cost_per_m_output=10.0,
            task_scores={"coding": 0.9},
            last_updated="2026-05-31",
            source="test",
        ),
        ModelEntry(
            model_id="gamma",
            provider="vllm",
            context_window=32000,
            cost_per_m_input=0.5,
            cost_per_m_output=0.5,
            task_scores={"coding": 0.7},
            last_updated="2026-05-31",
            source="test",
        ),
    ])
    # Default settings: available_models is empty → auto-derive engages
    settings = GatewaySettings()
    assert settings.available_models == []
    client = TestClient(build_app(settings, model_registry=registry))

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    # All three registry models are advertised (sorted alphabetically)
    assert "alpha" in ids
    assert "beta" in ids
    assert "gamma" in ids
    # The three policy aliases get auto-appended in both modes
    assert "anthropic/modelmeld-saver" in ids
    assert "anthropic/modelmeld-auto" in ids
    assert "anthropic/modelmeld-quality" in ids


def test_models_explicit_list_ignores_registry() -> None:
    """When operator pins `available_models=[...]`, that list wins —
    the registry is bypassed for the advertised set. Used to restrict
    surface area (hide deprecated-but-still-routable models, limit a
    tenant to a known-stable subset, etc.).
    """
    from modelmeld.config import GatewaySettings
    from modelmeld.scout.registry import ModelEntry, ModelRegistry

    # Registry has 3 models; operator restricts to just 1.
    registry = ModelRegistry([
        ModelEntry(
            model_id="model-in-registry-a",
            provider="openai",
            context_window=100000,
            cost_per_m_input=1.0,
            cost_per_m_output=3.0,
            task_scores={"coding": 0.8},
            last_updated="2026-05-31",
            source="test",
        ),
        ModelEntry(
            model_id="model-in-registry-b",
            provider="openai",
            context_window=100000,
            cost_per_m_input=1.0,
            cost_per_m_output=3.0,
            task_scores={"coding": 0.8},
            last_updated="2026-05-31",
            source="test",
        ),
    ])
    settings = GatewaySettings(available_models=["only-this-one"])
    client = TestClient(build_app(settings, model_registry=registry))

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    # The explicit pin wins — registry contents are NOT advertised.
    assert "only-this-one" in ids
    assert "model-in-registry-a" not in ids
    assert "model-in-registry-b" not in ids
    # Policy aliases still appended (they're not registry-backed).
    assert "anthropic/modelmeld-saver" in ids


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

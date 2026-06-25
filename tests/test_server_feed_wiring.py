# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""build_app wires RegistryFeedClient at startup (Pro feed consumer).

When `registry_feed_url` is configured the gateway fetches the curated, signed
registry at startup and routes on it; any fetch failure falls back to the
bundled registry; with no feed configured nothing is fetched.
"""
from __future__ import annotations

from starlette.testclient import TestClient

from modelmeld.api import server
from modelmeld.config import GatewaySettings
from modelmeld.scout.feed import FeedFetchResult
from modelmeld.scout.registry import ModelEntry, ModelRegistry


def _feed_registry() -> ModelRegistry:
    return ModelRegistry([
        ModelEntry(
            model_id="feed-only-model", provider="vllm", context_window=100_000,
            cost_per_m_input=0.1, cost_per_m_output=0.1,
            task_scores={"coding": 0.9},
        ),
    ])


class _FakeClient:
    calls = 0
    result: FeedFetchResult | None = None

    def __init__(self, **kwargs) -> None:
        type(self).calls += 1
        self.kwargs = kwargs

    async def fetch(self) -> FeedFetchResult | None:
        return type(self).result

    async def close(self) -> None:
        pass


def test_feed_unconfigured_does_not_fetch(monkeypatch) -> None:
    _FakeClient.calls = 0
    monkeypatch.setattr(server, "RegistryFeedClient", _FakeClient)
    app = server.build_app(settings=GatewaySettings())  # no registry_feed_url
    with TestClient(app):
        pass
    assert _FakeClient.calls == 0


def test_feed_loaded_swaps_registry(monkeypatch) -> None:
    _FakeClient.calls = 0
    _FakeClient.result = FeedFetchResult(
        registry=_feed_registry(), source="feed", fetched_at=None, feed_version=42,
    )
    monkeypatch.setattr(server, "RegistryFeedClient", _FakeClient)
    app = server.build_app(settings=GatewaySettings(
        registry_feed_url="https://feed.example/v1/feed",
        registry_feed_license_key="jwt",
        registry_feed_public_key_pem="pem",
    ))
    with TestClient(app):
        ids = [e.model_id for e in app.state.model_registry.all_entries()]
    assert _FakeClient.calls == 1
    assert "feed-only-model" in ids


def test_feed_failure_keeps_default(monkeypatch) -> None:
    _FakeClient.calls = 0
    # source="seed" signals the client fell back (fetch/verify failed).
    _FakeClient.result = FeedFetchResult(
        registry=_feed_registry(), source="seed", fetched_at=None, feed_version=None,
    )
    monkeypatch.setattr(server, "RegistryFeedClient", _FakeClient)
    app = server.build_app(settings=GatewaySettings(
        registry_feed_url="https://feed.example/v1/feed",
        registry_feed_public_key_pem="pem",
    ))
    default_ids = {e.model_id for e in app.state.model_registry.all_entries()}
    with TestClient(app):
        post_ids = {e.model_id for e in app.state.model_registry.all_entries()}
    assert "feed-only-model" not in post_ids
    assert post_ids == default_ids

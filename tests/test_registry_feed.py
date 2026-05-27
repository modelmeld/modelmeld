"""RegistryFeedClient — signed-fetch + cache + seed-fallback."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from modelmeld.scout import RegistryFeedClient
from modelmeld.scout.feed import (
    SIGNATURE_HEADER,
    SUPPORTED_SCHEMA_VERSIONS,
)

# ---------------------------------------------------------------------------
# Test fixtures: keypair + canned envelope
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ed25519_keys() -> tuple[Ed25519PrivateKey, bytes]:
    """Generate a keypair once per module. Returns (private_key, public_pem)."""
    private_key = Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_pem


def _make_envelope(
    *,
    schema_version: int = 1,
    feed_version: int = 42,
    valid_until: datetime | None = None,
    models: list[dict] | None = None,
) -> dict:
    """Build a feed envelope with sensible defaults."""
    if valid_until is None:
        valid_until = datetime.now(timezone.utc) + timedelta(days=1)
    if models is None:
        models = [{
            "model_id": "feed-model-1",
            "provider": "anthropic",
            "context_window": 200000,
            "cost_per_m_input": 3.0,
            "cost_per_m_output": 15.0,
            "task_scores": {"coding": 0.85},
            "last_updated": "2026-05-20",
            "source": "feed",
        }]
    return {
        "schema_version": schema_version,
        "feed_version": feed_version,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "valid_until": valid_until.isoformat(),
        "registry": {
            "version": 1,
            "last_updated": "2026-05-20",
            "models": models,
        },
    }


def _signed_response(
    private_key: Ed25519PrivateKey, envelope: dict,
) -> httpx.Response:
    """Build the HTTP response a real feed server would send."""
    body = json.dumps(envelope, sort_keys=True).encode("utf-8")
    signature = private_key.sign(body)
    return httpx.Response(
        200, content=body,
        headers={SIGNATURE_HEADER: base64.b64encode(signature).decode("ascii")},
    )


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Happy path: signed feed loads
# ---------------------------------------------------------------------------

async def test_signed_feed_loads_successfully(ed25519_keys) -> None:
    private_key, public_pem = ed25519_keys
    envelope = _make_envelope()

    def handler(request: httpx.Request) -> httpx.Response:
        # License key sent in the right place
        assert request.headers["authorization"] == "Bearer license-XYZ"
        return _signed_response(private_key, envelope)

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="license-XYZ",
        public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    try:
        result = await client.fetch()
    finally:
        await client.close()

    assert result.source == "feed"
    assert result.feed_version == 42
    assert result.fetched_at is not None
    assert "feed-model-1" in result.registry
    assert result.registry.get("feed-model-1").cost_per_m_input == 3.0


# ---------------------------------------------------------------------------
# Tampered signature rejected
# ---------------------------------------------------------------------------

async def test_tampered_payload_falls_back_to_seed(ed25519_keys, caplog) -> None:
    private_key, public_pem = ed25519_keys
    envelope = _make_envelope()

    def handler(request: httpx.Request) -> httpx.Response:
        response = _signed_response(private_key, envelope)
        # Tamper with the body AFTER signing — signature should no longer verify
        tampered_body = response.content.replace(b"feed-model-1", b"feed-model-X")
        return httpx.Response(
            200, content=tampered_body,
            headers={SIGNATURE_HEADER: response.headers[SIGNATURE_HEADER]},
        )

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k",
        public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    with caplog.at_level("WARNING"):
        result = await client.fetch()
    await client.close()

    assert result.source == "seed"
    # The seed is the bundled fallback, not the tampered model
    assert "feed-model-X" not in result.registry
    # Loud log about the failure
    assert any("signature" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Wrong signature key rejected
# ---------------------------------------------------------------------------

async def test_signature_from_wrong_key_falls_back_to_seed() -> None:
    # The handler signs with one key; we configure the client with a different
    # public key. Verify should fail.
    wrong_signer = Ed25519PrivateKey.generate()
    legit_signer = Ed25519PrivateKey.generate()
    legit_public_pem = legit_signer.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    envelope = _make_envelope()

    def handler(_request: httpx.Request) -> httpx.Response:
        return _signed_response(wrong_signer, envelope)

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=legit_public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


# ---------------------------------------------------------------------------
# Unsupported schema_version rejected
# ---------------------------------------------------------------------------

async def test_unknown_schema_version_falls_back_to_seed(ed25519_keys) -> None:
    private_key, public_pem = ed25519_keys
    envelope = _make_envelope(schema_version=99)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _signed_response(private_key, envelope)

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


def test_schema_version_1_is_supported() -> None:
    assert 1 in SUPPORTED_SCHEMA_VERSIONS


# ---------------------------------------------------------------------------
# Expired feed rejected
# ---------------------------------------------------------------------------

async def test_expired_feed_falls_back_to_seed(ed25519_keys) -> None:
    private_key, public_pem = ed25519_keys
    envelope = _make_envelope(
        valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _signed_response(private_key, envelope)

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


# ---------------------------------------------------------------------------
# Network failures
# ---------------------------------------------------------------------------

async def test_http_4xx_falls_back_to_seed(ed25519_keys) -> None:
    _private_key, public_pem = ed25519_keys

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="bad-key", public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


async def test_connection_error_falls_back_to_seed(ed25519_keys) -> None:
    _private_key, public_pem = ed25519_keys

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS resolution failed")

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


# ---------------------------------------------------------------------------
# No feed_url configured → seed without warning attempt
# ---------------------------------------------------------------------------

async def test_no_feed_url_returns_seed_without_network() -> None:
    """A self-hosted gateway with no feed configured should not attempt fetches."""
    network_calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        network_calls["count"] += 1
        return httpx.Response(200, content=b"{}")

    client = RegistryFeedClient(
        feed_url=None, license_key=None, public_key_pem=None,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"
    assert network_calls["count"] == 0   # no http calls attempted


# ---------------------------------------------------------------------------
# Missing signature header rejected when verification is required
# ---------------------------------------------------------------------------

async def test_missing_signature_header_falls_back(ed25519_keys) -> None:
    _private_key, public_pem = ed25519_keys
    envelope = _make_envelope()

    def handler(_request: httpx.Request) -> httpx.Response:
        # Return the body WITHOUT the signature header
        return httpx.Response(200, content=json.dumps(envelope).encode())

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "seed"


# ---------------------------------------------------------------------------
# Local cache: warm cache short-circuits the network
# ---------------------------------------------------------------------------

async def test_warm_cache_returns_cached_without_network(ed25519_keys, tmp_path) -> None:
    private_key, public_pem = ed25519_keys
    envelope = _make_envelope(feed_version=7)
    cache_path = tmp_path / "registry-cache.json"

    network_calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        network_calls["count"] += 1
        return _signed_response(private_key, envelope)

    # First fetch: hits network, populates cache
    client1 = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        cache_path=cache_path, cache_ttl_sec=3600,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    first = await client1.fetch()
    await client1.close()
    assert first.source == "feed"
    assert cache_path.exists()
    assert network_calls["count"] == 1

    # Second fetch (fresh client to prove cache is on disk, not in-memory):
    # should read from cache without touching network
    client2 = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        cache_path=cache_path, cache_ttl_sec=3600,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    second = await client2.fetch()
    await client2.close()
    assert second.source == "cached"
    assert second.feed_version == 7
    assert network_calls["count"] == 1   # still only one network call


# ---------------------------------------------------------------------------
# Expired cache triggers re-fetch
# ---------------------------------------------------------------------------

async def test_expired_cache_triggers_refetch(ed25519_keys, tmp_path) -> None:
    private_key, public_pem = ed25519_keys
    cache_path = tmp_path / "stale.json"
    cache_path.write_bytes(json.dumps(_make_envelope(feed_version=1)).encode())
    # Backdate the cache file to be older than the TTL
    old_time = time.time() - 10000
    import os
    os.utime(cache_path, (old_time, old_time))

    fresh_envelope = _make_envelope(feed_version=99)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _signed_response(private_key, fresh_envelope)

    client = RegistryFeedClient(
        feed_url="https://feed.test/v1/registry",
        license_key="k", public_key_pem=public_pem,
        cache_path=cache_path, cache_ttl_sec=60,
        http_client=httpx.AsyncClient(transport=_mock_transport(handler)),
    )
    result = await client.fetch()
    await client.close()
    assert result.source == "feed"
    assert result.feed_version == 99


# ---------------------------------------------------------------------------
# Public key validation at construction
# ---------------------------------------------------------------------------

def test_non_ed25519_public_key_rejected() -> None:
    # An RSA key PEM should be rejected
    from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
    rsa_priv = generate_private_key(public_exponent=65537, key_size=2048)
    rsa_pub_pem = rsa_priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(ValueError, match="Ed25519"):
        RegistryFeedClient(
            feed_url="x", license_key="x",
            public_key_pem=rsa_pub_pem,
        )


def test_public_key_accepts_str_and_bytes(ed25519_keys) -> None:
    _private_key, public_pem = ed25519_keys
    # As bytes
    RegistryFeedClient(
        feed_url="x", license_key="x", public_key_pem=public_pem,
    )
    # As str
    RegistryFeedClient(
        feed_url="x", license_key="x",
        public_key_pem=public_pem.decode("utf-8"),
    )


# ---------------------------------------------------------------------------
# `seed_only` flag in the bundled registry triggers a one-time warning
# ---------------------------------------------------------------------------

def test_legacy_seed_only_flag_emits_info(monkeypatch, caplog) -> None:
    """Legacy `seed_only: true` flag still triggers the one-
    time snapshot-info log (suppressed under pytest)."""
    import modelmeld.scout.registry as mod
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(mod, "_seed_warning_emitted", False)

    with caplog.at_level("INFO"):
        mod.ModelRegistry.from_json({
            "version": 1, "seed_only": True, "models": [],
        })
    assert any("snapshot" in r.message.lower() for r in caplog.records)


def test_snapshot_release_date_emits_info_with_date(monkeypatch, caplog) -> None:
    """The newer `snapshot_release_date` flag is the modern form;
    the log message quotes the date so operators know how stale their
    routing data is."""
    import modelmeld.scout.registry as mod
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(mod, "_seed_warning_emitted", False)

    with caplog.at_level("INFO"):
        mod.ModelRegistry.from_json({
            "version": 1, "snapshot_release_date": "2026-05-22", "models": [],
        })
    msgs = [r.message for r in caplog.records]
    assert any("snapshot" in m.lower() for m in msgs)
    assert any("2026-05-22" in m for m in msgs)


def test_snapshot_info_emits_only_once(monkeypatch, caplog) -> None:
    import modelmeld.scout.registry as mod
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(mod, "_seed_warning_emitted", False)

    with caplog.at_level("INFO"):
        mod.ModelRegistry.from_json({"version": 1, "seed_only": True, "models": []})
        mod.ModelRegistry.from_json({"version": 1, "snapshot_release_date": "2026-05-22", "models": []})
        mod.ModelRegistry.from_json({"version": 1, "seed_only": True, "models": []})
    snapshot_messages = [
        r for r in caplog.records if "snapshot" in r.message.lower()
    ]
    assert len(snapshot_messages) == 1   # exactly one emission across three loads


def test_seed_only_false_does_not_warn(monkeypatch, caplog) -> None:
    """A live-feed payload (seed_only=false or absent) doesn't emit the warning."""
    import modelmeld.scout.registry as mod
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.setattr(mod, "_seed_warning_emitted", False)

    with caplog.at_level("WARNING"):
        mod.ModelRegistry.from_json({"version": 1, "models": []})           # absent
        mod.ModelRegistry.from_json({"version": 1, "seed_only": False, "models": []})
    assert not any("seed" in r.message.lower() for r in caplog.records)

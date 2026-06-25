# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""RegistryFeedClient multi-key pinning — zero-downtime key rotation.

Pinning both the current and next public key (concatenated PEM) lets the
publisher cut over signing keys without any subscriber falling back to the
bundled seed mid-rotation: the feed verifies if ANY pinned key matches.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from modelmeld.scout import RegistryFeedClient
from modelmeld.scout.feed import SIGNATURE_HEADER


def _pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _envelope() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "feed_version": 7,
        "issued_at": now.isoformat(),
        "valid_until": (now + timedelta(days=1)).isoformat(),
        "registry": {"version": 1, "models": [{
            "model_id": "rot-model", "provider": "x", "context_window": 100_000,
            "cost_per_m_input": 1.0, "cost_per_m_output": 1.0,
            "task_scores": {"coding": 0.8},
        }]},
    }


def _signed(key: Ed25519PrivateKey) -> httpx.Response:
    body = json.dumps(_envelope(), sort_keys=True).encode("utf-8")
    sig = key.sign(body)
    return httpx.Response(200, content=body, headers={
        SIGNATURE_HEADER: base64.b64encode(sig).decode("ascii"),
    })


def _client(pinned: list[bytes], signing_key: Ed25519PrivateKey):
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: _signed(signing_key))
    )
    client = RegistryFeedClient(
        feed_url="http://feed/v1/feed", license_key="jwt",
        public_key_pem=b"\n".join(pinned), http_client=http,
    )
    return client, http


async def test_both_old_and_new_keys_verify_during_rotation() -> None:
    old, new = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    pinned = [_pem(old), _pem(new)]
    for signer in (old, new):
        client, http = _client(pinned, signer)
        try:
            result = await client.fetch()
        finally:
            await http.aclose()
        assert result.source == "feed"  # whichever key signed, a pin matches


async def test_unpinned_key_falls_back_to_seed() -> None:
    old, new, attacker = (Ed25519PrivateKey.generate() for _ in range(3))
    client, http = _client([_pem(old), _pem(new)], attacker)
    try:
        result = await client.fetch()
    finally:
        await http.aclose()
    assert result.source == "seed"  # attacker's signature matches no pinned key


def test_single_key_still_parses() -> None:
    key = Ed25519PrivateKey.generate()
    client = RegistryFeedClient(
        feed_url=None, license_key=None, public_key_pem=_pem(key),
    )
    assert len(client._public_keys) == 1

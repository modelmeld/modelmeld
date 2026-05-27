# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""RegistryFeedClient — fetch the live curated model registry.

The bundled `default_registry.json` is stale by design — frozen at package
release time. Production deployments fetch a continuously curated feed from
`feed.modelmeld.ai` (or equivalent) authenticated with a per-tenant license
key issued from `Subscription` activation.

Wire format:

    GET  <feed_url>
    Authorization: Bearer <license_key>
    Accept: application/json
    →
    200 OK
    X-Feed-Signature: <base64(ed25519_sig)>  ← signs the raw response body
    Content-Type: application/json
    {
      "schema_version": 1,
      "feed_version": 42,
      "issued_at": "2026-05-20T00:00:00Z",
      "valid_until": "2026-05-21T00:00:00Z",
      "registry": { ...ModelRegistry payload... }
    }

Verification:
  1. Signature over the raw body bytes against a pinned Ed25519 public key.
  2. `schema_version` ∈ {known supported set}; otherwise reject.
  3. `valid_until` ≥ now; otherwise treat as stale.

Failure modes:
  - Network error / 5xx / timeout       → fall back to bundled seed.
  - 401 / 403                            → fall back to bundled seed, log loudly.
  - Signature mismatch                   → fall back to bundled seed, log loudly.
  - Schema unsupported                   → fall back to bundled seed, log loudly.
  - Cached payload still within TTL     → return cached without network hit.

The client NEVER raises on fetch failure — degraded routing is always
better than a crashed gateway. Callers can inspect `FeedFetchResult.source`
to see what they got.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from modelmeld.scout.registry import ModelRegistry

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
SIGNATURE_HEADER = "x-feed-signature"
DEFAULT_TIMEOUT_SEC = 15.0
DEFAULT_CACHE_TTL_SEC = 3600


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedFetchResult:
    """What `RegistryFeedClient.fetch()` returns.

    `source` is the load path that actually produced the registry:
      - "feed"   : freshly fetched + verified from the upstream feed
      - "cached" : loaded from local cache within TTL (no network hit)
      - "seed"   : bundled seed (network failed, signature invalid, etc.)
    """

    registry: ModelRegistry
    source: str
    fetched_at: datetime | None
    feed_version: int | None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RegistryFeedClient:
    """Async client that fetches + verifies + caches the live registry feed."""

    def __init__(
        self,
        *,
        feed_url: str | None,
        license_key: str | None,
        public_key_pem: bytes | str | None,
        cache_path: Path | str | None = None,
        cache_ttl_sec: int = DEFAULT_CACHE_TTL_SEC,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.feed_url = feed_url
        self.license_key = license_key
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_ttl_sec = cache_ttl_sec
        self._timeout = timeout_sec
        self._http = http_client
        self._owns_client = http_client is None
        self._public_key: Ed25519PublicKey | None = None
        if public_key_pem is not None:
            if isinstance(public_key_pem, str):
                public_key_pem = public_key_pem.encode("utf-8")
            key = load_pem_public_key(public_key_pem)
            if not isinstance(key, Ed25519PublicKey):
                raise ValueError(
                    "public_key_pem must be an Ed25519 public key in PEM format"
                )
            self._public_key = key
        elif feed_url:
            # Refuse to fetch a signed feed without a verifier. Previous
            # behavior was to fetch + log a warning + accept whatever the
            # server returned — a misconfigured deployment could silently
            # exfiltrate the bearer license_key to an attacker-controlled
            # URL and accept any payload as the live registry. Fail-closed
            # at construction time so the misconfiguration surfaces at
            # startup, not at first fetch.
            raise ValueError(
                "RegistryFeedClient: public_key_pem is required when feed_url "
                "is set (refusing to fetch the curated feed without "
                "signature verification). For local dev with no feed, pass "
                "feed_url=None to use the bundled seed."
            )

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._owns_client and self._http is not None:
            await self._http.aclose()
            self._http = None

    # ----- main entrypoint ------------------------------------------------

    async def fetch(self) -> FeedFetchResult:
        """Try cache → network → fall back to bundled seed.

        Never raises. Inspect `result.source` to see what produced the registry.
        """
        # 1. Try local cache if still warm.
        cached = self._read_cache()
        if cached is not None:
            return cached

        # 2. If no feed_url configured, this deployment hasn't opted into the
        #    feed — go straight to seed. No warning here; the seed-load warning
        #    already fires from registry.from_json.
        if not self.feed_url:
            return self._fallback_to_seed(reason="no_feed_url_configured")

        # 3. Try network. Any failure mode → seed fallback.
        try:
            body_bytes, signature_b64 = await self._http_fetch()
        except Exception as e:
            return self._fallback_to_seed(reason=f"http_error: {type(e).__name__}: {e}")

        # 4. Verify signature. Construction already refused if feed_url was
        # set without a public key, so self._public_key is guaranteed non-None
        # whenever this code path runs. Defensive check kept for clarity.
        if self._public_key is None:
            return self._fallback_to_seed(reason="no_public_key_configured")
        try:
            self._verify_signature(body_bytes, signature_b64)
        except InvalidSignature:
            return self._fallback_to_seed(reason="signature_invalid")
        except Exception as e:
            return self._fallback_to_seed(reason=f"signature_error: {e}")

        # 5. Parse + schema-validate the envelope.
        try:
            envelope = json.loads(body_bytes)
        except json.JSONDecodeError as e:
            return self._fallback_to_seed(reason=f"json_decode: {e}")
        ok, reason = self._validate_envelope(envelope)
        if not ok:
            return self._fallback_to_seed(reason=reason)

        # 6. Construct the registry from the envelope's `registry` field.
        try:
            registry = ModelRegistry.from_json(envelope["registry"])
        except (KeyError, ValueError) as e:
            return self._fallback_to_seed(reason=f"registry_parse: {e}")

        # 7. Persist to cache for next time.
        if self.cache_path is not None:
            self._write_cache(body_bytes)

        return FeedFetchResult(
            registry=registry,
            source="feed",
            fetched_at=datetime.now(timezone.utc),
            feed_version=int(envelope.get("feed_version", 0)) or None,
        )

    # ----- internals ------------------------------------------------------

    async def _http_fetch(self) -> tuple[bytes, str]:
        """Single GET against feed_url. Returns (body_bytes, signature_b64).

        Raises on non-2xx or missing signature header.
        """
        assert self.feed_url is not None
        client = await self._client()
        headers = {"accept": "application/json"}
        if self.license_key:
            headers["authorization"] = f"Bearer {self.license_key}"
        response = await client.get(self.feed_url, headers=headers)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"feed returned {response.status_code}",
                request=response.request,
                response=response,
            )
        signature_b64 = response.headers.get(SIGNATURE_HEADER)
        if signature_b64 is None and self._public_key is not None:
            raise ValueError(
                f"feed response missing required {SIGNATURE_HEADER!r} header"
            )
        return response.content, signature_b64 or ""

    def _verify_signature(self, body: bytes, signature_b64: str) -> None:
        """Raises `InvalidSignature` if the signature doesn't match."""
        assert self._public_key is not None
        try:
            signature = base64.b64decode(signature_b64)
        except Exception as e:
            raise InvalidSignature(f"signature not valid base64: {e}") from e
        self._public_key.verify(signature, body)

    def _validate_envelope(self, envelope: Any) -> tuple[bool, str]:
        """Returns (ok, reason). reason populated only on failure."""
        if not isinstance(envelope, dict):
            return False, f"envelope_not_object: type={type(envelope).__name__}"
        schema_version = envelope.get("schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            return False, (
                f"schema_unsupported: got {schema_version!r}, "
                f"supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        valid_until = envelope.get("valid_until")
        if valid_until is not None:
            try:
                expiry = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return False, f"valid_until_unparseable: {valid_until!r}"
            if expiry < datetime.now(timezone.utc):
                return False, f"feed_expired: valid_until={valid_until}"
        if "registry" not in envelope:
            return False, "envelope_missing_registry_field"
        return True, ""

    # ----- cache helpers --------------------------------------------------

    def _read_cache(self) -> FeedFetchResult | None:
        """Load cached payload if within TTL. Returns None on any failure."""
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            age = time.time() - self.cache_path.stat().st_mtime
        except OSError:
            return None
        if age > self.cache_ttl_sec:
            return None
        try:
            body = self.cache_path.read_bytes()
            envelope = json.loads(body)
        except (OSError, json.JSONDecodeError):
            return None
        ok, _reason = self._validate_envelope(envelope)
        if not ok:
            return None
        try:
            registry = ModelRegistry.from_json(envelope["registry"])
        except (KeyError, ValueError):
            return None
        return FeedFetchResult(
            registry=registry,
            source="cached",
            fetched_at=None,
            feed_version=int(envelope.get("feed_version", 0)) or None,
        )

    def _write_cache(self, body: bytes) -> None:
        assert self.cache_path is not None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(body)
        except OSError as e:
            logger.warning("failed to write feed cache: %s", e)

    def _fallback_to_seed(self, *, reason: str) -> FeedFetchResult:
        """Return the bundled seed; log the reason at WARN."""
        logger.warning(
            "registry feed unavailable — falling back to bundled seed (reason=%s)",
            reason,
        )
        return FeedFetchResult(
            registry=ModelRegistry.load_default(),
            source="seed",
            fetched_at=None,
            feed_version=None,
        )

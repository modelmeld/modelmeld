# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""License-key JWT verification — OSS-side primitive.

Published in `modelmeld` (AGPL-3.0-or-later) so the registry feed
client can self-verify its own JWT *before* sending it. Self-verification at the
client side gives the feed server room to skip expensive re-checks on
the hot path; the server only needs to confirm the signature is one we
issued (matches `kid`) and that the JWT has not been revoked.

Algorithm: EdDSA over Ed25519. The header is fixed:

    {"alg":"EdDSA","typ":"JWT","kid":"<key_fingerprint>"}

Payload claims:
    iss        — issuer (constant: our domain)
    sub        — tenant_id
    plan       — subscription plan
    feed_tier  — registry feed tier ("growth" / "enterprise")
    iat        — issued at (unix seconds)
    exp        — expires at (unix seconds)
    jti        — unique JWT id (also the LicenseKey row id)

The issuer side (`modelmeld_enterprise.licensing`) signs; this module only
verifies. Keeping verify in the OSS surface means downstream tools
(feed-client, observability) can validate JWTs without a paid SDK.

No external JWT library — pyjwt and python-jose both pull broad
dependency trees, and the Ed25519 path is tiny (sign = 1 line, verify =
1 line via `cryptography`). Hand-rolling keeps the OSS dep graph clean.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Header `alg` field. Ed25519 sign is `EdDSA` per RFC 8037.
JWT_ALG = "EdDSA"
JWT_TYP = "JWT"


# ---------------------------------------------------------------------------
# Errors — distinct types so callers can branch on cause
# ---------------------------------------------------------------------------


class LicenseKeyError(ValueError):
    """Base class for every license-verification failure."""


class LicenseKeyMalformedError(LicenseKeyError):
    """JWT could not be parsed (bad encoding, wrong segment count, bad header)."""


class LicenseKeySignatureError(LicenseKeyError):
    """Signature did not verify against the public key."""


class LicenseKeyExpiredError(LicenseKeyError):
    """JWT's `exp` claim is in the past."""

    def __init__(self, expired_at: int, now: int) -> None:
        super().__init__(
            f"license expired at unix={expired_at}, now=unix={now} "
            f"({now - expired_at}s ago)",
        )
        self.expired_at = expired_at
        self.now = now


class LicenseKeyKidMismatchError(LicenseKeyError):
    """JWT's `kid` doesn't match the expected fingerprint."""


# ---------------------------------------------------------------------------
# Claims DTO — the verified payload, plus the kid that signed it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LicenseClaims:
    issuer: str
    tenant_id: str
    plan: str
    feed_tier: str
    issued_at: int
    expires_at: int
    jti: str
    kid: str


# ---------------------------------------------------------------------------
# Base64url helpers — no padding, per RFC 7515
# ---------------------------------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    # Re-pad so the standard decoder accepts the input.
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ---------------------------------------------------------------------------
# Public-key utilities
# ---------------------------------------------------------------------------


def load_public_key_pem(pem: str | bytes) -> Ed25519PublicKey:
    """Load an Ed25519 public key from PEM."""
    if isinstance(pem, str):
        pem = pem.encode("utf-8")
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise LicenseKeyError(
            f"expected Ed25519 public key, got {type(key).__name__}",
        )
    return key


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """Stable identifier for a public key (used as JWT `kid`).

    sha256 of the raw 32-byte public key, hex-encoded. Truncated to 32 hex
    chars (128 bits of entropy) — collision-resistant under any realistic
    multi-key scenario while keeping the JWT header compact. Earlier
    versions used [:16] (64 bits); that was sufficient for single-key
    deployments but undersized for future post-rotation defenses where
    `expected_kid` enforcement matters.
    """
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_license_jwt(
    token: str,
    public_key: Ed25519PublicKey,
    *,
    expected_kid: str | None = None,
    now: int | None = None,
    leeway_sec: int = 60,
) -> LicenseClaims:
    """Verify a license JWT end-to-end.

    Checks performed (in order — fail fast on the cheapest first):
      1. Three dot-separated segments.
      2. Header parses + `alg==EdDSA` + `typ==JWT`.
      3. `expected_kid` matches header.kid (when provided).
      4. Signature verifies against `public_key`.
      5. Payload parses to JSON.
      6. `exp` is in the future (with `leeway_sec` slack for clock skew).

    Raises one of the `LicenseKey*Error` subclasses on any failure.
    Returns `LicenseClaims` on success.
    """
    segments = token.split(".")
    if len(segments) != 3:
        raise LicenseKeyMalformedError(
            f"expected 3 JWT segments, got {len(segments)}",
        )
    header_b64, payload_b64, sig_b64 = segments

    try:
        header = json.loads(_b64url_decode(header_b64))
    except (ValueError, json.JSONDecodeError) as e:
        raise LicenseKeyMalformedError("invalid JWT header") from e

    if header.get("alg") != JWT_ALG or header.get("typ") != JWT_TYP:
        raise LicenseKeyMalformedError(
            f"unsupported header alg={header.get('alg')!r} typ={header.get('typ')!r}",
        )

    kid = header.get("kid")
    if not isinstance(kid, str):
        raise LicenseKeyMalformedError("missing kid in header")
    if expected_kid is not None and kid != expected_kid:
        raise LicenseKeyKidMismatchError(
            f"expected kid={expected_kid!r}, got {kid!r}",
        )

    signing_input = (header_b64 + "." + payload_b64).encode("ascii")
    try:
        signature = _b64url_decode(sig_b64)
    except ValueError as e:
        raise LicenseKeyMalformedError("invalid signature encoding") from e
    try:
        public_key.verify(signature, signing_input)
    except InvalidSignature as e:
        raise LicenseKeySignatureError("signature verification failed") from e

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as e:
        raise LicenseKeyMalformedError("invalid JWT payload") from e

    ts = now if now is not None else int(time.time())
    exp = payload.get("exp")
    if not isinstance(exp, int):
        raise LicenseKeyMalformedError("payload missing or invalid 'exp'")
    if ts > exp + leeway_sec:
        raise LicenseKeyExpiredError(expired_at=exp, now=ts)

    return LicenseClaims(
        issuer=str(payload.get("iss", "")),
        tenant_id=str(payload.get("sub", "")),
        plan=str(payload.get("plan", "")),
        feed_tier=str(payload.get("feed_tier", "")),
        issued_at=int(payload.get("iat", 0)),
        expires_at=int(exp),
        jti=str(payload.get("jti", "")),
        kid=kid,
    )


def peek_unverified(token: str) -> dict[str, Any]:
    """Return the payload claims WITHOUT verifying the signature.

    Useful for the feed client to read its own `exp` before fetching, so
    it can pre-renew rather than hit a 401. Never trust the returned
    claims for authorization decisions.
    """
    try:
        _, payload_b64, _ = token.split(".")
        return json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as e:
        raise LicenseKeyMalformedError("could not peek JWT payload") from e

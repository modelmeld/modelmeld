"""License JWT verification — OSS surface."""

from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from modelmeld.licensing import (
    JWT_ALG,
    JWT_TYP,
    LicenseClaims,
    LicenseKeyExpiredError,
    LicenseKeyKidMismatchError,
    LicenseKeyMalformedError,
    LicenseKeySignatureError,
    load_public_key_pem,
    peek_unverified,
    public_key_fingerprint,
    verify_license_jwt,
)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_jwt(
    private_key: Ed25519PrivateKey,
    *,
    payload: dict | None = None,
    header_override: dict | None = None,
    tamper_signature: bool = False,
) -> str:
    """Hand-roll a JWT so we can introduce specific malformations."""
    pub = private_key.public_key()
    kid = public_key_fingerprint(pub)
    header = header_override if header_override is not None else {
        "alg": JWT_ALG, "typ": JWT_TYP, "kid": kid,
    }
    if payload is None:
        payload = {
            "iss": "gateway-licensing-v1",
            "sub": "tenant-1",
            "plan": "growth",
            "feed_tier": "growth",
            "iat": int(time.time()) - 60,
            "exp": int(time.time()) + 3600,
            "jti": "lic-1",
        }
    h_b64 = _b64url(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    p_b64 = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    sig = private_key.sign((h_b64 + "." + p_b64).encode())
    if tamper_signature:
        sig = sig[:-1] + bytes([sig[-1] ^ 0x01])
    s_b64 = _b64url(sig)
    return h_b64 + "." + p_b64 + "." + s_b64


@pytest.fixture
def keypair():
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_verify_happy_path(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv)
    claims = verify_license_jwt(token, pub)
    assert isinstance(claims, LicenseClaims)
    assert claims.tenant_id == "tenant-1"
    assert claims.plan == "growth"
    assert claims.feed_tier == "growth"
    assert claims.kid == public_key_fingerprint(pub)


def test_verify_with_expected_kid(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv)
    claims = verify_license_jwt(
        token, pub, expected_kid=public_key_fingerprint(pub),
    )
    assert claims.jti == "lic-1"


# ---------------------------------------------------------------------------
# Failure modes — each distinct error type
# ---------------------------------------------------------------------------

def test_segments_wrong_count(keypair) -> None:
    _priv, pub = keypair
    with pytest.raises(LicenseKeyMalformedError, match="3 JWT segments"):
        verify_license_jwt("a.b", pub)


def test_header_wrong_alg(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv, header_override={
        "alg": "HS256", "typ": JWT_TYP, "kid": "x",
    })
    with pytest.raises(LicenseKeyMalformedError, match="unsupported header alg"):
        verify_license_jwt(token, pub)


def test_kid_mismatch(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv)
    with pytest.raises(LicenseKeyKidMismatchError):
        verify_license_jwt(token, pub, expected_kid="other-kid")


def test_signature_tampered(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv, tamper_signature=True)
    with pytest.raises(LicenseKeySignatureError):
        verify_license_jwt(token, pub)


def test_signature_wrong_key(keypair) -> None:
    priv, _pub = keypair
    token = _make_jwt(priv)
    # Verify with a DIFFERENT public key
    other_priv = Ed25519PrivateKey.generate()
    with pytest.raises(LicenseKeySignatureError):
        verify_license_jwt(token, other_priv.public_key())


def test_expired_jwt_rejected(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv, payload={
        "iss": "x", "sub": "tenant-1", "plan": "p", "feed_tier": "t",
        "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,   # expired 1h ago
        "jti": "lic-old",
    })
    with pytest.raises(LicenseKeyExpiredError) as ei:
        verify_license_jwt(token, pub)
    assert ei.value.expired_at < ei.value.now


def test_clock_skew_leeway(keypair) -> None:
    """A JWT that expired 30 seconds ago is still accepted with leeway=60."""
    priv, pub = keypair
    token = _make_jwt(priv, payload={
        "iss": "x", "sub": "t", "plan": "p", "feed_tier": "t",
        "iat": int(time.time()) - 100,
        "exp": int(time.time()) - 30,
        "jti": "lic",
    })
    claims = verify_license_jwt(token, pub, leeway_sec=60)
    assert claims.tenant_id == "t"


def test_missing_exp_rejected(keypair) -> None:
    priv, pub = keypair
    token = _make_jwt(priv, payload={
        "iss": "x", "sub": "t", "plan": "p", "feed_tier": "t",
        "iat": int(time.time()), "jti": "x",
    })
    with pytest.raises(LicenseKeyMalformedError, match="exp"):
        verify_license_jwt(token, pub)


def test_invalid_payload_json(keypair) -> None:
    priv, pub = keypair
    # Construct a token with a payload that isn't JSON
    h_b64 = _b64url(json.dumps({"alg": JWT_ALG, "typ": JWT_TYP, "kid": "x"}).encode())
    p_b64 = _b64url(b"not-json")
    sig = priv.sign((h_b64 + "." + p_b64).encode())
    token = h_b64 + "." + p_b64 + "." + _b64url(sig)
    with pytest.raises(LicenseKeyMalformedError, match="payload"):
        verify_license_jwt(token, pub)


# ---------------------------------------------------------------------------
# peek_unverified
# ---------------------------------------------------------------------------

def test_peek_returns_claims_without_verifying(keypair) -> None:
    priv, _pub = keypair
    token = _make_jwt(priv)
    claims = peek_unverified(token)
    assert claims["sub"] == "tenant-1"
    # No verification → returns even for tampered tokens
    tampered = _make_jwt(priv, tamper_signature=True)
    claims2 = peek_unverified(tampered)
    assert claims2["sub"] == "tenant-1"


def test_peek_malformed_raises() -> None:
    with pytest.raises(LicenseKeyMalformedError):
        peek_unverified("not-a-jwt")


# ---------------------------------------------------------------------------
# Key utilities
# ---------------------------------------------------------------------------

def test_fingerprint_stable_across_calls(keypair) -> None:
    _priv, pub = keypair
    f1 = public_key_fingerprint(pub)
    f2 = public_key_fingerprint(pub)
    assert f1 == f2
    # 32 hex chars (128 bits). Was 16/64-bit; bumped for M-4 in the
    # pre-launch security audit to support post-rotation kid pinning.
    assert len(f1) == 32


def test_different_keys_have_different_fingerprints() -> None:
    p1 = Ed25519PrivateKey.generate().public_key()
    p2 = Ed25519PrivateKey.generate().public_key()
    assert public_key_fingerprint(p1) != public_key_fingerprint(p2)


def test_load_public_key_pem_roundtrip(keypair) -> None:
    _priv, pub = keypair
    pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    loaded = load_public_key_pem(pem)
    assert public_key_fingerprint(loaded) == public_key_fingerprint(pub)

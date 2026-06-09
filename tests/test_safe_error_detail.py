# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Tests for error-detail sanitization.

`safe_error_detail` is client-facing: it redacts credentials AND bounds length
(an unauthenticated caller may observe the response body). `redact_sensitive`
is the server-side-log variant: it redacts credentials but does NOT truncate,
so operators get the provider's full error (e.g. a complete upstream
`invalid_parameter` message) in the logs without secrets leaking to disk.
"""

from __future__ import annotations

from modelmeld.api._safe_error_detail import (
    _MAX_DETAIL_LEN,
    redact_sensitive,
    safe_error_detail,
)


def test_redact_sensitive_strips_credentials() -> None:
    text = "auth failed for sk-ant-api03-" + "A" * 40 + " on request"
    out = redact_sensitive(text)
    assert "sk-ant" not in out
    assert "[REDACTED]" in out


def test_redact_sensitive_does_not_truncate() -> None:
    # The whole point of the log variant: full length preserved so the
    # provider's real reason survives.
    long_detail = "InternalError.Algo.InvalidParameter: " + "x" * 5000
    out = redact_sensitive(long_detail)
    assert len(out) == len(long_detail)
    assert "[truncated]" not in out


def test_safe_error_detail_still_truncates() -> None:
    out = safe_error_detail("y" * (_MAX_DETAIL_LEN + 500))
    assert len(out) <= _MAX_DETAIL_LEN
    assert out.endswith("...[truncated]")


def test_safe_error_detail_redacts_then_bounds() -> None:
    out = safe_error_detail("key sk-proj-" + "B" * 60 + " rejected")
    assert "sk-proj" not in out
    assert "[REDACTED]" in out


def test_safe_error_detail_empty_falls_back_to_default() -> None:
    assert safe_error_detail("") == "upstream error"
    assert safe_error_detail(None) == "upstream error"
    assert safe_error_detail("", default="custom") == "custom"

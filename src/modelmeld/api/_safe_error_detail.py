# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Safe error-detail sanitization for HTTP responses.

Route handlers raise `HTTPException` with details often derived from
upstream provider errors (httpx, anthropic SDK, openai SDK). Those
errors can carry credential prefixes, account IDs, internal org IDs,
or request-payload fragments. This module sanitizes those before they
echo into the response body that any unauthenticated caller could
observe.
"""

from __future__ import annotations

import re

# Credential-shape patterns to strip from error strings. Mirrors the
# privacy scrubber but applied to error messages, not request bodies.
# Order matters: more-specific patterns (sk-proj-, sk-svcacct-) come
# before the generic `sk-` pattern so they don't get partially matched.
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),       # Anthropic
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}"),      # OpenAI project keys
    re.compile(r"sk-svcacct-[A-Za-z0-9_\-]{40,}"),   # OpenAI service-account keys
    re.compile(r"sk-[A-Za-z0-9_\-]{40,}"),           # OpenAI legacy + Stripe restricted
    re.compile(r"gws_[A-Za-z0-9_\-]{20,}"),          # ModelMeld API keys
    re.compile(r"AKIA[0-9A-Z]{16}"),                 # AWS access key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),              # GitHub PAT classic
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),     # GitHub PAT fine-grained
)

_MAX_DETAIL_LEN = 240


def safe_error_detail(error: object, default: str = "upstream error") -> str:
    """Return a sanitized error-detail string for HTTP response echo.

    - Coerces `error` to string
    - Replaces credential-shape substrings with "[REDACTED]"
    - Truncates to a fixed max length to bound unbounded leakage
    - Falls back to `default` for empty input
    """
    text = str(error) if error is not None else ""
    if not text:
        return default

    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    if len(text) > _MAX_DETAIL_LEN:
        text = text[: _MAX_DETAIL_LEN - 16] + "...[truncated]"

    return text

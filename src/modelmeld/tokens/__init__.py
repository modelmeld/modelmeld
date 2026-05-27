# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""Token counting — pluggable per-model tokenizers.

Public surface:
    TokenCounter             — abstract base
    CharBasedTokenCounter    — heuristic (1 token ≈ 4 chars), default
    LiteLLMTokenCounter      — accurate via `litellm.token_counter()`
    build_token_counter(s)   — factory keyed on GatewaySettings
"""

from __future__ import annotations

from modelmeld.tokens.counter import (
    CharBasedTokenCounter,
    LiteLLMTokenCounter,
    TokenCounter,
    build_token_counter,
)

__all__ = [
    "CharBasedTokenCounter",
    "LiteLLMTokenCounter",
    "TokenCounter",
    "build_token_counter",
]

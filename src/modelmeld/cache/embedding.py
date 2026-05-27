# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""EmbeddingClient ABC + reference implementations.

Production wires a real embedding model (OpenAI `text-embedding-3-small`,
local sentence-transformers, vLLM embed endpoint, etc.). Tests use the
deterministic `HashedBagOfWordsEmbedder` so paraphrased prompts get
similar vectors without a network call.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Async client that converts text → unit-length float vector.

    Implementations MUST return a vector of length `self.dim` and SHOULD
    L2-normalize so cosine-similarity becomes a dot product.
    """

    name: str
    dim: int

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return a `dim`-length vector for `text`."""

    async def close(self) -> None:
        """Release held resources. Default no-op."""


# ---------------------------------------------------------------------------
# Test-only deterministic embedder
# ---------------------------------------------------------------------------

# Token splitter: alphanumeric runs, lowercased. Conservative enough that
# punctuation differences don't change vectors but real paraphrases (which
# share vocabulary) score high.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashedBagOfWordsEmbedder(EmbeddingClient):
    """Deterministic per-token hashing into a fixed-size vector. NOT FOR PROD.

    For every word in the input text, hash it to a fixed bucket via
    BLAKE2b (Python's `hash()` is randomized per-process — useless for a
    persisted cache). Accumulate counts, then L2-normalize.

    Properties this gives us (verified in tests):
      - Identical text → identical vector
      - Paraphrases sharing most words → cosine ≈ 0.85–1.00
      - Disjoint-vocabulary prompts → cosine < 0.4

    Used as the test fixture for QdrantSemanticCache. Production wires
    a real model.
    """

    name = "hashed_bow"

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            idx = self._bucket(token)
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between `a` and `b`. Both should be unit-length."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))

"""Embedding generation with LiteLLM and deterministic local fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Sequence

import numpy as np

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


def _local_hash_embedding(text: str, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Deterministic pseudo-embedding for offline/tests without API keys."""
    tokens = re.findall(r"\w+", text.lower())
    vector = np.zeros(dim, dtype=np.float32)
    if not tokens:
        vector[0] = 1.0
        return vector.tolist()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(0, min(len(digest), dim)):
            vector[i % dim] += (digest[i] / 255.0) - 0.5
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def embed_text(text: str, use_local_fallback: bool = True) -> list[float]:
    """Generate embedding vector for text."""
    cleaned = text.strip()
    if not cleaned:
        return _local_hash_embedding("empty")

    use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "false").lower() == "true"
    if use_local_fallback and not use_remote:
        logger.debug("Using local hash embedding by default")
        return _local_hash_embedding(cleaned)

    try:
        import litellm

        response = litellm.embedding(
            model=EMBEDDING_MODEL,
            input=[cleaned],
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        )
        embedding = response.data[0]["embedding"]
        return [float(x) for x in embedding]
    except Exception as exc:
        if use_local_fallback:
            logger.warning("Embedding API failed (%s); using local fallback", exc)
            return _local_hash_embedding(cleaned)
        raise EmbeddingError(f"Failed to embed text: {exc}") from exc


def embed_texts(texts: Sequence[str], use_local_fallback: bool = True) -> list[list[float]]:
    return [embed_text(t, use_local_fallback=use_local_fallback) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va / na, vb / nb))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, math.ceil(len(text) / 4))

"""Embedding/relevance model interfaces."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from scad_rag.models.dummy_models import DummyEmbedder
from scad_rag.utils.device import resolve_device


class BaseEmbedder(ABC):
    """Abstract embedder."""

    @abstractmethod
    def score_pair(self, left: str, right: str) -> float:
        """Score a text pair."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 8) -> None:
        """Load a local embedding model."""
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._cache: dict[str, Any] = {}
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name, device=self.device)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory. Try reducing batch_size, using cpu.yaml, or choosing a smaller embedding model."
                ) from exc
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not load local embedding model '{model_name}'. Use quick_test.yaml offline.") from exc

    def score_pair(self, left: str, right: str) -> float:
        """Return cosine similarity mapped to [0, 1]."""
        left_emb = self._embedding(left)
        right_emb = self._embedding(right)
        return (cosine(list(left_emb), list(right_emb)) + 1.0) / 2.0

    def _embedding(self, text: str):
        """Return a cached embedding for text."""
        if text not in self._cache:
            self._cache[text] = self.model.encode([text], batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)[0]
        return self._cache[text]


def cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity."""
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    return dot / (ln * rn) if ln and rn else 0.0


def build_embedder(config: dict) -> BaseEmbedder:
    """Build configured embedder."""
    if config.get("use_dummy_models", False):
        return DummyEmbedder()
    return SentenceTransformerEmbedder(
        str(config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        str(config.get("device", "auto")),
        int(config.get("batch_size", 8)),
    )

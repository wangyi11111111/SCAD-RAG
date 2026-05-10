"""Lexical overlap features."""

from __future__ import annotations

from scad_rag.utils.text import token_set


def jaccard_similarity(left: str, right: str) -> float:
    """Compute Jaccard similarity."""
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coefficient(left: str, right: str) -> float:
    """Compute overlap coefficient."""
    a, b = token_set(left), token_set(right)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def lexical_relevance(left: str, right: str) -> float:
    """Blend overlap metrics."""
    return min(1.0, 0.35 * jaccard_similarity(left, right) + 0.65 * overlap_coefficient(left, right))

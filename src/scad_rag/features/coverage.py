"""Keyword coverage."""

from __future__ import annotations

from scad_rag.utils.text import token_set


def keyword_coverage(claim: str, evidence: str) -> float:
    """Return claim keyword coverage by evidence."""
    claim_tokens = token_set(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & token_set(evidence)) / len(claim_tokens)

"""ESS and SCAD score."""

from __future__ import annotations


def compute_ess(relevance: float, entailment: float, contradiction: float, coverage: float, weights: dict) -> float:
    """Compute ESS without sufficient context."""
    score = (
        float(weights.get("alpha", 0.25)) * relevance
        + float(weights.get("beta", 0.35)) * entailment
        - float(weights.get("gamma", 0.20)) * contradiction
        + float(weights.get("delta", 0.10)) * coverage
    )
    return max(0.0, min(1.0, score))


def compute_scad_score(
    relevance: float,
    entailment: float,
    contradiction: float,
    coverage: float,
    sufficient_context_score: float,
    weights: dict,
) -> float:
    """Compute SCAD score with sufficient-context score."""
    score = compute_ess(relevance, entailment, contradiction, coverage, weights) + float(weights.get("eta", 0.10)) * sufficient_context_score
    return max(0.0, min(1.0, score))

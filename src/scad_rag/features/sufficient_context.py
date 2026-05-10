"""Sufficient Context Gate."""

from __future__ import annotations


def evaluate_sufficient_context(
    relevance: float,
    entailment: float,
    contradiction: float,
    coverage: float,
    thresholds: dict,
) -> tuple[str, float]:
    """Return context status and sufficient context score."""
    contradiction_threshold = float(thresholds.get("contradiction_threshold", 0.55))
    entailment_threshold = float(thresholds.get("entailment_threshold", 0.50))
    low_relevance_threshold = float(thresholds.get("low_relevance_threshold", 0.25))
    coverage_threshold = float(thresholds.get("coverage_threshold", 0.35))
    score = max(0.0, min(1.0, 0.30 * relevance + 0.40 * entailment - 0.25 * contradiction + 0.15 * coverage))
    near = _near(relevance, low_relevance_threshold) or _near(entailment, entailment_threshold) or _near(coverage, coverage_threshold)
    if contradiction >= contradiction_threshold:
        return "Conflicting", min(score, 0.25)
    if entailment >= entailment_threshold and coverage >= coverage_threshold:
        return ("Uncertain", score) if near and score < 0.65 else ("Sufficient", max(score, 0.65))
    if relevance < low_relevance_threshold:
        return "Insufficient", min(score, 0.35)
    if relevance >= low_relevance_threshold and entailment < entailment_threshold:
        return ("Uncertain", score) if near else ("Insufficient", min(score, 0.49))
    return "Uncertain", score


def _near(value: float, threshold: float, margin: float = 0.06) -> bool:
    """Return whether a value is near a threshold."""
    return abs(value - threshold) <= margin

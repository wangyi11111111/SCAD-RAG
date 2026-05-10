"""ML-feature baseline."""

from __future__ import annotations


def feature_vector(audit) -> list[float]:
    """Return interpretable feature vector."""
    return [
        audit.max_relevance,
        audit.entailment_score,
        audit.neutral_score,
        audit.contradiction_score,
        audit.coverage_score,
        audit.score_original,
        audit.sufficient_context_score,
        audit.uncertainty_score,
    ]


def predict(audit, thresholds):
    """Deterministic fallback predictor."""
    if audit.contradiction_score >= float(thresholds.get("contradiction_threshold", 0.55)):
        return "Contradicted", 1, "Evidence-contradicted", "ML-feature fallback predicted contradiction."
    if audit.score_original >= float(thresholds.get("supported_threshold", 0.60)):
        return "Supported", 0, "No hallucination", "ML-feature fallback predicted support."
    if audit.uncertainty_score >= float(thresholds.get("uncertainty_threshold", 0.65)):
        return "Uncertain", -1, "High-risk-abstain", "ML-feature fallback predicted high risk."
    return "Insufficient", 1, "Generation-inconsistent", "ML-feature fallback predicted insufficient support."

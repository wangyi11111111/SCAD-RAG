"""ESS-rule baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict with score but without SCAD-specific modules."""
    if audit.contradiction_score >= float(thresholds.get("contradiction_threshold", 0.55)):
        return "Contradicted", 1, "Evidence-contradicted", "ESS-rule found contradiction."
    if audit.score_original >= float(thresholds.get("supported_threshold", 0.60)) and audit.entailment_score >= float(thresholds.get("entailment_threshold", 0.50)):
        return "Supported", 0, "No hallucination", "ESS-rule found sufficient feature score."
    if audit.max_relevance < float(thresholds.get("low_relevance_threshold", 0.25)):
        return "Insufficient", 1, "Retrieval-insufficient", "ESS-rule found low relevance."
    return "Insufficient", 1, "Generation-inconsistent", "ESS-rule found relevance without entailment."

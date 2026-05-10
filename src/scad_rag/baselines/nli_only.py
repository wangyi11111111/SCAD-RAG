"""NLI-only baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict from NLI scores."""
    if audit.contradiction_score >= float(thresholds.get("contradiction_threshold", 0.55)):
        return "Contradicted", 1, "Evidence-contradicted", "NLI-only found contradiction."
    if audit.entailment_score >= float(thresholds.get("entailment_threshold", 0.50)):
        return "Supported", 0, "No hallucination", "NLI-only found entailment."
    return "Insufficient", 1, "Generation-inconsistent", "NLI-only found no entailment."

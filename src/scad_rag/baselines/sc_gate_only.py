"""Sufficient-context gate only baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict from context status only."""
    if audit.context_status_original == "Conflicting":
        return "Contradicted", 1, "Evidence-contradicted", "SC-Gate-only found conflicting context."
    if audit.context_status_original == "Sufficient":
        return "Supported", 0, "No hallucination", "SC-Gate-only found sufficient context."
    if audit.context_status_original == "Insufficient":
        return "Insufficient", 1, "Generation-inconsistent", "SC-Gate-only found insufficient context."
    return "Uncertain", -1, "High-risk-abstain", "SC-Gate-only abstains on uncertain context."

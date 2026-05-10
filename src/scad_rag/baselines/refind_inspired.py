"""REFIND-inspired context sensitivity baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Use lightweight CSR = score_with_context - score_without_context."""
    csr = audit.score_original - audit.score_removed
    if csr >= 0.20 and audit.entailment_score >= float(thresholds.get("entailment_threshold", 0.50)):
        return "Supported", 0, "No hallucination", f"REFIND-inspired CSR={csr:.3f} is high."
    return "Uncertain", -1, "Unstable-evidence-dependency", f"REFIND-inspired CSR={csr:.3f} is weak."

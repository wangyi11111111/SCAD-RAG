"""Risk-calibrated attribution rules."""

from __future__ import annotations

from scad_rag.attribution.risk_calibration import should_abstain
from scad_rag.schema import AuditResult


def predict_from_audit(audit: AuditResult, thresholds: dict, risk_enabled: bool = True) -> tuple[str, int, str, str]:
    """Predict relation, hallucination, attribution, and explanation."""
    contradiction_threshold = float(thresholds.get("contradiction_threshold", 0.55))
    supported_threshold = float(thresholds.get("supported_threshold", 0.60))
    entailment_threshold = float(thresholds.get("entailment_threshold", 0.50))
    low_relevance = float(thresholds.get("low_relevance_threshold", 0.25))
    min_delta = float(thresholds.get("min_dependency_delta", 0.10))
    min_gap = float(thresholds.get("min_hard_negative_gap", 0.10))
    if audit.contradiction_score >= contradiction_threshold:
        return "Contradicted", 1, "Evidence-contradicted", "The best evidence directly contradicts the claim."
    if audit.score_original >= supported_threshold and audit.evidence_dependency_delta < min_delta and audit.hard_negative_robustness_gap < min_gap:
        return "Uncertain", -1, "Unstable-evidence-dependency", "The score is high but barely changes after evidence removal or hard-negative replacement."
    if audit.context_status_original == "Sufficient" and audit.score_original >= supported_threshold:
        return "Supported", 0, "No hallucination", "The current evidence is sufficient and the SCAD score supports the claim."
    if audit.max_relevance < low_relevance:
        return "Insufficient", 1, "Retrieval-insufficient", "Retrieved evidence is weakly related to the claim."
    if audit.max_relevance >= low_relevance and audit.entailment_score < entailment_threshold:
        if risk_enabled and should_abstain(audit.uncertainty_score, audit.risk_score, thresholds):
            return "Uncertain", -1, "High-risk-abstain", "The evidence is related but uncertainty/risk is too high for a reliable decision."
        return "Insufficient", 1, "Generation-inconsistent", "Evidence is relevant but does not entail the generated claim."
    if risk_enabled and should_abstain(audit.uncertainty_score, audit.risk_score, thresholds):
        return "Uncertain", -1, "High-risk-abstain", "Risk calibration recommends abstention."
    return "Uncertain", -1, "High-risk-abstain", "The claim falls outside confident supported, insufficient, or contradicted regions."

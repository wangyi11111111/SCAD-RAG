"""Counterfactual evidence audit for SCAD-RAG."""

from __future__ import annotations

from statistics import mean

from scad_rag.features.coverage import keyword_coverage
from scad_rag.features.evidence_sufficiency import compute_scad_score
from scad_rag.features.hard_negative import select_hard_negative
from scad_rag.features.relevance import rank_evidences
from scad_rag.features.sufficient_context import evaluate_sufficient_context
from scad_rag.features.uncertainty import compute_risk, compute_uncertainty
from scad_rag.schema import AuditResult, ClaimRecord, Evidence, EvidenceScore


def score_claim_evidence(claim_text: str, evidence: Evidence, relevance: float, nli_model, weights: dict, thresholds: dict) -> EvidenceScore:
    """Score one claim/evidence pair."""
    nli = nli_model.score_pair(evidence.text, claim_text)
    coverage = keyword_coverage(claim_text, evidence.text)
    status, sc_score = evaluate_sufficient_context(relevance, nli.entailment, nli.contradiction, coverage, thresholds)
    score = compute_scad_score(relevance, nli.entailment, nli.contradiction, coverage, sc_score, weights)
    return EvidenceScore(
        evidence.id,
        evidence.text,
        evidence.type,
        relevance,
        nli.entailment,
        nli.neutral,
        nli.contradiction,
        coverage,
        sc_score,
        status,
        score,
    )


def audit_claim(
    claim: ClaimRecord,
    embedder,
    nli_model,
    top_k: int,
    weights: dict,
    thresholds: dict,
    counterfactual_config: dict,
    risk_config: dict,
    evidence_pool: list[Evidence] | None = None,
    allow_gold_hard_negatives: bool = False,
) -> AuditResult:
    """Run SCAD original, removal, hard-negative, conflict, uncertainty, and risk audit."""
    if not claim.evidences:
        return AuditResult()
    ranked = rank_evidences(claim.claim_text, claim.evidences, embedder, top_k)
    scores = [score_claim_evidence(claim.claim_text, ev, rel, nli_model, weights, thresholds) for ev, rel in ranked]
    best = max(scores, key=lambda item: item.scad_score)
    score_original = best.scad_score
    removed_score, removed_status = (score_original, best.context_status)
    if counterfactual_config.get("enable_removal", True):
        removed_score, removed_status = _removed_view(claim, best.evidence_id, embedder, nli_model, top_k, weights, thresholds)
    hard_score, hard_status = (score_original, best.context_status)
    hard_id, hard_text = "", ""
    if counterfactual_config.get("enable_hard_negative", True):
        candidates = list(claim.evidences)
        if evidence_pool:
            candidates.extend([ev for ev in evidence_pool if ev.text != best.evidence_text])
        hard_ev, hard_rel = select_hard_negative(
            claim.claim_text,
            best.evidence_id,
            candidates,
            embedder,
            nli_model,
            thresholds,
            allow_gold_labels=allow_gold_hard_negatives,
            strategy=str(counterfactual_config.get("hard_negative_strategy", "hard_negative")),
        )
        if hard_ev:
            hard = score_claim_evidence(claim.claim_text, hard_ev, hard_rel, nli_model, weights, thresholds)
            hard_score, hard_status, hard_id, hard_text = hard.scad_score, hard.context_status, hard.evidence_id, hard.evidence_text
    max_conflict = max(scores, key=lambda item: item.contradiction_score)
    has_conflict = bool(
        counterfactual_config.get("enable_contradiction_probe", True)
        and max_conflict.contradiction_score >= float(thresholds.get("contradiction_threshold", 0.55))
    )
    edd = score_original - removed_score
    hnrg = score_original - hard_score
    stability = dependency_stability(score_original, edd, hnrg, thresholds)
    uncertainty = 0.0
    risk = 0.0
    if risk_config.get("enabled", True):
        uncertainty = compute_uncertainty(
            best.entailment_score,
            best.neutral_score,
            best.contradiction_score,
            best.context_status,
            score_original,
            thresholds,
            edd,
            hnrg,
            has_conflict,
        )
        risk = compute_risk(uncertainty, best.sufficient_context_score, best.contradiction_score, hnrg, stability, thresholds)
    return AuditResult(
        best_evidence_id=best.evidence_id,
        best_evidence_text=best.evidence_text,
        hard_negative_evidence_id=hard_id,
        hard_negative_evidence_text=hard_text,
        top_evidence_ids=[item.evidence_id for item in scores],
        max_relevance=max(item.relevance_score for item in scores),
        mean_topk_relevance=mean(item.relevance_score for item in scores),
        entailment_score=best.entailment_score,
        neutral_score=best.neutral_score,
        contradiction_score=best.contradiction_score,
        coverage_score=best.coverage_score,
        sufficient_context_score=best.sufficient_context_score,
        context_status_original=best.context_status,
        context_status_removed=removed_status,
        context_status_hard_negative=hard_status,
        score_original=score_original,
        score_removed=removed_score,
        score_hard_negative=hard_score,
        evidence_dependency_delta=edd,
        hard_negative_robustness_gap=hnrg,
        max_contradiction_score=max_conflict.contradiction_score if has_conflict else 0.0,
        has_conflicting_evidence=has_conflict,
        contradiction_evidence_id=max_conflict.evidence_id if has_conflict else "",
        contradiction_evidence_text=max_conflict.evidence_text if has_conflict else "",
        dependency_stability_label=stability,
        uncertainty_score=uncertainty,
        risk_score=risk,
        evidence_scores=scores,
    )


def dependency_stability(score_original: float, edd: float, hnrg: float, thresholds: dict) -> str:
    """Classify dependency stability."""
    min_delta = float(thresholds.get("min_dependency_delta", 0.10))
    min_gap = float(thresholds.get("min_hard_negative_gap", 0.10))
    supported = float(thresholds.get("supported_threshold", 0.60))
    if score_original >= supported and edd < min_delta and hnrg < min_gap:
        return "Unstable"
    if edd >= min_delta:
        return "Stable-dependent"
    if edd < min_delta and hnrg >= min_gap:
        return "Stable-redundant"
    return "Unknown"


def _removed_view(claim: ClaimRecord, best_id: str, embedder, nli_model, top_k: int, weights: dict, thresholds: dict) -> tuple[float, str]:
    """Score after removing best evidence."""
    remaining = [ev for ev in claim.evidences if ev.id != best_id]
    if not remaining:
        return 0.0, "Insufficient"
    ranked = rank_evidences(claim.claim_text, remaining, embedder, top_k)
    scores = [score_claim_evidence(claim.claim_text, ev, rel, nli_model, weights, thresholds) for ev, rel in ranked]
    best = max(scores, key=lambda item: item.scad_score)
    return best.scad_score, best.context_status

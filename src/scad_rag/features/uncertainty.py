"""Uncertainty and risk scoring."""

from __future__ import annotations

import math


def nli_entropy(entailment: float, neutral: float, contradiction: float) -> float:
    """Compute normalized entropy for NLI probabilities."""
    probs = [max(1e-12, entailment), max(1e-12, neutral), max(1e-12, contradiction)]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(3)


def compute_uncertainty(
    entailment: float,
    neutral: float,
    contradiction: float,
    context_status: str,
    score_original: float,
    thresholds: dict,
    evidence_dependency_delta: float,
    hard_negative_robustness_gap: float,
    has_conflict: bool,
) -> float:
    """Compute uncertainty from entropy, threshold closeness, dependency, robustness, and conflict."""
    entropy = nli_entropy(entailment, neutral, contradiction)
    supported_threshold = float(thresholds.get("supported_threshold", 0.60))
    min_delta = float(thresholds.get("min_dependency_delta", 0.10))
    min_gap = float(thresholds.get("min_hard_negative_gap", 0.10))
    closeness = max(0.0, 1.0 - min(1.0, abs(score_original - supported_threshold) / 0.25))
    context_uncertain = 1.0 if context_status == "Uncertain" else 0.0
    low_delta = 1.0 if evidence_dependency_delta < min_delta else 0.0
    low_gap = 1.0 if hard_negative_robustness_gap < min_gap else 0.0
    conflict = 1.0 if has_conflict else 0.0
    value = 0.30 * entropy + 0.20 * closeness + 0.20 * context_uncertain + 0.10 * low_delta + 0.10 * low_gap + 0.10 * conflict
    return max(0.0, min(1.0, value))


def nli_reliability_score(
    entailment: float,
    neutral: float,
    contradiction: float,
    coverage: float,
    thresholds: dict,
) -> float:
    """Estimate whether the local NLI signal is reliable for downstream attribution.

    High entropy, high neutral probability, and low lexical coverage are treated
    as signs that the relation decision is under-specified. The score is not a
    learned probability; it is a transparent gate used to prevent error
    propagation from weak NLI regions into fine-grained attribution labels.
    """
    entropy = nli_entropy(entailment, neutral, contradiction)
    coverage_threshold = float(thresholds.get("coverage_threshold", 0.35))
    neutral_threshold = float(thresholds.get("neutral_threshold", 0.70))
    low_coverage_penalty = max(0.0, (coverage_threshold - coverage) / max(coverage_threshold, 1e-9))
    high_neutral_penalty = max(0.0, (neutral - neutral_threshold) / max(1.0 - neutral_threshold, 1e-9))
    value = 1.0 - (0.45 * entropy + 0.30 * high_neutral_penalty + 0.25 * low_coverage_penalty)
    return max(0.0, min(1.0, value))


def compute_risk(
    uncertainty_score: float,
    sufficient_context_score: float,
    contradiction_score: float,
    hard_negative_robustness_gap: float,
    dependency_stability_label: str,
    thresholds: dict,
    nli_reliability: float = 1.0,
) -> float:
    """Compute risk score for abstention."""
    min_gap = float(thresholds.get("min_hard_negative_gap", 0.10))
    unstable = 1.0 if dependency_stability_label == "Unstable" else 0.0
    low_sc = 1.0 - sufficient_context_score
    low_gap = 1.0 if hard_negative_robustness_gap < min_gap else 0.0
    low_nli_reliability = 1.0 - nli_reliability
    value = (
        0.30 * uncertainty_score
        + 0.20 * low_sc
        + 0.18 * contradiction_score
        + 0.10 * low_gap
        + 0.10 * unstable
        + 0.12 * low_nli_reliability
    )
    return max(0.0, min(1.0, value))

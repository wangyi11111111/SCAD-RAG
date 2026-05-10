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


def compute_risk(
    uncertainty_score: float,
    sufficient_context_score: float,
    contradiction_score: float,
    hard_negative_robustness_gap: float,
    dependency_stability_label: str,
    thresholds: dict,
) -> float:
    """Compute risk score for abstention."""
    min_gap = float(thresholds.get("min_hard_negative_gap", 0.10))
    unstable = 1.0 if dependency_stability_label == "Unstable" else 0.0
    low_sc = 1.0 - sufficient_context_score
    low_gap = 1.0 if hard_negative_robustness_gap < min_gap else 0.0
    value = 0.35 * uncertainty_score + 0.25 * low_sc + 0.20 * contradiction_score + 0.10 * low_gap + 0.10 * unstable
    return max(0.0, min(1.0, value))

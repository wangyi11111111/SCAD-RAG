"""SCAD-RAG metrics."""

from __future__ import annotations

import math
from typing import Any

RELATIONS = ["Supported", "Insufficient", "Contradicted", "Uncertain"]
CONTEXTS = ["Sufficient", "Insufficient", "Conflicting", "Uncertain"]


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute classification, attribution, context, audit, and risk metrics."""
    if not rows:
        return {"num_claims": 0, "reason": "No predictions available."}
    gold_rel = [row.get("gold_relation") for row in rows]
    pred_rel = [row.get("pred_relation") for row in rows]
    gold_h = [int(row.get("gold_hallucination", -1)) for row in rows]
    pred_h = [int(row.get("pred_hallucination", -1)) for row in rows]
    binary = [(g, p) for g, p in zip(gold_h, pred_h) if g in {0, 1} and p in {0, 1}]
    precision, recall, h_f1 = _binary_prf([g for g, _ in binary], [p for _, p in binary])
    binary_macro = _binary_macro_f1([g for g, _ in binary], [p for _, p in binary])
    metrics: dict[str, Any] = {
        "num_claims": len(rows),
        "accuracy": _accuracy(gold_rel, pred_rel),
        "precision": precision,
        "recall": recall,
        "macro_f1": _macro_f1(gold_rel, pred_rel, RELATIONS),
        "hallucination_macro_f1": binary_macro,
        "hallucination_f1": h_f1,
        "hallucination_auroc": _auroc([g for g, _ in binary], [_hallucination_score(row) for row in rows if int(row.get("gold_hallucination", -1)) in {0, 1} and int(row.get("pred_hallucination", -1)) in {0, 1}]),
        "hallucination_brier": _brier_score(rows),
        "hallucination_ece": _expected_calibration_error(rows),
        "confusion_matrix": _confusion_matrix(gold_rel, pred_rel, RELATIONS),
        "average_evidence_dependency_delta": _avg(rows, "evidence_dependency_delta"),
        "average_hard_negative_robustness_gap": _avg(rows, "hard_negative_robustness_gap"),
        "contradiction_probe_rate": sum(_bool(row.get("has_conflicting_evidence")) for row in rows) / len(rows),
        "unstable_dependency_rate": sum(row.get("pred_attribution") == "Unstable-evidence-dependency" for row in rows) / len(rows),
        "abstain_rate": sum(int(row.get("pred_hallucination", 0)) == -1 for row in rows) / len(rows),
    }
    for label in RELATIONS:
        metrics[f"{label.lower()}_f1"] = _label_f1(gold_rel, pred_rel, label)
    attr_rows = [row for row in rows if row.get("gold_attribution") not in {"Unknown", ""}]
    if attr_rows:
        labels = sorted(set(row.get("gold_attribution") for row in attr_rows) | set(row.get("pred_attribution") for row in attr_rows))
        metrics["attribution_accuracy"] = _accuracy([row.get("gold_attribution") for row in attr_rows], [row.get("pred_attribution") for row in attr_rows])
        metrics["attribution_macro_f1"] = _macro_f1([row.get("gold_attribution") for row in attr_rows], [row.get("pred_attribution") for row in attr_rows], labels)
    else:
        metrics["attribution_skipped_reason"] = "No gold attribution labels were available."
    ctx_rows = [row for row in rows if row.get("gold_context_status") not in {"Unknown", ""}]
    if ctx_rows:
        metrics["context_status_accuracy"] = _accuracy([row.get("gold_context_status") for row in ctx_rows], [row.get("pred_context_status") for row in ctx_rows])
        metrics["context_status_macro_f1"] = _macro_f1([row.get("gold_context_status") for row in ctx_rows], [row.get("pred_context_status") for row in ctx_rows], CONTEXTS)
    else:
        metrics["context_status_skipped_reason"] = "No gold context_status labels were available."
    non_abstain = [row for row in rows if int(row.get("pred_hallucination", -1)) != -1]
    metrics["coverage_after_abstention"] = len(non_abstain) / len(rows)
    metrics["accuracy_after_abstention"] = _accuracy([row.get("gold_relation") for row in non_abstain], [row.get("pred_relation") for row in non_abstain]) if non_abstain else 0.0
    non_abstain_binary = [(int(row.get("gold_hallucination", -1)), int(row.get("pred_hallucination", -1))) for row in non_abstain if int(row.get("gold_hallucination", -1)) in {0, 1} and int(row.get("pred_hallucination", -1)) in {0, 1}]
    metrics["hallucination_f1_after_abstention"] = _binary_prf([g for g, _ in non_abstain_binary], [p for _, p in non_abstain_binary])[2]
    metrics["risk_error_correlation"] = _risk_error_correlation(rows)
    risk_curve = _risk_coverage_curve(rows)
    metrics["risk_coverage_curve"] = risk_curve
    metrics["risk_coverage_accuracy_auc"] = _curve_auc(risk_curve, "accuracy")
    metrics["selective_risk_auc"] = _curve_auc(risk_curve, "error_rate")
    return metrics


def _accuracy(gold: list[Any], pred: list[Any]) -> float:
    """Accuracy."""
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def _binary_prf(gold: list[int], pred: list[int]) -> tuple[float, float, float]:
    """Binary PRF for hallucination=1."""
    if not gold:
        return 0.0, 0.0, 0.0
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _label_f1(gold: list[Any], pred: list[Any], label: Any) -> float:
    """One-vs-rest F1."""
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _binary_macro_f1(gold: list[int], pred: list[int]) -> float:
    """Macro-F1 over hallucinated and non-hallucinated binary labels."""
    if not gold:
        return 0.0
    return (_label_f1(gold, pred, 0) + _label_f1(gold, pred, 1)) / 2.0


def _hallucination_score(row: dict[str, Any]) -> float:
    """Use inverse support score as a hallucination score for AUROC diagnostics."""
    if "calibrated_hallucination_probability" in row:
        try:
            return float(row.get("calibrated_hallucination_probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return 1.0 - float(row.get("score_original", 0.0) or 0.0)
    except (TypeError, ValueError):
        return float(row.get("risk_score", 0.0) or 0.0)


def _auroc(gold: list[int], scores: list[float]) -> float:
    """Compute AUROC by pairwise ranking without external dependencies."""
    positives = [score for label, score in zip(gold, scores) if label == 1]
    negatives = [score for label, score in zip(gold, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _brier_score(rows: list[dict[str, Any]]) -> float:
    """Brier score for hallucination probability; lower is better."""
    pairs = [
        (int(row.get("gold_hallucination", -1)), _clamp(_hallucination_score(row)))
        for row in rows
        if int(row.get("gold_hallucination", -1)) in {0, 1}
    ]
    if not pairs:
        return 0.0
    return sum((score - gold) ** 2 for gold, score in pairs) / len(pairs)


def _expected_calibration_error(rows: list[dict[str, Any]], bins: int = 10) -> float:
    """Expected calibration error for binary hallucination confidence; lower is better."""
    examples = []
    for row in rows:
        gold = int(row.get("gold_hallucination", -1))
        pred = int(row.get("pred_hallucination", -1))
        if gold not in {0, 1} or pred not in {0, 1}:
            continue
        p_hall = _clamp(_hallucination_score(row))
        confidence = p_hall if pred == 1 else 1.0 - p_hall
        correct = 1.0 if pred == gold else 0.0
        examples.append((confidence, correct))
    if not examples:
        return 0.0
    total = len(examples)
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [(conf, corr) for conf, corr in examples if (lo <= conf < hi) or (idx == bins - 1 and conf == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(conf for conf, _ in bucket) / len(bucket)
        avg_acc = sum(corr for _, corr in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_acc - avg_conf)
    return ece


def _risk_coverage_curve(rows: list[dict[str, Any]], points: int = 10) -> list[dict[str, float]]:
    """Return selective-prediction risk-coverage points sorted by increasing risk."""
    usable = [
        row
        for row in rows
        if int(row.get("gold_hallucination", -1)) in {0, 1} and int(row.get("pred_hallucination", -1)) in {0, 1}
    ]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda row: float(row.get("risk_score", 0.0) or 0.0))
    curve: list[dict[str, float]] = []
    for idx in range(1, points + 1):
        coverage = idx / points
        keep = ordered[: max(1, int(round(len(ordered) * coverage)))]
        gold = [int(row.get("gold_hallucination", -1)) for row in keep]
        pred = [int(row.get("pred_hallucination", -1)) for row in keep]
        acc = sum(g == p for g, p in zip(gold, pred)) / len(keep)
        f1 = _binary_prf(gold, pred)[2]
        curve.append({"coverage": coverage, "accuracy": acc, "error_rate": 1.0 - acc, "hallucination_f1": f1})
    return curve


def _curve_auc(curve: list[dict[str, float]], key: str) -> float:
    """Approximate area under a coverage curve by averaging reported points."""
    if not curve:
        return 0.0
    return sum(float(point.get(key, 0.0)) for point in curve) / len(curve)


def _clamp(value: float) -> float:
    """Clamp a score into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _macro_f1(gold: list[Any], pred: list[Any], labels: list[Any]) -> float:
    """Macro-F1."""
    return sum(_label_f1(gold, pred, label) for label in labels) / len(labels) if labels else 0.0


def _confusion_matrix(gold: list[Any], pred: list[Any], labels: list[Any]) -> dict[str, dict[str, int]]:
    """Confusion matrix."""
    matrix = {str(g): {str(p): 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        matrix.setdefault(str(g), {str(label): 0 for label in labels})
        matrix[str(g)][str(p)] = matrix[str(g)].get(str(p), 0) + 1
    return matrix


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    """Average field."""
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows) if rows else 0.0


def _bool(value: Any) -> bool:
    """Bool conversion."""
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}


def _risk_error_correlation(rows: list[dict[str, Any]]) -> float:
    """Pearson correlation between risk and relation error."""
    risks = [float(row.get("risk_score", 0.0) or 0.0) for row in rows]
    binary_rows = [row for row in rows if int(row.get("gold_hallucination", -1)) in {0, 1} and int(row.get("pred_hallucination", -1)) in {0, 1}]
    if binary_rows:
        risks = [float(row.get("risk_score", 0.0) or 0.0) for row in binary_rows]
        errors = [0.0 if int(row.get("gold_hallucination", -1)) == int(row.get("pred_hallucination", -1)) else 1.0 for row in binary_rows]
    else:
        errors = [0.0 if row.get("gold_relation") == row.get("pred_relation") else 1.0 for row in rows]
    mr = sum(risks) / len(risks)
    me = sum(errors) / len(errors)
    num = sum((r - mr) * (e - me) for r, e in zip(risks, errors))
    den = math.sqrt(sum((r - mr) ** 2 for r in risks) * sum((e - me) ** 2 for e in errors))
    return num / den if den else 0.0

"""Method comparison helpers."""

from __future__ import annotations

from typing import Any


def summarize_method_metrics(method_metrics: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten method metrics."""
    return [
        {
            "method": method,
            "accuracy": metrics.get("accuracy", 0.0),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "hallucination_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
            "hallucination_f1": metrics.get("hallucination_f1", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "hallucination_auroc": metrics.get("hallucination_auroc", 0.0),
            "hallucination_ece": metrics.get("hallucination_ece", 0.0),
            "hallucination_brier": metrics.get("hallucination_brier", 0.0),
            "context_status_macro_f1": metrics.get("context_status_macro_f1", 0.0),
            "attribution_macro_f1": metrics.get("attribution_macro_f1", 0.0),
            "abstain_rate": metrics.get("abstain_rate", 0.0),
            "accuracy_after_abstention": metrics.get("accuracy_after_abstention", 0.0),
            "hallucination_f1_after_abstention": metrics.get("hallucination_f1_after_abstention", 0.0),
            "risk_error_correlation": metrics.get("risk_error_correlation", 0.0),
            "risk_coverage_accuracy_auc": metrics.get("risk_coverage_accuracy_auc", 0.0),
            "selective_risk_auc": metrics.get("selective_risk_auc", 0.0),
        }
        for method, metrics in method_metrics
    ]

"""LaTeX exporters."""

from __future__ import annotations

from typing import Any


def metrics_to_latex(metrics: dict[str, Any], method: str) -> str:
    """Render one metrics row."""
    keys = ["hallucination_f1", "hallucination_macro_f1", "precision", "recall", "accuracy", "abstain_rate", "risk_error_correlation"]
    return f"{method} & " + " & ".join(f"{float(metrics.get(key, 0.0)):.3f}" for key in keys) + r" \\"


def ablation_to_latex(rows: list[dict[str, Any]]) -> str:
    """Render ablation table."""
    lines = ["Ablation & Acc. & Macro-F1 & Hallucination-F1 & Abstain \\\\"]
    for row in rows:
        lines.append(f"{row.get('ablation')} & {float(row.get('accuracy', 0.0)):.3f} & {float(row.get('macro_f1', 0.0)):.3f} & {float(row.get('hallucination_f1', 0.0)):.3f} & {float(row.get('abstain_rate', 0.0)):.3f} \\\\")
    return "\n".join(lines)


def risk_calibration_to_latex(metrics: dict[str, Any], method: str = "scad_rag") -> str:
    """Render risk calibration metrics as a LaTeX row."""
    keys = ["abstain_rate", "coverage_after_abstention", "accuracy_after_abstention", "risk_error_correlation", "hallucination_ece", "hallucination_brier", "risk_coverage_accuracy_auc"]
    return f"{method} & " + " & ".join(f"{float(metrics.get(key, 0.0)):.3f}" for key in keys) + r" \\"

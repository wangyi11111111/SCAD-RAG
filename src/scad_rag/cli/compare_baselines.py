"""Compare baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from scad_rag.attribution.rules import predict_from_audit
from scad_rag.baselines import ess_rule, lexical_overlap, majority, ml_feature_classifier, nli_only, refind_inspired, sc_gate_only, similarity_only
from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import apply_thresholds_path, load_config
from scad_rag.evaluation.compare_methods import summarize_method_metrics
from scad_rag.evaluation.diagnosis import paper_main_result_check, threshold_application_report
from scad_rag.evaluation.latex_export import metrics_to_latex
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.features.lexical_overlap import lexical_relevance
from scad_rag.schema import AuditResult
from collections import Counter

from scad_rag.utils.io import ensure_dir, read_csv, readable_existing_path, writable_file_path, write_csv

METHODS = ["majority", "lexical_overlap", "similarity_only", "nli_only", "ess_rule", "sc_gate_only", "ml_feature_classifier", "refind_inspired", "scad_rag"]


def main() -> int:
    """Run baseline suite."""
    parser = argparse.ArgumentParser(description="Compare SCAD-RAG baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--thresholds_path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = apply_thresholds_path(load_config(args.config), args.thresholds_path)
    if args.dataset:
        config["dataset"] = args.dataset
    if args.split:
        dataset = str(config.get("dataset", ""))
        config.setdefault(dataset, {})
        config[dataset]["split"] = args.split
        config[dataset]["processed_path"] = f"data/processed/{dataset}/{args.split}_processed.jsonl"
    root = ensure_dir(Path(config.get("output_dir", "experiments/runs")) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_baseline_compare")
    feature_rows = _feature_rows(config, root, args.max_samples)
    method_metrics = []
    method_predictions = {}
    for method in METHODS:
        rows_for_method = _apply_method(feature_rows, method, config.get("thresholds", {}), bool(config.get("risk_calibration", {}).get("enabled", True)))
        method_predictions[method] = rows_for_method
        method_metrics.append((method, compute_metrics(rows_for_method)))
    rows = summarize_method_metrics(method_metrics)
    if str(config.get("dataset", "")) == "ragtruth":
        rows = sorted(rows, key=lambda row: (float(row.get("hallucination_f1", 0.0)), float(row.get("hallucination_macro_f1", row.get("macro_f1", 0.0))), float(row.get("risk_error_correlation", 0.0)), float(row.get("accuracy_after_abstention", 0.0))), reverse=True)
    write_csv(root / "baseline_summary.csv", rows)
    (root / "baseline_summary.md").write_text(_summary(rows), encoding="utf-8")
    (root / "baseline_fairness_report.md").write_text(_fairness_report(config), encoding="utf-8")
    (root / "latex_table_main_results.txt").write_text(_main_latex(rows), encoding="utf-8")
    (root / "paper_main_result_check.md").write_text(paper_main_result_check(rows), encoding="utf-8")
    (root / "paper_metric_interpretation.md").write_text(_metric_interpretation(rows, method_predictions), encoding="utf-8")
    (root / "class_imbalance_report.md").write_text(_class_imbalance_report(feature_rows, method_predictions, rows), encoding="utf-8")
    writable_file_path(root / "threshold_application_report.md").write_text(threshold_application_report(config), encoding="utf-8")
    print(f"Wrote baseline comparison to {root}")
    return 0


def _feature_rows(config: dict, root: Path, max_samples: int | None) -> list[dict]:
    """Reuse latest SCAD predictions when available, otherwise run one feature cache."""
    latest = readable_existing_path(Path(config.get("output_dir", "experiments/runs")) / "latest_run.txt")
    if latest.exists():
        run_dir = Path(latest.read_text(encoding="utf-8").strip().lstrip("\ufeff"))
        predictions = run_dir / "predictions.csv"
        used_config = run_dir / "config_used.yaml"
        if predictions.exists() and _config_matches(config, used_config):
            return read_csv(predictions)
    result = run_experiment(config, "scad_rag", root / "feature_cache", max_samples=max_samples)
    return result["predictions"]


def _apply_method(rows: list[dict], method: str, thresholds: dict, risk_enabled: bool) -> list[dict]:
    """Apply one baseline decision rule to cached audit rows."""
    out = []
    for row in rows:
        updated = dict(row)
        audit = _audit_from_row(row)
        if method == "lexical_overlap":
            audit.max_relevance = lexical_relevance(str(row.get("claim_text", "")), str(row.get("best_evidence_text", "")))
        relation, hallucination, attribution, explanation = _predict_cached(method, audit, thresholds, risk_enabled)
        updated["pred_relation"] = relation
        updated["pred_hallucination"] = hallucination
        updated["pred_attribution"] = attribution
        updated["explanation"] = explanation
        out.append(updated)
    return out


def _config_matches(config: dict, used_config: Path) -> bool:
    """Return true when a cached run is compatible with the requested config."""
    if not used_config.exists():
        return False
    cached = load_config(used_config)
    return (
        cached.get("dataset") == config.get("dataset")
        and bool(cached.get("use_dummy_models", False)) == bool(config.get("use_dummy_models", False))
        and int(cached.get("top_k", 0)) == int(config.get("top_k", 0))
        and cached.get(str(config.get("dataset", "")), {}).get("processed_path") == config.get(str(config.get("dataset", "")), {}).get("processed_path")
        and cached.get("thresholds") == config.get("thresholds")
        and cached.get("evidence_chunking") == config.get("evidence_chunking")
    )


def _audit_from_row(row: dict) -> AuditResult:
    """Reconstruct audit features from a prediction row."""
    return AuditResult(
        best_evidence_id=str(row.get("best_evidence_id", "")),
        best_evidence_text=str(row.get("best_evidence_text", "")),
        hard_negative_evidence_id=str(row.get("hard_negative_evidence_id", "")),
        hard_negative_evidence_text=str(row.get("hard_negative_evidence_text", "")),
        max_relevance=_float(row.get("relevance_score", 0.0)),
        entailment_score=_float(row.get("entailment_score", 0.0)),
        neutral_score=_float(row.get("neutral_score", 1.0)),
        contradiction_score=_float(row.get("contradiction_score", 0.0)),
        coverage_score=_float(row.get("coverage_score", 0.0)),
        sufficient_context_score=_float(row.get("sufficient_context_score", 0.0)),
        context_status_original=str(row.get("pred_context_status", "Uncertain")),
        score_original=_float(row.get("score_original", 0.0)),
        score_removed=_float(row.get("score_removed", 0.0)),
        score_hard_negative=_float(row.get("score_hard_negative", 0.0)),
        evidence_dependency_delta=_float(row.get("evidence_dependency_delta", 0.0)),
        hard_negative_robustness_gap=_float(row.get("hard_negative_robustness_gap", 0.0)),
        has_conflicting_evidence=str(row.get("has_conflicting_evidence", "")).lower() in {"true", "1", "yes"},
        uncertainty_score=_float(row.get("uncertainty_score", 0.0)),
        risk_score=_float(row.get("risk_score", 0.0)),
    )


def _predict_cached(method: str, audit: AuditResult, thresholds: dict, risk_enabled: bool):
    """Run a cached-feature predictor."""
    if method == "scad_rag":
        return predict_from_audit(audit, thresholds, risk_enabled)
    return {
        "majority": majority.predict,
        "lexical_overlap": lexical_overlap.predict,
        "similarity_only": similarity_only.predict,
        "nli_only": nli_only.predict,
        "ess_rule": ess_rule.predict,
        "sc_gate_only": sc_gate_only.predict,
        "ml_feature_classifier": ml_feature_classifier.predict,
        "refind_inspired": refind_inspired.predict,
    }[method](audit, thresholds)


def _float(value) -> float:
    """Parse a float from CSV/object values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary(rows: list[dict]) -> str:
    """Markdown summary."""
    lines = ["# Baseline Summary", "", "RAGTruth rows are sorted by Hallucination-F1, then binary Macro-F1, risk-error correlation, and accuracy after abstention.", "", "| Method | Hall-F1 | Binary Macro-F1 | Precision | Recall | Accuracy | Risk Corr. | Abstain |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {float(row.get('hallucination_f1', 0.0)):.3f} | {float(row.get('hallucination_macro_f1', row.get('macro_f1', 0.0))):.3f} | {float(row.get('precision', 0.0)):.3f} | {float(row.get('recall', 0.0)):.3f} | {float(row.get('accuracy', 0.0)):.3f} | {float(row.get('risk_error_correlation', 0.0)):.3f} | {float(row.get('abstain_rate', 0.0)):.3f} |")
    return "\n".join(lines)


def _main_latex(rows: list[dict]) -> str:
    """Render main result table with imbalance notes."""
    lines = [
        "% Accuracy is auxiliary under severe class imbalance.",
        "% Hallucination-F1 is the primary metric for RAGTruth hallucination detection.",
        "Method & Hall-F1 & Binary Macro-F1 & Precision & Recall & Accuracy & Risk Corr. & Abstain \\\\",
    ]
    for row in rows:
        lines.append(
            f"{row.get('method')} & {float(row.get('hallucination_f1', 0.0)):.3f} & {float(row.get('hallucination_macro_f1', row.get('macro_f1', 0.0))):.3f} & {float(row.get('precision', 0.0)):.3f} & {float(row.get('recall', 0.0)):.3f} & {float(row.get('accuracy', 0.0)):.3f} & {float(row.get('risk_error_correlation', 0.0)):.3f} & {float(row.get('abstain_rate', 0.0)):.3f} \\\\"
        )
    return "\n".join(lines)


def _metric_interpretation(rows: list[dict], method_predictions: dict[str, list[dict]]) -> str:
    """Explain primary metric choice under class imbalance."""
    by_method = {row["method"]: row for row in rows}
    sim = by_method.get("similarity_only", {})
    pred_dist = Counter(row.get("pred_hallucination") for row in method_predictions.get("similarity_only", []))
    total = sum(pred_dist.values()) or 1
    non_hall_ratio = pred_dist.get(0, 0) / total
    trap = float(sim.get("accuracy", 0.0)) > 0.8 and float(sim.get("hallucination_f1", 0.0)) < 0.1
    lines = [
        "# Paper Metric Interpretation",
        "",
        "For RAGTruth, the primary task is sentence-level hallucination detection. Hallucination-F1 and binary Macro-F1 are more informative than Accuracy under severe class imbalance.",
        "",
        f"similarity_only accuracy: {float(sim.get('accuracy', 0.0)):.4f}",
        f"similarity_only hallucination-F1: {float(sim.get('hallucination_f1', 0.0)):.4f}",
        f"similarity_only non-hallucinated prediction ratio: {non_hall_ratio:.4f}",
        "",
    ]
    if trap:
        lines.append("High accuracy may be caused by majority-class bias. A method that predicts nearly all claims as non-hallucinated can score high Accuracy while failing to detect hallucinations.")
    else:
        lines.append("No severe majority-trap warning was triggered by the current heuristic.")
    return "\n".join(lines)


def _class_imbalance_report(feature_rows: list[dict], method_predictions: dict[str, list[dict]], rows: list[dict]) -> str:
    """Render class imbalance and majority-trap diagnostics."""
    lines = ["# Class Imbalance Report", ""]
    lines.append(f"gold_hallucination distribution: {dict(Counter(row.get('gold_hallucination') for row in feature_rows))}")
    lines.append(f"gold_relation distribution: {dict(Counter(row.get('gold_relation') for row in feature_rows))}")
    lines.append(f"gold_attribution distribution: {dict(Counter(row.get('gold_attribution') for row in feature_rows))}")
    lines.extend(["", "## Method Prediction Distributions", ""])
    metrics_by_method = {row["method"]: row for row in rows}
    for method, preds in method_predictions.items():
        pred_h = Counter(row.get("pred_hallucination") for row in preds)
        pred_rel = Counter(row.get("pred_relation") for row in preds)
        total = len(preds) or 1
        metrics = metrics_by_method.get(method, {})
        lines.extend(
            [
                f"### {method}",
                f"- pred_relation distribution: {dict(pred_rel)}",
                f"- pred_hallucination distribution: {dict(pred_h)}",
                f"- predicted hallucination ratio: {pred_h.get(1, 0) / total:.4f}",
                f"- precision / recall / f1: {float(metrics.get('precision', 0.0)):.4f} / {float(metrics.get('recall', 0.0)):.4f} / {float(metrics.get('hallucination_f1', 0.0)):.4f}",
            ]
        )
        if float(metrics.get("accuracy", 0.0)) > 0.8 and float(metrics.get("recall", 0.0)) < 0.2:
            lines.append("- High accuracy may be caused by majority-class bias.")
        lines.append("")
    return "\n".join(lines)


def _fairness_report(config: dict) -> str:
    """Render baseline fairness assumptions."""
    return "\n".join(
        [
            "# Baseline Fairness Report",
            "",
            f"Dataset: {config.get('dataset', 'toy')}",
            f"top_k: {config.get('top_k', 3)}",
            "",
            "All baselines in `compare_baselines` are run through the same `run_experiment` entrypoint.",
            "They share the same processed dataset, claim decomposition, evidence set, thresholds, and top_k.",
            "Only the final decision rule differs by method.",
            "",
            "Strict no-gold inference is inherited from config and defaults to true.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

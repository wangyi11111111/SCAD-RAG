"""Threshold tuning on validation splits only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import dump_yaml, load_config
from scad_rag.attribution.rules import predict_from_audit
from scad_rag.data.preprocess import prepare_dataset
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.schema import AuditResult
from scad_rag.utils.io import ensure_dir, write_csv


def main() -> int:
    """Tune thresholds on validation split only."""
    parser = argparse.ArgumentParser(description="Tune SCAD-RAG thresholds on validation data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    if args.split != "validation":
        raise ValueError("Threshold tuning is allowed only on split=validation.")
    base = load_config(args.config)
    if args.dataset:
        base["dataset"] = args.dataset
    if str(base.get("dataset", "")) == "ragtruth":
        base.setdefault("ragtruth", {})
        base["ragtruth"]["split"] = "train" if args.split == "validation" else args.split
        base["ragtruth"]["requested_split"] = args.split
        base["ragtruth"]["processed_path"] = f"data/processed/ragtruth/{args.split}_processed.jsonl"
        prepare_dataset(base, "ragtruth", max_samples=args.max_samples)
    root = ensure_dir(Path(base.get("output_dir", "experiments/runs")) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_threshold_tuning")
    feature_cache = run_experiment(base, "scad_rag", root / "feature_cache", max_samples=args.max_samples)["predictions"]
    if not feature_cache:
        raise RuntimeError(f"Threshold tuning produced no predictions for requested split={args.split}.")
    rows = []
    best_score = -1.0
    best_config = None
    grid = _grid(base.get("thresholds", {}))
    for idx, thresholds in enumerate(grid):
        cfg = deepcopy(base)
        cfg["thresholds"].update(thresholds)
        trial_predictions = _apply_thresholds(feature_cache, cfg["thresholds"], bool(cfg.get("risk_calibration", {}).get("enabled", True)))
        metrics = compute_metrics(trial_predictions)
        score = float(metrics.get("hallucination_f1", metrics.get("macro_f1", 0.0)))
        row = {"trial": idx, "hallucination_f1": score, "binary_macro_f1": metrics.get("hallucination_macro_f1", 0.0), "relation_macro_f1": metrics.get("macro_f1", 0.0), **thresholds}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_config = deepcopy(cfg["thresholds"])
    write_csv(root / "threshold_tuning_report.csv", rows)
    dump_yaml(best_config or base.get("thresholds", {}), root / "best_thresholds.yaml")
    (root / "threshold_tuning_report.md").write_text(_report(rows, best_config or {}, best_score, args.split, base.get("ragtruth", {}).get("split", args.split)), encoding="utf-8")
    (root / "latest_thresholds.txt").write_text(str(root / "best_thresholds.yaml"), encoding="utf-8")
    print(f"Wrote threshold tuning results to {root}")
    return 0


def _apply_thresholds(rows: list[dict], thresholds: dict, risk_enabled: bool) -> list[dict]:
    """Recompute decision labels from cached audit features."""
    predictions = []
    for row in rows:
        updated = dict(row)
        audit = AuditResult(
            max_relevance=float(row.get("relevance_score", 0.0) or 0.0),
            entailment_score=float(row.get("entailment_score", 0.0) or 0.0),
            neutral_score=float(row.get("neutral_score", 1.0) or 1.0),
            contradiction_score=float(row.get("contradiction_score", 0.0) or 0.0),
            coverage_score=float(row.get("coverage_score", 0.0) or 0.0),
            sufficient_context_score=float(row.get("sufficient_context_score", 0.0) or 0.0),
            context_status_original=str(row.get("pred_context_status", "Uncertain")),
            score_original=float(row.get("score_original", 0.0) or 0.0),
            score_removed=float(row.get("score_removed", 0.0) or 0.0),
            score_hard_negative=float(row.get("score_hard_negative", 0.0) or 0.0),
            evidence_dependency_delta=float(row.get("evidence_dependency_delta", 0.0) or 0.0),
            hard_negative_robustness_gap=float(row.get("hard_negative_robustness_gap", 0.0) or 0.0),
            has_conflicting_evidence=bool(row.get("has_conflicting_evidence", False)),
            uncertainty_score=float(row.get("uncertainty_score", 0.0) or 0.0),
            risk_score=float(row.get("risk_score", 0.0) or 0.0),
        )
        relation, hallucination, attribution, explanation = predict_from_audit(audit, thresholds, risk_enabled)
        updated["pred_relation"] = relation
        updated["pred_hallucination"] = hallucination
        updated["pred_attribution"] = attribution
        updated["explanation"] = explanation
        predictions.append(updated)
    return predictions


def _grid(thresholds: dict) -> list[dict]:
    """Small reproducible threshold grid."""
    supported_values = sorted({float(thresholds.get("supported_threshold", 0.60)), 0.55, 0.65})
    entail_values = sorted({float(thresholds.get("entailment_threshold", 0.50)), 0.45, 0.55})
    risk_values = sorted({float(thresholds.get("risk_threshold", 0.70)), 0.65, 0.75})
    return [
        {"supported_threshold": s, "entailment_threshold": e, "risk_threshold": r}
        for s in supported_values
        for e in entail_values
        for r in risk_values
    ]


def _report(rows: list[dict], best: dict, best_score: float, requested_split: str, actual_split: str) -> str:
    """Render markdown report."""
    lines = [
        "# Threshold Tuning Report",
        "",
        f"Requested split: {requested_split}",
        f"Actual calibration split: {actual_split}",
        "RAGTruth public data has no validation split; `validation` is mapped to train-side calibration and never to test.",
        f"Best hallucination_f1: {best_score:.4f}",
        "",
        "## Best Thresholds",
        "",
    ]
    for key, value in best.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", f"Trials: {len(rows)}"])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

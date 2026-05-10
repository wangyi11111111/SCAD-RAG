"""Run ablation experiments."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from scad_rag.attribution.rules import predict_from_audit
from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import apply_thresholds_path, deep_update, load_config
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.evaluation.latex_export import ablation_to_latex
from scad_rag.schema import AuditResult
from scad_rag.utils.io import ensure_dir, read_csv, readable_existing_path, write_csv


def main() -> int:
    """Run all configured ablations."""
    parser = argparse.ArgumentParser(description="Run SCAD-RAG ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--thresholds_path", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    abl = load_config(args.config)
    base_config = os.environ.get("SCAD_RAG_ABLATION_BASE_CONFIG", abl.get("base_config", "configs/quick_test.yaml"))
    base = apply_thresholds_path(load_config(base_config), args.thresholds_path or os.environ.get("SCAD_RAG_THRESHOLDS_PATH"))
    if os.environ.get("SCAD_RAG_ABLATION_DATASET"):
        base["dataset"] = os.environ["SCAD_RAG_ABLATION_DATASET"]
    split = args.split or os.environ.get("SCAD_RAG_SPLIT")
    if split and str(base.get("dataset", "")) == "ragtruth":
        base.setdefault("ragtruth", {})
        base["ragtruth"]["split"] = split
        base["ragtruth"]["processed_path"] = f"data/processed/ragtruth/{split}_processed.jsonl"
    max_samples = args.max_samples
    if max_samples is None and os.environ.get("SCAD_RAG_MAX_SAMPLES"):
        max_samples = int(os.environ["SCAD_RAG_MAX_SAMPLES"])
    root = ensure_dir(Path(abl.get("output_dir", "experiments/runs")) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ablation")
    feature_rows = _feature_rows(base, root, max_samples)
    rows = []
    for name in abl.get("ablations", []):
        cfg = apply_ablation(base, str(name))
        predictions = _apply_ablation_to_rows(feature_rows, cfg, str(name))
        metrics = compute_metrics(predictions)
        rows.append({"ablation": name, "accuracy": metrics.get("accuracy", 0.0), "macro_f1": metrics.get("macro_f1", 0.0), "hallucination_f1": metrics.get("hallucination_f1", 0.0), "abstain_rate": metrics.get("abstain_rate", 0.0)})
    write_csv(root / "ablation_summary.csv", rows)
    (root / "ablation_summary.md").write_text(_summary(rows), encoding="utf-8")
    (root / "latex_table_ablation.txt").write_text(ablation_to_latex(rows), encoding="utf-8")
    print(f"Wrote ablation summary to {root}")
    return 0


def apply_ablation(config: dict, name: str) -> dict:
    """Apply ablation config."""
    if name == "full":
        return deep_update(config, {})
    if name == "w_o_sufficient_context":
        return deep_update(config, {"sufficient_context": {"enabled": False}, "scad_weights": {"eta": 0.0}})
    if name == "w_o_relevance":
        return deep_update(config, {"scad_weights": {"alpha": 0.0}})
    if name == "w_o_nli":
        return deep_update(config, {"scad_weights": {"beta": 0.0, "gamma": 0.0}})
    if name == "w_o_coverage":
        return deep_update(config, {"scad_weights": {"delta": 0.0}})
    if name == "w_o_counterfactual":
        return deep_update(config, {"counterfactual": {"enable_removal": False, "enable_hard_negative": False, "enable_contradiction_probe": False}})
    if name == "w_o_removal":
        return deep_update(config, {"counterfactual": {"enable_removal": False}})
    if name == "w_o_hard_negative":
        return deep_update(config, {"counterfactual": {"enable_hard_negative": False}})
    if name == "w_o_contradiction_probe":
        return deep_update(config, {"counterfactual": {"enable_contradiction_probe": False}})
    if name == "w_o_risk_calibration":
        return deep_update(config, {"risk_calibration": {"enabled": False}})
    raise ValueError(f"Unknown ablation: {name}")


def _feature_rows(config: dict, root: Path, max_samples: int | None) -> list[dict]:
    """Reuse latest SCAD predictions when available, otherwise run one feature cache."""
    latest = readable_existing_path(Path(config.get("output_dir", "experiments/runs")) / "latest_run.txt")
    if latest.exists():
        run_dir = Path(latest.read_text(encoding="utf-8").strip().lstrip("\ufeff"))
        predictions = run_dir / "predictions.csv"
        used_config = run_dir / "config_used.yaml"
        if predictions.exists() and _config_matches(config, used_config):
            return read_csv(predictions)
    return run_experiment(config, "scad_rag", root / "feature_cache", max_samples=max_samples)["predictions"]


def _apply_ablation_to_rows(rows: list[dict], config: dict, name: str) -> list[dict]:
    """Apply feature-level ablation to cached full SCAD rows."""
    out = []
    for row in rows:
        updated = dict(row)
        audit = _audit_from_row(row)
        weights = config.get("scad_weights", {})
        if name in {"w_o_sufficient_context", "w_o_relevance", "w_o_nli", "w_o_coverage"}:
            audit.score_original = _score_from_components(audit, weights)
        if name == "w_o_counterfactual":
            audit.score_removed = audit.score_original
            audit.score_hard_negative = audit.score_original
            audit.evidence_dependency_delta = 0.0
            audit.hard_negative_robustness_gap = 0.0
            audit.has_conflicting_evidence = False
        if name == "w_o_removal":
            audit.score_removed = audit.score_original
            audit.evidence_dependency_delta = 0.0
        if name == "w_o_hard_negative":
            audit.score_hard_negative = audit.score_original
            audit.hard_negative_robustness_gap = 0.0
        if name == "w_o_contradiction_probe":
            audit.has_conflicting_evidence = False
            audit.max_contradiction_score = 0.0
        risk_enabled = bool(config.get("risk_calibration", {}).get("enabled", True))
        relation, hallucination, attribution, explanation = predict_from_audit(audit, config.get("thresholds", {}), risk_enabled)
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


def _score_from_components(audit: AuditResult, weights: dict) -> float:
    """Recompute original score from available component features."""
    return (
        float(weights.get("alpha", 0.25)) * audit.max_relevance
        + float(weights.get("beta", 0.35)) * audit.entailment_score
        - float(weights.get("gamma", 0.20)) * audit.contradiction_score
        + float(weights.get("delta", 0.10)) * audit.coverage_score
        + float(weights.get("eta", 0.10)) * audit.sufficient_context_score
    )


def _audit_from_row(row: dict) -> AuditResult:
    """Reconstruct audit features from a prediction row."""
    return AuditResult(
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


def _float(value) -> float:
    """Parse a float from CSV/object values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary(rows: list[dict]) -> str:
    """Markdown table."""
    lines = ["# Ablation Summary", "", "| Ablation | Accuracy | Macro-F1 | Hallucination-F1 | Abstain |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['ablation']} | {float(row['accuracy']):.3f} | {float(row['macro_f1']):.3f} | {float(row['hallucination_f1']):.3f} | {float(row['abstain_rate']):.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

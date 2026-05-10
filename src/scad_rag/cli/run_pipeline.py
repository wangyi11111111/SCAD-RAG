"""Run SCAD-RAG and baselines."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from scad_rag.attribution.rules import predict_from_audit
from scad_rag.baselines import ess_rule, lettucedetect_adapter, lexical_overlap, majority, ml_feature_classifier, nli_only, refind_inspired, sc_gate_only, similarity_only
from scad_rag.config import apply_thresholds_path, dump_yaml, load_config
from scad_rag.data.preprocess import load_samples
from scad_rag.evaluation.case_study import generate_case_studies
from scad_rag.evaluation.diagnosis import evidence_quality_report, experiment_diagnosis, manual_check_instruction, manual_check_template, threshold_application_report
from scad_rag.evaluation.error_analysis import generate_error_analysis
from scad_rag.evaluation.latex_export import metrics_to_latex
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.features.counterfactual_audit import audit_claim
from scad_rag.models.dummy_models import DummyEmbedder, DummyNLIModel
from scad_rag.models.embedder import build_embedder
from scad_rag.models.nli import build_nli_model
from scad_rag.schema import AuditResult, Evidence, decompose_sample, inference_claim_view
from scad_rag.utils.io import ensure_dir, writable_file_path, write_csv, write_json, write_jsonl
from scad_rag.utils.logging import get_logger
from scad_rag.utils.seed import set_seed

METHODS = {"majority", "lexical_overlap", "similarity_only", "nli_only", "ess_rule", "sc_gate_only", "ml_feature_classifier", "lettucedetect", "refind_inspired", "scad_rag"}

PREDICTION_COLUMNS = [
    "sample_id",
    "claim_id",
    "question",
    "claim_text",
    "best_evidence_id",
    "best_evidence_text",
    "hard_negative_evidence_id",
    "hard_negative_evidence_text",
    "gold_context_status",
    "pred_context_status",
    "gold_relation",
    "pred_relation",
    "gold_hallucination",
    "pred_hallucination",
    "gold_attribution",
    "pred_attribution",
    "relevance_score",
    "entailment_score",
    "neutral_score",
    "contradiction_score",
    "coverage_score",
    "sufficient_context_score",
    "score_original",
    "score_removed",
    "score_hard_negative",
    "evidence_dependency_delta",
    "hard_negative_robustness_gap",
    "has_conflicting_evidence",
    "uncertainty_score",
    "risk_score",
    "nli_reliability_score",
    "explanation",
]


def run_experiment(config: dict, method: str, run_dir: str | Path | None = None, max_samples: int | None = None) -> dict[str, Any]:
    """Run one experiment and write artifacts."""
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    set_seed(int(config.get("seed", 42)))
    runtime = _method_config(config, method)
    samples = load_samples(runtime, str(runtime.get("dataset", "toy")), max_samples=max_samples)
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(runtime.get("output_dir", "experiments/runs")) / f"{timestamp}_{method}"
    run_path = ensure_dir(run_dir)
    embedder, nli = _models(runtime, method)
    pool = [Evidence(f"{sample.id}:{ev.id}", ev.text, ev.type) for sample in samples for ev in sample.evidences]
    predictions: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for sample in samples:
        for claim in decompose_sample(sample):
            feature_claim = inference_claim_view(claim, bool(runtime.get("strict_no_gold_inference", True)))
            audit = _empty_audit(claim.evidences) if method == "majority" else audit_claim(
                feature_claim,
                embedder,
                nli,
                int(runtime.get("top_k", 3)),
                runtime.get("scad_weights", {}),
                runtime.get("thresholds", {}),
                runtime.get("counterfactual", {}),
                runtime.get("risk_calibration", {}),
                pool,
                allow_gold_hard_negatives=bool(runtime.get("hard_negative", {}).get("allow_gold_labels", False)),
            )
            pred_relation, pred_hall, pred_attr, explanation = _predict(method, audit, runtime.get("thresholds", {}), runtime.get("risk_calibration", {}).get("enabled", True))
            row = {
                "sample_id": claim.sample_id,
                "claim_id": claim.claim_id,
                "question": claim.question,
                "claim_text": claim.claim_text,
                "best_evidence_id": audit.best_evidence_id,
                "best_evidence_text": audit.best_evidence_text,
                "hard_negative_evidence_id": audit.hard_negative_evidence_id,
                "hard_negative_evidence_text": audit.hard_negative_evidence_text,
                "gold_context_status": claim.gold_context_status,
                "pred_context_status": audit.context_status_original,
                "gold_relation": claim.gold_relation,
                "pred_relation": pred_relation,
                "gold_hallucination": claim.gold_hallucination,
                "pred_hallucination": pred_hall,
                "gold_attribution": claim.gold_attribution,
                "pred_attribution": pred_attr,
                "relevance_score": audit.max_relevance,
                "entailment_score": audit.entailment_score,
                "neutral_score": audit.neutral_score,
                "contradiction_score": audit.contradiction_score,
                "coverage_score": audit.coverage_score,
                "sufficient_context_score": audit.sufficient_context_score,
                "score_original": audit.score_original,
                "score_removed": audit.score_removed,
                "score_hard_negative": audit.score_hard_negative,
                "evidence_dependency_delta": audit.evidence_dependency_delta,
                "hard_negative_robustness_gap": audit.hard_negative_robustness_gap,
                "has_conflicting_evidence": audit.has_conflicting_evidence,
                "uncertainty_score": audit.uncertainty_score,
                "risk_score": audit.risk_score,
                "nli_reliability_score": audit.nli_reliability_score,
                "explanation": explanation,
                "dependency_stability_label": audit.dependency_stability_label,
                "max_contradiction_score": audit.max_contradiction_score,
                "contradiction_evidence_id": audit.contradiction_evidence_id,
                "contradiction_evidence_text": audit.contradiction_evidence_text,
            }
            predictions.append(row)
            context_rows.append({"sample_id": claim.sample_id, "claim_id": claim.claim_id, "gold_context_status": claim.gold_context_status, "pred_context_status": audit.context_status_original, "sufficient_context_score": audit.sufficient_context_score})
            audit_rows.append({key: row[key] for key in ["sample_id", "claim_id", "score_original", "score_removed", "score_hard_negative", "evidence_dependency_delta", "hard_negative_robustness_gap", "has_conflicting_evidence", "dependency_stability_label", "max_contradiction_score", "contradiction_evidence_id", "contradiction_evidence_text"]})
            risk_rows.append({"sample_id": claim.sample_id, "claim_id": claim.claim_id, "uncertainty_score": audit.uncertainty_score, "risk_score": audit.risk_score, "nli_reliability_score": audit.nli_reliability_score, "pred_relation": pred_relation, "pred_hallucination": pred_hall, "pred_attribution": pred_attr})
            evidence_rows.extend(_evidence_rows(row, audit))
    metrics = compute_metrics(predictions)
    dump_yaml(runtime, run_path / "config_used.yaml")
    write_jsonl(run_path / "predictions.jsonl", predictions)
    write_csv(run_path / "predictions.csv", predictions, PREDICTION_COLUMNS)
    write_csv(run_path / "sentence_level_results.csv", predictions, PREDICTION_COLUMNS)
    write_csv(run_path / "claim_evidence_scores.csv", evidence_rows)
    write_csv(run_path / "sufficient_context_results.csv", context_rows)
    write_csv(run_path / "counterfactual_audit.csv", audit_rows)
    write_csv(run_path / "risk_calibration.csv", risk_rows)
    write_json(run_path / "metrics.json", metrics)
    writable_file_path(run_path / "case_studies.md").write_text(generate_case_studies(predictions), encoding="utf-8")
    writable_file_path(run_path / "error_analysis.md").write_text(generate_error_analysis(predictions), encoding="utf-8")
    writable_file_path(run_path / "latex_table_metrics.txt").write_text(metrics_to_latex(metrics, method), encoding="utf-8")
    manual_examples = _manual_check_examples(predictions, int(runtime.get("seed", 42)))
    write_csv(run_path / "examples_for_manual_check.csv", manual_examples)
    write_csv(run_path / "manual_check_template.csv", manual_check_template(manual_examples))
    manual_100 = _manual_check_100(predictions, int(runtime.get("seed", 42)))
    write_csv(run_path / "manual_check_100.csv", manual_check_template(manual_100))
    writable_file_path(run_path / "manual_check_sampling_report.md").write_text(_manual_check_sampling_report(manual_100), encoding="utf-8")
    writable_file_path(run_path / "manual_check_instruction.md").write_text(manual_check_instruction(), encoding="utf-8")
    writable_file_path(run_path / "experiment_diagnosis.md").write_text(experiment_diagnosis(predictions, metrics, runtime), encoding="utf-8")
    writable_file_path(run_path / "evidence_quality_report.md").write_text(evidence_quality_report(samples, predictions, runtime), encoding="utf-8")
    writable_file_path(run_path / "threshold_application_report.md").write_text(threshold_application_report(runtime), encoding="utf-8")
    latest = writable_file_path(Path(runtime.get("output_dir", "experiments/runs")) / "latest_run.txt")
    latest.write_text(str(run_path), encoding="utf-8")
    get_logger(__name__).info("Wrote %s predictions to %s", len(predictions), run_path)
    return {"run_dir": str(run_path), "metrics": metrics, "predictions": predictions}


def _method_config(config: dict, method: str) -> dict:
    """Apply method-specific config changes."""
    cfg = deepcopy(config)
    cfg["method"] = method
    if method == "ess_rule":
        cfg["scad_weights"] = deepcopy(cfg.get("scad_weights", {}))
        cfg["scad_weights"]["eta"] = 0.0
        cfg["counterfactual"] = {"enable_removal": False, "enable_hard_negative": False, "enable_contradiction_probe": False}
        cfg["risk_calibration"] = {"enabled": False}
    if method == "sc_gate_only":
        cfg["counterfactual"] = {"enable_removal": False, "enable_hard_negative": False, "enable_contradiction_probe": True}
    return cfg


def _models(config: dict, method: str):
    """Build required models."""
    if method in {"majority", "lexical_overlap"}:
        return DummyEmbedder(), DummyNLIModel()
    if method == "similarity_only":
        return build_embedder(config), DummyNLIModel()
    return build_embedder(config), build_nli_model(config)


def _predict(method: str, audit: AuditResult, thresholds: dict, risk_enabled: bool):
    """Dispatch predictor."""
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
        "lettucedetect": lettucedetect_adapter.predict,
        "refind_inspired": refind_inspired.predict,
    }[method](audit, thresholds)


def _empty_audit(evidences: list[Evidence]) -> AuditResult:
    """Return an empty audit for majority baseline."""
    if evidences:
        return AuditResult(best_evidence_id=evidences[0].id, best_evidence_text=evidences[0].text)
    return AuditResult()


def _evidence_rows(base: dict[str, Any], audit: AuditResult) -> list[dict[str, Any]]:
    """Flatten evidence scores."""
    return [
        {
            "sample_id": base["sample_id"],
            "claim_id": base["claim_id"],
            "evidence_id": score.evidence_id,
            "evidence_text": score.evidence_text,
            "evidence_type": score.evidence_type,
            "relevance_score": score.relevance_score,
            "entailment_score": score.entailment_score,
            "neutral_score": score.neutral_score,
            "contradiction_score": score.contradiction_score,
            "coverage_score": score.coverage_score,
            "sufficient_context_score": score.sufficient_context_score,
            "context_status": score.context_status,
            "scad_score": score.scad_score,
        }
        for score in audit.evidence_scores
    ]


def _manual_check_examples(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Sample examples for manual inspection by predicted attribution type."""
    import random

    rng = random.Random(seed)
    buckets = [
        ("predicted_supported", lambda row: row.get("pred_relation") == "Supported"),
        ("retrieval_insufficient", lambda row: row.get("pred_attribution") == "Retrieval-insufficient"),
        ("generation_inconsistent", lambda row: row.get("pred_attribution") == "Generation-inconsistent"),
        ("evidence_contradicted", lambda row: row.get("pred_attribution") == "Evidence-contradicted"),
        ("high_risk_abstain", lambda row: row.get("pred_attribution") == "High-risk-abstain"),
    ]
    sampled: list[dict[str, Any]] = []
    for bucket_name, predicate in buckets:
        candidates = [dict(row, manual_check_bucket=bucket_name) for row in rows if predicate(row)]
        rng.shuffle(candidates)
        sampled.extend(candidates[:20])
    return sampled


def _manual_check_100(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Sample up to 100 examples across predicted attribution buckets."""
    import random

    rng = random.Random(seed)
    buckets = [
        ("predicted_supported", lambda row: row.get("pred_relation") == "Supported"),
        ("retrieval_insufficient", lambda row: row.get("pred_attribution") == "Retrieval-insufficient"),
        ("generation_inconsistent", lambda row: row.get("pred_attribution") == "Generation-inconsistent"),
        ("evidence_contradicted", lambda row: row.get("pred_attribution") == "Evidence-contradicted"),
        ("high_risk_abstain", lambda row: row.get("pred_attribution") == "High-risk-abstain"),
    ]
    selected: list[dict[str, Any]] = []
    used = set()
    for bucket_name, predicate in buckets:
        candidates = [dict(row, manual_check_bucket=bucket_name) for row in rows if predicate(row)]
        rng.shuffle(candidates)
        for row in candidates:
            key = (row.get("sample_id"), row.get("claim_id"), bucket_name)
            if key not in used:
                selected.append(row)
                used.add(key)
            if sum(item.get("manual_check_bucket") == bucket_name for item in selected) >= 20:
                break
    if len(selected) < 100:
        fillers = [dict(row, manual_check_bucket="filler") for row in rows]
        rng.shuffle(fillers)
        seen_pairs = {(row.get("sample_id"), row.get("claim_id")) for row in selected}
        for row in fillers:
            pair = (row.get("sample_id"), row.get("claim_id"))
            if pair in seen_pairs:
                continue
            selected.append(row)
            seen_pairs.add(pair)
            if len(selected) >= 100:
                break
    return selected[:100]


def _manual_check_sampling_report(rows: list[dict[str, Any]]) -> str:
    """Report manual-check bucket counts."""
    from collections import Counter

    counts = Counter(row.get("manual_check_bucket", "unknown") for row in rows)
    lines = ["# Manual Check Sampling Report", "", f"Total rows: {len(rows)}", f"Bucket counts: {dict(counts)}"]
    missing = [name for name in ["predicted_supported", "retrieval_insufficient", "generation_inconsistent", "evidence_contradicted", "high_risk_abstain"] if counts.get(name, 0) < 20]
    if missing:
        lines.append(f"Some requested buckets had fewer than 20 rows and were backfilled with filler examples: {missing}")
    else:
        lines.append("All requested buckets reached 20 rows.")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run SCAD-RAG pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--thresholds_path", default=None)
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
    result = run_experiment(config, args.method or str(config.get("method", "scad_rag")), max_samples=args.max_samples)
    print(json.dumps({"run_dir": result["run_dir"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

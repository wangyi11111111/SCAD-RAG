"""Research experiment diagnosis reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


def experiment_diagnosis(rows: list[dict[str, Any]], metrics: dict[str, Any], config: dict) -> str:
    """Render a per-run experiment diagnosis report."""
    evidence_counts = Counter()
    claim_counts = Counter(row.get("sample_id") for row in rows)
    for row in rows:
        if row.get("best_evidence_text"):
            evidence_counts["has_best_evidence"] += 1
        if row.get("hard_negative_evidence_text"):
            evidence_counts["has_hard_negative"] += 1
    relation_dist = Counter(row.get("pred_relation") for row in rows)
    attribution_dist = Counter(row.get("pred_attribution") for row in rows)
    gold_relation_dist = Counter(row.get("gold_relation") for row in rows)
    warnings = []
    if metrics.get("abstain_rate", 0.0) > 0.4:
        warnings.append("Abstain rate is high; consider increasing risk_threshold or uncertainty_threshold.")
    if metrics.get("average_hard_negative_robustness_gap", 0.0) < 0.05:
        warnings.append("Hard-negative robustness gap is very low; hard-negative construction or thresholds may be weak.")
    if metrics.get("contradiction_probe_rate", 0.0) == 0.0:
        warnings.append("No conflict probes fired; NLI contradiction threshold or evidence mapping may need inspection.")
    if not warnings:
        warnings.append("No obvious automatic anomaly detected.")
    lines = [
        "# Experiment Diagnosis",
        "",
        "## Data Conversion",
        "",
        f"Dataset: {config.get('dataset', 'unknown')}",
        "Data conversion succeeded if this report was generated after `run_pipeline`.",
        "",
        "## Size",
        "",
        f"Claim count: {len(rows)}",
        f"Sample count: {len(claim_counts)}",
        f"Claims per sample distribution: {dict(Counter(claim_counts.values()))}",
        f"Evidence availability: {dict(evidence_counts)}",
        "",
        "## Label Distributions",
        "",
        f"Gold relation distribution: {dict(gold_relation_dist)}",
        f"Predicted relation distribution: {dict(relation_dist)}",
        f"Predicted attribution distribution: {dict(attribution_dist)}",
        "",
        "## Risk",
        "",
        f"Abstain rate: {float(metrics.get('abstain_rate', 0.0)):.4f}",
        f"Accuracy after abstention: {float(metrics.get('accuracy_after_abstention', 0.0)):.4f}",
        f"Coverage after abstention: {float(metrics.get('coverage_after_abstention', 0.0)):.4f}",
        f"Risk-error correlation: {float(metrics.get('risk_error_correlation', 0.0)):.4f}",
        "",
        "## SCAD-RAG vs Baselines",
        "",
        "Run `compare_baselines` and inspect `paper_main_result_check.md` for direct baseline comparisons.",
        "",
        "## Ablation Story",
        "",
        "Run `run_ablation` and inspect `ablation_summary.md`. The method story is supported when full SCAD-RAG improves over removing sufficient context, counterfactual audit, hard negative audit, and risk calibration.",
        "",
        "## Automatic Anomaly Check",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Next Steps", "", "- Inspect `manual_check_template.csv` for human validation.", "- If RAGTruth mapping is coarse, review `data/processed/ragtruth/conversion_report.md`.", "- Tune thresholds only on validation split before test evaluation."])
    return "\n".join(lines)


def manual_check_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert sampled manual-check examples into annotation template rows."""
    output = []
    for row in rows:
        output.append(
            {
                "sample_id": row.get("sample_id", ""),
                "claim_id": row.get("claim_id", ""),
                "question": row.get("question", ""),
                "claim_text": row.get("claim_text", ""),
                "best_evidence_text": row.get("best_evidence_text", ""),
                "hard_negative_text": row.get("hard_negative_evidence_text", ""),
                "pred_relation": row.get("pred_relation", ""),
                "pred_attribution": row.get("pred_attribution", ""),
                "explanation": row.get("explanation", ""),
                "human_judgment_correct": "",
                "human_hallucination_label": "",
                "human_relation_label": "",
                "human_attribution_label": "",
                "human_comment": "",
            }
        )
    return output


def manual_check_instruction() -> str:
    """Return manual annotation instructions."""
    return "\n".join(
        [
            "# Manual Check Instructions",
            "",
            "Annotate each row by reading the claim and the evidence shown in the CSV.",
            "",
            "1. Decide whether the claim is fully supported by the evidence.",
            "2. Mark hallucination as 0 only when the evidence directly supports the claim; mark 1 when the claim is unsupported, over-generated, or contradicted.",
            "3. If hallucinated, choose the most likely relation: Insufficient when evidence is missing or too weak; Contradicted when evidence directly conflicts; Supported only when fully supported; Uncertain when manual judgment is not reliable.",
            "4. Choose attribution: Retrieval-insufficient for unrelated/missing evidence; Generation-inconsistent for relevant evidence that does not entail the claim; Evidence-contradicted for direct conflict; High-risk-abstain when a human cannot decide confidently.",
            "5. Use `human_comment` for ambiguity, evidence truncation, or label-mapping concerns.",
        ]
    )


def evidence_quality_report(samples: list[Any], rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    """Render evidence construction and truncation diagnostics."""
    evidence_counts = [len(getattr(sample, "evidences", [])) for sample in samples]
    evidence_lengths = [len(ev.text) for sample in samples for ev in getattr(sample, "evidences", [])]
    token_lengths = [len(str(row.get("best_evidence_text", "")).split()) for row in rows]
    relevance_all = [float(row.get("relevance_score", 0.0) or 0.0) for row in rows]
    hall_rel = [float(row.get("relevance_score", 0.0) or 0.0) for row in rows if int(row.get("gold_hallucination", -1)) == 1]
    non_rel = [float(row.get("relevance_score", 0.0) or 0.0) for row in rows if int(row.get("gold_hallucination", -1)) == 0]
    long_evidence = sum(length > int(config.get("evidence_chunking", {}).get("max_chunk_chars", 500)) for length in evidence_lengths)
    trunc = sum(length > 512 for length in token_lengths)
    lines = [
        "# Evidence Quality Report",
        "",
        f"Evidence chunking enabled: {bool(config.get('evidence_chunking', {}).get('enabled', False))}",
        f"max_chunk_chars: {config.get('evidence_chunking', {}).get('max_chunk_chars', 500)}",
        f"top_k_chunks: {config.get('evidence_chunking', {}).get('top_k_chunks', 5)}",
        "",
        f"Evidence count distribution per sample: {dict(Counter(evidence_counts))}",
        f"Average evidence length (chars): {_avg(evidence_lengths):.2f}",
        f"Evidence chunks longer than configured max chars: {long_evidence}",
        f"Average NLI premise token length: {_avg(token_lengths):.2f}",
        f"Approximate NLI truncation ratio (>512 whitespace tokens): {(trunc / len(token_lengths) if token_lengths else 0.0):.4f}",
        f"Average top-evidence relevance: {_avg(relevance_all):.4f}",
        f"Average relevance for hallucinated claims: {_avg(hall_rel):.4f}",
        f"Average relevance for non-hallucinated claims: {_avg(non_rel):.4f}",
        "",
    ]
    if trunc:
        lines.append("Warning: Some NLI premises are likely to be truncated. Sentence/passage chunking should remain enabled.")
    else:
        lines.append("No severe NLI truncation risk was detected by the whitespace-token heuristic.")
    return "\n".join(lines)


def threshold_application_report(config: dict[str, Any]) -> str:
    """Render threshold application status."""
    app = config.get("_threshold_application", {})
    lines = [
        "# Threshold Application Report",
        "",
        f"Requested thresholds_path: {config.get('thresholds_path', '')}",
        f"Applied: {bool(app.get('applied', False))}",
        f"Applied file: {app.get('thresholds_path', '')}",
        "",
        "## Run Pipeline Thresholds",
        "",
    ]
    for key, value in config.get("thresholds", {}).items():
        lines.append(f"- {key}: {value}")
    if app.get("applied"):
        lines.extend(["", "Threshold tuning output was successfully applied to this run."])
    else:
        lines.extend(["", f"Threshold tuning output was not applied: {app.get('reason', 'unknown')}"])
    return "\n".join(lines)


def risk_calibration_diagnosis(metrics: dict[str, Any]) -> str:
    """Render risk calibration diagnosis."""
    abstain = float(metrics.get("abstain_rate", 0.0))
    acc = float(metrics.get("accuracy", 0.0))
    acc_after = float(metrics.get("accuracy_after_abstention", 0.0))
    coverage = float(metrics.get("coverage_after_abstention", 0.0))
    corr = float(metrics.get("risk_error_correlation", 0.0))
    ece = float(metrics.get("hallucination_ece", 0.0))
    brier = float(metrics.get("hallucination_brier", 0.0))
    risk_auc = float(metrics.get("risk_coverage_accuracy_auc", 0.0))
    selective_risk_auc = float(metrics.get("selective_risk_auc", 0.0))
    verdict = "Risk calibration appears useful because accuracy improves after abstention." if acc_after > acc else "Risk calibration does not yet improve accuracy after abstention."
    if abstain > 0.4:
        action = "Abstain rate is high; consider increasing risk_threshold or uncertainty_threshold."
    elif abstain < 0.02:
        action = "Abstain rate is very low; consider lowering risk_threshold if risky cases remain unfiltered."
    else:
        action = "Abstain rate is in a reasonable diagnostic range."
    return "\n".join(
        [
            "# Risk Calibration Diagnosis",
            "",
            f"abstain_rate: {abstain:.4f}",
            f"accuracy_after_abstention: {acc_after:.4f}",
            f"hallucination_f1_after_abstention: {float(metrics.get('hallucination_f1_after_abstention', 0.0)):.4f}",
            f"coverage_after_abstention: {coverage:.4f}",
            f"risk_error_correlation: {corr:.4f}",
            f"hallucination_ece: {ece:.4f}",
            f"hallucination_brier: {brier:.4f}",
            f"risk_coverage_accuracy_auc: {risk_auc:.4f}",
            f"selective_risk_auc: {selective_risk_auc:.4f}",
            "",
            verdict,
            action,
            "",
            "ECE and Brier score evaluate probability calibration; lower values are better. The risk-coverage statistics evaluate whether retaining lower-risk claims improves selective accuracy.",
        ]
    )


def paper_main_result_check(rows: list[dict[str, Any]]) -> str:
    """Check whether SCAD-RAG beats key baselines."""
    by_method = {row["method"]: row for row in rows}
    scad = by_method.get("scad_rag", {})
    checks = []
    for baseline in ["similarity_only", "nli_only", "ess_rule"]:
        base = by_method.get(baseline, {})
        scad_score = float(scad.get("hallucination_f1", scad.get("macro_f1", 0.0)) or 0.0)
        base_score = float(base.get("hallucination_f1", base.get("macro_f1", 0.0)) or 0.0)
        passed = scad_score > base_score
        checks.append((baseline, passed, scad_score, base_score))
    lines = ["# Paper Main Result Check", "", "Primary comparison metric for RAGTruth: Hallucination-F1.", ""]
    for baseline, passed, scad_score, base_score in checks:
        lines.append(f"- SCAD-RAG > {baseline}: {'yes' if passed else 'no'} (SCAD={scad_score:.4f}, baseline={base_score:.4f})")
    if not all(item[1] for item in checks):
        lines.extend(
            [
                "",
                "## Possible Causes",
                "",
                "- thresholds are not tuned for validation data",
                "- RAGTruth adapter mapping is too coarse",
                "- the local NLI model is weak for the domain",
                "- risk abstention is too frequent or too rare",
                "- hard negative construction is unstable",
            ]
        )
    else:
        lines.extend(["", "The main-result direction supports the SCAD-RAG method story on these metrics."])
    return "\n".join(lines)


def _avg(values: list[float | int]) -> float:
    """Average a numeric list."""
    return sum(values) / len(values) if values else 0.0

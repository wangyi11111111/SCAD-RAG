"""Run optional LettuceDetect baseline on cached SCAD-RAG prediction rows."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.utils.io import ensure_dir, write_csv, write_json


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run LettuceDetect as an optional local baseline.")
    parser.add_argument("--predictions", required=True, help="Existing SCAD-RAG predictions.csv used only as claim/evidence/gold carrier.")
    parser.add_argument("--output_dir", default="experiments/runs")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--model_path", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()
    root = ensure_dir(Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lettucedetect")
    rows = _read_rows(args.predictions, args.max_rows)
    report: dict[str, Any] = {
        "input_predictions": args.predictions,
        "model_path": args.model_path,
        "confidence_threshold": args.confidence_threshold,
        "max_rows": args.max_rows,
    }
    try:
        from lettucedetect.models.inference import HallucinationDetector  # type: ignore

        detector = HallucinationDetector(method="transformer", model_path=args.model_path)
    except Exception as exc:
        report["status"] = "failed_to_load"
        report["error"] = repr(exc)
        write_json(root / "lettucedetect_report.json", report)
        (root / "lettucedetect_report.md").write_text(_failure_report(report), encoding="utf-8")
        raise RuntimeError(f"LettuceDetect could not be loaded: {exc}") from exc
    predictions = []
    for index, row in enumerate(rows, start=1):
        claim = str(row.get("claim_text", ""))
        evidence = str(row.get("best_evidence_text", "") or "")
        if evidence.lower() == "nan":
            evidence = ""
        question = str(row.get("question", "") or "")
        try:
            spans = detector.predict(context=[evidence] if evidence else [], answer=claim, question=question, output_format="spans")
        except Exception as exc:
            spans = []
            row["lettucedetect_error"] = repr(exc)
        max_confidence = _max_confidence(spans)
        pred_hallucination = 1 if spans and max_confidence >= args.confidence_threshold else 0
        item = dict(row)
        item.update(
            {
                "pred_relation": "Insufficient" if pred_hallucination else "Supported",
                "pred_hallucination": pred_hallucination,
                "pred_attribution": "Generation-inconsistent" if pred_hallucination else "No hallucination",
                "pred_context_status": "Insufficient" if pred_hallucination else "Sufficient",
                "calibrated_hallucination_probability": max_confidence,
                "score_original": 1.0 - max_confidence,
                "risk_score": max_confidence,
                "uncertainty_score": max_confidence,
                "lettucedetect_num_spans": len(spans),
                "lettucedetect_max_confidence": max_confidence,
                "lettucedetect_spans_json": json.dumps(spans, ensure_ascii=False),
                "explanation": f"LettuceDetect predicted {len(spans)} unsupported span(s), max_confidence={max_confidence:.4f}.",
            }
        )
        predictions.append(item)
        if index % 100 == 0:
            print(f"Processed {index}/{len(rows)} rows")
    metrics = compute_metrics(predictions)
    report["status"] = "completed"
    report["num_rows"] = len(predictions)
    report["metrics"] = metrics
    write_csv(root / "lettucedetect_predictions.csv", predictions)
    write_json(root / "lettucedetect_metrics.json", metrics)
    write_json(root / "lettucedetect_report.json", report)
    (root / "lettucedetect_report.md").write_text(_success_report(report), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "metrics": metrics}, indent=2))
    return 0


def _read_rows(path: str | Path, max_rows: int | None) -> list[dict[str, str]]:
    """Read prediction rows without loading more than requested."""
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _max_confidence(spans: Any) -> float:
    """Return maximum span confidence."""
    if not isinstance(spans, list) or not spans:
        return 0.0
    values = []
    for span in spans:
        if isinstance(span, dict):
            try:
                values.append(float(span.get("confidence", 1.0)))
            except (TypeError, ValueError):
                values.append(1.0)
        else:
            values.append(1.0)
    return max(values) if values else 0.0


def _success_report(report: dict[str, Any]) -> str:
    """Render success report."""
    metrics = report.get("metrics", {})
    return "\n".join(
        [
            "# LettuceDetect Baseline Report",
            "",
            f"status: {report.get('status')}",
            f"model_path: {report.get('model_path')}",
            f"num_rows: {report.get('num_rows')}",
            f"confidence_threshold: {report.get('confidence_threshold')}",
            "",
            "## Metrics",
            "",
            f"- Hallucination-F1: {float(metrics.get('hallucination_f1', 0.0)):.4f}",
            f"- Binary Macro-F1: {float(metrics.get('hallucination_macro_f1', 0.0)):.4f}",
            f"- AUROC: {float(metrics.get('hallucination_auroc', 0.0)):.4f}",
            f"- Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}",
            f"- Risk-error correlation: {float(metrics.get('risk_error_correlation', 0.0)):.4f}",
        ]
    )


def _failure_report(report: dict[str, Any]) -> str:
    """Render failure report."""
    return "\n".join(
        [
            "# LettuceDetect Baseline Report",
            "",
            f"status: {report.get('status')}",
            f"model_path: {report.get('model_path')}",
            f"error: {report.get('error')}",
            "",
            "LettuceDetect is optional and is not required for quick_test or SCAD-RAG reproducibility.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Run NLI backbone sensitivity experiments."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import apply_thresholds_path, load_config
from scad_rag.utils.io import ensure_dir, write_csv, write_json


DEFAULT_MODELS = [
    "typeform/distilbert-base-uncased-mnli",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-v3-base",
]


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate SCAD-RAG sensitivity to local NLI backbones.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated Hugging Face NLI model names.")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--thresholds_path", default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()
    base = apply_thresholds_path(load_config(args.config), args.thresholds_path)
    root = ensure_dir(Path(base.get("output_dir", "experiments/runs")) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_nli_sensitivity")
    rows: list[dict] = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        config = deepcopy(base)
        config["nli_model_name"] = model_name
        config["method"] = "scad_rag"
        safe_name = model_name.replace("/", "__").replace(":", "_")
        try:
            result = run_experiment(config, "scad_rag", root / safe_name, max_samples=args.max_samples)
            row = _metric_row(model_name, result["metrics"], result["run_dir"])
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row = {"nli_model_name": model_name, "status": "failed", "error": repr(exc)}
        rows.append(row)
        write_csv(root / "nli_sensitivity_summary.csv", rows)
    write_json(root / "nli_sensitivity_summary.json", rows)
    (root / "nli_sensitivity_report.md").write_text(_report(rows, args.max_samples), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "rows": rows}, indent=2))
    return 0


def _metric_row(model_name: str, metrics: dict, run_dir: str) -> dict:
    """Convert metrics into a compact summary row."""
    return {
        "nli_model_name": model_name,
        "status": "completed",
        "run_dir": run_dir,
        "num_claims": metrics.get("num_claims", 0),
        "hallucination_f1": metrics.get("hallucination_f1", 0.0),
        "hallucination_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
        "hallucination_auroc": metrics.get("hallucination_auroc", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "risk_error_correlation": metrics.get("risk_error_correlation", 0.0),
        "hallucination_ece": metrics.get("hallucination_ece", 0.0),
        "hallucination_brier": metrics.get("hallucination_brier", 0.0),
        "risk_coverage_accuracy_auc": metrics.get("risk_coverage_accuracy_auc", 0.0),
        "selective_risk_auc": metrics.get("selective_risk_auc", 0.0),
    }


def _report(rows: list[dict], max_samples: int) -> str:
    """Render markdown report."""
    lines = [
        "# NLI Backbone Sensitivity",
        "",
        f"max_samples: {max_samples}",
        "",
        "This experiment evaluates whether SCAD-RAG conclusions depend on one local NLI backbone. All runs use the same processed data, evidence alignment, top-k setting, and no-gold inference mode.",
        "",
        "| NLI model | Status | Hall-F1 | AUROC | Risk Corr. | ECE | Brier |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('nli_model_name')} | {row.get('status')} | {float(row.get('hallucination_f1', 0.0)):.4f} | {float(row.get('hallucination_auroc', 0.0)):.4f} | {float(row.get('risk_error_correlation', 0.0)):.4f} | {float(row.get('hallucination_ece', 0.0)):.4f} | {float(row.get('hallucination_brier', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: large variance across rows indicates that local NLI quality is a bottleneck and should be discussed as a deployment trade-off. Stable trends indicate that SCAD-RAG audit features are not tied to a single NLI checkpoint.",
        ]
    )
    failures = [row for row in rows if row.get("status") != "completed"]
    if failures:
        lines.extend(["", "## Failed Models", ""])
        for row in failures:
            lines.append(f"- {row.get('nli_model_name')}: {row.get('error')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

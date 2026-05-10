"""Evaluate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scad_rag.config import load_config
from scad_rag.evaluation.diagnosis import risk_calibration_diagnosis
from scad_rag.evaluation.latex_export import metrics_to_latex, risk_calibration_to_latex
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.utils.io import read_jsonl, writable_file_path, write_json


def main() -> int:
    """Evaluate latest or specified run."""
    parser = argparse.ArgumentParser(description="Evaluate SCAD-RAG predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(config)
    rows = read_jsonl(run_dir / "predictions.jsonl")
    metrics = compute_metrics(rows)
    write_json(run_dir / "metrics.json", metrics)
    writable_file_path(run_dir / "latex_table_metrics.txt").write_text(metrics_to_latex(metrics, str(config.get("method", "scad_rag"))), encoding="utf-8")
    writable_file_path(run_dir / "latex_table_risk_calibration.txt").write_text(risk_calibration_to_latex(metrics, str(config.get("method", "scad_rag"))), encoding="utf-8")
    writable_file_path(run_dir / "risk_calibration_diagnosis.md").write_text(risk_calibration_diagnosis(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def _latest_run(config: dict) -> Path:
    """Resolve latest run."""
    latest = Path(config.get("output_dir", "experiments/runs")) / "latest_run.txt"
    if not latest.exists():
        latest = writable_file_path(latest)
    if not latest.exists():
        raise FileNotFoundError("No latest_run.txt found. Run pipeline first or pass --run_dir.")
    return Path(latest.read_text(encoding="utf-8").strip().lstrip("\ufeff"))


if __name__ == "__main__":
    raise SystemExit(main())

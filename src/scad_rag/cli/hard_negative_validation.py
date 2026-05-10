"""Validate counterfactual replacement strategies."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import apply_thresholds_path, load_config
from scad_rag.utils.io import ensure_dir, write_csv, write_json


STRATEGIES = ["hard_negative", "low_relevance", "random"]


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare hard-negative counterfactual replacement strategies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--thresholds_path", default=None)
    args = parser.parse_args()
    base = apply_thresholds_path(load_config(args.config), args.thresholds_path)
    root = ensure_dir(Path(base.get("output_dir", "experiments/runs")) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_hard_negative_validation")
    rows = []
    for strategy in [item.strip() for item in args.strategies.split(",") if item.strip()]:
        config = deepcopy(base)
        config.setdefault("counterfactual", {})
        config["counterfactual"]["enable_hard_negative"] = True
        config["counterfactual"]["hard_negative_strategy"] = strategy
        result = run_experiment(config, "scad_rag", root / strategy, max_samples=args.max_samples)
        rows.append(_metric_row(strategy, result["metrics"], result["run_dir"]))
        write_csv(root / "hard_negative_validation_summary.csv", rows)
    write_json(root / "hard_negative_validation_summary.json", rows)
    (root / "hard_negative_validation_report.md").write_text(_report(rows, args.max_samples), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "rows": rows}, indent=2))
    return 0


def _metric_row(strategy: str, metrics: dict, run_dir: str) -> dict:
    """Convert run metrics into one summary row."""
    return {
        "strategy": strategy,
        "run_dir": run_dir,
        "num_claims": metrics.get("num_claims", 0),
        "hallucination_f1": metrics.get("hallucination_f1", 0.0),
        "hallucination_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
        "hallucination_auroc": metrics.get("hallucination_auroc", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "average_hard_negative_robustness_gap": metrics.get("average_hard_negative_robustness_gap", 0.0),
        "risk_error_correlation": metrics.get("risk_error_correlation", 0.0),
        "hallucination_ece": metrics.get("hallucination_ece", 0.0),
        "hallucination_brier": metrics.get("hallucination_brier", 0.0),
    }


def _report(rows: list[dict], max_samples: int) -> str:
    """Render markdown report."""
    best = max(rows, key=lambda row: float(row.get("average_hard_negative_robustness_gap", 0.0))) if rows else {}
    lines = [
        "# Hard Negative Validation",
        "",
        f"max_samples: {max_samples}",
        "",
        "This control experiment checks whether semantic hard negatives are more informative than random or low-relevance replacements. Prediction uses strict no-gold inference.",
        "",
        "| Strategy | Hall-F1 | AUROC | HNRG | Risk Corr. | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('strategy')} | {float(row.get('hallucination_f1', 0.0)):.4f} | {float(row.get('hallucination_auroc', 0.0)):.4f} | {float(row.get('average_hard_negative_robustness_gap', 0.0)):.4f} | {float(row.get('risk_error_correlation', 0.0)):.4f} | {float(row.get('hallucination_ece', 0.0)):.4f} | {float(row.get('hallucination_brier', 0.0)):.4f} |"
        )
    if best:
        lines.extend(
            [
                "",
                f"Largest average HNRG is obtained by `{best.get('strategy')}`. If the semantic hard-negative strategy improves HNRG or risk correlation over random replacement, it supports the counterfactual audit design. If not, the paper should present it as a diagnostic feature rather than a causal proof.",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

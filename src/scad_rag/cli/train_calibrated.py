"""Train and evaluate SCAD-RAG-Calibrated."""

from __future__ import annotations

import argparse

from scad_rag.training.calibrated import run_calibrated_experiment


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a calibrated head over cached SCAD audit features.")
    parser.add_argument("--train_predictions", required=True, help="CSV predictions/features from train or calibration data.")
    parser.add_argument("--test_predictions", required=True, help="CSV predictions/features from held-out evaluation data.")
    parser.add_argument("--baseline_summary", default=None, help="Optional baseline_summary.csv to merge into the report.")
    parser.add_argument("--output_dir", default="experiments/runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_feature_ablation", action="store_true")
    args = parser.parse_args()
    output = run_calibrated_experiment(
        train_predictions=args.train_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        baseline_summary=args.baseline_summary,
        seed=args.seed,
        feature_ablation=not args.no_feature_ablation,
    )
    print(f"Wrote calibrated experiment to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

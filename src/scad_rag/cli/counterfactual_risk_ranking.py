"""Run fixed-prediction EDD/HNRG risk-ranking diagnostics."""

from __future__ import annotations

import argparse

from scad_rag.training.calibrated import run_counterfactual_risk_ranking_experiment


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate EDD/HNRG as selective-risk ordering signals with fixed predictions."
    )
    parser.add_argument("--train_predictions", required=True, help="Train-side SCAD prediction CSV.")
    parser.add_argument("--test_predictions", required=True, help="Held-out SCAD prediction CSV.")
    parser.add_argument("--output_dir", default="experiments/runs/counterfactual_risk_ranking")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = run_counterfactual_risk_ranking_experiment(
        train_predictions=args.train_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Wrote counterfactual risk-ranking experiment to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

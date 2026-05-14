"""Run fixed-classifier EDD/HNRG incremental diagnostics."""

from __future__ import annotations

import argparse

from scad_rag.training.calibrated import run_counterfactual_incremental_experiment


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate EDD/HNRG under a fixed classifier and fixed base feature set."
    )
    parser.add_argument("--train_predictions", required=True, help="Train-side SCAD prediction CSV.")
    parser.add_argument("--test_predictions", required=True, help="Held-out SCAD prediction CSV.")
    parser.add_argument("--output_dir", default="experiments/runs/counterfactual_incremental")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = run_counterfactual_incremental_experiment(
        train_predictions=args.train_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Wrote counterfactual incremental experiment to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

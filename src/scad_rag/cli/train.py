"""Train CLI."""

from __future__ import annotations

import argparse

from scad_rag.config import load_config
from scad_rag.training.train_classifier import train_feature_classifier


def main() -> int:
    """Train a lightweight model."""
    parser = argparse.ArgumentParser(description="Train SCAD-RAG classifiers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="ml_feature_classifier")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    if args.method != "ml_feature_classifier":
        raise ValueError("Only ml_feature_classifier training is implemented.")
    path = train_feature_classifier(load_config(args.config), max_samples=args.max_samples)
    print(f"Saved classifier to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

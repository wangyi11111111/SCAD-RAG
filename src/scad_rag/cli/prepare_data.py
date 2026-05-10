"""Prepare datasets."""

from __future__ import annotations

import argparse

from scad_rag.config import load_config
from scad_rag.data.preprocess import prepare_dataset
from scad_rag.utils.logging import get_logger


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prepare SCAD-RAG datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = args.dataset or str(config.get("dataset", "toy"))
    if args.split and dataset == "ragtruth":
        config.setdefault(dataset, {})
        config[dataset]["split"] = args.split
        config[dataset]["processed_path"] = f"data/processed/{dataset}/{args.split}_processed.jsonl"
    elif args.split:
        config.setdefault(dataset, {})
        config[dataset]["split"] = args.split
        config[dataset]["processed_path"] = f"data/processed/{dataset}/{args.split}_processed.jsonl"
    path = prepare_dataset(config, dataset, max_samples=args.max_samples)
    get_logger(__name__).info("Prepared dataset at %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

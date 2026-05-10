"""Single example prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import load_config
from scad_rag.schema import Evidence, RAGSample, SentenceLabel, sample_to_dict
from scad_rag.utils.io import writable_file_path, write_jsonl


def main() -> int:
    """Predict one RAG answer."""
    parser = argparse.ArgumentParser(description="Predict one example.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidences", required=True, help="Evidence strings separated by ||.")
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evidences = [Evidence(f"e{i + 1}", text.strip(), "retrieved") for i, text in enumerate(args.evidences.split("||")) if text.strip()]
    sample = RAGSample("single_prediction", args.question, evidences, args.answer, [SentenceLabel("s1", args.answer)])
    target = writable_file_path(Path("data/processed/single/processed.jsonl"))
    write_jsonl(target, [sample_to_dict(sample)])
    config["dataset"] = "single"
    config["single"] = {"processed_path": str(target)}
    result = run_experiment(config, str(config.get("method", "scad_rag")))
    print(json.dumps(result["predictions"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

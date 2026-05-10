"""Generate case studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from scad_rag.config import load_config
from scad_rag.evaluation.case_study import generate_case_studies
from scad_rag.utils.io import read_jsonl, writable_file_path


def main() -> int:
    """Generate case_studies.md."""
    parser = argparse.ArgumentParser(description="Generate case studies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(config)
    rows = read_jsonl(run_dir / "predictions.jsonl")
    output = writable_file_path(run_dir / "case_studies.md")
    output.write_text(generate_case_studies(rows), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def _latest_run(config: dict) -> Path:
    """Resolve latest run."""
    latest = Path(config.get("output_dir", "experiments/runs")) / "latest_run.txt"
    if not latest.exists():
        latest = writable_file_path(latest)
    if not latest.exists():
        raise FileNotFoundError("No latest run found.")
    return Path(latest.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    raise SystemExit(main())

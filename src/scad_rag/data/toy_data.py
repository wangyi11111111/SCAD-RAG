"""Toy data preparation."""

from __future__ import annotations

from pathlib import Path

from scad_rag.schema import sample_from_dict, sample_to_dict
from scad_rag.utils.io import ensure_dir, read_jsonl, write_jsonl


def prepare_toy_data(config: dict) -> Path:
    """Validate and copy toy data to processed JSONL."""
    cfg = config.get("toy", {})
    source = Path(cfg.get("path", "data/samples/toy_rag.jsonl"))
    target = Path(cfg.get("processed_path", "data/processed/toy/processed.jsonl"))
    target = ensure_dir(target.parent) / target.name
    rows = [sample_to_dict(sample_from_dict(row)) for row in read_jsonl(source)]
    return write_jsonl(target, rows)

"""Optional HaluBench adapter scaffold."""

from __future__ import annotations

from pathlib import Path

from scad_rag.schema import Evidence, RAGSample, SentenceLabel, sample_to_dict
from scad_rag.utils.io import read_any_table, write_jsonl


def prepare_halubench(config: dict) -> Path:
    """Convert simple binary HaluBench-style rows."""
    cfg = config.get("halubench", {})
    raw = Path(cfg.get("raw_path", "data/raw/halubench"))
    target = Path(cfg.get("processed_path", "data/processed/halubench/processed.jsonl"))
    if not raw.exists():
        raise FileNotFoundError(f"HaluBench raw path does not exist: {raw}")
    rows = []
    for source in _sources(raw):
        rows.extend(read_any_table(source))
    samples = []
    for i, row in enumerate(rows):
        answer = str(row.get("answer", row.get("response", "")))
        hallucination = int(row.get("hallucination", row.get("label", 1)))
        relation = "Insufficient" if hallucination else "Supported"
        evidences = [Evidence("e1", str(row.get("evidence", row.get("context", ""))), "retrieved")]
        samples.append(RAGSample(str(row.get("id", f"halubench_{i:06d}")), str(row.get("question", row.get("prompt", ""))), evidences, answer, [SentenceLabel("s1", answer, relation, hallucination, "Unknown" if hallucination else "No hallucination", "Uncertain")]))
    return write_jsonl(target, [sample_to_dict(sample) for sample in samples])


def _sources(path: Path) -> list[Path]:
    """List source files."""
    if path.is_file():
        return [path]
    files = []
    for suffix in ["*.jsonl", "*.json", "*.csv"]:
        files.extend(path.rglob(suffix))
    return sorted(files)

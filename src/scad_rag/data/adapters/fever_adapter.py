"""FEVER adapter."""

from __future__ import annotations

from pathlib import Path

from scad_rag.schema import Evidence, RAGSample, SentenceLabel, sample_to_dict
from scad_rag.utils.io import read_any_table, readable_existing_path, write_jsonl

MAP = {"SUPPORTS": "Supported", "SUPPORTED": "Supported", "REFUTES": "Contradicted", "REFUTED": "Contradicted", "NOT ENOUGH INFO": "Insufficient", "NOT_ENOUGH_INFO": "Insufficient"}


def prepare_fever(config: dict, max_samples: int | None = None) -> Path:
    """Convert FEVER rows."""
    cfg = config.get("fever", {})
    raw = readable_existing_path(cfg.get("raw_path", "data/raw/fever"))
    target = Path(cfg.get("processed_path", "data/processed/fever/processed.jsonl"))
    if not raw.exists():
        raise FileNotFoundError(f"FEVER raw path does not exist: {raw}")
    rows = []
    for source in _sources(raw):
        rows.extend(read_any_table(source))
        if max_samples is not None and len(rows) >= max_samples:
            rows = rows[:max_samples]
            break
    samples = []
    for i, row in enumerate(rows):
        claim = str(row.get("claim", row.get("claim_text", "")))
        label = MAP.get(str(row.get("label", "")).upper(), "Insufficient")
        ev_raw = row.get("evidence", row.get("evidences", []))
        evidences = [Evidence(f"e{j + 1}", _evidence_text(item), "gold") for j, item in enumerate(ev_raw if isinstance(ev_raw, list) else [ev_raw])]
        evidences = [ev for ev in evidences if ev.text.strip()]
        attribution = _attribution_from_label(label)
        context_status = "Sufficient" if label == "Supported" else ("Conflicting" if label == "Contradicted" else "Insufficient")
        samples.append(
            RAGSample(
                str(row.get("id", f"fever_{i:06d}")),
                claim,
                evidences,
                claim,
                [SentenceLabel("s1", claim, label, 0 if label == "Supported" else 1, attribution, context_status)],
            )
        )
    return write_jsonl(target, [sample_to_dict(sample) for sample in samples])


def _evidence_text(item) -> str:
    """Extract evidence sentence text from FEVER-style evidence records."""
    if isinstance(item, dict):
        for key in ["text", "sentence", "evidence_text"]:
            if item.get(key):
                return str(item[key])
        return " ".join(str(value) for value in item.values() if isinstance(value, str))
    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            return str(item[2])
        return " ".join(str(value) for value in item)
    return str(item)


def _attribution_from_label(label: str) -> str:
    """Map FEVER relation labels to SCAD-RAG attribution labels."""
    if label == "Supported":
        return "No hallucination"
    if label == "Contradicted":
        return "Evidence-contradicted"
    return "Retrieval-insufficient"


def _sources(path: Path) -> list[Path]:
    """List source files."""
    if path.is_file():
        return [path]
    files = []
    for suffix in ["*.jsonl", "*.json", "*.csv"]:
        files.extend(path.rglob(suffix))
    return sorted(files)

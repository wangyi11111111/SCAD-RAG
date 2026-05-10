"""Dataset preparation and loading."""

from __future__ import annotations

from pathlib import Path

from scad_rag.data.adapters.fever_adapter import prepare_fever
from scad_rag.data.adapters.halubench_adapter import prepare_halubench
from scad_rag.data.adapters.ragtruth_adapter import prepare_ragtruth
from scad_rag.data.adapters.scifact_adapter import prepare_scifact
from scad_rag.data.toy_data import prepare_toy_data
from scad_rag.schema import RAGSample, sample_from_dict
import json

from scad_rag.utils.io import read_jsonl, readable_existing_path


def prepare_dataset(config: dict, dataset: str | None = None, max_samples: int | None = None) -> Path:
    """Prepare a dataset and return processed path."""
    name = dataset or str(config.get("dataset", "toy"))
    if name == "toy":
        return prepare_toy_data(config)
    if name == "ragtruth":
        return prepare_ragtruth(config, max_samples=max_samples)
    if name == "fever":
        return prepare_fever(config, max_samples=max_samples)
    if name == "scifact":
        return prepare_scifact(config)
    if name == "halubench":
        return prepare_halubench(config)
    if isinstance(config.get(name), dict) and config[name].get("processed_path"):
        path = Path(config[name]["processed_path"])
        if path.exists():
            return path
    raise ValueError(f"Unsupported dataset: {name}")


def processed_path_for(config: dict, dataset: str | None = None) -> Path:
    """Return expected processed path."""
    name = dataset or str(config.get("dataset", "toy"))
    if name == "toy":
        return readable_existing_path(config.get("toy", {}).get("processed_path", "data/processed/toy/processed.jsonl"))
    return readable_existing_path(config.get(name, {}).get("processed_path", f"data/processed/{name}/processed.jsonl"))


def load_samples(config: dict, dataset: str | None = None, max_samples: int | None = None) -> list[RAGSample]:
    """Load processed samples, preparing them when needed."""
    path = processed_path_for(config, dataset)
    if not path.exists():
        path = prepare_dataset(config, dataset, max_samples=max_samples)
    rows = _read_jsonl_limited(path, max_samples)
    return [sample_from_dict(row) for row in rows]


def _read_jsonl_limited(path: Path, max_samples: int | None = None) -> list[dict]:
    """Read at most max_samples JSONL rows without loading a full file when limited."""
    if max_samples is None:
        return read_jsonl(path)
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= max_samples:
                    break
    return rows

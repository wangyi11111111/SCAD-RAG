"""UTF-8 I/O helpers with Windows restricted-workspace fallback."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create and return a writable directory."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _probe(directory)
        return directory
    except OSError:
        fallback = _fallback_path(directory)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def writable_file_path(path: str | Path) -> Path:
    """Return a writable file path, falling back to temp when needed."""
    output = Path(path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        _probe(output.parent)
        return output
    except OSError:
        fallback = _fallback_path(output)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def fallback_path(path: str | Path) -> Path:
    """Return the deterministic temp fallback path for a project path."""
    return _fallback_path(Path(path))


def readable_existing_path(path: str | Path) -> Path:
    """Return path if it exists, otherwise its fallback if available."""
    source = Path(path)
    if source.exists():
        return source
    fallback = fallback_path(source)
    if fallback.exists():
        return fallback
    return source


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write JSONL and return the actual output path."""
    output = writable_file_path(path)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_json(path: str | Path) -> Any:
    """Read JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    """Write JSON and return output path."""
    output = writable_file_path(path)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """Write CSV and return output path."""
    output = writable_file_path(path)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_any_table(path: str | Path) -> list[dict[str, Any]]:
    """Read json, jsonl, or csv records."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    if source.suffix.lower() == ".json":
        data = read_json(source)
        return data if isinstance(data, list) else [data]
    if source.suffix.lower() == ".csv":
        return read_csv(source)
    raise ValueError(f"Unsupported file type: {source}")


def _probe(directory: Path) -> None:
    """Check writability."""
    probe = directory / ".scad_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def _fallback_path(path: Path) -> Path:
    """Map a project path into a deterministic temp fallback."""
    raw = str(path).replace(":", "").replace("\\", "/").strip("/")
    return Path(tempfile.gettempdir()) / "scad_rag_workspace_fallback" / Path(*[p for p in raw.split("/") if p])

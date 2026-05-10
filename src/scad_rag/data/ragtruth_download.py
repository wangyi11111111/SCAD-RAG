"""Download and verify the public RAGTruth dataset files."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scad_rag.utils.io import fallback_path, writable_file_path

RESPONSE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
SOURCE_INFO_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
REQUIRED_FILES = {
    "response.jsonl": RESPONSE_URL,
    "source_info.jsonl": SOURCE_INFO_URL,
}


@dataclass
class DownloadResult:
    """Result of a RAGTruth download attempt."""

    out_dir: Path
    requested_out_dir: Path
    method: str
    verified: bool
    files: dict[str, dict[str, Any]]
    errors: list[str]
    report_path: Path


def download_ragtruth(out_dir: str | Path = "data/raw/ragtruth", force: bool = False, verify: bool = True) -> DownloadResult:
    """Download RAGTruth response/source files and write a report."""
    requested = Path(out_dir)
    target = _writable_dir(requested)
    errors: list[str] = []
    method = "existing"
    try:
        needed = [name for name in REQUIRED_FILES if force or not _nonempty(target / name)]
        if needed:
            method = _download_needed(target, needed, errors)
        file_report = _verify_files(target) if verify else _file_report_without_verify(target)
        verified = bool(file_report) and all(item.get("passed", False) for item in file_report.values()) if verify else True
    except Exception as exc:
        errors.append(str(exc))
        file_report = _file_report_without_verify(target)
        verified = False
    report_path = _write_report(requested, target, method, verified, file_report, errors)
    if verify and not verified:
        raise RuntimeError(f"RAGTruth download verification failed. See report: {report_path}")
    return DownloadResult(target, requested, method, verified, file_report, errors, report_path)


def _writable_dir(requested: Path) -> Path:
    """Create a writable output directory, using fallback if the workspace is read-only."""
    try:
        requested.mkdir(parents=True, exist_ok=True)
        probe = requested / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return requested
    except OSError:
        target = fallback_path(requested)
        target.mkdir(parents=True, exist_ok=True)
        return target


def _download_needed(target: Path, needed: list[str], errors: list[str]) -> str:
    """Download missing files with urllib, PowerShell, then git clone fallback."""
    try:
        for name in needed:
            _download_urllib(REQUIRED_FILES[name], target / name)
        return "python urllib.request"
    except Exception as exc:
        errors.append(f"urllib failed: {exc}")
    try:
        for name in needed:
            _download_powershell(REQUIRED_FILES[name], target / name)
        return "PowerShell Invoke-WebRequest"
    except Exception as exc:
        errors.append(f"Invoke-WebRequest failed: {exc}")
    try:
        _download_git_clone(target, needed)
        return "git clone"
    except Exception as exc:
        errors.append(f"git clone failed: {exc}")
    raise RuntimeError("Could not download RAGTruth files from GitHub.")


def _download_urllib(url: str, output: Path) -> None:
    """Download one file with urllib."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    output.write_bytes(data)


def _download_powershell(url: str, output: Path) -> None:
    """Download one file with PowerShell."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        f"Invoke-WebRequest -Uri '{url}' -OutFile '{str(output)}'",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _download_git_clone(target: Path, needed: list[str]) -> None:
    """Clone the public repo to a temp directory and copy dataset files."""
    with tempfile.TemporaryDirectory(prefix="scad_rag_ragtruth_") as tmp:
        repo = Path(tmp) / "RAGTruth"
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ParticleMedia/RAGTruth.git", str(repo)], check=True, capture_output=True, text=True)
        for name in needed:
            source = repo / "dataset" / name
            if not source.exists():
                raise FileNotFoundError(f"Missing {source} in cloned repository.")
            shutil.copyfile(source, target / name)


def _nonempty(path: Path) -> bool:
    """Return true when a file exists and is non-trivial."""
    return path.exists() and path.is_file() and path.stat().st_size > 1024


def _verify_files(target: Path) -> dict[str, dict[str, Any]]:
    """Verify file existence, size, JSONL validity, and expected fields."""
    reports: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        path = target / name
        report = _inspect_jsonl(path)
        if name == "response.jsonl":
            candidates = {"response", "labels", "source_id", "id", "prompt"}
        else:
            candidates = {"source_id", "source_info", "prompt", "question", "id"}
        observed = set()
        for fields in report.get("first_three_fields", []):
            observed.update(fields)
        report["candidate_fields_found"] = sorted(observed.intersection(candidates))
        report["passed"] = bool(report.get("exists")) and int(report.get("size_bytes", 0)) > 1024 and bool(report.get("first_three_json_valid")) and bool(report["candidate_fields_found"])
        reports[name] = report
    return reports


def _inspect_jsonl(path: Path) -> dict[str, Any]:
    """Inspect the first three JSONL rows."""
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "first_three_json_valid": False,
        "first_three_fields": [],
    }
    if not path.exists():
        return report
    fields = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 3:
                break
            data = json.loads(line)
            fields.append(sorted(data.keys()) if isinstance(data, dict) else [])
    report["first_three_fields"] = fields
    report["first_three_json_valid"] = len(fields) > 0
    return report


def _file_report_without_verify(target: Path) -> dict[str, dict[str, Any]]:
    """Return basic file metadata when verification is disabled or failed."""
    return {
        name: {
            "path": str(target / name),
            "exists": (target / name).exists(),
            "size_bytes": (target / name).stat().st_size if (target / name).exists() else 0,
            "passed": False,
        }
        for name in REQUIRED_FILES
    }


def _write_report(requested: Path, target: Path, method: str, verified: bool, files: dict[str, dict[str, Any]], errors: list[str]) -> Path:
    """Write a markdown download report."""
    lines = [
        "# RAGTruth Download Report",
        "",
        f"Download time: {datetime.now().isoformat(timespec='seconds')}",
        f"Download method: {method}",
        f"Requested output directory: {requested}",
        f"Actual output directory: {target}",
        f"Integrity check passed: {verified}",
        "",
        "## Files",
        "",
    ]
    for name, item in files.items():
        lines.extend(
            [
                f"### {name}",
                f"- Path: {item.get('path', target / name)}",
                f"- Exists: {item.get('exists', False)}",
                f"- Size bytes: {item.get('size_bytes', 0)}",
                f"- First 3 row fields: {item.get('first_three_fields', [])}",
                f"- Candidate fields found: {item.get('candidate_fields_found', [])}",
                f"- Passed: {item.get('passed', False)}",
                "",
            ]
        )
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.extend(["", "## Manual Recovery", "", "Check network access to GitHub, then place response.jsonl and source_info.jsonl in the requested or actual output directory."])
    report = writable_file_path(target / "download_report.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report

"""Download the public SciFact release."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path

from scad_rag.utils.io import ensure_dir

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download SciFact public data.")
    parser.add_argument("--out_dir", default="data/raw/scifact")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    report = out_dir / "download_report.md"
    if not args.force and (out_dir / "data").exists():
        report.write_text(_report(out_dir, "skipped-existing", True, ""), encoding="utf-8")
        print(f"SciFact already exists at {out_dir}")
        return 0
    archive = out_dir / "data.tar.gz"
    try:
        urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(out_dir)
        archive.unlink(missing_ok=True)
        ok = any(out_dir.rglob("claims_train.jsonl")) and any(out_dir.rglob("corpus.jsonl"))
        report.write_text(_report(out_dir, "urllib", bool(ok), ""), encoding="utf-8")
        if not ok:
            raise RuntimeError("Downloaded archive did not contain expected SciFact files.")
        print(f"Downloaded SciFact to {out_dir}")
        return 0
    except Exception as exc:
        report.write_text(_report(out_dir, "urllib", False, str(exc)), encoding="utf-8")
        raise RuntimeError(f"SciFact download failed. Check network access or manually place official SciFact data under {out_dir}.") from exc


def _report(out_dir: Path, method: str, ok: bool, error: str) -> str:
    """Render download report."""
    files = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*.jsonl"))[:20] if out_dir.exists() else []
    return "\n".join(
        [
            "# SciFact Download Report",
            "",
            f"time: {datetime.now().isoformat()}",
            f"url: {URL}",
            f"out_dir: {out_dir}",
            f"method: {method}",
            f"integrity_ok: {ok}",
            f"error: {error}",
            "",
            "## JSONL Files",
            *[f"- {item}" for item in files],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

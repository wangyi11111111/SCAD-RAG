"""Download the public RAGTruth dataset files."""

from __future__ import annotations

import argparse
import json
import sys

from scad_rag.data.ragtruth_download import download_ragtruth


def _as_bool(value: str | bool) -> bool:
    """Parse common CLI boolean strings."""
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Download public RAGTruth files from GitHub.")
    parser.add_argument("--out_dir", default="data/raw/ragtruth")
    parser.add_argument("--force", type=_as_bool, default=False)
    parser.add_argument("--verify", type=_as_bool, default=True)
    args = parser.parse_args()
    try:
        result = download_ragtruth(args.out_dir, force=args.force, verify=args.verify)
    except Exception as exc:
        print(f"RAGTruth download failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "requested_out_dir": str(result.requested_out_dir),
                "actual_out_dir": str(result.out_dir),
                "method": result.method,
                "verified": result.verified,
                "report_path": str(result.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

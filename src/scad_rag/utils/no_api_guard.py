"""Static guard against commercial large-model API usage."""

from __future__ import annotations

import sys
from pathlib import Path


def blocked_patterns() -> list[str]:
    """Return blocked strings without keeping every literal contiguous in tests."""
    return [
        "import " + "open" + "ai",
        "from " + "open" + "ai" + " import",
        "import " + "anth" + "ropic",
        "import " + "google.generative" + "ai",
        "import " + "co" + "here",
        "api." + "open" + "ai" + ".com",
        "anth" + "ropic.com",
        "generativelanguage.google" + "apis.com",
        "api." + "co" + "here.ai",
    ]


def iter_python_files(root: Path) -> list[Path]:
    """List Python files under source, scripts, and tests."""
    files = []
    for folder in ["src", "scripts", "tests"]:
        target = root / folder
        if target.exists():
            files.extend(target.rglob("*.py"))
    return files


def scan_paths(paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Return blocked pattern findings."""
    findings = []
    patterns = blocked_patterns()
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            lowered = line.lower()
            for pattern in patterns:
                if pattern in lowered:
                    findings.append((path, line_no, pattern))
    return findings


def run_guard(root: str | Path = ".") -> None:
    """Raise if blocked API usage is found."""
    findings = scan_paths(iter_python_files(Path(root)))
    if findings:
        rendered = "\n".join(f"{path}:{line_no}: {pattern}" for path, line_no, pattern in findings)
        raise RuntimeError(f"Blocked commercial API usage detected:\n{rendered}")


def main() -> int:
    """CLI entrypoint."""
    try:
        run_guard(Path.cwd())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("No blocked commercial API usage found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

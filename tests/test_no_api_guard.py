from pathlib import Path

from scad_rag.utils.no_api_guard import scan_paths


def test_no_api_guard_detects_blocked_import(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("import " + "open" + "ai\n", encoding="utf-8")
    assert scan_paths([bad])

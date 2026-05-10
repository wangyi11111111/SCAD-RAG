"""Error analysis report."""

from __future__ import annotations

from collections import Counter
from typing import Any


def generate_error_analysis(rows: list[dict[str, Any]]) -> str:
    """Generate markdown error analysis."""
    relation_errors = [row for row in rows if row.get("gold_relation") != row.get("pred_relation")]
    attr_errors = [row for row in rows if row.get("gold_attribution") not in {"Unknown", ""} and row.get("gold_attribution") != row.get("pred_attribution")]
    lines = ["# Error Analysis", "", f"Total claims: {len(rows)}", f"Relation errors: {len(relation_errors)}", f"Attribution errors: {len(attr_errors)}", "", "## Relation Error Types", ""]
    rel_counter = Counter((row.get("gold_relation"), row.get("pred_relation")) for row in relation_errors)
    lines.extend([f"- {g} -> {p}: {c}" for (g, p), c in rel_counter.most_common()] or ["- No relation errors."])
    lines.extend(["", "## Attribution Error Types", ""])
    attr_counter = Counter((row.get("gold_attribution"), row.get("pred_attribution")) for row in attr_errors)
    lines.extend([f"- {g} -> {p}: {c}" for (g, p), c in attr_counter.most_common()] or ["- No attribution errors with available gold labels."])
    return "\n".join(lines)

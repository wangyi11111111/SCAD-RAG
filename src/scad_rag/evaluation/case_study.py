"""Case study generation."""

from __future__ import annotations

from typing import Any

CASE_ORDER = [
    "No hallucination",
    "Retrieval-insufficient",
    "Generation-inconsistent",
    "Evidence-contradicted",
    "Unstable-evidence-dependency",
    "High-risk-abstain",
]


def generate_case_studies(rows: list[dict[str, Any]], max_cases: int = 6) -> str:
    """Generate representative cases."""
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[tuple[str, str]] = set()
    for label in CASE_ORDER:
        found = next(
            (
                row
                for row in rows
                if (row.get("pred_attribution") == label or row.get("gold_attribution") == label)
                and (str(row.get("sample_id")), str(row.get("claim_id"))) not in selected_ids
            ),
            None,
        )
        if found:
            key = (str(found.get("sample_id")), str(found.get("claim_id")))
            if key not in selected_ids:
                selected.append((label, found))
                selected_ids.add(key)
    for row in rows:
        if len(selected) >= max_cases:
            break
        key = (str(row.get("sample_id")), str(row.get("claim_id")))
        if key not in selected_ids:
            selected.append((str(row.get("pred_attribution", "Unknown")), row))
            selected_ids.add(key)
    lines = ["# Case Studies", ""]
    for i, (case_label, row) in enumerate(selected[:max_cases], start=1):
        lines.extend(
            [
                f"## Case {i}: {case_label}",
                "",
                "Question:",
                str(row.get("question", "")),
                "",
                "Claim:",
                str(row.get("claim_text", "")),
                "",
                "Gold:",
                f"{row.get('gold_relation')} / {row.get('gold_hallucination')} / {row.get('gold_attribution')}",
                "",
                "Prediction:",
                f"{row.get('pred_relation')} / {row.get('pred_hallucination')} / {row.get('pred_attribution')}",
                "",
                "Context Status:",
                f"{row.get('gold_context_status')} -> {row.get('pred_context_status')}",
                "",
                "Best Evidence:",
                str(row.get("best_evidence_text", "")),
                "",
                "Hard Negative Evidence:",
                str(row.get("hard_negative_evidence_text", "")),
                "",
                "Scores:",
                f"- relevance: {float(row.get('relevance_score', 0.0)):.3f}",
                f"- entailment: {float(row.get('entailment_score', 0.0)):.3f}",
                f"- contradiction: {float(row.get('contradiction_score', 0.0)):.3f}",
                f"- coverage: {float(row.get('coverage_score', 0.0)):.3f}",
                f"- sufficient_context_score: {float(row.get('sufficient_context_score', 0.0)):.3f}",
                f"- score_original: {float(row.get('score_original', 0.0)):.3f}",
                f"- score_removed: {float(row.get('score_removed', 0.0)):.3f}",
                f"- score_hard_negative: {float(row.get('score_hard_negative', 0.0)):.3f}",
                f"- EDD: {float(row.get('evidence_dependency_delta', 0.0)):.3f}",
                f"- HNRG: {float(row.get('hard_negative_robustness_gap', 0.0)):.3f}",
                f"- uncertainty: {float(row.get('uncertainty_score', 0.0)):.3f}",
                f"- risk: {float(row.get('risk_score', 0.0)):.3f}",
                "",
                "Explanation:",
                str(row.get("explanation", "")),
                "",
            ]
        )
    return "\n".join(lines)

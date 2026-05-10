"""Hard negative evidence selection."""

from __future__ import annotations

from scad_rag.features.coverage import keyword_coverage
from scad_rag.features.lexical_overlap import lexical_relevance
from scad_rag.features.sufficient_context import evaluate_sufficient_context
from scad_rag.schema import Evidence


def select_hard_negative(
    claim_text: str,
    best_evidence_id: str,
    candidate_evidences: list[Evidence],
    embedder,
    nli_model,
    thresholds: dict,
    allow_gold_labels: bool = False,
    strategy: str = "hard_negative",
) -> tuple[Evidence | None, float]:
    """Select replacement evidence for counterfactual hard-negative auditing.

    Strategies:
    - hard_negative: high relevance and low lexical coverage, without gold labels by default.
    - low_relevance: least relevant candidate.
    - random: deterministic first candidate after caller-side seeding/sorting.
    """
    candidates = [ev for ev in candidate_evidences if ev.id != best_evidence_id]
    if not candidates:
        return None, 0.0
    if strategy == "low_relevance":
        low = min(((float(embedder.score_pair(claim_text, ev.text)), ev) for ev in candidates), key=lambda item: item[0])
        return low[1], low[0]
    if strategy == "random":
        chosen = sorted(candidates, key=lambda ev: ev.id)[0]
        return chosen, float(embedder.score_pair(claim_text, chosen.text))
    preferred = [ev for ev in candidates if ev.type in {"hard_negative", "distractor"}] if allow_gold_labels else []
    pool = preferred or candidates
    max_candidates = int(thresholds.get("max_hard_negative_candidates", 8))
    if len(pool) > max_candidates:
        pool = sorted(pool, key=lambda ev: lexical_relevance(claim_text, ev.text), reverse=True)[:max_candidates]
    scored = []
    for ev in pool:
        relevance = float(embedder.score_pair(claim_text, ev.text))
        coverage = keyword_coverage(claim_text, ev.text)
        unsupported_bonus = 1.0 if coverage < float(thresholds.get("coverage_threshold", 0.35)) else 0.0
        scored.append((unsupported_bonus, relevance, ev))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = scored[0]
    if chosen[0] <= 0.0:
        low = min(((float(embedder.score_pair(claim_text, ev.text)), ev) for ev in candidates), key=lambda item: item[0])
        return low[1], low[0]
    return chosen[2], chosen[1]

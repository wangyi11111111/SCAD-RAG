"""Hard negative evidence selection.

The selector intentionally avoids gold labels in prediction mode. It first
keeps semantically close candidates, then prefers candidates that remain
related to the claim but are weakly entailing or contradictory under the local
NLI model. This makes the hard-negative view a reproducible robustness probe
rather than a label-leaking replacement.
"""

from __future__ import annotations

from scad_rag.features.coverage import keyword_coverage
from scad_rag.features.lexical_overlap import lexical_relevance
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
    - hard_negative: high relevance, weak entailment, and limited coverage, without gold labels by default.
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
    min_relevance = float(thresholds.get("hard_negative_min_relevance", thresholds.get("low_relevance_threshold", 0.25)))
    coverage_threshold = float(thresholds.get("coverage_threshold", 0.35))
    for ev in pool:
        relevance = float(embedder.score_pair(claim_text, ev.text))
        coverage = keyword_coverage(claim_text, ev.text)
        if relevance < min_relevance:
            continue
        nli = nli_model.score_pair(ev.text, claim_text)
        low_coverage = 1.0 if coverage < coverage_threshold else max(0.0, 1.0 - coverage)
        weak_entailment = 1.0 - float(nli.entailment)
        # Keep the candidate close to the claim but unsupported by relation evidence.
        hard_negative_score = (
            0.45 * relevance
            + 0.25 * weak_entailment
            + 0.20 * low_coverage
            + 0.10 * float(nli.contradiction)
        )
        scored.append((hard_negative_score, relevance, ev))
    if not scored:
        low = min(((float(embedder.score_pair(claim_text, ev.text)), ev) for ev in candidates), key=lambda item: item[0])
        return low[1], low[0]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = scored[0]
    return chosen[2], chosen[1]

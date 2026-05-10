"""Evidence relevance ranking."""

from __future__ import annotations

from scad_rag.schema import Evidence


def rank_evidences(claim: str, evidences: list[Evidence], embedder, top_k: int) -> list[tuple[Evidence, float]]:
    """Rank evidences by relevance."""
    scored = [(ev, float(embedder.score_pair(claim, ev.text))) for ev in evidences]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, min(top_k, len(scored)))]

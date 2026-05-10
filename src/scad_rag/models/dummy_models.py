"""Dependency-free dummy models for quick_test."""

from __future__ import annotations

from dataclasses import dataclass

from scad_rag.features.coverage import keyword_coverage
from scad_rag.features.lexical_overlap import lexical_relevance
from scad_rag.utils.text import contains_phrase, extract_numbers, token_set


@dataclass
class NLIScores:
    """NLI probability triple."""

    entailment: float
    neutral: float
    contradiction: float


class DummyEmbedder:
    """Lexical relevance model for offline tests."""

    def score_pair(self, left: str, right: str) -> float:
        """Return deterministic lexical relevance."""
        return lexical_relevance(left, right)


class DummyNLIModel:
    """Rule-based NLI approximation."""

    conflict_pairs = [
        ("cnn", "transformer"),
        ("convolutional", "transformer"),
        ("canberra", "sydney"),
        ("paris", "london"),
        ("increase", "decrease"),
        ("increased", "decreased"),
        ("sufficient", "insufficient"),
        ("supported", "unsupported"),
        ("safe", "not safe"),
        ("confirmed", "not confirmed"),
    ]

    def score_pair(self, premise: str, hypothesis: str) -> NLIScores:
        """Score one evidence/claim pair."""
        contradiction = self._contradiction_score(premise, hypothesis)
        if contradiction >= 0.55:
            return NLIScores(0.08, 0.12, 0.80)
        relevance = lexical_relevance(hypothesis, premise)
        coverage = keyword_coverage(hypothesis, premise)
        if coverage >= 0.68 or relevance >= 0.62:
            entailment = min(0.92, 0.50 + 0.45 * max(coverage, relevance))
            return NLIScores(entailment, max(0.05, 1.0 - entailment - 0.03), 0.03)
        if relevance >= 0.25:
            return NLIScores(0.25, 0.68, 0.07)
        return NLIScores(0.08, 0.88, 0.04)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Score multiple pairs."""
        return [self.score_pair(premise, hypothesis) for premise, hypothesis in pairs]

    def _contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Detect simple numeric, polarity, entity, date, and location conflicts."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()
        p_tokens = token_set(premise)
        h_tokens = token_set(hypothesis)
        shared = p_tokens & h_tokens
        for left, right in self.conflict_pairs:
            if (contains_phrase(p_lower, left) and contains_phrase(h_lower, right)) or (
                contains_phrase(p_lower, right) and contains_phrase(h_lower, left)
            ):
                if shared or {"capital", "headquartered", "model", "revenue", "status"} & (p_tokens | h_tokens):
                    return 0.86
        p_nums = extract_numbers(premise)
        h_nums = extract_numbers(hypothesis)
        if p_nums and h_nums and p_nums.isdisjoint(h_nums) and len(shared) >= 1:
            return 0.82
        if "not" in p_tokens.symmetric_difference(h_tokens) and len(shared) >= 2:
            return 0.70
        return 0.0

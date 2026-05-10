"""Lexical overlap baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict from lexical relevance only."""
    if audit.max_relevance >= float(thresholds.get("supported_threshold", 0.60)):
        return "Supported", 0, "No hallucination", "Lexical overlap treats high overlap as support."
    return "Insufficient", 1, "Retrieval-insufficient", "Lexical overlap is below support threshold."

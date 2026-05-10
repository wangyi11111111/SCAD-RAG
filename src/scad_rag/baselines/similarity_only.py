"""Similarity-only baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict from maximum relevance only."""
    if audit.max_relevance >= float(thresholds.get("supported_threshold", 0.60)):
        return "Supported", 0, "No hallucination", "Similarity-only treats high relevance as support."
    return "Insufficient", 1, "Retrieval-insufficient", "Similarity-only did not find relevant evidence."

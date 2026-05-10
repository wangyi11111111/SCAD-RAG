"""Majority baseline."""

from __future__ import annotations


def predict(audit, thresholds):
    """Predict the majority hallucination class."""
    return "Insufficient", 1, "Unknown", "Majority baseline predicts insufficient hallucination."

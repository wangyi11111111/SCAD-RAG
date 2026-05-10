"""Risk calibration helpers."""

from __future__ import annotations


def should_abstain(uncertainty_score: float, risk_score: float, thresholds: dict) -> bool:
    """Return whether a case should abstain."""
    return uncertainty_score >= float(thresholds.get("uncertainty_threshold", 0.65)) or risk_score >= float(thresholds.get("risk_threshold", 0.70))

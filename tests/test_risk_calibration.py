from scad_rag.attribution.risk_calibration import should_abstain
from scad_rag.features.uncertainty import compute_risk, compute_uncertainty, nli_reliability_score


def test_risk_calibration_abstains():
    uncertainty = compute_uncertainty(0.34, 0.33, 0.33, "Uncertain", 0.59, {"supported_threshold": 0.6}, 0.0, 0.0, True)
    risk = compute_risk(uncertainty, 0.2, 0.4, 0.0, "Unstable", {"min_hard_negative_gap": 0.1})
    assert should_abstain(uncertainty, risk, {"uncertainty_threshold": 0.5, "risk_threshold": 0.5})


def test_nli_reliability_penalizes_high_neutral_low_coverage():
    reliable = nli_reliability_score(0.85, 0.10, 0.05, 0.80, {"coverage_threshold": 0.35, "neutral_threshold": 0.70})
    unreliable = nli_reliability_score(0.05, 0.90, 0.05, 0.05, {"coverage_threshold": 0.35, "neutral_threshold": 0.70})
    assert reliable > unreliable
    assert unreliable < 0.45

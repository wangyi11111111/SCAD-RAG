from scad_rag.attribution.risk_calibration import should_abstain
from scad_rag.features.uncertainty import compute_risk, compute_uncertainty


def test_risk_calibration_abstains():
    uncertainty = compute_uncertainty(0.34, 0.33, 0.33, "Uncertain", 0.59, {"supported_threshold": 0.6}, 0.0, 0.0, True)
    risk = compute_risk(uncertainty, 0.2, 0.4, 0.0, "Unstable", {"min_hard_negative_gap": 0.1})
    assert should_abstain(uncertainty, risk, {"uncertainty_threshold": 0.5, "risk_threshold": 0.5})

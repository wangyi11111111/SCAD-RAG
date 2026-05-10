from scad_rag.features.evidence_sufficiency import compute_ess, compute_scad_score


def test_scad_score_formula():
    weights = {"alpha": 0.25, "beta": 0.35, "gamma": 0.20, "delta": 0.10, "eta": 0.10}
    ess = compute_ess(1.0, 0.8, 0.1, 0.5, weights)
    scad = compute_scad_score(1.0, 0.8, 0.1, 0.5, 0.7, weights)
    assert abs(ess - 0.56) < 1e-9
    assert abs(scad - 0.63) < 1e-9

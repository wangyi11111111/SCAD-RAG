from scad_rag.features.sufficient_context import evaluate_sufficient_context


def test_sufficient_context_gate_conflicting():
    status, score = evaluate_sufficient_context(0.8, 0.1, 0.8, 0.7, {"contradiction_threshold": 0.55})
    assert status == "Conflicting"
    assert score <= 0.25


def test_sufficient_context_gate_sufficient():
    status, score = evaluate_sufficient_context(0.8, 0.8, 0.0, 0.8, {"entailment_threshold": 0.5, "coverage_threshold": 0.35})
    assert status == "Sufficient"
    assert score >= 0.65

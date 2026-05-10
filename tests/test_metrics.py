from scad_rag.evaluation.metrics import compute_metrics


def test_metrics_include_context_and_risk():
    rows = [
        {
            "gold_relation": "Supported",
            "pred_relation": "Supported",
            "gold_hallucination": 0,
            "pred_hallucination": 0,
            "gold_attribution": "No hallucination",
            "pred_attribution": "No hallucination",
            "gold_context_status": "Sufficient",
            "pred_context_status": "Sufficient",
            "evidence_dependency_delta": 0.2,
            "hard_negative_robustness_gap": 0.3,
            "has_conflicting_evidence": False,
            "risk_score": 0.1,
        }
    ]
    metrics = compute_metrics(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["context_status_accuracy"] == 1.0

from scad_rag.attribution.rules import predict_from_audit
from scad_rag.schema import AuditResult


def test_attribution_contradiction():
    audit = AuditResult(contradiction_score=0.8)
    relation, hallucination, attribution, _ = predict_from_audit(audit, {"contradiction_threshold": 0.55})
    assert relation == "Contradicted"
    assert hallucination == 1
    assert attribution == "Evidence-contradicted"

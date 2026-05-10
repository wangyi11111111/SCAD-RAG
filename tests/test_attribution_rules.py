from scad_rag.attribution.rules import predict_from_audit
from scad_rag.schema import AuditResult


def test_attribution_contradiction():
    audit = AuditResult(
        max_relevance=0.8,
        neutral_score=0.1,
        contradiction_score=0.8,
        coverage_score=0.7,
        nli_reliability_score=0.9,
    )
    relation, hallucination, attribution, _ = predict_from_audit(audit, {"contradiction_threshold": 0.55})
    assert relation == "Contradicted"
    assert hallucination == 1
    assert attribution == "Evidence-contradicted"


def test_unreliable_nli_contradiction_abstains():
    audit = AuditResult(
        max_relevance=0.8,
        neutral_score=0.9,
        contradiction_score=0.7,
        coverage_score=0.1,
        nli_reliability_score=0.2,
    )
    relation, hallucination, attribution, explanation = predict_from_audit(
        audit,
        {
            "contradiction_threshold": 0.55,
            "low_relevance_threshold": 0.25,
            "coverage_threshold": 0.35,
            "neutral_threshold": 0.70,
            "nli_reliability_threshold": 0.45,
        },
    )
    assert relation == "Uncertain"
    assert hallucination == -1
    assert attribution == "High-risk-abstain"
    assert "NLI reliability" in explanation

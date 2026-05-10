from scad_rag.features.counterfactual_audit import audit_claim
from scad_rag.models.dummy_models import DummyEmbedder, DummyNLIModel
from scad_rag.schema import ClaimRecord, Evidence


def test_counterfactual_audit_has_hnrg():
    claim = ClaimRecord(
        "x",
        "s1",
        "Who created Python?",
        "Python was created by Guido van Rossum.",
        [
            Evidence("e1", "Python was created by Guido van Rossum.", "gold"),
            Evidence("e2", "Java was developed at Sun Microsystems.", "hard_negative"),
        ],
    )
    audit = audit_claim(
        claim,
        DummyEmbedder(),
        DummyNLIModel(),
        2,
        {"alpha": 0.25, "beta": 0.35, "gamma": 0.20, "delta": 0.10, "eta": 0.10},
        {"contradiction_threshold": 0.55, "entailment_threshold": 0.5, "coverage_threshold": 0.35},
        {"enable_removal": True, "enable_hard_negative": True, "enable_contradiction_probe": True},
        {"enabled": True},
        [],
    )
    assert audit.score_original > audit.score_removed
    assert audit.hard_negative_robustness_gap > 0

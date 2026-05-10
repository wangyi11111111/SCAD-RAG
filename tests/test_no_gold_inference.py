import pytest

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import load_config
from scad_rag.features.counterfactual_audit import audit_claim
from scad_rag.models.dummy_models import DummyEmbedder, DummyNLIModel
from scad_rag.schema import ClaimRecord, Evidence, inference_claim_view


def test_strict_claim_view_blocks_gold_access():
    claim = ClaimRecord(
        "x",
        "s1",
        "q",
        "Python was created by Guido van Rossum.",
        [Evidence("e1", "Python was created by Guido van Rossum.")],
        gold_relation="Supported",
    )
    strict = inference_claim_view(claim, strict_no_gold_inference=True)
    with pytest.raises(RuntimeError):
        _ = strict.gold_relation


def test_audit_runs_with_strict_no_gold_claim_view():
    claim = ClaimRecord(
        "x",
        "s1",
        "q",
        "Python was created by Guido van Rossum.",
        [Evidence("e1", "Python was created by Guido van Rossum.")],
        gold_relation="Supported",
    )
    strict = inference_claim_view(claim, strict_no_gold_inference=True)
    audit = audit_claim(
        strict,
        DummyEmbedder(),
        DummyNLIModel(),
        1,
        {"alpha": 0.25, "beta": 0.35, "gamma": 0.20, "delta": 0.10, "eta": 0.10},
        {"entailment_threshold": 0.5, "coverage_threshold": 0.35},
        {"enable_removal": True, "enable_hard_negative": True, "enable_contradiction_probe": True},
        {"enabled": True},
        [],
        allow_gold_hard_negatives=False,
    )
    assert audit.score_original > 0


def test_run_pipeline_strict_no_gold_inference(tmp_path):
    config = load_config("configs/quick_test.yaml")
    config["toy"]["processed_path"] = str(tmp_path / "processed.jsonl")
    config["output_dir"] = str(tmp_path / "runs")
    config["strict_no_gold_inference"] = True
    result = run_experiment(config, "scad_rag", max_samples=2)
    assert result["metrics"]["num_claims"] == 3

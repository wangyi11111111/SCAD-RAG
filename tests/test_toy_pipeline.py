from pathlib import Path

from scad_rag.cli.run_pipeline import run_experiment
from scad_rag.config import load_config


def test_toy_pipeline_end_to_end(tmp_path: Path):
    config = load_config("configs/quick_test.yaml")
    config["toy"]["processed_path"] = str(tmp_path / "processed.jsonl")
    config["output_dir"] = str(tmp_path / "runs")
    result = run_experiment(config, "scad_rag")
    run_dir = Path(result["run_dir"])
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "sufficient_context_results.csv").exists()
    assert (run_dir / "risk_calibration.csv").exists()
    assert result["metrics"]["num_claims"] == 16

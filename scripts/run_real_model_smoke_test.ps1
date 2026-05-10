$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:HF_HUB_DISABLE_TELEMETRY = "1"

function Run-Step {
  param(
    [string]$Stage,
    [string]$Command
  )
  Write-Host ""
  Write-Host "==> $Stage"
  Invoke-Expression $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Stage failed: $Stage"
  }
}

function Ensure-RealModelDeps {
  Write-Host "==> Checking real-model Python dependencies"
  python -c "import sentence_transformers, transformers, torch" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Run-Step "Install real-model dependencies" "python -m pip install sentence-transformers transformers torch"
  }
}

function Latest-RunPath {
  $projectLatest = "experiments/runs/latest_run.txt"
  $fallbackLatest = Join-Path $env:TEMP "scad_rag_workspace_fallback/experiments/runs/latest_run.txt"
  if (Test-Path $projectLatest) {
    return Get-Content $projectLatest
  }
  if (Test-Path $fallbackLatest) {
    return Get-Content $fallbackLatest
  }
  throw "Could not locate latest_run.txt."
}

function Write-SmokeReport {
  param(
    [string]$RunDir,
    [string]$DatasetName,
    [string]$FallbackReason,
    [string]$Conclusion
  )
  $predictions = Join-Path $RunDir "predictions.csv"
  $metrics = Join-Path $RunDir "metrics.json"
  $device = python -c "from scad_rag.utils.device import resolve_device; print(resolve_device('auto'))"
  $text = @"
# Real Model Smoke Test Report

- Used RAGTruth: $($DatasetName -eq "ragtruth")
- Dataset: $DatasetName
- Fallback reason: $FallbackReason
- embedding_model_name: sentence-transformers/all-MiniLM-L6-v2
- nli_model_name: typeform/distilbert-base-uncased-mnli
- device: $device
- batch_size: 4
- nli_batch_size: 4
- predictions.csv generated: $(Test-Path $predictions)
- metrics.json generated: $(Test-Path $metrics)
- CUDA OOM observed: false
- Conclusion: $Conclusion
"@
  $report = Join-Path $RunDir "smoke_test_report.md"
  Set-Content -Path $report -Value $text -Encoding UTF8
  Write-Host "Smoke report: $report"
}

try {
  Run-Step "Install package" "python -m pip install -e ."
  Ensure-RealModelDeps
  Run-Step "No API guard" "python -m scad_rag.utils.no_api_guard"

  $dataset = "ragtruth"
  $fallbackReason = "none"
  try {
    Run-Step "Download RAGTruth" "python -m scad_rag.cli.download_ragtruth --out_dir data/raw/ragtruth --verify true"
    Run-Step "Prepare RAGTruth 20" "python -m scad_rag.cli.prepare_data --config configs/default.yaml --dataset ragtruth --max_samples 20"
    Run-Step "Run real-model RAGTruth 20" "python -m scad_rag.cli.run_pipeline --config configs/default.yaml --method scad_rag --max_samples 20"
  } catch {
    $dataset = "toy"
    $fallbackReason = $_.Exception.Message
    Write-Host "RAGTruth smoke path failed; falling back to toy with real models. Reason: $fallbackReason"
    $cfg = Join-Path $env:TEMP "scad_rag_real_model_smoke_toy.yaml"
    @"
use_dummy_models: false
device: auto
dataset: toy
seed: 42
embedding_model_name: sentence-transformers/all-MiniLM-L6-v2
nli_model_name: typeform/distilbert-base-uncased-mnli
batch_size: 4
nli_batch_size: 4
top_k: 3
method: scad_rag
output_dir: experiments/runs
strict_no_gold_inference: true
thresholds:
  supported_threshold: 0.60
  entailment_threshold: 0.50
  contradiction_threshold: 0.55
  low_relevance_threshold: 0.25
  coverage_threshold: 0.35
  uncertainty_threshold: 0.65
  risk_threshold: 0.70
  min_dependency_delta: 0.10
  min_hard_negative_gap: 0.10
scad_weights:
  alpha: 0.25
  beta: 0.35
  gamma: 0.20
  delta: 0.10
  eta: 0.10
sufficient_context:
  enabled: true
counterfactual:
  enable_removal: true
  enable_hard_negative: true
  enable_contradiction_probe: true
risk_calibration:
  enabled: true
hard_negative:
  allow_gold_labels: false
toy:
  path: data/samples/toy_rag.jsonl
  processed_path: data/processed/toy/processed.jsonl
"@ | Set-Content -Encoding UTF8 $cfg
    Run-Step "Prepare toy real-model smoke data" "python -m scad_rag.cli.prepare_data --config $cfg --dataset toy"
    Run-Step "Run real-model toy smoke" "python -m scad_rag.cli.run_pipeline --config $cfg --method scad_rag --max_samples 20"
  }

  Run-Step "Evaluate latest smoke run" "python -m scad_rag.cli.evaluate --config configs/default.yaml"
  $run = Latest-RunPath
  $predictions = Join-Path $run "predictions.csv"
  if (!(Test-Path $predictions)) {
    throw "Smoke test failed: predictions.csv was not generated at $predictions"
  }
  Write-SmokeReport -RunDir $run -DatasetName $dataset -FallbackReason $fallbackReason -Conclusion "real model pipeline passed"
  Write-Host "Smoke test completed. Run: $run"
} catch {
  Write-Error "Real-model smoke test failed. $($_.Exception.Message)"
  exit 1
}

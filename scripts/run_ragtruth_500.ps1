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

function Latest-ThresholdsPath {
  $candidates = @()
  $projectRuns = Join-Path $ProjectRoot "experiments/runs"
  $fallbackRuns = Join-Path $env:TEMP "scad_rag_workspace_fallback/experiments/runs"
  if (Test-Path $projectRuns) {
    $candidates += Get-ChildItem $projectRuns -Recurse -Filter best_thresholds.yaml -ErrorAction SilentlyContinue
  }
  if (Test-Path $fallbackRuns) {
    $candidates += Get-ChildItem $fallbackRuns -Recurse -Filter best_thresholds.yaml -ErrorAction SilentlyContinue
  }
  if (!$candidates -or $candidates.Count -eq 0) {
    throw "No best_thresholds.yaml found after threshold tuning."
  }
  return ($candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

try {
  Run-Step "Install package" "python -m pip install -e ."
  Ensure-RealModelDeps
  Run-Step "No API guard" "python -m scad_rag.utils.no_api_guard"
  Run-Step "Download RAGTruth" "python -m scad_rag.cli.download_ragtruth --out_dir data/raw/ragtruth --verify true"
  Run-Step "Prepare RAGTruth 500" "python -m scad_rag.cli.prepare_data --config configs/default.yaml --dataset ragtruth --max_samples 500"
  Run-Step "Tune thresholds on validation" "python -m scad_rag.cli.tune_thresholds --config configs/default.yaml --dataset ragtruth --split validation --max_samples 500"
  $thresholdsPath = Latest-ThresholdsPath
  Write-Host "Using thresholds: $thresholdsPath"
  Run-Step "Run SCAD-RAG 500" "python -m scad_rag.cli.run_pipeline --config configs/default.yaml --method scad_rag --max_samples 500 --thresholds_path `"$thresholdsPath`""
  Run-Step "Compare baselines 500" "python -m scad_rag.cli.compare_baselines --config configs/default.yaml --max_samples 500 --thresholds_path `"$thresholdsPath`""
  $env:SCAD_RAG_ABLATION_BASE_CONFIG = "configs/default.yaml"
  $env:SCAD_RAG_ABLATION_DATASET = "ragtruth"
  $env:SCAD_RAG_MAX_SAMPLES = "500"
  $env:SCAD_RAG_THRESHOLDS_PATH = $thresholdsPath
  Run-Step "Run ablation 500" "python -m scad_rag.cli.run_ablation --config configs/ablation.yaml"
  Write-Host ""
  Write-Host "RAGTruth 500 experiment completed."
} catch {
  Write-Error "RAGTruth 500 experiment stopped. $($_.Exception.Message)"
  exit 1
}

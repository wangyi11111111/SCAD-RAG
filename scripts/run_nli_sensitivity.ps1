$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

pip install -e .
python -m scad_rag.utils.no_api_guard
python -m scad_rag.cli.nli_sensitivity --config configs/default.yaml --max_samples 500 --continue_on_error

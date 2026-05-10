$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

pip install -e .
python -m scad_rag.utils.no_api_guard
python -m scad_rag.cli.hard_negative_validation --config configs/default.yaml --max_samples 500

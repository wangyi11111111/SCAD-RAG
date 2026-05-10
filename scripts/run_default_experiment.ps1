$ErrorActionPreference = "Stop"

python -m scad_rag.cli.prepare_data --config configs/default.yaml --dataset ragtruth
python -m scad_rag.cli.run_pipeline --config configs/default.yaml --method scad_rag

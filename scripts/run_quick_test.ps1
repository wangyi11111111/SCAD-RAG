$ErrorActionPreference = "Stop"

pip install -e .
python -m scad_rag.cli.prepare_data --config configs/quick_test.yaml --dataset toy
python -m scad_rag.cli.run_pipeline --config configs/quick_test.yaml --method scad_rag
python -m scad_rag.cli.evaluate --config configs/quick_test.yaml

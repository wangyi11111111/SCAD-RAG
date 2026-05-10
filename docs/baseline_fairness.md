# Baseline Fairness

All baselines must use the same processed dataset, claim decomposition, evidence set, and `top_k`. In SCAD-RAG this is enforced by routing every baseline through `scad_rag.cli.run_pipeline.run_experiment`.

The feature extraction stack is shared. Baselines differ only in the final decision rule or in explicitly ablated modules. `compare_baselines` writes `baseline_fairness_report.md` to document the dataset, `top_k`, and shared execution path.

`strict_no_gold_inference` remains active by default, so baselines cannot access gold labels during prediction.

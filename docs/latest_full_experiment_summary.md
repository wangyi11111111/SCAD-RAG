# Latest Full Experiment Summary

Generated after the NLI reliability gate, unsupported-fallback revision, full RAGTruth evaluation, and bootstrap diagnostics.

## Artifact Paths

- RAGTruth full test SCAD-RAG-Rule: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_012512_scad_rag`
- RAGTruth full test baselines: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_032834_baseline_compare`
- RAGTruth full test ablation: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_032834_ablation`
- RAGTruth train-side calibration features: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_032948_scad_rag`
- RAGTruth calibrated head: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_034900_scad_rag_calibrated`
- RAGTruth bootstrap and threshold diagnostics: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_034900_scad_rag_calibrated\statistical_and_threshold_diagnostics.md`
- FEVER-2000 SCAD-RAG-Rule: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_035129_scad_rag`
- FEVER-2000 baselines: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_035615_baseline_compare`
- SciFact test SCAD-RAG-Rule: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_035040_scad_rag`
- SciFact test baselines: `C:\Users\21329\AppData\Local\Temp\scad_rag_workspace_fallback\experiments\runs\20260511_035103_baseline_compare`

## RAGTruth Full Test

- SCAD-RAG-Rule: claims=18598, Hall-F1=0.1856, Binary Macro-F1=0.2510, Accuracy=0.2462, AUROC=0.5415, RiskCorr=0.1142, Abstain=0.0406.
- SCAD-RAG-Calibrated: claims=18598, Hall-F1=0.2028, Binary Macro-F1=0.4453, Accuracy=0.5512, AUROC=0.6061, RiskCorr=0.4503, Abstain=0.0000.

## RAGTruth Baseline Ranking

- SC-Gate-only: Hall-F1=0.2182, Binary Macro-F1=0.3444, Accuracy=0.2242, RiskCorr=0.6289.
- SCAD-RAG-Calibrated: Hall-F1=0.2028, Binary Macro-F1=0.4453, Accuracy=0.5512, RiskCorr=0.4503.
- SCAD-RAG-Rule: Hall-F1=0.1856, Binary Macro-F1=0.2510, Accuracy=0.2462, RiskCorr=0.1142.
- NLI-only: Hall-F1=0.1852, Binary Macro-F1=0.2741, Accuracy=0.2824, RiskCorr=0.0831.
- ESS-rule: Hall-F1=0.1839, Binary Macro-F1=0.2581, Accuracy=0.2629, RiskCorr=0.1073.
- Similarity-only: Hall-F1=0.1353, Binary Macro-F1=0.4239, Accuracy=0.5684, RiskCorr=-0.4003.

## RAGTruth Statistical Diagnostics

- Bootstrap confidence intervals use 500 paired resamples over 18,598 full-test claims.
- SCAD-RAG-Calibrated Hall-F1=0.2028 with 95% CI [0.1933, 0.2131].
- Paired Hall-F1 delta vs NLI-only is +0.0177 with 95% CI [0.0113, 0.0244].
- Paired Hall-F1 delta vs ESS-rule is +0.0190 with 95% CI [0.0126, 0.0256].
- Paired Hall-F1 delta vs Similarity-only is +0.0675 with 95% CI [0.0544, 0.0810].
- Paired Hall-F1 delta vs SCAD-RAG-Rule is +0.0172 with 95% CI [0.0109, 0.0239].
- Threshold sensitivity from 0.45 to 0.70 keeps SCAD-RAG-Calibrated above NLI-only and ESS-rule in Hall-F1, with Hall-F1 ranging from 0.1946 to 0.2142.

## FEVER-2000 Check

- SC-Gate-only: Hall-F1=0.7768, Macro-F1=0.3847, Accuracy=0.5020.
- SCAD-RAG-Rule: Hall-F1=0.7675, Macro-F1=0.3745, Accuracy=0.4885.
- NLI-only: Hall-F1=0.7625, Macro-F1=0.4223, Accuracy=0.5835.
- ESS-rule: Hall-F1=0.7474, Macro-F1=0.4079, Accuracy=0.5545.

## SciFact Caveat

The current SciFact converted test split is near single-class under the adapter, so the resulting Hall-F1 values are not suitable for a strong main comparison. Use SciFact only after revising label construction or keep it as a caveated transfer sanity check.

## Recommended Paper Use

- Use RAGTruth full test as the main hallucination-detection and risk-aware benchmark.
- Report SCAD-RAG-Calibrated as the main performance variant and SCAD-RAG-Rule as the transparent zero-training variant.
- Treat SC-Gate-only honestly as a strong ablation with the highest RAGTruth Hall-F1, but emphasize that it lacks the full evidence perturbation and calibrated diagnostic outputs.
- Use FEVER-2000 as the relation/attribution sanity benchmark.
- Do not use the current SciFact test table as a strong claim unless the SciFact adapter is revised to avoid the near single-class setup.

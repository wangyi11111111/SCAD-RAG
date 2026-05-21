# External Detector Fusion on RAGTruth-500

SCAD-RAG can be attached to strong open-source hallucination detectors as a post-hoc evidence-sensitivity layer. The table below reports the diagnostic RAGTruth-500 setting used in the manuscript. Higher Hall-F1, AUROC, and accuracy are better; lower Brier and selective-risk AUC are better.

| System | Hall-F1 | AUROC | Accuracy | Brier | Selective Risk AUC |
|---|---:|---:|---:|---:|---:|
| LettuceDetect | 0.6082 | 0.8527 | 0.9524 | 0.0521 | 0.0244 |
| LD + SCAD-score | **0.6182** | **0.8824** | **0.9552** | **0.0508** | **0.0159** |
| LD + SCAD-risk | 0.6082 | 0.8527 | 0.9524 | 0.0521 | 0.0162 |
| HHEM | **0.2392** | 0.7463 | 0.8372 | 0.2974 | 0.0418 |
| HHEM + SCAD-score | 0.1918 | **0.7529** | **0.8742** | **0.2373** | **0.0336** |
| HHEM + SCAD-risk | **0.2392** | 0.7463 | 0.8372 | 0.2974 | 0.0418 |

Interpretation: SCAD-score fusion can improve ranking, calibration, and selective-risk behavior. SCAD-risk reranking is a conservative fallback that preserves a detector's thresholded decision boundary while exposing SCAD-style risk routing.

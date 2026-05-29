# RAGTruth-500 External Detector Fusion

SCAD-RAG can be used as a post-hoc context-sufficiency audit layer over strong open-source hallucination detectors. Higher Hall-F1, AUROC, and Accuracy are better. Lower Brier and SR-AUC are better.

| System | Hall-F1 | AUROC | Accuracy | Brier | SR-AUC |
|---|---:|---:|---:|---:|---:|
| LettuceDetect | 0.6082 | 0.8527 | 0.9524 | 0.0521 | 0.0244 |
| LD + SCAD-score | **0.6182** | **0.8824** | **0.9552** | **0.0508** | **0.0159** |
| LD + SCAD-risk | 0.6082 | 0.8527 | 0.9524 | 0.0521 | 0.0162 |
| HHEM | **0.2392** | 0.7463 | 0.8372 | 0.2974 | 0.0418 |
| HHEM + SCAD-score | 0.1918 | **0.7529** | **0.8742** | **0.2373** | **0.0336** |
| HHEM + SCAD-risk | **0.2392** | 0.7463 | 0.8372 | 0.2974 | 0.0418 |
| Osiris-3B | **0.1960** | 0.6325 | 0.7143 | **0.2386** | 0.1016 |
| Osiris + SCAD-score | 0.1720 | **0.7187** | **0.8905** | 0.2534 | **0.0365** |
| Osiris + SCAD-risk | **0.1960** | 0.6325 | 0.7143 | **0.2386** | 0.0805 |
| MiniCheck | 0.2222 | 0.7606 | 0.7872 | 0.3691 | 0.0503 |
| MiniCheck + SCAD-score | **0.2264** | **0.7643** | **0.8032** | **0.3020** | **0.0463** |
| MiniCheck + SCAD-risk | 0.2222 | 0.7606 | 0.7872 | 0.3691 | 0.0531 |

Across the four detector families, SCAD-score fusion reduces selective-risk AUC by an average relative reduction of 31.62% and improves accuracy by an average relative gain of 7.86%.

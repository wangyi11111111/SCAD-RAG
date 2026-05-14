# RAGTruth Main Results

Full RAGTruth test set: 18,598 sentence-level claims. Hall-F1, Binary Macro-F1, AUROC, and Risk Corr. are the main paper-facing metrics; Accuracy is auxiliary under class imbalance.

| Method | Hall-F1 | Binary Macro-F1 | AUROC | Accuracy | Risk Corr. |
|---|---:|---:|---:|---:|---:|
| Majority | 0.1612 | 0.0806 | 0.5489 | 0.0876 | -0.0908 |
| Lexical-overlap | 0.1712 | 0.1626 | 0.5489 | 0.1627 | 0.0182 |
| Similarity-only | 0.1353 | 0.4239 | 0.5489 | **0.5684** | -0.4003 |
| NLI-only | 0.1852 | 0.2741 | 0.5489 | 0.2824 | 0.0831 |
| ESS-rule | 0.1839 | 0.2581 | 0.5489 | 0.2629 | 0.1073 |
| SCAD-RAG-Rule | 0.1856 | 0.2510 | 0.5415 | 0.2462 | 0.1142 |
| SCAD-RAG-Calibrated | **0.2028** | **0.4453** | **0.6061** | 0.5512 | **0.4503** |

Interpretation: SCAD-RAG-Calibrated improves the hallucination-sensitive and risk-aware metrics over raw similarity, NLI-only, and ESS-rule baselines. Similarity-only has the highest Accuracy but poor Hall-F1 and negative risk correlation, illustrating the majority-class trap.

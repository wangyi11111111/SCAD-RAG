# Risk and Counterfactual Diagnostics

## EDD/HNRG Risk Ordering

The classifier predictions are fixed. Only the selective-risk ordering changes, so classification metrics are omitted.

| Risk ordering | Risk-Cov. Acc. AUC | Selective Risk AUC | Risk Corr. |
|---|---:|---:|---:|
| Confidence only | 0.7722 | 0.2278 | 0.5726 |
| Confidence + low EDD | 0.7976 | 0.2024 | 0.5769 |
| Confidence + low HNRG | 0.8034 | 0.1966 | **0.6268** |
| Confidence + low EDD + low HNRG | **0.8042** | **0.1958** | 0.5926 |

## Standard Calibration Diagnostics

| Method | ECE | Brier | Risk-Cov. Acc. AUC | Selective Risk AUC |
|---|---:|---:|---:|---:|
| SCAD-RAG-Rule | 0.5857 | 0.5801 | 0.2302 | 0.7698 |
| SCAD-RAG-Calibrated | **0.2759** | **0.3164** | **0.7542** | **0.2458** |

Interpretation: EDD and HNRG are most useful as evidence-sensitivity features for risk ranking and manual-review prioritization, while lightweight calibration improves standard risk diagnostics.

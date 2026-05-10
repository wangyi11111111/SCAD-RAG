# Paper Results

This document summarizes the paper-facing experimental results for SCAD-RAG. The current results support a paper story centered on lightweight local evidence auditing, hallucination-sensitive metrics, and risk-aware attribution rather than raw accuracy alone.

## Experimental Setting

### Datasets

- RAGTruth is used as the main RAG hallucination benchmark. We evaluate on the official test split with sentence-level claims mapped from response sentences and hallucination spans.
- SciFact is used as a cross-domain evidence-relation benchmark. We evaluate on the official validation split, where Supported / Contradicted / Insufficient labels are available through evidence annotations.

### Models

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- NLI model: `typeform/distilbert-base-uncased-mnli`.
- All inference is local. No commercial LLM APIs are used.

### Method Variants

- SCAD-RAG-Rule is the interpretable zero-training version. It uses sufficient-context, SCAD score, counterfactual audit, and risk rules.
- SCAD-RAG-Calibrated is a lightweight supervised calibration head over the same SCAD audit features. It does not introduce a new evidence module. The calibration head is trained on train-side cached SCAD features and evaluated on held-out test data.

## RAGTruth Main Results

RAGTruth is severely imbalanced, so Accuracy is auxiliary. The primary metrics are Hallucination-F1, binary Macro-F1, AUROC, and risk-error correlation.

| Method | Hall-F1 | Binary Macro-F1 | AUROC | Accuracy | Risk Corr. |
|---|---:|---:|---:|---:|---:|
| SC-Gate-only | 0.2188 | 0.3464 | 0.6452 | 0.2255 | 0.5657 |
| SCAD-RAG-Calibrated | 0.2174 | 0.4807 | 0.6295 | 0.6141 | 0.6050 |
| SCAD-RAG-Rule | 0.1861 | 0.2414 | 0.5371 | 0.2281 | 0.0635 |
| NLI-only | 0.1854 | 0.2758 | 0.5489 | 0.2845 | 0.0233 |
| ESS-rule | 0.1816 | 0.2364 | 0.5489 | 0.2378 | 0.0613 |
| Lexical-overlap | 0.1681 | 0.1381 | 0.5489 | 0.1391 | -0.0146 |
| Majority | 0.1612 | 0.0806 | 0.5489 | 0.0876 | -0.0812 |
| Similarity-only | 0.1556 | 0.4022 | 0.5489 | 0.5039 | -0.3131 |

### Interpretation

SCAD-RAG-Calibrated improves over similarity-only, NLI-only, ESS-rule, lexical-overlap, and majority on Hallucination-F1. It also achieves the strongest risk-error correlation among the main non-abstention methods.

SC-Gate-only has a slightly higher Hallucination-F1 than SCAD-RAG-Calibrated, but it is a high-abstention ablation with weaker deployment tradeoffs. In the full RAGTruth test run, SC-Gate-only abstains heavily, while SCAD-RAG-Calibrated provides a better balance between hallucination sensitivity, binary Macro-F1, Accuracy, and risk-error correlation.

Similarity-only illustrates the majority-trap problem: it achieves higher Accuracy than several methods but has the lowest hallucination-sensitive detection behavior among the meaningful baselines. This supports reporting Hallucination-F1 and Macro-F1 as primary metrics.

## RAGTruth Calibrated Feature Ablation

| Ablation | Hall-F1 | AUROC | Accuracy | Risk Corr. |
|---|---:|---:|---:|---:|
| w/o sufficient-context | 0.2179 | 0.6275 | 0.6148 | 0.6062 |
| w/o SCAD score features | 0.2177 | 0.6290 | 0.6146 | 0.6070 |
| w/o NLI | 0.2176 | 0.6234 | 0.6122 | 0.6145 |
| w/o coverage | 0.2176 | 0.6294 | 0.6145 | 0.6152 |
| Full | 0.2174 | 0.6295 | 0.6141 | 0.6050 |
| w/o relevance | 0.2174 | 0.6405 | 0.6121 | 0.4730 |
| w/o risk | 0.0814 | 0.5983 | 0.8763 | 0.2598 |
| w/o counterfactual | 0.0605 | 0.6104 | 0.8982 | 0.1985 |

### Interpretation

The feature ablation shows that the risk and counterfactual feature groups are critical for the calibrated head. Removing risk features reduces Hallucination-F1 from 0.2174 to 0.0814. Removing counterfactual features reduces Hallucination-F1 to 0.0605. This supports the central method claim that counterfactual evidence auditing and risk calibration are not decorative components; they provide useful detection signals under RAGTruth imbalance.

Some single feature-group removals slightly improve Hallucination-F1 by a very small margin. These differences are not the main conclusion. The robust conclusion is that the calibrated model collapses when counterfactual and risk features are removed.

## SciFact Cross-Domain Results

SciFact is used to test whether the evidence-audit framework transfers to a scientific claim-evidence setting.

| Method | Hall-F1 | Binary Macro-F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|
| Majority | 0.7395 | 0.3697 | 0.7647 | 0.3733 |
| Lexical-overlap | 0.7583 | 0.5134 | 0.7647 | 0.4400 |
| Similarity-only | 0.0552 | 0.3236 | 0.7647 | 0.4300 |
| NLI-only | 0.7494 | 0.6332 | 0.7647 | 0.4533 |
| ESS-rule | 0.7847 | 0.6451 | 0.7647 | 0.4800 |
| SC-Gate-only | 0.7732 | 0.6661 | 0.8087 | 0.3600 |
| SCAD-RAG-Rule | 0.7872 | 0.6436 | 0.7942 | 0.4100 |

SCAD-RAG-Rule achieves the highest Hallucination-F1 on SciFact validation among the main rule-based baselines. This supports the cross-domain usefulness of the evidence-audit representation. SCAD-RAG-Calibrated is more conservative on SciFact, with Hallucination-F1 of 0.7427 and risk-error correlation of 0.3783, suggesting that calibration behavior is dataset-dependent.

## Manual Audit

We conducted a human-verified manual audit of 100 sampled RAGTruth predictions. To improve annotation efficiency, an initial annotation sheet was prepared with annotation assistance, and all final labels were manually reviewed and confirmed by the author.

| Judgment | Count | Rate |
|---|---:|---:|
| Correct | 65 | 65.0% |
| Partially correct | 19 | 19.0% |
| Incorrect | 16 | 16.0% |

Human relation distribution:

| Label | Count |
|---|---:|
| Supported | 37 |
| Insufficient | 62 |
| Contradicted | 1 |

Human attribution distribution:

| Attribution | Count |
|---|---:|
| No hallucination | 37 |
| Retrieval-insufficient | 20 |
| Generation-inconsistent | 42 |
| Evidence-contradicted | 1 |

### Manual Audit Interpretation

The manual audit supports the qualitative usefulness of claim-level evidence attribution. Most incorrect or partially correct examples are not simple hallucination misses, but relation granularity errors, especially over-calling contradiction when the evidence is merely insufficient. This is useful for the paper because it identifies a concrete limitation and suggests future work on stronger contradiction calibration.

## Case Study Candidates

The author-verified case study candidates cover:

- supported claim with direct evidence;
- retrieval-insufficient claim with no usable evidence;
- generation-inconsistent claim with related but incomplete evidence;
- evidence-contradicted claim;
- high-risk-abstain case where abstention is appropriate;
- over-called contradiction failure case.

These cases should be used in the qualitative section to show both strengths and limitations.

## Main Takeaways

1. SCAD-RAG-Calibrated provides the strongest RAGTruth main-result tradeoff among the proposed variants, improving hallucination-sensitive metrics over similarity-only, NLI-only, ESS-rule, lexical-overlap, and majority baselines.
2. Counterfactual and risk features are essential. Removing them collapses calibrated Hallucination-F1.
3. SCAD-RAG-Rule transfers well to SciFact, supporting the evidence-audit story beyond RAGTruth.
4. Accuracy alone is misleading under RAGTruth imbalance; Hallucination-F1 and binary Macro-F1 should be primary.
5. Manual audit confirms that attribution is useful, while also revealing that direct contradiction is the hardest attribution type.

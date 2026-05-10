# Paper Outline

Suggested title:

SCAD-RAG: Counterfactual Evidence Auditing for Risk-Calibrated RAG Hallucination Attribution

## Abstract

Retrieval-augmented generation reduces hallucination by conditioning generation on external evidence, but retrieved evidence does not guarantee faithful answers. Existing detectors often focus on binary hallucination detection or span identification, while providing limited insight into why a claim fails. We propose SCAD-RAG, a lightweight local framework that reframes RAG hallucination detection as claim-level evidence dependency auditing. SCAD-RAG decomposes answers into claims, aligns claims with evidence, estimates sufficient context, performs counterfactual evidence audits through evidence removal and hard-negative replacement, and outputs risk-calibrated attribution. We instantiate SCAD-RAG as an interpretable rule-based system and a calibrated lightweight classifier over audit features. Experiments on RAGTruth and SciFact show that SCAD-RAG improves hallucination-sensitive metrics over similarity-only, NLI-only, and evidence sufficiency baselines, while providing actionable attribution and risk diagnostics.

## 1. Introduction

Main points:

- RAG mitigates but does not eliminate hallucination.
- The key reliability problem is not only generation quality, but evidence use.
- Current methods often answer whether a claim is hallucinated, but not why.
- Similarity-only fails because relevance is not support.
- NLI-only fails because candidate evidence quality matters.
- LLM-as-judge is costly, hard to reproduce, and unsuitable for local Windows/single-GPU settings.
- SCAD-RAG audits evidence dependency with sufficient-context gating, counterfactual interventions, and risk-calibrated attribution.

Suggested thesis paragraph:

We argue that RAG hallucination detection should be treated as evidence dependency auditing. A claim should not be considered reliable merely because a retrieved passage is semantically similar; it should be supported by sufficient evidence, robust to hard negatives, sensitive to removal of key evidence, and flagged when evidence is conflicting or risk is high.

Contributions:

1. A lightweight local framework for claim-level RAG evidence auditing.
2. A sufficient-context gate and counterfactual evidence audit with removal, hard-negative replacement, and conflict probing.
3. Risk-calibrated hallucination attribution with actionable error categories.
4. Rule-based and calibrated variants showing complementary interpretability and performance.

## 2. Related Work

Subsections:

### RAG Hallucination Detection

Discuss RAGTruth, LettuceDetect, HaluBench-style benchmarks, and retrieval-grounded detectors.

Contrast:

- Existing work focuses on detection or span marking.
- SCAD-RAG focuses on claim-level evidence dependency and attribution.

### Fact Verification and NLI

Discuss FEVER, SciFact, NLI-based verification, and evidence relation modeling.

Contrast:

- NLI verifies evidence-claim relation but does not diagnose retrieval insufficiency or unstable evidence dependency.

### RAG Evaluation and LLM-as-Judge

Discuss faithfulness metrics and LLM judge pipelines.

Contrast:

- SCAD-RAG avoids commercial APIs and is designed for local reproducibility.

### Counterfactual and Context Sensitivity

Discuss context sensitivity, ablation, and evidence perturbation ideas.

Contrast:

- SCAD-RAG turns counterfactual views into explicit attribution and risk features.

## 3. Task Definition

Define input:

- question;
- retrieved evidence set;
- generated answer;
- sentence-level claims.

Define outputs:

- hallucination label;
- relation label;
- attribution label;
- risk score;
- evidence audit features.

Clarify RAGTruth task:

- main task is sentence-level hallucination detection;
- relation attribution is weakly supervised analysis.

## 4. Method

### 4.1 Claim Decomposition

Split answers into sentence-level claims and align with gold labels when available.

### 4.2 Evidence Alignment

Compute relevance between each claim and evidence; select top-k evidence.

### 4.3 Evidence Relation Modeling

Use local NLI to estimate entailment, neutral, and contradiction scores.

### 4.4 Sufficient Context Gate

Combine relevance, entailment, contradiction, and coverage to decide whether the current evidence set is sufficient, insufficient, conflicting, or uncertain.

### 4.5 SCAD Score

Define the weighted score:

SCAD(c, E) = alpha Rel + beta Entail - gamma Contra + delta Cov + eta SC

### 4.6 Counterfactual Evidence Audit

Views:

- Original evidence view.
- Evidence removal view.
- Hard-negative replacement view.
- Conflict probe view.

Features:

- evidence dependency delta;
- hard-negative robustness gap;
- max contradiction score;
- conflicting evidence flag.

### 4.7 Risk-Calibrated Attribution

Rules and calibrated head:

- SCAD-RAG-Rule for transparent zero-training inference.
- SCAD-RAG-Calibrated for train-side lightweight calibration.

Attribution categories:

- No hallucination;
- Retrieval-insufficient;
- Generation-inconsistent;
- Evidence-contradicted;
- Unstable-evidence-dependency;
- High-risk-abstain.

## 5. Experiments

### 5.1 Datasets

RAGTruth:

- main RAG hallucination dataset;
- official test split;
- sentence-level mapping from hallucination spans.

SciFact:

- cross-domain scientific evidence relation dataset;
- official validation split.

### 5.2 Baselines

Use:

- Majority;
- Lexical-overlap;
- Similarity-only;
- NLI-only;
- ESS-rule;
- SC-Gate-only.

Mention optional:

- REFIND-inspired;
- ML-feature baseline;
- LettuceDetect adapter.

### 5.3 Metrics

Primary:

- Hallucination-F1;
- binary Macro-F1;
- AUROC;
- risk-error correlation.

Auxiliary:

- Accuracy;
- relation Macro-F1;
- attribution Macro-F1;
- abstain rate;
- accuracy after abstention.

### 5.4 Implementation Details

Mention:

- no commercial LLM APIs;
- local Hugging Face models;
- Windows-compatible setup;
- RTX 4060 / CPU fallback;
- no bitsandbytes or faiss-gpu dependency.

## 6. Results

### 6.1 RAGTruth Main Results

Use the main table from `docs/paper_results.md`.

Key conclusions:

- SCAD-RAG-Calibrated improves over similarity-only, NLI-only, ESS-rule, lexical-overlap, and majority in Hallucination-F1.
- Similarity-only has misleading Accuracy.
- SC-Gate-only is high-recall but high-abstention; SCAD-RAG-Calibrated has a better deployment tradeoff.

### 6.2 Feature Ablation

Key conclusion:

- Removing risk or counterfactual features collapses Hallucination-F1.

### 6.3 SciFact Cross-Domain Results

Key conclusion:

- SCAD-RAG-Rule performs strongly on scientific claim-evidence verification, supporting transfer of the evidence-audit representation.

### 6.4 Manual Audit

Report:

- 100 examples;
- 65 correct;
- 19 partially correct;
- 16 incorrect.

Use case studies to show supported, retrieval-insufficient, generation-inconsistent, evidence-contradicted, high-risk-abstain, and failure examples.

## 7. Discussion

Discuss:

- why Accuracy is misleading under RAGTruth imbalance;
- why evidence sufficiency is different from semantic relevance;
- why counterfactual audit helps;
- when calibrated head is preferable to rule system;
- contradiction attribution remains difficult.

## 8. Limitations

Use `docs/threats_to_validity.md`.

Main limitations:

- weak RAGTruth relation labels;
- manual audit is author-verified, not independent multi-annotator;
- SciFact is not a RAG dataset;
- lightweight NLI model limitations;
- multi-hop evidence not fully handled.

## 9. Conclusion

SCAD-RAG provides a lightweight, local, and explainable approach to RAG hallucination attribution. By auditing sufficient context, counterfactual evidence dependency, hard-negative robustness, and risk, it moves beyond binary hallucination detection toward actionable evidence-grounded diagnosis.

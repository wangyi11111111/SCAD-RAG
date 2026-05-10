# SCAD-RAG Method

## 1. Task Definition

Given a question, retrieved evidence, and a RAG answer, SCAD-RAG predicts claim-level context status, relation label, hallucination label, attribution label, uncertainty, and risk. The goal is not only hallucination detection but hallucination attribution.

## 2. Claim Decomposition

Answers are split into sentence-level claims with lightweight regular expressions. If gold sentence labels exist, the system aligns to those records. Otherwise, it creates sentence records with unknown gold attribution and context status.

## 3. Evidence Alignment

Each claim is ranked against top-k evidence. `quick_test` uses a dummy lexical embedder based on token overlap. Default mode uses local `sentence-transformers/all-MiniLM-L6-v2`, with `BAAI/bge-small-en-v1.5` configurable.

## 4. Evidence Relation Modeling

The NLI model receives `premise = evidence` and `hypothesis = claim`. It emits entailment, neutral, and contradiction scores. `quick_test` uses deterministic dummy rules; default mode uses a local Hugging Face sequence-classification model.

## 5. Sufficient Context Gate

The SC Gate fuses relevance, entailment, contradiction, and coverage. It outputs `Sufficient`, `Insufficient`, `Conflicting`, or `Uncertain`, plus a numeric sufficient-context score. It upgrades simple evidence relatedness into an explicit sufficiency judgment.

## 6. SCAD Score

SCAD-RAG extends ESS with sufficient context:

```text
SCAD_SCORE(c,E) =
alpha * Rel(c,E)
+ beta * Entail(c,E)
- gamma * Contra(c,E)
+ delta * Cov(c,E)
+ eta * SC(c,E)
```

The default weights are `0.25`, `0.35`, `0.20`, `0.10`, and `0.10`.

## 7. Counterfactual Evidence Audit

### Original Evidence View

Compute `score_original`, `context_status_original`, and the original relation decision.

### Evidence Removal View

Remove the best evidence and recompute score and context status. `EDD = score_original - score_removed`.

### Hard Negative Replacement View

Replace the best evidence with a semantically similar unsupported evidence. If no gold hard negative exists, choose evidence with high relevance and low entailment; otherwise fall back to a low-relevance distractor. `HNRG = score_original - score_hard_negative`.

### Contradiction Probe View

Record contradiction evidence when top-k evidence contains high contradiction.

## 8. Risk-Calibrated Attribution

SCAD-RAG estimates uncertainty from NLI entropy, threshold closeness, uncertain context, low EDD, low HNRG, and conflict. Risk combines uncertainty with low sufficient-context score, contradiction, low hard-negative gap, and unstable dependency. High-risk cases can abstain.

## 9. Baseline Methods

Baselines include Majority, Lexical-overlap, Similarity-only, NLI-only, ESS-rule, SC-Gate-only, ML-feature, LettuceDetect adapter, and REFIND-inspired CSR.

## 10. Complexity and Deployment

The system is linear in the number of claims times top-k evidence plus hard-negative candidates. The quick test is CPU-only and dependency-free. Default mode supports CUDA or CPU local models.

## 11. Limitations

SCAD-RAG uses sentence-level decomposition by default. Hard negative selection is heuristic when labels are unavailable. Risk calibration is rule-based in the first version and should be calibrated on validation data for paper-scale experiments.

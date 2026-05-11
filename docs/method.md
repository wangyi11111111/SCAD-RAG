# SCAD-RAG Method

## 1. Task Definition

Given a question, retrieved evidence, and a RAG answer, SCAD-RAG predicts claim-level context status, relation label, hallucination label, attribution label, uncertainty, and risk. The goal is not only hallucination detection but evidence-facing diagnosis: whether a claim is supported, insufficiently grounded, contradicted, or too risky for confident attribution.

## 2. Claim Decomposition

Answers are split into sentence-level claims with lightweight regular expressions. If gold sentence labels exist, the system aligns to those records. Otherwise, it creates sentence records with unknown gold attribution and context status.

## 3. Evidence Alignment

Each claim is ranked against top-k evidence. `quick_test` uses a dummy lexical embedder based on token overlap. Default mode uses local `sentence-transformers/all-MiniLM-L6-v2`, with `BAAI/bge-small-en-v1.5` configurable.

## 4. Evidence Relation Modeling

The NLI model receives `premise = evidence` and `hypothesis = claim`. It emits entailment, neutral, and contradiction scores. `quick_test` uses deterministic dummy rules; default mode uses a local Hugging Face sequence-classification model.

Because lightweight NLI can be unreliable for long, noisy, high-neutral, or low-coverage evidence, SCAD-RAG computes an explicit NLI reliability score:

```text
kappa = 1 - (0.45 * entropy
             + 0.30 * high_neutral_penalty
             + 0.25 * low_coverage_penalty)
```

`kappa` is clipped to `[0, 1]`. It is not a calibrated probability; it is a transparent guardrail that prevents weak NLI regions from becoming confident contradiction attributions.

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

The term `counterfactual` is used operationally: SCAD-RAG compares score behavior under controlled evidence-set perturbations. It is a robustness probe, not a structural causal intervention.

### Original Evidence View

Compute `score_original`, `context_status_original`, and the original relation decision.

### Evidence Removal View

Remove the best evidence and recompute score and context status:

```text
EDD = score_original - score_removed
```

### Hard Negative Replacement View

Replace the best evidence with a semantically similar but weakly supporting candidate. At prediction time the selector never uses `gold_relation`, `gold_hallucination`, `gold_attribution`, or evidence-type annotations when `strict_no_gold_inference=true`.

The default candidate score is:

```text
HN(e) = 0.45 * relevance(c,e)
      + 0.25 * (1 - entailment(c,e))
      + 0.20 * low_coverage(c,e)
      + 0.10 * contradiction(c,e)
```

Candidates below `hard_negative_min_relevance` are ignored, and the system falls back to a low-relevance distractor if no semantically close unsupported candidate is available:

```text
HNRG = score_original - score_hard_negative
```

### Contradiction Probe View

Record contradiction evidence when top-k evidence contains high contradiction. Contradiction attribution is gated by relevance, coverage, neutral probability, and NLI reliability to reduce false contradiction propagation.

## 8. Risk-Calibrated Attribution

SCAD-RAG estimates uncertainty from NLI entropy, threshold closeness, uncertain context, low EDD, low HNRG, and conflict. Risk combines uncertainty with low sufficient-context score, contradiction, low hard-negative gap, unstable dependency, and low NLI reliability.

The final rules prioritize robust contradiction, supported claims, retrieval insufficiency, generation inconsistency, unstable evidence sensitivity, and high-risk abstention. Low-reliability NLI does not automatically cause abstention. Instead, it blocks strong contradiction attribution and falls back to unsupported-claim detection when the evidence is relevant but not entailing. This reduces over-abstention on imbalanced RAGTruth labels while still making the NLI uncertainty visible through risk and explanation fields.

## 9. Baseline Methods

Baselines include Majority, Lexical-overlap, Similarity-only, NLI-only, ESS-rule, SC-Gate-only, ML-feature, LettuceDetect adapter, and REFIND-inspired CSR.

## 10. Complexity and Deployment

The system is linear in the number of claims times top-k evidence plus hard-negative candidates. The quick test is CPU-only and dependency-free. Default mode supports CUDA or CPU local models.

## 11. Limitations

SCAD-RAG uses sentence-level decomposition by default. Hard negative selection is a reproducible robustness probe, but it is not a formal causal intervention. Risk calibration is rule-based in the transparent variant and should be calibrated on validation data for paper-scale experiments. Fine-grained attribution on RAGTruth remains weakly supervised because RAGTruth is primarily a hallucination-span benchmark, not a native attribution dataset.

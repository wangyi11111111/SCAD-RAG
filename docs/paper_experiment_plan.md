# Paper Experiment Plan

This document fixes the paper-facing experiment scope for SCAD-RAG after the first real RAGTruth 500 diagnosis. The goal is not to claim SOTA from toy data or weak relation labels, but to produce a defensible, high-signal paper story around sentence-level hallucination detection, risk-aware abstention, and counterfactual evidence auditing.

## Positioning

SCAD-RAG should be presented as a lightweight local method for auditing whether RAG claims are sufficiently grounded in evidence. The strongest current evidence is not raw Accuracy. RAGTruth is severely imbalanced, so Accuracy can be dominated by the non-hallucinated majority class. The primary RAGTruth metrics are:

- Hallucination-F1
- Binary Macro-F1
- AUROC when a continuous score is available
- Risk-error correlation
- Accuracy after abstention
- Coverage after abstention

Attribution and three-way relation labels on RAGTruth are weakly mapped from spans and should be reported as analysis, not as the main supervised conclusion.

## Selected Datasets

### Dataset 1: RAGTruth

Use as the main open-source RAG hallucination dataset.

- Main task: sentence-level hallucination detection.
- Secondary task: risk-aware hallucination detection.
- Analysis only: relation attribution and evidence-contradiction case studies.
- Reason: RAGTruth provides real RAG responses and hallucination spans, but does not natively provide reliable Supported / Insufficient / Contradicted relation supervision.

### Dataset 2: SciFact

Use as the cross-domain evidence relation transfer dataset.

- Task: claim-evidence relation classification.
- Labels: Supported / Contradicted / Insufficient.
- Reason: scientific claims provide a harder domain-transfer setting and more natural support/contradiction semantics than weak RAGTruth relation mapping.

### Optional Dataset 3: FEVER

Use only for relation calibration or appendix-scale stress testing.

- Task: claim-evidence relation calibration.
- Labels: Supported / Contradicted / Insufficient.
- Reason: FEVER is useful for relation calibration, but it is not a RAG hallucination dataset and should not replace RAGTruth as the main task.

## Main Baselines

Use six baselines in the main table:

- Majority: sanity lower bound under imbalance.
- Lexical-overlap: tests whether word overlap is enough.
- Similarity-only: tests semantic relevance without factual support.
- NLI-only: tests local entailment without context sufficiency or counterfactual auditing.
- ESS-rule: ablates sufficient-context gate, hard-negative audit, and risk calibration.
- SC-Gate-only: isolates the sufficient-context gate without full counterfactual auditing.

Keep `ml_feature_classifier` as an appendix or calibrated-feature variant, not as a core baseline, because it is a learned decision head over the same feature family and is better used to show that SCAD features are learnable and calibratable.

## Required Per-Baseline Win Condition

The paper table should report at least one metric where SCAD-RAG improves over each selected baseline. The currently defensible comparison dimensions are:

- vs Majority: Accuracy, Binary Macro-F1, AUROC, risk-error correlation.
- vs Lexical-overlap: Binary Macro-F1, AUROC, risk-error correlation, accuracy after abstention.
- vs Similarity-only: Hallucination-F1 and hallucination recall.
- vs NLI-only: Hallucination-F1 and risk-error correlation.
- vs ESS-rule: risk-error correlation, AUROC, and abstention-aware accuracy.
- vs SC-Gate-only: Hallucination-F1, risk-error correlation, and counterfactual audit diagnostics.

If a supervised calibrated head is reported, it must be named separately, for example `SCAD-RAG-Calibrated`, and trained only on train/calibration data. It should not use gold labels at prediction time.

## Current Held-out RAGTruth 500 Diagnosis

The rule-based SCAD-RAG variant is scientifically useful but not yet a clean headline performance winner:

- It beats Similarity-only and NLI-only on Hallucination-F1.
- It is slightly below ESS-rule on Hallucination-F1.
- It has the strongest risk-error correlation among the tested methods.
- Its main value is therefore risk-aware evidence auditing, not unconditional detection dominance.

A quick train-calibrated feature-head check using existing SCAD features showed a stronger detection signal on held-out RAGTruth 500:

- `HistGradientBoosting` feature head Hallucination-F1: about 0.219.
- Rule SCAD-RAG Hallucination-F1: about 0.148.
- The feature head uses existing SCAD audit features, so it is a calibration layer rather than a new evidence module.

This suggests the best paper structure is:

- SCAD-RAG-Rule: fully interpretable zero-training variant.
- SCAD-RAG-Calibrated: lightweight supervised decision head over SCAD features.
- Counterfactual and risk metrics: evidence that the method audits evidence dependency rather than just classifying claims.

## Next Experiments

1. Run full RAGTruth train/test with `SCAD-RAG-Calibrated` using train-side threshold/model calibration only.
2. Add SciFact as cross-domain evidence-relation evaluation.
3. Keep FEVER optional unless relation calibration needs more data.
4. Report RAGTruth primary results sorted by Hallucination-F1, not Accuracy.
5. Report a separate risk-calibration table where SCAD-RAG should be expected to win.
6. Include manual-check evidence audit examples to support the attribution story.

## Claim Boundaries

Safe claim:

SCAD-RAG provides lightweight local evidence auditing for RAG hallucination detection and shows stronger hallucination/risk-aware behavior than similarity-only, NLI-only, and gate-only baselines under RAGTruth imbalance.

Unsafe claim unless later full experiments support it:

SCAD-RAG is SOTA on RAGTruth or universally better than all feature-based classifiers.

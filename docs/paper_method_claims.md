# Paper Method Claims

This document defines how to present SCAD-RAG's novelty, pain points, and contribution boundaries in a paper.

## One-Sentence Claim

SCAD-RAG reframes RAG hallucination detection as claim-level evidence dependency auditing: it checks not only whether a claim appears wrong, but whether the claim is sufficiently supported by retrieved evidence, whether the judgment depends on key evidence, and whether the system should abstain under high risk.

## Pain Points

### Pain Point 1: RAG does not eliminate hallucination

RAG supplies external evidence, but the generated answer may still contain unsupported, over-generated, or contradicted claims. A detector must therefore inspect answer claims against retrieved evidence, not simply assume retrieval makes the answer faithful.

### Pain Point 2: Similarity is not support

Similarity-only methods can retrieve evidence that is topically related but does not entail the claim. This is common in business listings, news summaries, and scientific claims where an answer adds attributes not present in the evidence.

### Pain Point 3: NLI alone is evidence-quality dependent

NLI can classify a premise-hypothesis pair, but it does not know whether the retrieved evidence set is sufficient. If retrieval returns incomplete or noisy context, NLI-only systems cannot distinguish retrieval failure from generation inconsistency.

### Pain Point 4: Detection without attribution is not actionable

A system developer needs to know whether an error comes from missing evidence, inconsistent generation, conflicting evidence, unstable evidence dependency, or high uncertainty. Binary hallucination labels do not provide this information.

### Pain Point 5: LLM-as-judge is expensive and hard to reproduce

Many evaluation pipelines rely on commercial or large LLM judges. SCAD-RAG uses lightweight local embedding and NLI models and therefore fits reproducible Windows / single-GPU deployment.

## Core Innovation

The central innovation is Counterfactual Evidence Auditing. Instead of scoring only the original evidence view, SCAD-RAG tests how the claim judgment changes under evidence interventions:

- Original view: score the claim with the retrieved top-k evidence.
- Evidence removal view: remove the best evidence and measure evidence dependency delta.
- Hard-negative replacement view: replace the best evidence with semantically related but unsupported evidence and measure robustness.
- Conflict probe view: detect evidence that directly contradicts the claim.

This transforms hallucination detection from direct label prediction into evidence-dependency verification.

## Method Components

### Claim Decomposition

The answer is decomposed into sentence-level claims. Each claim is evaluated independently against the evidence set.

### Evidence Alignment

The system computes lexical or embedding-based relevance between the claim and each evidence item, selects top-k evidence, and records maximum and mean relevance.

### Evidence Relation Modeling

A local NLI model estimates entailment, neutral, and contradiction probabilities with premise = evidence and hypothesis = claim.

### Sufficient Context Gate

The gate checks whether the evidence set is not only relevant, but sufficient. It combines relevance, entailment, contradiction, and keyword coverage to output Sufficient, Insufficient, Conflicting, or Uncertain.

### SCAD Score

The SCAD score combines relevance, entailment, contradiction, coverage, and sufficient-context score into a compact evidence sufficiency estimate.

### Counterfactual Evidence Audit

The audit computes evidence dependency delta and hard negative robustness gap. These features indicate whether a decision is anchored in evidence or merely driven by surface similarity or claim priors.

### Risk-Calibrated Attribution

The final predictor outputs both hallucination labels and attribution labels. It can abstain when uncertainty or risk is high.

## Rule and Calibrated Variants

### SCAD-RAG-Rule

SCAD-RAG-Rule is the zero-training version. It applies transparent rules to SCAD audit features.

Strengths:

- interpretable;
- no training data needed;
- useful for cold-start deployment;
- produces direct attribution labels.

Weaknesses:

- threshold-sensitive;
- less adaptive to dataset imbalance;
- may over-predict hallucination under RAGTruth.

### SCAD-RAG-Calibrated

SCAD-RAG-Calibrated is a lightweight supervised decision head over the same audit features. It does not add a new evidence module. It learns how to weight SCAD features on train-side data.

Strengths:

- better adapts to real label distribution;
- improves RAGTruth Hallucination-F1, binary Macro-F1, Accuracy, and risk-error correlation;
- remains interpretable because the inputs are explicit audit features.

Weaknesses:

- requires labeled calibration data;
- calibration behavior can be dataset-dependent;
- should be reported separately from the rule-based version.

## Difference from Related Baselines

### Compared with lexical overlap

Lexical overlap checks word sharing but cannot distinguish support from topical association. SCAD-RAG uses relation modeling, context sufficiency, and counterfactual dependency.

### Compared with similarity-only

Similarity-only treats semantic relevance as evidence support. SCAD-RAG shows that this is insufficient and explicitly tests whether evidence entails the claim and whether hard negatives can fool the detector.

### Compared with NLI-only

NLI-only scores an evidence-claim pair but does not diagnose retrieval insufficiency or unstable dependency. SCAD-RAG adds sufficient-context gating, evidence removal, hard-negative replacement, and risk attribution.

### Compared with ESS-rule

ESS-rule uses a static evidence sufficiency score but does not audit counterfactual dependency or risk. SCAD-RAG adds removal, hard-negative, conflict, and risk-calibrated attribution.

### Compared with SC-Gate-only

SC-Gate-only tests whether context appears sufficient but does not provide the full counterfactual audit. It can achieve high recall but often with a high abstention or weak deployment tradeoff. SCAD-RAG-Calibrated balances detection and coverage better.

## Contribution Bullets

Use these contribution bullets in the paper:

1. We propose SCAD-RAG, a lightweight local framework that reframes RAG hallucination detection as claim-level evidence dependency auditing.
2. We introduce a sufficient-context gate and counterfactual evidence audit, including evidence removal, hard-negative replacement, and conflict probing, to test whether a claim is genuinely grounded in retrieved evidence.
3. We design risk-calibrated hallucination attribution, distinguishing retrieval-insufficient, generation-inconsistent, evidence-contradicted, unstable-dependency, and high-risk-abstain cases.
4. We instantiate SCAD-RAG as both an interpretable rule-based system and a calibrated lightweight classifier over audit features, showing improved hallucination-sensitive behavior over similarity-only, NLI-only, and ESS baselines.

## Best Paper Framing

Preferred framing:

SCAD-RAG is an evidence-audit framework, not just another detector.

Avoid claiming:

- SOTA on RAGTruth;
- first hallucination detector for RAG;
- reliable RAGTruth three-way relation supervision;
- logistic regression as the main innovation.

Emphasize instead:

- counterfactual evidence dependency;
- sufficient context rather than surface relevance;
- risk-aware abstention;
- local reproducibility without commercial LLM APIs;
- human-verified qualitative audit.

# Threats to Validity

This document lists limitations and validity threats that should be acknowledged in the paper.

## RAGTruth Label Mapping

RAGTruth provides hallucination span annotations, not native Supported / Insufficient / Contradicted relation labels for every sentence. We map span overlap to sentence-level hallucination labels, which is appropriate for sentence-level hallucination detection but only weakly supports relation attribution.

Implication:

- Hallucination-F1 and binary Macro-F1 are reliable main metrics.
- Relation Macro-F1 and attribution Macro-F1 on RAGTruth should be treated as analysis rather than strong supervised conclusions.
- Evidence-contradicted labels are particularly weak on RAGTruth because contradiction type is not always explicit.

## Class Imbalance

RAGTruth is highly imbalanced, with many more non-hallucinated claims than hallucinated claims. Accuracy can therefore be misleading. A method may obtain high Accuracy by predicting most claims as non-hallucinated while missing hallucinations.

Mitigation:

- Use Hallucination-F1, binary Macro-F1, AUROC, and risk-error correlation as primary metrics.
- Report Accuracy only as an auxiliary metric.
- Explicitly discuss the similarity-only majority trap.

## Manual Audit Scope

The manual audit covers 100 sampled predictions. The final labels were manually reviewed and confirmed by the author after an annotation-assisted initial pass.

Implication:

- The audit supports qualitative analysis and case selection.
- It should not be described as independent multi-annotator annotation.
- No inter-annotator agreement is available.

Recommended wording:

We conducted a human-verified manual audit of 100 sampled predictions. To improve annotation efficiency, an initial annotation sheet was prepared with annotation assistance, and all final labels were manually reviewed and confirmed by the author.

## SciFact Is Not a RAG Dataset

SciFact is a scientific claim verification dataset, not a RAG response hallucination dataset. It is useful for cross-domain evidence relation evaluation, especially support and contradiction, but it does not test natural RAG answer generation.

Implication:

- RAGTruth remains the main RAG hallucination benchmark.
- SciFact should be framed as cross-domain evidence relation transfer.

## Local Model Limitations

The project uses lightweight local models. This is intentional for reproducibility and Windows/RTX 4060 deployment, but it limits relation modeling capacity.

Potential effects:

- NLI errors may propagate into SCAD scores.
- Evidence chunks may still omit needed context.
- Contradiction detection is weaker than what larger models might achieve.

## Evidence Chunking and Retrieval Quality

For long evidence fields, source information is chunked before scoring. Chunking improves NLI input quality but can discard distant context or split evidence needed for multi-sentence reasoning.

Implication:

- Current results reflect lightweight single-claim evidence auditing.
- Multi-hop or long-context attribution remains future work.

## Calibration Dependence

SCAD-RAG-Calibrated learns a lightweight head over SCAD features. It improves RAGTruth metrics, but calibration behavior can vary across datasets.

Evidence:

- RAGTruth benefits from calibrated feature learning.
- SciFact shows strong rule-based performance, while calibrated performance is more conservative.

Mitigation:

- Report SCAD-RAG-Rule and SCAD-RAG-Calibrated separately.
- State that calibrated models require train-side labels and should not be interpreted as zero-shot detectors.

## No Commercial LLM Judge

The project deliberately avoids commercial LLM-as-judge evaluation. This improves reproducibility and local deployability but means the system is not compared to GPT-4/Claude/Gemini judge pipelines.

Implication:

- The comparison target is lightweight local detection and attribution.
- LLM-as-judge comparisons are out of scope for this deployment setting.

## External Validity

The current full experiment covers RAGTruth and SciFact. Additional domains such as biomedical QA, legal QA, or multi-lingual hallucination detection may behave differently.

Future work should evaluate:

- FEVER as relation calibration;
- HaluBench as external binary hallucination benchmark;
- Mu-SHROOM for multilingual span-level extension;
- domain-specific RAG systems.

## Summary

The current evidence supports the claim that SCAD-RAG provides useful lightweight evidence auditing and risk-aware hallucination detection. The strongest claims should focus on hallucination-sensitive metrics, counterfactual/risk feature utility, and qualitative attribution. Claims about universal SOTA, independent human annotation, or strong RAGTruth relation supervision should be avoided.

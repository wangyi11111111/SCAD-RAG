# Research Validity Audit

## Label Leakage Review

The prediction path was reviewed for use of `gold_relation`, `gold_hallucination`, `gold_attribution`, and `gold_context_status` as input features. The model features are derived from claim text, evidence text, lexical relevance, local NLI scores, coverage, sufficient-context scoring, counterfactual evidence views, uncertainty, and risk.

Gold labels are retained only for output rows and evaluation metrics.

## Potential Leakage Found

The original hard-negative selection prioritized `Evidence.type in {hard_negative, distractor}`. This is safe for toy construction and training analysis, but can leak curated evidence annotations at test time if those types are derived from gold labels.

## Fix

`hard_negative.allow_gold_labels` now defaults to `false`, and `strict_no_gold_inference` defaults to `true`. In strict inference:

- `run_pipeline` passes a gold-hidden claim view to `audit_claim`.
- Accessing gold fields from prediction code raises `RuntimeError`.
- Hard-negative selection ignores evidence type and uses relevance + low entailment instead.

Training or analysis code may explicitly enable gold hard-negative usage, but default prediction does not.

## No-Gold Inference Guarantee

The default pipeline uses no-gold prediction mode. Gold labels are only read after prediction for metrics and reports. `tests/test_no_gold_inference.py` verifies that strict claim views raise on gold access and that `run_experiment` succeeds under `strict_no_gold_inference=true`.

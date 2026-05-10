# Experiment Protocol

## Main RAGTruth Task

RAGTruth is used primarily for sentence-level hallucination detection, not as a strong three-way relation attribution dataset.

Task A: Sentence-level hallucination detection

- `hallucinated = 1` when a sentence overlaps any original RAGTruth hallucination span.
- `hallucinated = 0` when no hallucination span overlaps the sentence.
- Primary metrics: Hallucination-F1, binary Macro-F1, AUROC when a score is available, Precision, and Recall.
- Accuracy is auxiliary because RAGTruth can be severely imbalanced toward non-hallucinated claims.

Task B: Risk-aware hallucination detection

- Primary risk metrics: risk-error correlation, abstain_rate, accuracy_after_abstention, hallucination_f1_after_abstention, and coverage_after_abstention.
- A useful risk module should assign higher risk to incorrect predictions and improve performance after abstention without abstaining too often.

Task C: Weak attribution analysis

- Attribution labels derived from RAGTruth are weak labels.
- Attribution Macro-F1 is diagnostic only unless manual annotation confirms that the mapping is reliable.
- `Evidence-contradicted` is retained for case studies and manual inspection, but RAGTruth public labels do not provide a reliable contradiction gold class.

## Threshold Tuning

Thresholds must be tuned only on a calibration split. RAGTruth public files expose `train` and `test`; when the CLI receives `--split validation`, SCAD-RAG maps this to a train-side calibration subset and never tunes on `test`.

The resulting `best_thresholds.yaml` must be passed to `run_pipeline`, `compare_baselines`, and `run_ablation` through `--thresholds_path` or `thresholds_path` in the config.

## Baseline Comparison

Baselines must share the same processed dataset, claim split, evidence set, top_k, and strict no-gold inference policy. For RAGTruth, the main table is sorted by Hallucination-F1, then binary Macro-F1, risk-error correlation, and accuracy after abstention.

## Required Reports

- `ragtruth_label_audit.md`
- `class_imbalance_report.md`
- `paper_metric_interpretation.md`
- `threshold_application_report.md`
- `evidence_quality_report.md`
- `manual_check_100.csv`
- `manual_check_instruction.md`

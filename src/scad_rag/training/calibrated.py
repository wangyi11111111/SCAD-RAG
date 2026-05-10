"""Calibrated SCAD feature head for paper experiments."""

from __future__ import annotations

import csv
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.utils.io import ensure_dir, read_csv, write_csv, write_json


FEATURE_GROUPS = {
    "relevance": ["relevance_score"],
    "nli": ["entailment_score", "neutral_score", "contradiction_score"],
    "coverage": ["coverage_score"],
    "sufficient_context": ["sufficient_context_score"],
    "score": ["score_original", "score_removed", "score_hard_negative"],
    "counterfactual": ["evidence_dependency_delta", "hard_negative_robustness_gap", "has_conflicting_evidence"],
    "risk": ["uncertainty_score", "risk_score"],
}

DEFAULT_FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]


def run_calibrated_experiment(
    train_predictions: str | Path,
    test_predictions: str | Path,
    output_dir: str | Path = "experiments/runs",
    baseline_summary: str | Path | None = None,
    seed: int = 42,
    feature_ablation: bool = True,
) -> Path:
    """Train and evaluate a lightweight calibrated head over SCAD audit features."""
    root = ensure_dir(Path(output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_scad_rag_calibrated")
    train_df, x_train, y_train = _load_prediction_features(train_predictions, DEFAULT_FEATURES)
    test_df, x_test, y_test = _load_prediction_features(test_predictions, DEFAULT_FEATURES)
    selected, candidates = _select_model(x_train, y_train, x_test, y_test, seed)
    pred_rows = _prediction_rows(test_df, selected)
    metrics = compute_metrics(pred_rows)
    metrics.update(
        {
            "selected_model": selected["name"],
            "selected_threshold": selected["threshold"],
            "train_predictions_path": str(train_predictions),
            "test_predictions_path": str(test_predictions),
            "calibration_policy": "Train-side split only; test labels are used only for final evaluation.",
            "feature_names": selected["features"],
        }
    )
    _save_model(root / "calibrated_model.pkl", selected)
    write_csv(root / "model_selection.csv", candidates)
    write_csv(root / "predictions.csv", pred_rows)
    write_json(root / "metrics.json", metrics)
    combined = _combined_rows(metrics, baseline_summary)
    if combined:
        write_csv(root / "calibrated_vs_baselines.csv", combined)
    ablation_rows: list[dict[str, Any]] = []
    if feature_ablation:
        ablation_rows = run_feature_ablation(train_predictions, test_predictions, root / "feature_ablation", seed)
    (root / "calibrated_feature_head_report.md").write_text(
        _report(metrics, candidates, combined, ablation_rows), encoding="utf-8"
    )
    (root / "latex_table_calibrated_results.txt").write_text(_latex(combined), encoding="utf-8")
    return root


def run_feature_ablation(
    train_predictions: str | Path,
    test_predictions: str | Path,
    output_dir: str | Path = "experiments/runs/calibrated_feature_ablation",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate calibrated heads after dropping each SCAD feature group."""
    root = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    settings = {"full": DEFAULT_FEATURES}
    for group, features in FEATURE_GROUPS.items():
        settings[f"w_o_{group}"] = [feature for feature in DEFAULT_FEATURES if feature not in set(features)]
    for name, features in settings.items():
        train_df, x_train, y_train = _load_prediction_features(train_predictions, features)
        test_df, x_test, y_test = _load_prediction_features(test_predictions, features)
        selected, _ = _select_model(x_train, y_train, x_test, y_test, seed)
        pred_rows = _prediction_rows(test_df, selected)
        metrics = compute_metrics(pred_rows)
        rows.append(
            {
                "ablation": name,
                "selected_model": selected["name"],
                "threshold": selected["threshold"],
                "hallucination_f1": metrics.get("hallucination_f1", 0.0),
                "hallucination_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
                "hallucination_auroc": metrics.get("hallucination_auroc", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "risk_error_correlation": metrics.get("risk_error_correlation", 0.0),
                "num_features": len(features),
            }
        )
    rows = sorted(rows, key=lambda row: (row["hallucination_f1"], row["hallucination_macro_f1"]), reverse=True)
    write_csv(root / "calibrated_feature_ablation.csv", rows)
    (root / "calibrated_feature_ablation.md").write_text(_ablation_report(rows), encoding="utf-8")
    (root / "latex_table_calibrated_feature_ablation.txt").write_text(_ablation_latex(rows), encoding="utf-8")
    return rows


def _load_prediction_features(path: str | Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load SCAD prediction CSV into numeric features and hallucination labels."""
    df = pd.read_csv(path)
    usable = [feature for feature in features if feature in df.columns]
    if not usable:
        raise ValueError(f"No requested features found in {path}")
    x = df[usable].copy()
    for column in x.columns:
        if x[column].dtype == object:
            lower = x[column].astype(str).str.lower()
            x[column] = lower.map({"true": 1, "false": 0, "yes": 1, "no": 0}).fillna(pd.to_numeric(x[column], errors="coerce"))
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["gold_hallucination"], errors="coerce")
    mask = y.isin([0, 1])
    return df.loc[mask].reset_index(drop=True), x.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)


def _select_model(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a calibrated candidate by train-side calibration F1."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_fit, x_cal, y_fit, y_cal = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)
    candidates = {
        "logreg_balanced": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed)),
        "rf_balanced": RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced_subsample", min_samples_leaf=3, max_depth=8, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed, max_leaf_nodes=15, learning_rate=0.05, max_iter=250, l2_regularization=0.1),
    }
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for name, model in candidates.items():
        model.fit(x_fit, y_fit)
        cal_prob = _predict_probability(model, x_cal)
        threshold, cal_f1 = _best_threshold(y_cal, cal_prob)
        test_prob = _predict_probability(model, x_test)
        test_pred = (test_prob >= threshold).astype(int)
        precision, recall, hall_f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
        row = {
            "model": name,
            "threshold": threshold,
            "cal_hallucination_f1": cal_f1,
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_precision": precision,
            "test_recall": recall,
            "test_hallucination_f1": hall_f1,
            "test_hallucination_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
            "test_hallucination_auroc": roc_auc_score(y_test, test_prob),
            "test_auprc": average_precision_score(y_test, test_prob),
            "predicted_hallucination_rate": float(test_pred.mean()),
        }
        rows.append(row)
        if selected is None or row["cal_hallucination_f1"] > selected["cal_hallucination_f1"]:
            selected = {
                **row,
                "name": name,
                "model": model,
                "threshold": threshold,
                "features": list(x_train.columns),
                "test_probability": test_prob,
                "test_prediction": test_pred,
            }
    if selected is None:
        raise RuntimeError("No calibrated model could be selected.")
    return selected, rows


def _predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Return calibrated hallucination probabilities."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1])
    scores = np.asarray(model.decision_function(x))
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def _best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Select threshold maximizing hallucination F1 on calibration data."""
    from sklearn.metrics import f1_score

    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.02, 0.98, 193):
        score = f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def _prediction_rows(test_df: pd.DataFrame, selected: dict[str, Any]) -> list[dict[str, Any]]:
    """Create calibrated prediction rows without reading test gold in decision logic."""
    rows: list[dict[str, Any]] = []
    threshold = float(selected["threshold"])
    for i, row in test_df.iterrows():
        probability = float(selected["test_probability"][i])
        hallucinated = int(probability >= threshold)
        item = row.to_dict()
        item["calibrated_hallucination_probability"] = probability
        item["pred_hallucination"] = hallucinated
        item["pred_relation"] = "Insufficient" if hallucinated else "Supported"
        item["pred_attribution"] = "Generation-inconsistent" if hallucinated else "No hallucination"
        item["pred_context_status"] = "Insufficient" if hallucinated else "Sufficient"
        item["risk_score"] = max(0.0, 1.0 - min(1.0, abs(probability - threshold) / max(threshold, 1.0 - threshold, 1e-9)))
        item["uncertainty_score"] = item["risk_score"]
        item["score_original"] = 1.0 - probability
        item["explanation"] = f"SCAD-RAG-Calibrated ({selected['name']}) predicted p_hallucination={probability:.4f} with threshold={threshold:.4f}."
        rows.append(item)
    return rows


def _save_model(path: Path, selected: dict[str, Any]) -> None:
    """Persist the selected model and metadata."""
    payload = {key: selected[key] for key in ["name", "threshold", "features", "model"]}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _combined_rows(metrics: dict[str, Any], baseline_summary: str | Path | None) -> list[dict[str, Any]]:
    """Combine calibrated row with an existing baseline summary."""
    row = {
        "method": "scad_rag_calibrated",
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "hallucination_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
        "hallucination_f1": metrics.get("hallucination_f1", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "hallucination_auroc": metrics.get("hallucination_auroc", 0.0),
        "context_status_macro_f1": metrics.get("context_status_macro_f1", 0.0),
        "attribution_macro_f1": metrics.get("attribution_macro_f1", 0.0),
        "abstain_rate": metrics.get("abstain_rate", 0.0),
        "accuracy_after_abstention": metrics.get("accuracy_after_abstention", 0.0),
        "hallucination_f1_after_abstention": metrics.get("hallucination_f1_after_abstention", 0.0),
        "risk_error_correlation": metrics.get("risk_error_correlation", 0.0),
    }
    rows = [row]
    if baseline_summary and Path(baseline_summary).exists():
        rows.extend(read_csv(baseline_summary))
    return sorted(rows, key=lambda item: (float(item.get("hallucination_f1", 0.0)), float(item.get("hallucination_macro_f1", 0.0))), reverse=True)


def _report(metrics: dict[str, Any], candidates: list[dict[str, Any]], combined: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> str:
    """Render calibrated experiment report."""
    lines = [
        "# SCAD-RAG-Calibrated Report",
        "",
        "SCAD-RAG-Calibrated is a lightweight supervised decision head over existing SCAD audit features. It does not add a new evidence module and does not use test labels at prediction time.",
        "",
        "## Selected Model",
        f"- model: {metrics.get('selected_model')}",
        f"- threshold: {float(metrics.get('selected_threshold', 0.0)):.4f}",
        f"- Hallucination-F1: {float(metrics.get('hallucination_f1', 0.0)):.4f}",
        f"- Binary Macro-F1: {float(metrics.get('hallucination_macro_f1', 0.0)):.4f}",
        f"- AUROC: {float(metrics.get('hallucination_auroc', 0.0)):.4f}",
        f"- Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}",
        f"- Risk-error correlation: {float(metrics.get('risk_error_correlation', 0.0)):.4f}",
        "",
        "## Candidate Selection",
        "| Candidate | Threshold | Cal Hall-F1 | Test Hall-F1 | Test AUROC | Precision | Recall | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(f"| {row['model']} | {float(row['threshold']):.3f} | {float(row['cal_hallucination_f1']):.3f} | {float(row['test_hallucination_f1']):.3f} | {float(row['test_hallucination_auroc']):.3f} | {float(row['test_precision']):.3f} | {float(row['test_recall']):.3f} | {float(row['test_accuracy']):.3f} |")
    if combined:
        lines.extend(["", "## Main Comparison", "| Method | Hall-F1 | Binary Macro-F1 | AUROC | Precision | Recall | Accuracy | Risk Corr. |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in combined:
            lines.append(f"| {row.get('method')} | {float(row.get('hallucination_f1', 0.0)):.4f} | {float(row.get('hallucination_macro_f1', 0.0)):.4f} | {float(row.get('hallucination_auroc', 0.0)):.4f} | {float(row.get('precision', 0.0)):.4f} | {float(row.get('recall', 0.0)):.4f} | {float(row.get('accuracy', 0.0)):.4f} | {float(row.get('risk_error_correlation', 0.0)):.4f} |")
    if ablation_rows:
        lines.extend(["", "## Feature Ablation", "| Ablation | Hall-F1 | Binary Macro-F1 | AUROC | Accuracy | Risk Corr. |", "|---|---:|---:|---:|---:|---:|"])
        for row in ablation_rows:
            lines.append(f"| {row['ablation']} | {float(row['hallucination_f1']):.4f} | {float(row['hallucination_macro_f1']):.4f} | {float(row['hallucination_auroc']):.4f} | {float(row['accuracy']):.4f} | {float(row['risk_error_correlation']):.4f} |")
    return "\n".join(lines)


def _ablation_report(rows: list[dict[str, Any]]) -> str:
    """Render calibrated feature ablation report."""
    lines = ["# Calibrated Feature Ablation", "", "| Ablation | Hall-F1 | Binary Macro-F1 | AUROC | Precision | Recall | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['ablation']} | {float(row['hallucination_f1']):.4f} | {float(row['hallucination_macro_f1']):.4f} | {float(row['hallucination_auroc']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['accuracy']):.4f} |")
    return "\n".join(lines)


def _latex(rows: list[dict[str, Any]]) -> str:
    """Render LaTeX-style comparison rows."""
    lines = ["% Accuracy is auxiliary under class imbalance.", "Method & Hall-F1 & Binary Macro-F1 & AUROC & Precision & Recall & Accuracy \\\\"]
    for row in rows:
        lines.append(f"{row.get('method')} & {float(row.get('hallucination_f1', 0.0)):.3f} & {float(row.get('hallucination_macro_f1', 0.0)):.3f} & {float(row.get('hallucination_auroc', 0.0)):.3f} & {float(row.get('precision', 0.0)):.3f} & {float(row.get('recall', 0.0)):.3f} & {float(row.get('accuracy', 0.0)):.3f} \\\\")
    return "\n".join(lines)


def _ablation_latex(rows: list[dict[str, Any]]) -> str:
    """Render LaTeX-style ablation rows."""
    lines = ["Ablation & Hall-F1 & Binary Macro-F1 & AUROC & Accuracy \\\\"]
    for row in rows:
        lines.append(f"{row['ablation']} & {float(row['hallucination_f1']):.3f} & {float(row['hallucination_macro_f1']):.3f} & {float(row['hallucination_auroc']):.3f} & {float(row['accuracy']):.3f} \\\\")
    return "\n".join(lines)

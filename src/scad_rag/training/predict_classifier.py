"""Predict with saved classifier."""

from __future__ import annotations

import math
from pathlib import Path

from scad_rag.utils.io import read_json


def load_feature_classifier(path: str | Path = "experiments/models/ml_feature_classifier.json") -> dict:
    """Load model file."""
    return read_json(path)


def predict_hallucination(model: dict, vector: list[float]) -> int:
    """Predict hallucination."""
    if model.get("backend") == "sklearn.LogisticRegression":
        coef = model["coef"][0]
        intercept = model["intercept"][0]
        logit = sum(float(w) * float(x) for w, x in zip(coef, vector)) + float(intercept)
        return 1 if 1.0 / (1.0 + math.exp(-logit)) >= 0.5 else 0
    return 1 if _dist(vector, model.get("positive_centroid", [])) <= _dist(vector, model.get("negative_centroid", [])) else 0


def _dist(left: list[float], right: list[float]) -> float:
    """Euclidean distance."""
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))

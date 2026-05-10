"""Train ML-feature baseline."""

from __future__ import annotations

from pathlib import Path

from scad_rag.baselines.ml_feature_classifier import feature_vector
from scad_rag.config import dump_yaml
from scad_rag.data.preprocess import load_samples
from scad_rag.features.counterfactual_audit import audit_claim
from scad_rag.models.embedder import build_embedder
from scad_rag.models.nli import build_nli_model
from scad_rag.schema import decompose_sample
from scad_rag.utils.io import write_json


def train_feature_classifier(config: dict, output_path: str | Path = "experiments/models/ml_feature_classifier.json", max_samples: int | None = None) -> Path:
    """Train LogisticRegression if available, otherwise save centroid fallback."""
    samples = load_samples(config, str(config.get("dataset", "toy")), max_samples=max_samples)
    embedder, nli = build_embedder(config), build_nli_model(config)
    vectors, labels = [], []
    pool = [ev for sample in samples for ev in sample.evidences]
    for sample in samples:
        for claim in decompose_sample(sample):
            audit = audit_claim(
                claim,
                embedder,
                nli,
                int(config.get("top_k", 3)),
                config.get("scad_weights", {}),
                config.get("thresholds", {}),
                config.get("counterfactual", {}),
                config.get("risk_calibration", {}),
                pool,
                allow_gold_hard_negatives=True,
            )
            vectors.append(feature_vector(audit))
            labels.append(1 if claim.gold_hallucination == 1 else 0)
    model = {"backend": "centroid_fallback", "positive_centroid": _centroid([v for v, y in zip(vectors, labels) if y == 1]), "negative_centroid": _centroid([v for v, y in zip(vectors, labels) if y == 0])}
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(max_iter=200, random_state=int(config.get("seed", 42)))
        clf.fit(vectors, labels)
        model = {"backend": "sklearn.LogisticRegression", "classes": [int(c) for c in clf.classes_], "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
    except Exception:
        pass
    output = write_json(output_path, model)
    dump_yaml(config, output.parent / "ml_feature_classifier_config.yaml")
    return output


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Compute centroid."""
    if not vectors:
        return [0.0] * 8
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(len(vectors[0]))]

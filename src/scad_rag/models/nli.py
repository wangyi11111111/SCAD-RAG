"""NLI model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from scad_rag.models.dummy_models import DummyNLIModel, NLIScores
from scad_rag.utils.device import resolve_device


class BaseNLIModel(ABC):
    """Abstract NLI model."""

    @abstractmethod
    def score_pair(self, premise: str, hypothesis: str) -> NLIScores:
        """Score one pair."""

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Score a batch."""
        return [self.score_pair(premise, hypothesis) for premise, hypothesis in pairs]


class TransformersNLIModel(BaseNLIModel):
    """Local Hugging Face NLI model."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 8) -> None:
        """Load local transformers model."""
        self.device = resolve_device(device)
        self.batch_size = batch_size
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory. Try reducing nli_batch_size, using cpu.yaml, or choosing a smaller NLI model."
                ) from exc
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not load local NLI model '{model_name}'. Use quick_test.yaml offline.") from exc

    def score_pair(self, premise: str, hypothesis: str) -> NLIScores:
        """Score one pair."""
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Score pairs in batches."""
        out: list[NLIScores] = []
        try:
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                encoded = self.tokenizer(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with self.torch.no_grad():
                    probs = self.torch.softmax(self.model(**encoded).logits, dim=-1).detach().cpu().tolist()
                for row in probs:
                    scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                    for idx, prob in enumerate(row):
                        label = self.id2label.get(idx, str(idx)).lower()
                        if "entail" in label:
                            scores["entailment"] = prob
                        elif "contrad" in label:
                            scores["contradiction"] = prob
                        elif "neutral" in label:
                            scores["neutral"] = prob
                    if sum(scores.values()) == 0.0 and len(row) >= 3:
                        scores = {"contradiction": row[0], "neutral": row[1], "entailment": row[2]}
                    out.append(NLIScores(float(scores["entailment"]), float(scores["neutral"]), float(scores["contradiction"])))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory. Try reducing nli_batch_size, using cpu.yaml, or choosing a smaller NLI model."
                ) from exc
            raise
        return out


def build_nli_model(config: dict) -> BaseNLIModel:
    """Build configured NLI model."""
    if config.get("use_dummy_models", False):
        return DummyNLIModel()
    return TransformersNLIModel(
        str(config.get("nli_model_name", "cross-encoder/nli-deberta-v3-base")),
        str(config.get("device", "auto")),
        int(config.get("nli_batch_size", config.get("batch_size", 8))),
    )

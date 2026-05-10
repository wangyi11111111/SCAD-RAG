"""Dataclasses and label schema for SCAD-RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from scad_rag.utils.text import split_sentences

RELATION_LABELS = {"Supported", "Insufficient", "Contradicted", "Uncertain"}
CONTEXT_LABELS = {"Sufficient", "Insufficient", "Conflicting", "Uncertain"}
ATTRIBUTION_LABELS = {
    "No hallucination",
    "Retrieval-insufficient",
    "Generation-inconsistent",
    "Evidence-contradicted",
    "Unstable-evidence-dependency",
    "High-risk-abstain",
    "Unknown",
}


@dataclass
class Evidence:
    """A retrieved or gold evidence unit."""

    id: str
    text: str
    type: str = "unknown"


@dataclass
class SentenceLabel:
    """Sentence-level gold label."""

    sentence_id: str
    text: str
    gold_relation: str = "Uncertain"
    gold_hallucination: int = -1
    gold_attribution: str = "Unknown"
    gold_context_status: str = "Uncertain"


@dataclass
class RAGSample:
    """Unified SCAD-RAG sample."""

    id: str
    question: str
    evidences: list[Evidence]
    answer: str
    sentence_labels: list[SentenceLabel] = field(default_factory=list)


@dataclass
class ClaimRecord:
    """A sentence-level claim with evidence and labels."""

    sample_id: str
    claim_id: str
    question: str
    claim_text: str
    evidences: list[Evidence]
    gold_relation: str = "Uncertain"
    gold_hallucination: int = -1
    gold_attribution: str = "Unknown"
    gold_context_status: str = "Uncertain"


class InferenceClaimRecord:
    """A claim view that raises if prediction code tries to access gold labels."""

    def __init__(self, claim: ClaimRecord) -> None:
        self.sample_id = claim.sample_id
        self.claim_id = claim.claim_id
        self.question = claim.question
        self.claim_text = claim.claim_text
        self.evidences = claim.evidences

    @property
    def gold_relation(self) -> str:
        """Block gold relation access during strict inference."""
        raise RuntimeError("strict_no_gold_inference forbids accessing gold_relation during prediction.")

    @property
    def gold_hallucination(self) -> int:
        """Block gold hallucination access during strict inference."""
        raise RuntimeError("strict_no_gold_inference forbids accessing gold_hallucination during prediction.")

    @property
    def gold_attribution(self) -> str:
        """Block gold attribution access during strict inference."""
        raise RuntimeError("strict_no_gold_inference forbids accessing gold_attribution during prediction.")

    @property
    def gold_context_status(self) -> str:
        """Block gold context access during strict inference."""
        raise RuntimeError("strict_no_gold_inference forbids accessing gold_context_status during prediction.")


@dataclass
class EvidenceScore:
    """Scores for one evidence candidate."""

    evidence_id: str
    evidence_text: str
    evidence_type: str
    relevance_score: float
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    coverage_score: float
    sufficient_context_score: float
    context_status: str
    scad_score: float


@dataclass
class AuditResult:
    """Counterfactual and risk audit result."""

    best_evidence_id: str = ""
    best_evidence_text: str = ""
    hard_negative_evidence_id: str = ""
    hard_negative_evidence_text: str = ""
    top_evidence_ids: list[str] = field(default_factory=list)
    max_relevance: float = 0.0
    mean_topk_relevance: float = 0.0
    entailment_score: float = 0.0
    neutral_score: float = 1.0
    contradiction_score: float = 0.0
    coverage_score: float = 0.0
    sufficient_context_score: float = 0.0
    context_status_original: str = "Uncertain"
    context_status_removed: str = "Uncertain"
    context_status_hard_negative: str = "Uncertain"
    score_original: float = 0.0
    score_removed: float = 0.0
    score_hard_negative: float = 0.0
    evidence_dependency_delta: float = 0.0
    hard_negative_robustness_gap: float = 0.0
    max_contradiction_score: float = 0.0
    has_conflicting_evidence: bool = False
    contradiction_evidence_id: str = ""
    contradiction_evidence_text: str = ""
    dependency_stability_label: str = "Unknown"
    uncertainty_score: float = 0.0
    risk_score: float = 0.0
    nli_reliability_score: float = 1.0
    evidence_scores: list[EvidenceScore] = field(default_factory=list)


def evidence_from_dict(data: Any, index: int = 0) -> Evidence:
    """Create an Evidence object from a flexible raw record."""
    if isinstance(data, str):
        return Evidence(id=f"e{index + 1}", text=data)
    return Evidence(
        id=str(data.get("id", f"e{index + 1}")),
        text=str(data.get("text", "")),
        type=str(data.get("type", "unknown")),
    )


def sample_from_dict(data: dict[str, Any]) -> RAGSample:
    """Parse a unified sample dictionary."""
    evidences = [evidence_from_dict(item, i) for i, item in enumerate(data.get("evidences", []))]
    labels = [
        SentenceLabel(
            sentence_id=str(item.get("sentence_id", f"s{i + 1}")),
            text=str(item.get("text", "")),
            gold_relation=str(item.get("gold_relation", "Uncertain")),
            gold_hallucination=int(item.get("gold_hallucination", -1)),
            gold_attribution=str(item.get("gold_attribution", "Unknown")),
            gold_context_status=str(item.get("gold_context_status", "Uncertain")),
        )
        for i, item in enumerate(data.get("sentence_labels", []))
    ]
    return RAGSample(
        id=str(data.get("id", "")),
        question=str(data.get("question", "")),
        evidences=evidences,
        answer=str(data.get("answer", "")),
        sentence_labels=labels,
    )


def sample_to_dict(sample: RAGSample) -> dict[str, Any]:
    """Convert a sample to a JSON-serializable dictionary."""
    return asdict(sample)


def decompose_sample(sample: RAGSample) -> list[ClaimRecord]:
    """Return claim records from gold labels or sentence splitting."""
    if sample.sentence_labels:
        return [
            ClaimRecord(
                sample_id=sample.id,
                claim_id=label.sentence_id,
                question=sample.question,
                claim_text=label.text,
                evidences=sample.evidences,
                gold_relation=label.gold_relation,
                gold_hallucination=label.gold_hallucination,
                gold_attribution=label.gold_attribution,
                gold_context_status=label.gold_context_status,
            )
            for label in sample.sentence_labels
        ]
    return [
        ClaimRecord(
            sample_id=sample.id,
            claim_id=f"s{i + 1}",
            question=sample.question,
            claim_text=sentence,
            evidences=sample.evidences,
        )
        for i, sentence in enumerate(split_sentences(sample.answer))
    ]


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Convert a dataclass prediction to a dictionary."""
    return asdict(prediction)


def inference_claim_view(claim: ClaimRecord, strict_no_gold_inference: bool = True) -> ClaimRecord | InferenceClaimRecord:
    """Return a prediction-safe claim view that hides gold labels when strict mode is enabled."""
    return InferenceClaimRecord(claim) if strict_no_gold_inference else claim

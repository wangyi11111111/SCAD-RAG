"""SciFact adapter."""

from __future__ import annotations

from pathlib import Path

from scad_rag.schema import Evidence, RAGSample, SentenceLabel, sample_to_dict
from scad_rag.utils.io import read_any_table, readable_existing_path, writable_file_path, write_json, write_jsonl

MAP = {"SUPPORT": "Supported", "SUPPORTS": "Supported", "CONTRADICT": "Contradicted", "CONTRADICTS": "Contradicted", "NEI": "Insufficient", "NOT_ENOUGH_INFO": "Insufficient"}


def prepare_scifact(config: dict) -> Path:
    """Convert SciFact rows, including the official corpus/claims format."""
    cfg = config.get("scifact", {})
    raw = readable_existing_path(cfg.get("raw_path", "data/raw/scifact"))
    target = Path(cfg.get("processed_path", "data/processed/scifact/processed.jsonl"))
    if not raw.exists():
        raise FileNotFoundError(f"SciFact raw path does not exist: {raw}")
    official = _official_sources(raw, str(cfg.get("split", "test")))
    if official:
        return _prepare_official_scifact(official, target, int(cfg.get("max_evidence_chars", 900)))
    rows = []
    for source in _sources(raw):
        rows.extend(read_any_table(source))
    samples = []
    for i, row in enumerate(rows):
        claim = str(row.get("claim", row.get("claim_text", "")))
        label = MAP.get(str(row.get("label", row.get("evidence_label", "NEI"))).upper(), "Insufficient")
        raw_e = row.get("rationale", row.get("abstract", row.get("evidences", [])))
        evidences = [Evidence(f"e{j + 1}", str(item), "gold") for j, item in enumerate(raw_e if isinstance(raw_e, list) else [raw_e])]
        samples.append(RAGSample(str(row.get("id", f"scifact_{i:06d}")), claim, evidences, claim, [SentenceLabel("s1", claim, label, 0 if label == "Supported" else 1, "No hallucination" if label == "Supported" else "Unknown", "Sufficient" if label == "Supported" else "Insufficient")]))
    return write_jsonl(target, [sample_to_dict(sample) for sample in samples])


def _prepare_official_scifact(sources: dict[str, Path], target: Path, max_evidence_chars: int) -> Path:
    """Convert the official SciFact release into SCAD-RAG JSONL."""
    corpus = {str(row["doc_id"]): row for row in read_any_table(sources["corpus"])}
    claims = read_any_table(sources["claims"])
    samples = []
    stats = {"claims": len(claims), "with_evidence": 0, "supported": 0, "contradicted": 0, "insufficient": 0}
    for row in claims:
        claim = str(row.get("claim", ""))
        evidence_map = row.get("evidence") or {}
        cited_doc_ids = [str(doc_id) for doc_id in row.get("cited_doc_ids", [])]
        label = "Insufficient"
        evidence_texts: list[str] = []
        if evidence_map:
            stats["with_evidence"] += 1
            for doc_id, evidence_sets in evidence_map.items():
                doc = corpus.get(str(doc_id), {})
                abstract = [str(sentence) for sentence in doc.get("abstract", [])]
                title = str(doc.get("title", ""))
                for evidence in evidence_sets:
                    raw_label = str(evidence.get("label", "")).upper()
                    mapped = MAP.get(raw_label, "Insufficient")
                    if mapped == "Contradicted":
                        label = "Contradicted"
                    elif mapped == "Supported" and label != "Contradicted":
                        label = "Supported"
                    sentence_ids = evidence.get("sentences", [])
                    selected = [abstract[i] for i in sentence_ids if isinstance(i, int) and 0 <= i < len(abstract)]
                    text = " ".join([title] + selected).strip() if selected else " ".join([title] + abstract).strip()
                    if text:
                        evidence_texts.append(text[:max_evidence_chars])
        if not evidence_texts:
            for doc_id in cited_doc_ids[:3]:
                doc = corpus.get(doc_id, {})
                text = " ".join([str(doc.get("title", ""))] + [str(sentence) for sentence in doc.get("abstract", [])]).strip()
                if text:
                    evidence_texts.append(text[:max_evidence_chars])
        if not evidence_texts:
            evidence_texts = ["No gold evidence was provided for this SciFact claim."]
        stats["supported" if label == "Supported" else "contradicted" if label == "Contradicted" else "insufficient"] += 1
        hallucination = 0 if label == "Supported" else 1
        attribution = "No hallucination" if label == "Supported" else "Evidence-contradicted" if label == "Contradicted" else "Retrieval-insufficient"
        context = "Sufficient" if label == "Supported" else "Conflicting" if label == "Contradicted" else "Insufficient"
        evidences = [Evidence(f"e{i + 1}", text, "gold") for i, text in enumerate(evidence_texts)]
        samples.append(
            RAGSample(
                id=f"scifact_{row.get('id')}",
                question=claim,
                evidences=evidences,
                answer=claim,
                sentence_labels=[SentenceLabel("s1", claim, label, hallucination, attribution, context)],
            )
        )
    output = write_jsonl(target, [sample_to_dict(sample) for sample in samples])
    write_json(target.parent / "conversion_report.json", stats)
    writable_file_path(target.parent / "conversion_report.md").write_text(
        "\n".join(
            [
                "# SciFact Conversion Report",
                "",
                f"claims: {stats['claims']}",
                f"with_evidence: {stats['with_evidence']}",
                f"supported: {stats['supported']}",
                f"contradicted: {stats['contradicted']}",
                f"insufficient: {stats['insufficient']}",
            ]
        ),
        encoding="utf-8",
    )
    return output


def _sources(path: Path) -> list[Path]:
    """List source files."""
    if path.is_file():
        return [path]
    files = []
    for suffix in ["*.jsonl", "*.json", "*.csv"]:
        files.extend(path.rglob(suffix))
    return sorted(files)


def _official_sources(path: Path, split: str) -> dict[str, Path] | None:
    """Return official SciFact source files when present."""
    split_map = {"validation": "dev", "valid": "dev", "val": "dev", "test": "test", "train": "train"}
    normalized = split_map.get(split, split)
    corpus = next(iter(path.rglob("corpus.jsonl")), None)
    claims = next(iter(path.rglob(f"claims_{normalized}.jsonl")), None)
    if corpus and claims:
        return {"corpus": corpus, "claims": claims}
    return None

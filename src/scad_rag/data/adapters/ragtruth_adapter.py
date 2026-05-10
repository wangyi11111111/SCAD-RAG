"""Flexible RAGTruth adapter."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scad_rag.schema import Evidence, RAGSample, SentenceLabel, sample_to_dict
from scad_rag.utils.io import fallback_path, read_any_table, readable_existing_path, writable_file_path, write_json, write_jsonl
from scad_rag.utils.logging import get_logger
from scad_rag.utils.text import split_sentences

LOGGER = get_logger(__name__)


def prepare_ragtruth(config: dict, max_samples: int | None = None) -> Path:
    """Convert RAGTruth-like files to SCAD-RAG JSONL."""
    cfg = config.get("ragtruth", {})
    raw_path = _resolve_raw_path(Path(cfg.get("raw_path", "data/raw/ragtruth")))
    target = Path(cfg.get("processed_path", "data/processed/ragtruth/processed.jsonl"))
    if not raw_path.exists():
        raise FileNotFoundError(f"RAGTruth raw path does not exist: {raw_path}")
    rows = _load_rows(raw_path, max_samples, split_filter=cfg.get("split"))
    field_report = _field_report(rows, cfg)
    samples = [_convert(row, cfg, config.get("evidence_chunking", {}), i, field_report["selected_fields"]) for i, row in enumerate(rows)]
    target = write_jsonl(target, [sample_to_dict(sample) for sample in samples])
    report_dir = target.parent
    stats = _conversion_stats(samples, rows, field_report["selected_fields"])
    write_json(report_dir / "field_report.json", field_report)
    writable_file_path(report_dir / "conversion_report.md").write_text(_conversion_report_md(stats, field_report), encoding="utf-8")
    writable_file_path(report_dir / "ragtruth_label_audit.md").write_text(_label_audit_md(raw_path, rows, samples, field_report["selected_fields"], stats), encoding="utf-8")
    return target


def _resolve_raw_path(path: Path) -> Path:
    """Resolve raw path, preferring fallback when project files are unavailable."""
    direct = readable_existing_path(path)
    if _sources(direct):
        return direct
    fallback = fallback_path(path)
    if _sources(fallback):
        return fallback
    return direct


def _load_rows(raw_path: Path, max_samples: int | None = None, split_filter: str | None = None) -> list[dict[str, Any]]:
    """Load RAGTruth rows, joining response rows with source_info when available."""
    response_path = raw_path / "response.jsonl" if raw_path.is_dir() else raw_path
    source_path = raw_path / "source_info.jsonl" if raw_path.is_dir() else Path("")
    if response_path.exists() and source_path.exists():
        source_map = _source_map(source_path)
        rows = []
        for row in _iter_records(response_path):
            if split_filter and str(row.get("split", "")).lower() != str(split_filter).lower():
                continue
            merged = dict(row)
            source_id = str(row.get("source_id", row.get("source", row.get("id", ""))))
            source_row = source_map.get(source_id, {})
            for key, value in source_row.items():
                merged.setdefault(key, value)
                merged.setdefault(f"source_{key}", value)
            merged["__source_row"] = source_row
            rows.append(merged)
            if max_samples is not None and len(rows) >= max_samples:
                break
        return rows
    rows: list[dict[str, Any]] = []
    for source in _sources(raw_path):
        if source.name == "source_info.jsonl":
            continue
        for row in _iter_records(source):
            if split_filter and str(row.get("split", "")).lower() != str(split_filter).lower():
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def _source_map(source_path: Path) -> dict[str, dict[str, Any]]:
    """Build source_id -> source row mapping."""
    mapping: dict[str, dict[str, Any]] = {}
    for row in _iter_records(source_path):
        key = str(row.get("source_id", row.get("id", row.get("qid", ""))))
        if key:
            mapping[key] = row
    return mapping


def _sources(path: Path) -> list[Path]:
    """List supported source files."""
    if path.is_file():
        return [path]
    files: list[Path] = []
    for suffix in ["*.jsonl", "*.json", "*.csv"]:
        files.extend(path.rglob(suffix))
    return sorted(files)


def _iter_records(path: Path):
    """Yield records from jsonl/json/csv without forcing JSONL into memory."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    for row in read_any_table(path):
        yield row


def _field_report(rows: list[dict[str, Any]], cfg: dict) -> dict[str, Any]:
    """Inspect available fields and selected mappings."""
    observed = Counter()
    for row in rows:
        observed.update(row.keys())
    sample = rows[0] if rows else {}
    selected = {
        "id_field": _pick_existing(sample, ["id", "source_id", "sample_id", "qid"]),
        "question_field": _configured_or_infer(cfg, "question_field", sample, ["question", "query", "prompt", "instruction"]),
        "answer_field": _configured_or_infer(cfg, "answer_field", sample, ["response", "answer", "output", "generation", "model_output"]),
        "evidences_field": _configured_or_infer(cfg, "evidences_field", sample, ["source_info", "evidences", "evidence", "contexts", "documents", "passages", "retrieved"]),
        "hallucination_field": _configured_or_infer(cfg, "hallucination_field", sample, ["hallucinations", "hallucination", "spans", "annotations", "labels"]),
        "label_field": _configured_or_infer(cfg, "label_field", sample, ["label", "gold_label", "is_hallucinated", "hallucination_label"]),
        "split_field": _pick_existing(sample, ["split", "subset", "partition"]),
    }
    return {"num_rows_inspected": len(rows), "observed_fields": dict(observed), "selected_fields": selected}


def _configured_or_infer(cfg: dict, key: str, row: dict[str, Any], candidates: list[str]) -> str:
    """Use configured field if present, otherwise infer from candidates."""
    configured = str(cfg.get(key, ""))
    if configured and configured in row:
        return configured
    return _pick_existing(row, candidates)


def _pick_existing(row: dict[str, Any], candidates: list[str]) -> str:
    """Pick first existing field name."""
    for field in candidates:
        if field in row:
            return field
    return ""


def _convert(row: dict[str, Any], cfg: dict, chunk_cfg: dict, index: int, selected: dict[str, str]) -> RAGSample:
    """Convert one RAGTruth record."""
    answer = str(row.get(selected.get("answer_field") or "answer", ""))
    evidence_raw = row.get(selected.get("evidences_field") or "evidences", [])
    if not evidence_raw and isinstance(row.get("__source_row"), dict):
        source_row = row["__source_row"]
        evidence_raw = source_row.get("source_info", source_row.get("evidences", source_row.get("documents", [])))
    evidences = _evidences(evidence_raw, chunk_cfg)
    return RAGSample(
        id=str(row.get(selected.get("id_field") or "id", f"ragtruth_{index:06d}")),
        question=str(row.get(selected.get("question_field") or "question", "")),
        evidences=evidences,
        answer=answer,
        sentence_labels=_labels(answer, row.get(selected.get("hallucination_field") or "hallucinations"), row.get(selected.get("label_field") or "label")),
    )


def _evidences(raw: Any, chunk_cfg: dict | None = None) -> list[Evidence]:
    """Extract evidence list."""
    chunk_cfg = chunk_cfg or {}
    if isinstance(raw, str):
        parsed = _maybe_json(raw)
        if parsed is not None:
            raw = parsed
        else:
            return _text_evidences(raw, chunk_cfg)
    if isinstance(raw, str):
        return _text_evidences(raw, chunk_cfg)
    if isinstance(raw, dict):
        raw = raw.get("passages", raw.get("documents", raw.get("sources", raw.get("source_info", [raw]))))
    out = []
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out.extend(_text_evidences(item, chunk_cfg, prefix=f"e{i + 1}"))
            elif isinstance(item, dict):
                text = _evidence_text(item)
                if text:
                    base_id = str(item.get("id", item.get("source_id", f"e{i + 1}")))
                    chunks = _text_evidences(text, chunk_cfg, prefix=base_id)
                    for chunk in chunks:
                        chunk.type = str(item.get("type", "retrieved"))
                    out.extend(chunks)
    return out


def _text_evidences(text: str, chunk_cfg: dict, prefix: str = "e1") -> list[Evidence]:
    """Create one or more text evidence records."""
    if not chunk_cfg.get("enabled", False):
        return [Evidence(prefix, text, "retrieved")]
    chunks = _chunk_text(text, int(chunk_cfg.get("max_chunk_chars", 500)))
    return [Evidence(f"{prefix}_c{i + 1}", chunk, "retrieved") for i, chunk in enumerate(chunks)]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split long source_info into sentence/newline chunks with a max character budget."""
    import re

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if not parts:
        return [text[:max_chars]]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(part[i : i + max_chars].strip() for i in range(0, len(part), max_chars) if part[i : i + max_chars].strip())
            continue
        if current and len(current) + 1 + len(part) > max_chars:
            chunks.append(current.strip())
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _maybe_json(text: str) -> Any | None:
    """Parse JSON-looking strings."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _evidence_text(item: dict[str, Any]) -> str:
    """Extract text from an evidence object."""
    for key in ["text", "content", "passage", "source", "snippet", "document", "raw"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    primitive_parts = [str(value) for value in item.values() if isinstance(value, (str, int, float)) and str(value).strip()]
    return " ".join(primitive_parts)


def _labels(answer: str, hallucinations: Any, coarse: Any) -> list[SentenceLabel]:
    """Map annotations coarsely to sentence labels."""
    sentences = split_sentences(answer)
    if not sentences:
        return []
    hallucinations = _maybe_json(hallucinations) if isinstance(hallucinations, str) else hallucinations
    if hallucinations:
        LOGGER.warning("RAGTruth span-level annotations were mapped coarsely to sentence labels.")
    if isinstance(hallucinations, list):
        return _span_labels(answer, sentences, hallucinations)
    if hallucinations:
        relation = "Insufficient"
    else:
        is_supported = str(coarse).lower() in {"", "0", "false", "supported", "non-hallucinated", "none", "[]"}
        relation = "Supported" if is_supported else "Insufficient"
    return [_make_label(i, sent, relation) for i, sent in enumerate(sentences)]


def _span_labels(answer: str, sentences: list[str], hallucinations: list[Any]) -> list[SentenceLabel]:
    """Map span annotations to overlapping sentence labels when offsets exist."""
    spans = []
    for item in hallucinations:
        if isinstance(item, dict):
            start = item.get("start", item.get("start_idx", item.get("start_index")))
            end = item.get("end", item.get("end_idx", item.get("end_index")))
            if isinstance(start, int) and isinstance(end, int):
                spans.append((start, end))
    labels = []
    cursor = 0
    for i, sent in enumerate(sentences):
        start = answer.find(sent, cursor)
        start = cursor if start < 0 else start
        end = start + len(sent)
        cursor = end
        relation = "Insufficient" if any(not (span_end <= start or span_start >= end) for span_start, span_end in spans) else "Supported"
        if hallucinations and not spans:
            relation = "Insufficient"
        labels.append(_make_label(i, sent, relation))
    return labels


def _make_label(index: int, sentence: str, relation: str) -> SentenceLabel:
    """Create a sentence label."""
    return SentenceLabel(
        f"s{index + 1}",
        sentence,
        relation,
        0 if relation == "Supported" else 1,
        "No hallucination" if relation == "Supported" else "Unknown",
        "Sufficient" if relation == "Supported" else "Uncertain",
    )


def _conversion_stats(samples: list[RAGSample], rows: list[dict[str, Any]], selected: dict[str, str]) -> dict[str, Any]:
    """Compute conversion statistics."""
    evidence_counts = [len(sample.evidences) for sample in samples]
    sentence_counts = [len(sample.sentence_labels) for sample in samples]
    labeled = sum(1 for sample in samples if any(label.gold_attribution != "Unknown" for label in sample.sentence_labels))
    mapping = _mapping_quality(rows, selected)
    return {
        "num_samples": len(samples),
        "num_claims": sum(sentence_counts),
        "evidence_count_distribution": dict(Counter(evidence_counts)),
        "answer_sentence_count_distribution": dict(Counter(sentence_counts)),
        "num_labeled_samples": labeled,
        "num_unlabeled_samples": len(samples) - labeled,
        "num_unmapped_span_samples": mapping["span_unable_to_align_count"],
        **mapping,
    }


def _conversion_report_md(stats: dict[str, Any], field_report: dict[str, Any]) -> str:
    """Render conversion report markdown."""
    lines = [
        "# RAGTruth Conversion Report",
        "",
        f"Samples: {stats['num_samples']}",
        f"Claims: {stats['num_claims']}",
        f"Labeled samples: {stats['num_labeled_samples']}",
        f"Unlabeled samples: {stats['num_unlabeled_samples']}",
        f"Unmapped span samples: {stats['num_unmapped_span_samples']}",
        "",
        "## Selected Fields",
        "",
    ]
    for key, value in field_report.get("selected_fields", {}).items():
        lines.append(f"- {key}: {value or 'not found'}")
    lines.extend(["", "## Evidence Count Distribution", ""])
    for key, value in sorted(stats["evidence_count_distribution"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Answer Sentence Count Distribution", ""])
    for key, value in sorted(stats["answer_sentence_count_distribution"].items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _raw_response_rows(raw_path: Path) -> list[dict[str, Any]]:
    """Read all response rows for label auditing."""
    response_path = raw_path / "response.jsonl" if raw_path.is_dir() else raw_path
    if not response_path.exists():
        return []
    return list(_iter_records(response_path))


def _mapping_quality(rows: list[dict[str, Any]], selected: dict[str, str]) -> dict[str, Any]:
    """Measure how labels map from spans to sentences."""
    answer_field = selected.get("answer_field") or "response"
    label_field = selected.get("hallucination_field") or "labels"
    hit_sentences = 0
    no_hit_sentences = 0
    unable = 0
    cross_sentence = 0
    fallback = 0
    for row in rows:
        answer = str(row.get(answer_field, ""))
        sentences = split_sentences(answer)
        labels = _maybe_json(row.get(label_field)) if isinstance(row.get(label_field), str) else row.get(label_field)
        spans = _extract_spans(labels)
        if labels and not spans:
            fallback += len(sentences)
            unable += len(labels) if isinstance(labels, list) else 1
            continue
        sentence_ranges = _sentence_ranges(answer, sentences)
        touched = set()
        for span_start, span_end, _ in spans:
            overlapped = [i for i, (start, end) in enumerate(sentence_ranges) if not (span_end <= start or span_start >= end)]
            if not overlapped:
                unable += 1
            if len(overlapped) > 1:
                cross_sentence += 1
            touched.update(overlapped)
        hit_sentences += len(touched)
        no_hit_sentences += max(0, len(sentences) - len(touched))
    return {
        "span_hit_sentence_count": hit_sentences,
        "span_no_hit_sentence_count": no_hit_sentences,
        "span_unable_to_align_count": unable,
        "span_cross_sentence_count": cross_sentence,
        "response_level_fallback_sentence_count": fallback,
    }


def _extract_spans(labels: Any) -> list[tuple[int, int, dict[str, Any]]]:
    """Extract offset spans from RAGTruth labels."""
    spans = []
    if not isinstance(labels, list):
        return spans
    for item in labels:
        if not isinstance(item, dict):
            continue
        start = item.get("start", item.get("start_idx", item.get("start_index")))
        end = item.get("end", item.get("end_idx", item.get("end_index")))
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end, item))
    return spans


def _sentence_ranges(answer: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Return character ranges for split sentences."""
    ranges = []
    cursor = 0
    for sent in sentences:
        start = answer.find(sent, cursor)
        start = cursor if start < 0 else start
        end = start + len(sent)
        ranges.append((start, end))
        cursor = end
    return ranges


def _label_audit_md(raw_path: Path, rows: list[dict[str, Any]], samples: list[RAGSample], selected: dict[str, str], stats: dict[str, Any]) -> str:
    """Render RAGTruth label mapping audit."""
    raw_rows = _raw_response_rows(raw_path)
    label_counter = Counter()
    label_field_counter = Counter()
    label_type_counter = Counter()
    examples = []
    nonempty = 0
    span_total = 0
    for row in raw_rows:
        labels = row.get("labels", [])
        if labels:
            nonempty += 1
            span_total += len(labels) if isinstance(labels, list) else 1
        if isinstance(labels, list):
            label_counter[len(labels)] += 1
            for item in labels:
                if isinstance(item, dict):
                    label_field_counter.update(item.keys())
                    for key in ["type", "label_type", "category"]:
                        if key in item:
                            label_type_counter[str(item.get(key))] += 1
                    if len(examples) < 3:
                        examples.append(item)
    raw_count = len(raw_rows)
    processed_claims = sum(len(sample.sentence_labels) for sample in samples)
    offset_supported = "start" in label_field_counter and "end" in label_field_counter
    relation_reliability = "unreliable"
    binary_reliability = "partially reliable" if offset_supported else "unreliable"
    overall = "partially reliable" if binary_reliability == "partially reliable" else "unreliable"
    lines = [
        "# RAGTruth Label Mapping Audit",
        "",
        "## Raw `labels` Structure",
        "",
        f"Total response rows: {raw_count}",
        f"Labels empty: {raw_count - nonempty} ({((raw_count - nonempty) / raw_count if raw_count else 0):.4f})",
        f"Labels non-empty: {nonempty} ({(nonempty / raw_count if raw_count else 0):.4f})",
        f"Average hallucination spans per response: {(span_total / raw_count if raw_count else 0):.4f}",
        f"Average hallucination spans per non-empty response: {(span_total / nonempty if nonempty else 0):.4f}",
        f"Span field names: {dict(label_field_counter)}",
        f"Span type/category values: {dict(label_type_counter)}",
        f"Has start/end offsets: {offset_supported}",
        f"Example span records: {json.dumps(examples, ensure_ascii=False)[:2000]}",
        "",
        "## Sentence-Level Mapping Rules",
        "",
        "- A sentence is `hallucinated=1` when its character range overlaps any RAGTruth hallucination span with usable offsets.",
        "- A sentence is `hallucinated=0` when no hallucination span overlaps it.",
        "- `Supported` is currently used as the weak relation label for non-overlapping sentences.",
        "- `Insufficient` is currently used as the weak relation label for overlapping hallucination spans.",
        "- `Contradicted` cannot be reliably derived from the public RAGTruth span labels and is not a trustworthy gold class.",
        "- If labels are present but offsets cannot be parsed, the converter falls back to response-level weak hallucination labels.",
        "",
        "## Mapping Quality Statistics",
        "",
        f"Processed response count: {len(rows)}",
        f"Processed sentence/claim count: {processed_claims}",
        f"Sentences hit by spans: {stats.get('span_hit_sentence_count', 0)}",
        f"Sentences not hit by spans: {stats.get('span_no_hit_sentence_count', 0)}",
        f"Spans unable to align to a sentence: {stats.get('span_unable_to_align_count', 0)}",
        f"Spans crossing multiple sentences: {stats.get('span_cross_sentence_count', 0)}",
        f"Sentence labels from response-level fallback: {stats.get('response_level_fallback_sentence_count', 0)}",
        "",
        "## Reliability Judgment",
        "",
        f"Sentence-level hallucination detection reliability: {binary_reliability}",
        f"Supported / Insufficient / Contradicted relation reliability: {relation_reliability}",
        f"Overall mapping reliability: {overall}",
        "",
        "RAGTruth 原始标注更适合 hallucination span detection，而不是天然的 three-way relation attribution。当前 relation/attribution 标签是弱映射，不应作为强监督主结论。",
    ]
    return "\n".join(lines)

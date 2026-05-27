"""Fast external-detector + SCAD audit feature fusion for pilot/full diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from scad_rag.config import load_config
from scad_rag.data.preprocess import load_samples
from scad_rag.evaluation.metrics import compute_metrics
from scad_rag.features.counterfactual_audit import audit_claim
from scad_rag.models.dummy_models import DummyEmbedder
from scad_rag.models.nli import build_nli_model
from scad_rag.schema import Evidence, decompose_sample, inference_claim_view
from scad_rag.utils.io import ensure_dir, write_csv, write_json
from scad_rag.utils.seed import set_seed


def main() -> int:
    """Run a lightweight external-detector fusion experiment."""
    parser = argparse.ArgumentParser(description="Run fast external detector + SCAD fusion.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_claims", type=int, default=2500)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--pool_limit", type=int, default=80)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--detector", choices=["lettuce", "hhem", "osiris"], default="lettuce")
    parser.add_argument("--detector_batch_size", type=int, default=8)
    parser.add_argument("--lettuce_model_path", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    parser.add_argument("--hhem_model_path", default="vectara/hallucination_evaluation_model")
    parser.add_argument("--osiris_model_path", default="judgmentlabs/Qwen2.5-Osiris-3B-Instruct")
    parser.add_argument("--osiris_gguf_repo", default="mradermacher/Qwen2.5-Osiris-3B-Instruct-GGUF")
    parser.add_argument("--osiris_gguf_file", default="Qwen2.5-Osiris-3B-Instruct.Q4_K_M.gguf")
    parser.add_argument("--osiris_max_input_chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", args.seed)))
    config["use_dummy_models"] = False
    config["top_k"] = args.top_k
    config.setdefault("counterfactual", {})["enable_contradiction_probe"] = False
    config.setdefault("thresholds", {})["max_hard_negative_candidates"] = 4
    root = ensure_dir(
        Path(args.output_dir or config.get("output_dir", "experiments/runs"))
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.detector}_fusion_fast"
    )

    samples = load_samples(config, str(config.get("dataset", "ragtruth")), max_samples=args.max_samples)
    claims = []
    for sample in samples:
        for claim in decompose_sample(sample):
            claims.append(claim)
            if args.max_claims and len(claims) >= args.max_claims:
                break
        if args.max_claims and len(claims) >= args.max_claims:
            break

    embedder = DummyEmbedder()
    nli = build_nli_model(config)
    pool = _small_pool(samples, args.pool_limit)
    detector = _load_external_detector(
        args.detector,
        args.lettuce_model_path,
        args.hhem_model_path,
        args.osiris_model_path,
        args.osiris_gguf_repo,
        args.osiris_gguf_file,
    )
    rows = []
    hhem_pairs: list[tuple[str, str]] = []
    osiris_pairs: list[tuple[str, str]] = []
    for idx, claim in enumerate(claims, start=1):
        feature_claim = inference_claim_view(claim, bool(config.get("strict_no_gold_inference", True)))
        audit = audit_claim(
            feature_claim,
            embedder,
            nli,
            args.top_k,
            config.get("scad_weights", {}),
            config.get("thresholds", {}),
            config.get("counterfactual", {}),
            config.get("risk_calibration", {}),
            pool,
            allow_gold_hard_negatives=False,
        )
        if args.detector == "hhem":
            detector_score = 0.0
            hhem_pairs.append((audit.best_evidence_text or "", claim.claim_text))
        elif args.detector == "osiris":
            detector_score = 0.0
            osiris_pairs.append((audit.best_evidence_text or "", claim.claim_text))
        else:
            detector_score = _external_score(detector, claim.question, claim.claim_text, audit.best_evidence_text)
        row = _row_from_audit(claim, audit, detector_score, args.detector)
        rows.append(row)
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(claims)} claims")
    if args.detector == "hhem":
        scores = _hhem_scores_batch(detector["model"], hhem_pairs, max(1, args.detector_batch_size))
        for row, score in zip(rows, scores):
            row["external_score"] = score
            row["detector_score"] = score
    if args.detector == "osiris":
        scores = _osiris_scores_batch(detector, osiris_pairs, max(1, args.detector_batch_size), args.osiris_max_input_chars)
        for row, score in zip(rows, scores):
            row["external_score"] = score
            row["detector_score"] = score

    result = _run_fusion(rows, int(config.get("seed", args.seed)), _detector_label(args.detector))
    write_csv(root / "external_fusion_features.csv", rows)
    write_csv(root / "external_detector_fusion_results.csv", result["summary"])
    for name, preds in result["predictions"].items():
        write_csv(root / f"{_safe(name)}_predictions.csv", preds)
    write_json(root / "external_detector_fusion_report.json", {k: v for k, v in result.items() if k != "predictions"})
    (root / "external_detector_fusion_report.md").write_text(_report_md(result), encoding="utf-8")
    print(json.dumps({"run_dir": str(root), "detector": args.detector, "summary": result["summary"]}, indent=2))
    return 0


def _small_pool(samples: list[Any], limit: int) -> list[Evidence]:
    """Build a bounded cross-sample evidence pool."""
    pool = []
    for sample in samples:
        for ev in sample.evidences[:2]:
            pool.append(Evidence(f"{sample.id}:{ev.id}", ev.text, ev.type))
            if len(pool) >= limit:
                return pool
    return pool


def _load_lettuce(model_path: str):
    """Load LettuceDetect detector."""
    try:
        from lettucedetect.models.inference import HallucinationDetector  # type: ignore

        return HallucinationDetector(method="transformer", model_path=model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load LettuceDetect model '{model_path}': {exc}") from exc


def _load_external_detector(
    detector: str,
    lettuce_model_path: str,
    hhem_model_path: str,
    osiris_model_path: str,
    osiris_gguf_repo: str,
    osiris_gguf_file: str,
) -> dict[str, Any]:
    """Load the selected external detector."""
    if detector == "lettuce":
        return {"type": "lettuce", "model": _load_lettuce(lettuce_model_path)}
    if detector == "hhem":
        return {"type": "hhem", "model": _load_hhem(hhem_model_path)}
    if detector == "osiris":
        return _load_osiris(osiris_model_path, osiris_gguf_repo, osiris_gguf_file)
    raise ValueError(f"Unsupported detector: {detector}")


def _load_hhem(model_path: str):
    """Load Vectara HHEM local detector."""
    try:
        from transformers import AutoModelForSequenceClassification  # type: ignore

        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        try:
            model.eval()
        except Exception:
            pass
        return model
    except Exception as exc:
        raise RuntimeError(f"Could not load HHEM model '{model_path}': {exc}") from exc


def _load_osiris(model_path: str, gguf_repo: str, gguf_file: str) -> dict[str, Any]:
    """Load Osiris local causal detector."""
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return {"type": "osiris", "model": model, "tokenizer": tokenizer, "device": device}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA out of memory while loading Osiris. Try reducing detector_batch_size, "
                "using the 1.5B Osiris model, or running on CPU/offload."
            ) from exc
        raise RuntimeError(f"Could not load Osiris model '{model_path}': {exc}") from exc
    except Exception as exc:
        print(f"Could not load Transformers Osiris model '{model_path}'. Falling back to GGUF: {exc}")
        return _load_osiris_gguf(gguf_repo, gguf_file)


def _load_osiris_gguf(repo_id: str, filename: str) -> dict[str, Any]:
    """Load public Osiris GGUF with llama-cpp-python."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        from llama_cpp import Llama  # type: ignore

        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        except OSError as exc:
            if "Consistency check failed" not in str(exc):
                raise
            print("Incomplete Osiris GGUF download detected; retrying with force_download=True.")
            path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=True)
        threads = max(2, min(8, (os.cpu_count() or 4) // 2))
        llm = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=threads,
            n_batch=128,
            n_gpu_layers=-1,
            verbose=False,
        )
        return {"type": "osiris_gguf", "model": llm, "model_path": path, "device": "cpu"}
    except Exception as exc:
        raise RuntimeError(f"Could not load Osiris GGUF model '{repo_id}/{filename}': {exc}") from exc


def _external_score(detector: dict[str, Any], question: str, claim: str, evidence: str) -> float:
    """Return a hallucination-oriented external detector score."""
    kind = str(detector.get("type"))
    model = detector.get("model")
    if kind == "lettuce":
        return _lettuce_score(model, question, claim, evidence)
    if kind == "hhem":
        return _hhem_score(model, claim, evidence)
    if kind in {"osiris", "osiris_gguf"}:
        return _osiris_scores_batch(detector, [(evidence, claim)], 1, 2200)[0]
    raise ValueError(f"Unsupported detector type: {kind}")


def _lettuce_score(detector: Any, question: str, claim: str, evidence: str) -> float:
    """Return max LettuceDetect span confidence."""
    try:
        spans = detector.predict(context=[evidence] if evidence else [], answer=claim, question=question, output_format="spans")
    except Exception:
        spans = []
    if not isinstance(spans, list) or not spans:
        return 0.0
    vals = []
    for span in spans:
        if isinstance(span, dict):
            vals.append(_float(span.get("confidence", 1.0), 1.0))
        else:
            vals.append(1.0)
    return max(vals) if vals else 0.0


def _hhem_score(model: Any, claim: str, evidence: str) -> float:
    """Return HHEM hallucination score as 1 - consistency."""
    if not evidence:
        return 1.0
    try:
        pred = model.predict([(evidence, claim)])
        try:
            value = float(pred[0].detach().cpu().item())
        except Exception:
            value = float(pred[0])
        return max(0.0, min(1.0, 1.0 - value))
    except Exception:
        return 0.5


def _hhem_scores_batch(model: Any, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Return HHEM hallucination scores in batches."""
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        prepared = [(evidence, claim) if evidence else ("", claim) for evidence, claim in batch]
        try:
            pred = model.predict(prepared)
            for item, (evidence, _claim) in zip(pred, prepared):
                if not evidence:
                    scores.append(1.0)
                    continue
                try:
                    value = float(item.detach().cpu().item())
                except Exception:
                    value = float(item)
                scores.append(max(0.0, min(1.0, 1.0 - value)))
        except Exception:
            for evidence, claim in prepared:
                scores.append(_hhem_score(model, claim, evidence))
        print(f"Scored HHEM {min(start + batch_size, len(pairs))}/{len(pairs)} claims")
    return scores


def _osiris_scores_batch(detector: dict[str, Any], pairs: list[tuple[str, str]], batch_size: int, max_input_chars: int) -> list[float]:
    """Return Osiris hallucination probabilities from local next-token label scores."""
    if detector.get("type") == "osiris_gguf":
        return _osiris_gguf_scores(detector["model"], pairs, max_input_chars)

    import torch  # type: ignore

    model = detector["model"]
    tokenizer = detector["tokenizer"]
    device = detector["device"]
    hall_ids = _label_token_ids(tokenizer, [" HALLUCINATED", "HALLUCINATED", " hallucinated", "Hallucinated"])
    support_ids = _label_token_ids(tokenizer, [" SUPPORTED", "SUPPORTED", " supported", "Supported"])
    if not hall_ids or not support_ids:
        raise RuntimeError("Could not derive Osiris label token ids for HALLUCINATED/SUPPORTED.")
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        prompts = [_osiris_prompt(evidence, claim, max_input_chars) for evidence, claim in batch]
        try:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits[:, -1, :]
            for row_logits in logits:
                hall_logit = torch.logsumexp(row_logits[hall_ids], dim=0)
                support_logit = torch.logsumexp(row_logits[support_ids], dim=0)
                prob = torch.softmax(torch.stack([support_logit, hall_logit]), dim=0)[1].item()
                scores.append(max(0.0, min(1.0, float(prob))))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                torch.cuda.empty_cache()
                scores.extend(_osiris_scores_batch(detector, batch, 1, max_input_chars))
            elif "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory during Osiris scoring. Try reducing osiris_max_input_chars, "
                    "using the 1.5B Osiris model, or running fewer claims."
                ) from exc
            else:
                raise
        print(f"Scored Osiris {min(start + batch_size, len(pairs))}/{len(pairs)} claims")
    return scores


def _osiris_gguf_scores(llm: Any, pairs: list[tuple[str, str]], max_input_chars: int) -> list[float]:
    """Return Osiris GGUF hallucination probabilities by short deterministic generation."""
    scores: list[float] = []
    for idx, (evidence, claim) in enumerate(pairs, start=1):
        prompt = _osiris_prompt(evidence, claim, max_input_chars)
        result = llm(prompt, max_tokens=3, temperature=0.0, echo=False)
        choice = result["choices"][0]
        text = str(choice.get("text", "")).strip().upper()
        score = _score_osiris_text(text)
        scores.append(score)
        print(f"Scored Osiris-GGUF {idx}/{len(pairs)} claims: {text[:20]} -> {score:.4f}")
    return scores


def _score_osiris_text(text: str) -> float:
    """Convert an Osiris textual label into a hallucination probability."""
    if "HALLUCINATED" in text or text.startswith("HALLUC") or text.startswith("1"):
        return 0.9
    if "SUPPORTED" in text or text.startswith("0"):
        return 0.1
    return 0.5


def _score_from_logprobs(top_logprobs: dict[str, float], fallback: float) -> float:
    """Estimate binary hallucination probability from first-token logprobs."""
    hall_vals = []
    support_vals = []
    for token, value in top_logprobs.items():
        norm = str(token).strip().upper()
        if norm.startswith("1") or norm.startswith("H"):
            hall_vals.append(float(value))
        if norm.startswith("0") or norm.startswith("S"):
            support_vals.append(float(value))
    if not hall_vals or not support_vals:
        return fallback
    h = _logsumexp(hall_vals)
    s = _logsumexp(support_vals)
    return 1.0 / (1.0 + math.exp(s - h))


def _logsumexp(values: list[float]) -> float:
    """Stable logsumexp for small lists."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _label_token_ids(tokenizer: Any, labels: list[str]) -> list[int]:
    """Return unique first-token ids for a set of textual labels."""
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def _osiris_prompt(evidence: str, claim: str, max_input_chars: int) -> str:
    """Build a short Osiris binary hallucination prompt."""
    evidence = (evidence or "")[:max_input_chars]
    claim = (claim or "")[:700]
    return (
        "You are a hallucination detector for retrieval-augmented generation.\n"
        "Given the context and the claim, decide whether the claim is supported by the context.\n"
        "Answer with exactly one label: SUPPORTED or HALLUCINATED.\n\n"
        f"Context:\n{evidence}\n\n"
        f"Claim:\n{claim}\n\n"
        "Label:"
    )


def _row_from_audit(claim: Any, audit: Any, detector_score: float, detector: str) -> dict[str, Any]:
    """Convert audit features into a fusion row."""
    return {
        "sample_id": claim.sample_id,
        "claim_id": claim.claim_id,
        "question": claim.question,
        "claim_text": claim.claim_text,
        "best_evidence_text": audit.best_evidence_text,
        "gold_relation": claim.gold_relation,
        "gold_hallucination": claim.gold_hallucination,
        "gold_attribution": claim.gold_attribution,
        "detector": detector,
        "external_score": detector_score,
        "detector_score": detector_score,
        "relevance_score": audit.max_relevance,
        "entailment_score": audit.entailment_score,
        "neutral_score": audit.neutral_score,
        "contradiction_score": audit.contradiction_score,
        "coverage_score": audit.coverage_score,
        "sufficient_context_score": audit.sufficient_context_score,
        "score_original": audit.score_original,
        "score_removed": audit.score_removed,
        "score_hard_negative": audit.score_hard_negative,
        "evidence_dependency_delta": audit.evidence_dependency_delta,
        "hard_negative_robustness_gap": audit.hard_negative_robustness_gap,
        "uncertainty_score": audit.uncertainty_score,
        "risk_score": audit.risk_score,
        "nli_reliability_score": audit.nli_reliability_score,
    }


def _run_fusion(rows: list[dict[str, Any]], seed: int, detector_label: str) -> dict[str, Any]:
    """Tune simple fusion weights on validation and evaluate on held-out rows."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(len(idx) * 0.5))
    val_idx = set(idx[:cut])
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    test = [rows[i] for i in range(len(rows)) if i not in val_idx]
    ld_val = [_float(r["external_score"]) for r in val]
    ld_test = [_float(r["external_score"]) for r in test]
    scad_val = [1.0 - _float(r["score_original"]) for r in val]
    scad_test = [1.0 - _float(r["score_original"]) for r in test]
    ld_th = _tune_threshold(ld_val, val)
    scad_th = _tune_threshold(scad_val, val)
    best = None
    for w_ld in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for w_scad in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for w_risk in [0.0, 0.1, 0.2, 0.3]:
                for w_dep in [0.0, 0.05, 0.1, 0.2]:
                    if w_scad + w_risk + w_dep == 0:
                        continue
                    scores = _scores(val, w_ld, w_scad, w_risk, w_dep)
                    th = _tune_threshold(scores, val)
                    metrics = compute_metrics(_pred_rows(val, scores, th, None, "fusion-val"))
                    key = (metrics.get("hallucination_f1", 0.0), metrics.get("hallucination_auroc", 0.0))
                    candidate = (key, w_ld, w_scad, w_risk, w_dep, th)
                    if best is None or candidate > best:
                        best = candidate
    _, w_ld, w_scad, w_risk, w_dep, fusion_th = best
    best_risk = None
    for w in [x / 10 for x in range(0, 11)]:
        risk_val = _risk_scores(val, w)
        metrics = compute_metrics(_pred_rows(val, ld_val, ld_th, risk_val, "risk-val"))
        key = (metrics.get("risk_coverage_accuracy_auc", 0.0), -metrics.get("selective_risk_auc", 0.0))
        if best_risk is None or key > best_risk[0]:
            best_risk = (key, w)
    risk_w = best_risk[1]
    systems = {
        "SCAD-only": _pred_rows(test, scad_test, scad_th, None, "SCAD-only"),
        f"{detector_label}-only": _pred_rows(test, ld_test, ld_th, None, f"{detector_label}-only"),
        f"{detector_label}+SCAD-score-fusion": _pred_rows(test, _scores(test, w_ld, w_scad, w_risk, w_dep), fusion_th, None, "fusion"),
        f"{detector_label}+SCAD-risk-rerank": _pred_rows(test, ld_test, ld_th, _risk_scores(test, risk_w), "risk-rerank"),
    }
    summary = []
    for name, pred in systems.items():
        metrics = compute_metrics(pred)
        summary.append(
            {
                "system": name,
                "n_test": len(pred),
                "hall_f1": metrics.get("hallucination_f1", 0.0),
                "binary_macro_f1": metrics.get("hallucination_macro_f1", 0.0),
                "auroc": metrics.get("hallucination_auroc", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "ece": metrics.get("hallucination_ece", 0.0),
                "brier": metrics.get("hallucination_brier", 0.0),
                "risk_corr": metrics.get("risk_error_correlation", 0.0),
                "risk_cov_acc_auc": metrics.get("risk_coverage_accuracy_auc", 0.0),
                "selective_risk_auc": metrics.get("selective_risk_auc", 0.0),
            }
        )
    return {
        "n_total": len(rows),
        "n_val": len(val),
        "n_test": len(test),
        "ld_threshold": ld_th,
        "scad_threshold": scad_th,
        "fusion_weights": {"lettuce": w_ld, "scad_score": w_scad, "risk": w_risk, "low_dependency": w_dep, "threshold": fusion_th},
        "risk_rerank_weight_lettuce": risk_w,
        "summary": summary,
        "predictions": systems,
    }


def _scores(rows: list[dict[str, Any]], w_ld: float, w_scad: float, w_risk: float, w_dep: float) -> list[float]:
    """Weighted hallucination scores."""
    out = []
    denom = w_ld + w_scad + w_risk + w_dep
    for r in rows:
        dep = 1.0 - min(1.0, max(0.0, (_float(r["evidence_dependency_delta"]) + _float(r["hard_negative_robustness_gap"])) / 2.0))
        s = w_ld * _float(r["external_score"]) + w_scad * (1.0 - _float(r["score_original"])) + w_risk * _float(r["risk_score"]) + w_dep * dep
        out.append(s / denom if denom else _float(r["external_score"]))
    return out


def _risk_scores(rows: list[dict[str, Any]], lettuce_weight: float) -> list[float]:
    """Risk reranking scores."""
    out = []
    for r in rows:
        dep = 1.0 - min(1.0, max(0.0, (_float(r["evidence_dependency_delta"]) + _float(r["hard_negative_robustness_gap"])) / 2.0))
        scad_risk = 0.65 * _float(r["risk_score"]) + 0.35 * dep
        out.append(lettuce_weight * _float(r["external_score"]) + (1.0 - lettuce_weight) * scad_risk)
    return out


def _tune_threshold(scores: list[float], rows: list[dict[str, Any]]) -> float:
    """Tune threshold by validation Hall-F1."""
    best_th, best_f1 = 0.5, -1.0
    for idx in range(101):
        th = idx / 100
        f1 = compute_metrics(_pred_rows(rows, scores, th, None, "tune")).get("hallucination_f1", 0.0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th


def _pred_rows(rows: list[dict[str, Any]], scores: list[float], threshold: float, risks: list[float] | None, name: str) -> list[dict[str, Any]]:
    """Create prediction rows from scores."""
    out = []
    for i, (row, score) in enumerate(zip(rows, scores)):
        pred = 1 if score >= threshold else 0
        item = dict(row)
        item["pred_hallucination"] = pred
        item["pred_relation"] = "Insufficient" if pred else "Supported"
        item["pred_attribution"] = "Generation-inconsistent" if pred else "No hallucination"
        item["pred_context_status"] = "Insufficient" if pred else "Sufficient"
        item["calibrated_hallucination_probability"] = score
        item["risk_score"] = risks[i] if risks is not None else score
        item["explanation"] = f"{name}: score={score:.4f}, threshold={threshold:.4f}"
        out.append(item)
    return out


def _report_md(result: dict[str, Any]) -> str:
    """Render markdown report."""
    lines = [
        "# External Detector Fusion Fast Report",
        "",
        f"Total rows: {result['n_total']}; validation: {result['n_val']}; test: {result['n_test']}",
        f"External detector threshold: {result['ld_threshold']}",
        f"SCAD threshold: {result['scad_threshold']}",
        f"Fusion weights: {result['fusion_weights']}",
        f"Risk rerank external-detector weight: {result['risk_rerank_weight_lettuce']}",
        "",
        "| System | Hall-F1 | AUROC | ECE | Brier | Risk-Cov Acc AUC | Selective Risk AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['system']} | {row['hall_f1']:.4f} | {row['auroc']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} | {row['risk_cov_acc_auc']:.4f} | {row['selective_risk_auc']:.4f} |"
        )
    return "\n".join(lines)


def _safe(name: str) -> str:
    """Safe filename."""
    return name.replace("+", "_").replace("-", "_").replace(" ", "_")


def _detector_label(detector: str) -> str:
    """Return display label for an external detector."""
    if detector == "lettuce":
        return "LettuceDetect"
    if detector == "hhem":
        return "HHEM"
    return "Osiris-3B"


def _float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion."""
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())

# SCAD-RAG Code Release Notes

This repository contains the reproducible implementation of SCAD-RAG, a local evidence-auditing framework for RAG hallucination diagnosis.

## Release Scope

- Claim-level RAG hallucination pipeline with sufficient-context scoring, evidence perturbation diagnostics, and risk-calibrated prediction.
- Local inference only: no commercial LLM APIs or API-key services are used.
- Toy data and `quick_test` mode for CPU-only, offline verification.
- RAGTruth, FEVER, SciFact, and optional HaluBench data adapters.
- Baselines, ablations, threshold tuning, no-gold inference checks, and no-API guard.
- Paper-facing tables and SVG figures under `paper_assets/`.

## Recommended Commit Scope

The Git root contains other local project material. For a clean SCAD-RAG release, stage only:

```powershell
git add scad_rag
```

Do not use `git add .` from the parent directory unless the older local projects are intentionally part of the release.

## Pre-Commit Checks

```powershell
cd scad_rag
python -m scad_rag.utils.no_api_guard
pytest tests/
```

Expected status for this release: `18 passed`; pytest cache warnings on Windows do not affect correctness.

## Large Files

The repository intentionally excludes raw datasets, model weights, generated experiment runs, PDFs, and zip archives. Users should download public datasets with the provided CLIs and regenerate results locally.

# No API Policy

SCAD-RAG does not call commercial large-model APIs. The guard scans `src/`, `scripts/`, and `tests/` for blocked imports and endpoints.

```powershell
python -m scad_rag.utils.no_api_guard
```

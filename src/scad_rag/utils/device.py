"""Device resolution."""

from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda."""
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

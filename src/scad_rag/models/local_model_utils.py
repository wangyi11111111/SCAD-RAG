"""Local model helper messages."""

from __future__ import annotations


def format_local_model_error(model_name: str, kind: str) -> str:
    """Return a friendly local model error."""
    return f"Could not load local {kind} model '{model_name}'. Cache it locally or run configs/quick_test.yaml."

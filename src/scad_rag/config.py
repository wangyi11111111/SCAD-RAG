"""Small YAML config loader used by SCAD-RAG."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    """Parse a scalar config value."""
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    try:
        return float(value) if any(ch in value for ch in ".eE") else int(value)
    except ValueError:
        return value


def _lines(text: str) -> list[tuple[int, str]]:
    """Return non-empty indentation/content pairs."""
    out = []
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        out.append((indent, raw.strip()))
    return out


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    """Parse a tiny YAML subset sufficient for project configs."""
    is_list = index < len(lines) and lines[index][1].startswith("- ")
    container: Any = [] if is_list else {}
    while index < len(lines):
        current, content = lines[index]
        if current < indent:
            break
        if current > indent:
            raise ValueError(f"Invalid indentation near {content}")
        if is_list:
            if not content.startswith("- "):
                break
            item = content[2:].strip()
            container.append(_parse_scalar(item))
            index += 1
            continue
        key, sep, value = content.partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {content}")
        index += 1
        if value.strip():
            container[key.strip()] = _parse_scalar(value)
        elif index < len(lines) and lines[index][0] > current:
            child, index = _parse_block(lines, index, lines[index][0])
            container[key.strip()] = child
        else:
            container[key.strip()] = {}
    return container, index


def loads_yaml(text: str) -> dict[str, Any]:
    """Load a minimal YAML mapping without requiring PyYAML."""
    lines = _lines(text)
    if not lines:
        return {}
    parsed, _ = _parse_block(lines, 0, lines[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a mapping.")
    return parsed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        data = loads_yaml(text)
    data["_config_path"] = str(config_path)
    return data


def apply_thresholds_path(config: dict[str, Any], thresholds_path: str | Path | None = None) -> dict[str, Any]:
    """Apply thresholds from a YAML file to a config copy."""
    result = copy.deepcopy(config)
    selected = thresholds_path or result.get("thresholds_path")
    if selected in {None, "", "null"}:
        result["_threshold_application"] = {
            "thresholds_path": "",
            "applied": False,
            "reason": "No thresholds_path was provided.",
        }
        return result
    path = Path(str(selected))
    if not path.exists():
        from scad_rag.utils.io import readable_existing_path

        path = readable_existing_path(path)
    if not path.exists():
        raise FileNotFoundError(f"thresholds_path does not exist: {selected}")
    loaded = load_config(path)
    thresholds = {key: value for key, value in loaded.items() if not str(key).startswith("_")}
    result.setdefault("thresholds", {})
    result["thresholds"].update(thresholds)
    result["thresholds_path"] = str(path)
    result["_threshold_application"] = {
        "thresholds_path": str(path),
        "applied": True,
        "best_thresholds": thresholds,
    }
    return result


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively updated copy of a config."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def to_yaml(data: Any, indent: int = 0) -> str:
    """Serialize simple values into YAML-like text."""
    pad = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_format(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        return "\n".join(f"{pad}- {_format(item)}" for item in data)
    return f"{pad}{_format(data)}"


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write config text using the project's safe I/O fallback."""
    from scad_rag.utils.io import writable_file_path

    writable_file_path(path).write_text(to_yaml(data), encoding="utf-8")


def _format(value: Any) -> str:
    """Format one scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)

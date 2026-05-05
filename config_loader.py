"""Load and query the shared training configuration.

Purpose:
    Parse `configs/train.yaml` into a nested Python dictionary without adding
    a runtime dependency on a full YAML package.

Input / Output:
    Input: `configs/train.yaml` in the repository root.
    Output: a cached nested `dict[str, Any]` plus `get_config_value(path, default)`
    for safe dotted-path lookups.

Pipeline fit:
    Upstream input is the single project config file.
    Downstream output is consumed by preprocessing, cache building, model setup,
    and vocab fitting so every stage reads the same canonical settings.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train.yaml"
_CONFIG_PATH_ENV = "TRAIN_CONFIG_PATH"
_CONFIG_CACHE: Dict[str, Any] | None = None


def _resolve_config_path() -> Path:
    raw = os.environ.get(_CONFIG_PATH_ENV, "").strip()
    if not raw:
        return DEFAULT_CONFIG_PATH
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()
    return candidate


CONFIG_PATH = _resolve_config_path()


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out_chars: List[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
            out_chars.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out_chars.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out_chars.append(ch)
    return "".join(out_chars).rstrip()


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    if (
        (value.startswith("[") and value.endswith("]"))
        or (value.startswith("(") and value.endswith(")"))
        or (value.startswith("{") and value.endswith("}"))
    ):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        else:
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            return parsed

    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_lines(lines: List[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for idx, raw in enumerate(lines, start=1):
        clean = _strip_comment(raw)
        if not clean.strip():
            continue

        indent = len(clean) - len(clean.lstrip(" "))
        if clean.lstrip().startswith("- "):
            raise ValueError(f"Unsupported YAML list syntax at line {idx}: {raw!r}")

        content = clean.strip()
        if ":" not in content:
            raise ValueError(f"Invalid YAML line {idx}, expected key:value: {raw!r}")
        key, rest = content.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if not key:
            raise ValueError(f"Empty YAML key at line {idx}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if rest == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(rest)

    return root


def load_train_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load and cache the project training config.

    Args:
        force_reload (bool):
            When `True`, bypass the in-memory cache and reread
            `configs/train.yaml` from disk.

    Returns:
        Dict[str, Any]:
            Nested mapping representing the parsed config tree.

    Important notes:
        - Returns `{}` when the config file does not exist.
        - The parsed result is cached across calls unless reloaded explicitly.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    if not CONFIG_PATH.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    text = CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    _CONFIG_CACHE = _parse_lines(lines)
    return _CONFIG_CACHE


def get_config_value(path: str, default: Any) -> Any:
    """Resolve a dotted config path with a fallback.

    Args:
        path (str):
            Dotted lookup path such as ``"audio.sample_rate"`` or
            ``"vocab.special_tokens.pad"``.
        default (Any):
            Value to return when any path segment is missing.

    Returns:
        Any:
            The resolved config value, or `default` if the path is absent.

    Important notes:
        - This is read-only and does not mutate the cached config tree.
        - Missing intermediate nodes short-circuit to `default`.
    """
    node: Any = load_train_config()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

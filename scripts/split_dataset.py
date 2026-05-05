"""Stable group-aware dataset split utility.

Purpose:
    Build reproducible train/validation CSV splits from a manifest-like index
    (for example `manifest.csv` or `cache_index.csv`) while enforcing
    group-level isolation (e.g., beatmap set/song isolation).

Input / Output:
    Input:
        - CSV index configured by `split.input_index_path`
        - Required columns are flexible; common columns include:
          `sample_id`, `valid`, `set_id`, `audio_path`, `osu_path`
    Output:
        - `split.train_path`: train CSV (preserves all original columns)
        - `split.val_path`: val CSV (preserves all original columns)
        - optional `split.summary_path`: JSON summary with split metadata/stats

Pipeline fit:
    Upstream input:
        - Produced by indexing stages such as:
          `build_manifest.py` (manifest) or cache index generation
          (`scripts/build_train_cache.py`)
    Downstream output:
        - Consumed by training/validation data loaders and orchestration scripts
        - Used by vocab fitting and other train-only preprocessing to avoid
          train/val leakage
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import load_train_config

LOGGER = logging.getLogger("split_dataset")


def normalize_bool(value: Any, default: bool = False) -> bool:
    """Normalize heterogeneous boolean-like values.

    Args:
        value (Any):
            Raw value from config/CSV (bool/int/float/string/None).
        default (bool):
            Fallback value for empty/unrecognized inputs.

    Returns:
        bool:
            Normalized boolean value.

    Important notes:
        - String matching is case-insensitive.
        - Unrecognized text returns `default` instead of raising.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _canonical_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(project_root.resolve())
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class SplitSettings:
    project_root: Path
    input_index_path: Path
    output_dir: Path
    train_path: Path
    val_path: Path
    summary_path: Path | None
    val_ratio: float
    seed: int
    group_by: str
    fallback_group_by: str
    valid_only: bool
    shuffle: bool
    overwrite: bool


def load_split_settings(force_reload: bool = True) -> SplitSettings:
    """Load and validate split settings from `configs/train.yaml`.

    Args:
        force_reload (bool):
            Passed to config loader; when true, bypasses config cache.

    Returns:
        SplitSettings:
            Fully validated and path-resolved split configuration.

    Important notes:
        - Relative paths are resolved against project root.
        - `val_ratio` must be strictly between 0 and 1.
        - Missing required path fields raise `ValueError`.
    """
    config = load_train_config(force_reload=force_reload)
    if not isinstance(config, dict):
        raise ValueError("Invalid train config.")
    split_cfg = config.get("split", {})
    if not isinstance(split_cfg, dict):
        raise ValueError("split config must be a mapping.")

    input_index_path = _resolve_path(split_cfg.get("input_index_path"), PROJECT_ROOT)
    output_dir = _resolve_path(split_cfg.get("output_dir"), PROJECT_ROOT)
    train_path = _resolve_path(split_cfg.get("train_path"), PROJECT_ROOT)
    val_path = _resolve_path(split_cfg.get("val_path"), PROJECT_ROOT)
    summary_path = _resolve_path(split_cfg.get("summary_path"), PROJECT_ROOT)
    if input_index_path is None:
        raise ValueError("split.input_index_path must be configured.")
    if output_dir is None:
        raise ValueError("split.output_dir must be configured.")
    if train_path is None:
        raise ValueError("split.train_path must be configured.")
    if val_path is None:
        raise ValueError("split.val_path must be configured.")

    val_ratio = _as_float(split_cfg.get("val_ratio"), 0.1)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"split.val_ratio must be in (0,1), got {val_ratio!r}")

    seed = _as_int(split_cfg.get("seed"), 42)
    group_by = str(split_cfg.get("group_by", "set_id")).strip()
    fallback_group_by = str(split_cfg.get("fallback_group_by", "audio_path")).strip()
    if not group_by:
        raise ValueError("split.group_by must not be empty.")
    if not fallback_group_by:
        raise ValueError("split.fallback_group_by must not be empty.")

    return SplitSettings(
        project_root=PROJECT_ROOT,
        input_index_path=input_index_path,
        output_dir=output_dir,
        train_path=train_path,
        val_path=val_path,
        summary_path=summary_path,
        val_ratio=val_ratio,
        seed=seed,
        group_by=group_by,
        fallback_group_by=fallback_group_by,
        valid_only=normalize_bool(split_cfg.get("valid_only"), True),
        shuffle=normalize_bool(split_cfg.get("shuffle"), True),
        overwrite=normalize_bool(split_cfg.get("overwrite"), False),
    )


def load_index(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load an input index CSV with UTF-8 BOM compatibility.

    Args:
        path (Path):
            Path to CSV index (manifest/cache index/split-like file).

    Returns:
        Tuple[List[Dict[str, str]], List[str]]:
            - rows: list of row dicts keyed by CSV header fields
            - fieldnames: CSV header order for downstream write-back

    Important notes:
        - Raises `FileNotFoundError` if the path does not exist.
        - Raises `ValueError` when header is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input index file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fieldnames


def _row_valid(row: Mapping[str, Any]) -> bool:
    return normalize_bool(row.get("valid"), default=False)


def _norm_key(value: Any) -> str:
    return str(value or "").strip()


def _fallback_from_audio_or_osu_parent(row: Mapping[str, Any]) -> str:
    audio_path = _norm_key(row.get("audio_path"))
    if audio_path:
        return audio_path.replace("\\", "/").lower()
    osu_path = _norm_key(row.get("osu_path"))
    if not osu_path:
        return ""
    p = Path(osu_path)
    parent = p.parent.as_posix() if p.parent != Path(".") else ""
    return parent.lower()


def get_group_key(row: Mapping[str, Any], config: SplitSettings) -> str:
    """Resolve deterministic split group key for one row.

    Args:
        row (Mapping[str, Any]):
            Input sample row from the index CSV.
        config (SplitSettings):
            Split configuration with `group_by` and fallback strategy.

    Returns:
        str:
            Stable group key with explicit namespace prefix
            (e.g., `set_id:12345`, `audio_path:./x.mp3`).

    Important notes:
        - Primary key uses `config.group_by`.
        - When primary key is empty, fallback uses `config.fallback_group_by`.
        - For `audio_path`/`osu_path` fallback, path-based normalization is used.
        - Final fallback prevents empty grouping by using `sample_id`/`osu_path`.
    """
    primary = _norm_key(row.get(config.group_by))
    if primary:
        return f"{config.group_by}:{primary}"

    fallback = _norm_key(row.get(config.fallback_group_by))
    if not fallback and config.fallback_group_by in {"audio_path", "osu_path"}:
        fallback = _fallback_from_audio_or_osu_parent(row)
    if fallback:
        return f"{config.fallback_group_by}:{fallback}"

    sample_id = _norm_key(row.get("sample_id"))
    if sample_id:
        return f"sample_id:{sample_id}"
    osu_path = _norm_key(row.get("osu_path"))
    if osu_path:
        return f"osu_path:{osu_path}"
    return "unknown:"


def _row_sort_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    primary = [
        str(row.get("sample_id", "")).strip(),
        str(row.get("set_id", "")).strip(),
        str(row.get("audio_path", "")).strip(),
        str(row.get("osu_path", "")).strip(),
    ]
    stable_rest = [f"{k}={row[k]}" for k in sorted(row.keys())]
    return tuple(primary + stable_rest)


def split_groups(
    grouped_rows: Mapping[str, List[Dict[str, str]]],
    val_ratio: float,
    seed: int,
    shuffle: bool,
) -> Tuple[set[str], set[str]]:
    """Split grouped samples into train/val group key sets.

    Args:
        grouped_rows (Mapping[str, List[Dict[str, str]]]):
            Mapping from group key to all rows in that group.
        val_ratio (float):
            Target validation fraction, applied at group granularity.
        seed (int):
            Random seed used when `shuffle=True`.
        shuffle (bool):
            Whether to shuffle sorted group keys before partitioning.

    Returns:
        Tuple[set[str], set[str]]:
            `(train_keys, val_keys)` as disjoint group-key sets.

    Important notes:
        - Ratio target is approximate because whole groups are indivisible.
        - At least one group is forced into each split when possible.
        - Determinism comes from sort + seeded shuffle.
    """
    group_keys = sorted(grouped_rows.keys())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(group_keys)

    total_rows = sum(len(grouped_rows[key]) for key in group_keys)
    target_val_rows = total_rows * val_ratio

    val_keys: set[str] = set()
    val_rows = 0
    for key in group_keys:
        # Stop only after at least one val group exists; this avoids empty-val
        # when the first group already exceeds the target row budget.
        if val_rows >= target_val_rows and val_keys:
            break
        val_keys.add(key)
        val_rows += len(grouped_rows[key])

    train_keys = set(group_keys) - val_keys
    # Guardrails: keep both splits non-empty for downstream train/val flows.
    if not train_keys:
        largest_key = max(val_keys, key=lambda k: len(grouped_rows[k]))
        val_keys.remove(largest_key)
        train_keys.add(largest_key)
    if not val_keys:
        smallest_key = min(train_keys, key=lambda k: len(grouped_rows[k]))
        train_keys.remove(smallest_key)
        val_keys.add(smallest_key)
    return train_keys, val_keys


def write_split(rows: Iterable[Dict[str, str]], path: Path, fieldnames: List[str]) -> None:
    """Write split rows to CSV while preserving caller-provided column order.

    Args:
        rows (Iterable[Dict[str, str]]):
            Row dictionaries to write.
        path (Path):
            Target CSV path.
        fieldnames (List[str]):
            Header/column order, typically from `load_index`.

    Returns:
        None

    Important notes:
        - Parent directory is created automatically.
        - Existing file handling is controlled by caller-side overwrite checks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_no_group_overlap(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    config: SplitSettings,
) -> Tuple[set[str], set[str]]:
    """Validate train/val group isolation and return observed group sets.

    Args:
        train_rows (List[Dict[str, str]]):
            Rows assigned to train split.
        val_rows (List[Dict[str, str]]):
            Rows assigned to validation split.
        config (SplitSettings):
            Group-key resolution settings.

    Returns:
        Tuple[set[str], set[str]]:
            `(train_groups, val_groups)` computed from rows.

    Important notes:
        - Raises `RuntimeError` if any group key appears in both splits.
        - Error payload shows a small deterministic overlap sample.
    """
    train_groups = {get_group_key(row, config) for row in train_rows}
    val_groups = {get_group_key(row, config) for row in val_rows}
    overlap = train_groups & val_groups
    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(f"Detected train/val group overlap: {examples}")
    return train_groups, val_groups


def write_split_summary(summary: Dict[str, Any], path: Path) -> None:
    """Persist split metadata and statistics as JSON.

    Args:
        summary (Dict[str, Any]):
            Summary payload generated by `run_split`.
        path (Path):
            Output JSON path.

    Returns:
        None

    Important notes:
        - Uses UTF-8 and trailing newline for stable text artifacts.
        - Key order follows insertion order from summary construction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False, sort_keys=False)
        fp.write("\n")


def _check_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [p for p in paths if p.exists()]
    if existing:
        joined = ", ".join(str(p) for p in existing)
        raise FileExistsError(f"Output exists; set split.overwrite=true to overwrite: {joined}")


def run_split(settings: SplitSettings) -> Dict[str, Any]:
    """Execute full split pipeline and return split summary.

    Args:
        settings (SplitSettings):
            Split configuration containing paths, grouping, ratio, and seed.

    Returns:
        Dict[str, Any]:
            Summary metadata/statistics, including counts and realized val ratio.

    Important notes:
        - If `valid_only=True`, only rows with `valid=true` are considered.
        - Group overlap is checked and treated as a hard error.
        - Within-group rows are canonically sorted for input-order robustness.
        - Raises `ValueError` when no eligible rows remain after filtering.
    """
    rows, fieldnames = load_index(settings.input_index_path)
    source_rows = rows
    if settings.valid_only:
        source_rows = [row for row in source_rows if _row_valid(row)]

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in source_rows:
        key = get_group_key(row, settings)
        grouped.setdefault(key, []).append(row)
    # Canonical per-group ordering minimizes sensitivity to input row order.
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=_row_sort_key)

    if not grouped:
        raise ValueError("No rows available for splitting after filters.")

    train_keys, val_keys = split_groups(
        grouped,
        val_ratio=settings.val_ratio,
        seed=settings.seed,
        shuffle=settings.shuffle,
    )

    train_rows = [row for key in sorted(train_keys) for row in grouped[key]]
    val_rows = [row for key in sorted(val_keys) for row in grouped[key]]
    train_groups, val_groups = validate_no_group_overlap(train_rows, val_rows, settings)

    outputs = [settings.train_path, settings.val_path]
    if settings.summary_path is not None:
        outputs.append(settings.summary_path)
    _check_overwrite(outputs, settings.overwrite)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    write_split(train_rows, settings.train_path, fieldnames)
    write_split(val_rows, settings.val_path, fieldnames)

    total = len(train_rows) + len(val_rows)
    val_actual_ratio = (len(val_rows) / float(total)) if total else 0.0

    summary = {
        "input_index_path": _canonical_path(settings.input_index_path, settings.project_root),
        "train_path": _canonical_path(settings.train_path, settings.project_root),
        "val_path": _canonical_path(settings.val_path, settings.project_root),
        "summary_path": _canonical_path(settings.summary_path, settings.project_root)
        if settings.summary_path
        else "",
        "seed": settings.seed,
        "val_ratio": settings.val_ratio,
        "val_actual_ratio": val_actual_ratio,
        "group_by": settings.group_by,
        "fallback_group_by": settings.fallback_group_by,
        "valid_only": settings.valid_only,
        "shuffle": settings.shuffle,
        "total_rows_read": len(rows),
        "total_rows_split": total,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "total_groups": len(grouped),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
    }

    if settings.summary_path is not None:
        write_split_summary(summary, settings.summary_path)
    return summary


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """CLI entrypoint for split generation.

    Args:
        None

    Returns:
        None

    Important notes:
        - Reads `split.*` settings from `configs/train.yaml`.
        - Logs summary statistics and output artifact paths.
    """
    _configure_logging()
    settings = load_split_settings(force_reload=True)
    summary = run_split(settings)
    LOGGER.info(
        "Split done: train_rows=%d val_rows=%d train_groups=%d val_groups=%d val_ratio=%.6f",
        summary["train_rows"],
        summary["val_rows"],
        summary["train_groups"],
        summary["val_groups"],
        summary["val_actual_ratio"],
    )
    LOGGER.info("Train CSV: %s", settings.train_path)
    LOGGER.info("Val CSV: %s", settings.val_path)
    if settings.summary_path is not None:
        LOGGER.info("Summary JSON: %s", settings.summary_path)


if __name__ == "__main__":
    main()

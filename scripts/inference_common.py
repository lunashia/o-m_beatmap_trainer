"""Shared inference/evaluation utilities for cache-backed sample loading.

Purpose:
    Provide a single reusable implementation for:
      - resolving split sample ids
      - collecting raw `(x, y)` pairs from cache index + shards
      - validating records via the training dataset adapter
      - loading model state dict from flexible checkpoint layouts
      - normalizing paths/device handling

Input / Output format:
    Upstream input:
      - split CSV rows with `sample_id`
      - cache index rows (`sample_id`, `shard_path`, `valid`, ...)
      - cache shard records:
          {
            "sample_id": str,
            "valid": bool,
            "payload": {"samples": [(x, y), ...]}
          }
    Output:
      - `CacheSampleRecord` list with metadata + raw sample pair
      - filtered valid `CacheSampleRecord` list
      - normalized checkpoint state dict for model loading

Pipeline fit:
    - Upstream producers:
      `scripts/split_dataset.py`, `scripts/build_train_cache.py`, training checkpoints.
    - Downstream consumers:
      `scripts/eval_minimal.py` and `scripts/infer_minimal.py`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from scripts.train_minimal import (
    NextEventDataset,
    _is_valid_row,
    _load_shard_records,
    _read_cache_index,
    _resolve_path,
)
from vocab.delta_tick_vocab import DeltaTickVocab


@dataclass(frozen=True)
class CacheSampleRecord:
    """One cache sample pair with source metadata.

    Args:
        sample_id: Logical sample id from manifest/split.
        input_source: Canonical shard path that provided this record.
        sample_index_in_record: Index of `(x, y)` inside record payload samples.
        x: Raw model input dict from cache payload.
        y: Raw target dict from cache payload.

    Returns:
        Immutable metadata container used by eval/infer stages.

    Important notes:
        - `x` is expected to include `past_events`, `grid_stats`, `audio_window`.
        - `y` is expected to include `delta_tick`, `lane_mask`.
        - Additional optional encoded fields are tolerated.
    """

    sample_id: str
    input_source: str
    sample_index_in_record: int
    x: Dict[str, Any]
    y: Dict[str, Any]
    # Interface note:
    # - x should contain raw `past_events` and may also contain encoded
    #   `past_delta_id` / `past_lane_mask_id`.
    # - y should contain raw `delta_tick` + `lane_mask`; `delta_tick_id` is optional.


def canonical_path(path: Path, project_root: Path) -> str:
    """Convert an absolute path to project-relative canonical text when possible.

    Args:
        path: Absolute or relative filesystem path.
        project_root: Project root used for relative conversion.

    Returns:
        `./...` style relative path if inside project root; absolute path otherwise.

    Important notes:
        - Uses `.resolve()` first, so symlinks/relative segments are normalized.
    """
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(project_root.resolve())
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(resolved)


def read_split_sample_ids(split_path: Path) -> set[str]:
    """Load unique `sample_id` values from a split/index CSV.

    Args:
        split_path: CSV path containing a `sample_id` column.

    Returns:
        Set of non-empty `sample_id` values.

    Important notes:
        - Raises if file is missing or contains no usable `sample_id`.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        sample_ids: set[str] = set()
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                sample_ids.add(sample_id)
    if not sample_ids:
        raise ValueError(f"No sample_id found in split/index file: {split_path}")
    return sample_ids


def collect_cache_samples_for_ids(
    cache_index_path: Path,
    project_root: Path,
    sample_ids: set[str],
) -> Tuple[List[CacheSampleRecord], int]:
    """Collect raw cache samples for a selected set of sample ids.

    Args:
        cache_index_path: Path to cache index CSV.
        project_root: Root path used to resolve relative shard paths.
        sample_ids: Selected sample ids to collect.

    Returns:
        Tuple:
          - list of `CacheSampleRecord`
          - count of failed/unresolved sample ids

    Important notes:
        - Index rows with `valid=false`, missing shard path, or unresolved records
          are counted as failures.
        - Uses a per-sample first-seen policy across shard records to avoid
          duplicate sample records when index/shards are inconsistent.
    """
    index_rows = _read_cache_index(cache_index_path)
    if not index_rows:
        raise ValueError(f"Cache index is empty or missing: {cache_index_path}")

    ids_with_any_row: set[str] = set()
    failure_sample_ids: set[str] = set()
    shard_to_sample_ids: Dict[str, set[str]] = {}
    shard_path_lookup: Dict[str, Path] = {}

    for row in index_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id not in sample_ids:
            continue
        ids_with_any_row.add(sample_id)

        if not _is_valid_row(row):
            failure_sample_ids.add(sample_id)
            continue
        shard_path = _resolve_path(row.get("shard_path"), project_root)
        if shard_path is None:
            failure_sample_ids.add(sample_id)
            continue

        # Lower-cased resolved path is used as stable dedupe key on Windows.
        key = str(shard_path.resolve()).lower()
        shard_path_lookup[key] = shard_path
        shard_to_sample_ids.setdefault(key, set()).add(sample_id)

    missing_ids = sample_ids - ids_with_any_row
    failure_sample_ids.update(missing_ids)

    all_records: List[CacheSampleRecord] = []
    seen_record_ids: set[str] = set()

    for shard_key in sorted(shard_to_sample_ids.keys()):
        shard_path = shard_path_lookup[shard_key]
        allowed_ids = shard_to_sample_ids[shard_key]
        for record in _load_shard_records(shard_path):
            sample_id = str(record.get("sample_id", "")).strip()
            if not sample_id or sample_id not in allowed_ids:
                continue
            if sample_id in seen_record_ids:
                continue
            seen_record_ids.add(sample_id)

            if not _is_valid_row(record):
                failure_sample_ids.add(sample_id)
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                failure_sample_ids.add(sample_id)
                continue
            raw_samples = payload.get("samples")
            if not isinstance(raw_samples, (list, tuple)):
                failure_sample_ids.add(sample_id)
                continue

            for sample_idx, sample in enumerate(raw_samples):
                if not (isinstance(sample, (list, tuple)) and len(sample) >= 2):
                    continue
                all_records.append(
                    CacheSampleRecord(
                        sample_id=sample_id,
                        input_source=canonical_path(shard_path, project_root),
                        sample_index_in_record=sample_idx,
                        x=sample[0],
                        y=sample[1],
                    )
                )

    unresolved = (sample_ids - failure_sample_ids) - seen_record_ids
    failure_sample_ids.update(unresolved)
    return all_records, len(failure_sample_ids)


def filter_valid_cache_sample_records(
    records: Sequence[CacheSampleRecord],
    vocab: DeltaTickVocab,
) -> Tuple[List[CacheSampleRecord], int]:
    """Filter cache records by probing dataset conversion compatibility.

    Args:
        records: Raw cache sample records to validate.
        vocab: Frozen delta-tick vocab used by dataset encoding.

    Returns:
        Tuple:
          - valid records that can be converted by `NextEventDataset`
          - count of failed pairs

    Important notes:
        - Validation is performed by invoking `NextEventDataset.__getitem__`,
          ensuring parity with train/eval/infer tensorization assumptions.
    """
    raw_pairs = [(record.x, record.y) for record in records]
    probe_dataset = NextEventDataset(raw_pairs, vocab)
    valid_records: List[CacheSampleRecord] = []
    failed_pairs = 0
    for idx in range(len(probe_dataset)):
        try:
            _ = probe_dataset[idx]
        except Exception:  # noqa: BLE001
            failed_pairs += 1
            continue
        valid_records.append(records[idx])
    return valid_records, failed_pairs


def resolve_device(device_value: str) -> torch.device:
    """Resolve runtime device with CUDA fallback to CPU.

    Args:
        device_value: Requested device string, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`.

    Returns:
        A `torch.device` that is valid in current runtime.

    Important notes:
        - Requests starting with `cuda` fall back to CPU if CUDA is unavailable.
    """
    requested = device_value.strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_value)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract model state dict from supported checkpoint layouts.

    Args:
        checkpoint: Loaded checkpoint object (usually `torch.load(...)` output).

    Returns:
        Model state dict mapping parameter names to tensors.

    Important notes:
        - Supports:
          1) `{"model_state_dict": {...}}`
          2) `{"state_dict": {...}}`
          3) plain tensor dict state dict
        - Raises for unsupported structures.
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected state_dict or model_state_dict.")


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common distributed-training prefixes in checkpoint keys.

    Args:
        state_dict: Raw model state dict.

    Returns:
        State dict where leading `module.` is stripped when present.

    Important notes:
        - Keeps keys unchanged when they do not start with `module.`.
    """
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[7:]] = value
        else:
            normalized[key] = value
    return normalized

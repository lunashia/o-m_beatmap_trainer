from __future__ import annotations

"""Data loading utilities for training and validation from offline cache.

Purpose:
    Convert split sample IDs and cache shards into validated PyTorch DataLoaders
    for train/val phases, while reporting sample counts and cache/filter failures.

Input / Output format:
    Input:
        - Parsed training config dictionary.
        - Frozen `DeltaTickVocab` used by `NextEventDataset`.
        - Split CSV files containing `sample_id`.
        - Cache index file and shard files.
    Output:
        - `TrainDataSettings` with resolved paths and loader settings.
        - `DataBundle` containing train/val DataLoaders and summary counts.

Pipeline fit:
    Upstream input:
        - Split files from `scripts/split_dataset.py`.
        - Cache index/shards from `scripts/build_train_cache.py`.
    Downstream output:
        - `DataBundle` consumed by `train.py` runtime loop.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from scripts.inference_common import (
    collect_cache_samples_for_ids,
    filter_valid_cache_sample_records,
    read_split_sample_ids,
)
from scripts.train_minimal import NextEventDataset
from scripts.train_minimal import _resolve_path as _resolve_training_path
from vocab.delta_tick_vocab import DeltaTickVocab


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainDataSettings:
    """Resolved data-related training settings.

    Fields:
        project_root: Project root used for relative-path resolution.
        cache_index_path: Path to cache index CSV.
        train_split_path: Path to train split CSV.
        val_split_path: Path to validation split CSV.
        batch_size: Batch size for both train and val loaders.
        num_workers: DataLoader worker count.
        pin_memory: DataLoader pin-memory flag.
        persistent_workers: DataLoader persistent worker flag.
        prefetch_factor: Optional prefetch factor (valid when workers > 0).
    """

    project_root: Path
    cache_index_path: Path
    train_split_path: Path
    val_split_path: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None


@dataclass(frozen=True)
class DataBundle:
    """Prepared train/val data objects and counters.

    Fields:
        train_loader: Shuffled DataLoader for training.
        val_loader: Non-shuffled DataLoader for validation.
        train_sample_count: Number of validated train samples.
        val_sample_count: Number of validated val samples.
        train_failed_sample_count: Count of missing/invalid train samples.
        val_failed_sample_count: Count of missing/invalid val samples.
        grid_stats_dim: Feature dimension inferred from first train sample.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    train_sample_count: int
    val_sample_count: int
    train_failed_sample_count: int
    val_failed_sample_count: int
    grid_stats_dim: int


def _as_bool(raw: Any, default: bool = False) -> bool:
    """Normalize a loosely-typed value into bool.

    Args:
        raw: Candidate value from config.
        default: Fallback value when parsing fails.

    Returns:
        Parsed boolean value.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(raw: Any, default: int) -> int:
    """Normalize a loosely-typed value into int.

    Args:
        raw: Candidate value from config.
        default: Fallback integer when conversion fails.

    Returns:
        Parsed integer value.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def load_train_data_settings(config: Dict[str, Any]) -> TrainDataSettings:
    """Parse and validate data-loading settings from global config.

    Args:
        config: Full train config mapping loaded from `configs/train.yaml`.

    Returns:
        `TrainDataSettings` with resolved absolute paths and normalized loader values.

    Important notes:
        - `split.train_path`, `split.val_path`, and `cache.cache_index_path` are required.
        - `prefetch_factor` is optional and validated only when provided.
    """
    cache_cfg = config.get("cache", {}) if isinstance(config, dict) else {}
    split_cfg = config.get("split", {}) if isinstance(config, dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    dataloader_cfg = train_cfg.get("dataloader", {}) if isinstance(train_cfg, dict) else {}

    cache_index_path = _resolve_training_path(
        cache_cfg.get("cache_index_path", "./data/cache/train_cache/cache_index.csv"),
        PROJECT_ROOT,
    )
    train_split_path = _resolve_training_path(split_cfg.get("train_path"), PROJECT_ROOT)
    val_split_path = _resolve_training_path(split_cfg.get("val_path"), PROJECT_ROOT)
    if cache_index_path is None:
        raise ValueError("cache.cache_index_path must be configured.")
    if train_split_path is None:
        raise ValueError("split.train_path must be configured.")
    if val_split_path is None:
        raise ValueError("split.val_path must be configured.")

    batch_size = _as_int(train_cfg.get("batch_size", 16), 16)
    num_workers = _as_int(train_cfg.get("num_workers", 0), 0)
    if batch_size <= 0:
        raise ValueError(f"train.batch_size must be positive, got {batch_size!r}")
    if num_workers < 0:
        raise ValueError(f"train.num_workers must be >=0, got {num_workers!r}")

    prefetch_factor_raw = dataloader_cfg.get("prefetch_factor", None)
    prefetch_factor = None
    if prefetch_factor_raw is not None:
        prefetch_factor = _as_int(prefetch_factor_raw, 2)
        if prefetch_factor <= 0:
            raise ValueError("train.dataloader.prefetch_factor must be > 0 when set.")

    return TrainDataSettings(
        project_root=PROJECT_ROOT,
        cache_index_path=cache_index_path,
        train_split_path=train_split_path,
        val_split_path=val_split_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_as_bool(dataloader_cfg.get("pin_memory", True), True),
        persistent_workers=_as_bool(dataloader_cfg.get("persistent_workers", False), False),
        prefetch_factor=prefetch_factor,
    )


def _build_loader(
    *,
    records: list[Any],
    vocab: DeltaTickVocab,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    """Build a `DataLoader` from validated cache records.

    Args:
        records: Sequence of cache sample records exposing `.x` and `.y`.
        vocab: Frozen delta-tick vocab used by dataset encoding.
        batch_size: DataLoader batch size.
        shuffle: Whether to shuffle records.
        num_workers: Worker process count.
        pin_memory: DataLoader pin-memory flag.
        persistent_workers: DataLoader persistent-worker flag.
        prefetch_factor: Optional prefetch factor when workers > 0.

    Returns:
        Constructed `DataLoader` for `NextEventDataset`.

    Important notes:
        - `prefetch_factor` and `persistent_workers` are applied only when
          `num_workers > 0`, matching PyTorch constraints.
    """
    sample_pairs = [(record.x, record.y) for record in records]
    dataset = NextEventDataset(sample_pairs, vocab)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def build_train_val_loaders(
    settings: TrainDataSettings,
    vocab: DeltaTickVocab,
) -> DataBundle:
    """Build validated train/val DataLoaders from split IDs and cache.

    Args:
        settings: Resolved data-loading settings.
        vocab: Frozen delta-tick vocab used by dataset encoding.

    Returns:
        `DataBundle` containing loaders, sample counts, failure counts,
        and inferred `grid_stats_dim`.

    Important notes:
        - Raises when no valid samples remain for either split.
        - Uses the same validation filter as eval/infer for interface consistency.
    """
    train_ids = read_split_sample_ids(settings.train_split_path)
    val_ids = read_split_sample_ids(settings.val_split_path)

    train_records_raw, train_missing = collect_cache_samples_for_ids(
        settings.cache_index_path,
        settings.project_root,
        train_ids,
    )
    val_records_raw, val_missing = collect_cache_samples_for_ids(
        settings.cache_index_path,
        settings.project_root,
        val_ids,
    )
    train_records, train_failed_pairs = filter_valid_cache_sample_records(train_records_raw, vocab)
    val_records, val_failed_pairs = filter_valid_cache_sample_records(val_records_raw, vocab)

    train_failed_sample_count = train_missing + train_failed_pairs
    val_failed_sample_count = val_missing + val_failed_pairs
    if not train_records:
        raise ValueError("No valid training samples found after cache filtering.")
    if not val_records:
        raise ValueError("No valid validation samples found after cache filtering.")

    train_loader = _build_loader(
        records=train_records,
        vocab=vocab,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers,
        prefetch_factor=settings.prefetch_factor,
    )
    val_loader = _build_loader(
        records=val_records,
        vocab=vocab,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers,
        prefetch_factor=settings.prefetch_factor,
    )

    # Infer model input dimension from one encoded sample to avoid config drift.
    first_x, _ = train_loader.dataset[0]
    grid_stats_dim = int(first_x["grid_stats"].shape[0])

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sample_count=len(train_loader.dataset),
        val_sample_count=len(val_loader.dataset),
        train_failed_sample_count=train_failed_sample_count,
        val_failed_sample_count=val_failed_sample_count,
        grid_stats_dim=grid_stats_dim,
    )

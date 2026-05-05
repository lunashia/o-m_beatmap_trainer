"""Minimal validation entry for cache-backed next-event prediction.

Purpose:
    Run offline validation for a trained `NextEventPredictor` checkpoint using
    val split sample ids and cached `(x, y)` records, then report aggregate loss.

Input / Output format:
    Upstream input:
      - config `eval.*`, `split.val_path`, `cache.cache_index_path`, `vocab.path`
      - val split CSV with `sample_id`
      - cache shards containing `payload.samples = [(x, y), ...]`
      - checkpoint with model state dict and `vocab_path`
    Output:
      - metrics JSON with `val_loss`, `sample_count`, `step_count`,
        `failed_sample_count`, and provenance paths
      - optional per-sample prediction CSV

Pipeline fit:
    - Upstream producers:
      `scripts/split_dataset.py`, `scripts/build_train_cache.py`, training script checkpoints.
    - Downstream consumers:
      experiment tracking, model selection, and later inference runs.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("torch is required for validation. Install PyTorch first.") from exc

from config_loader import load_train_config
from model.next_event_model import NextEventPredictor, compute_next_event_loss
from scripts.inference_common import (
    canonical_path,
    collect_cache_samples_for_ids,
    extract_state_dict,
    filter_valid_cache_sample_records,
    normalize_state_dict_keys,
    read_split_sample_ids,
    resolve_device,
)
from scripts.train_minimal import NextEventDataset, _resolve_path
from vocab.delta_tick_vocab import load_vocab_from_config

LOGGER = logging.getLogger("eval_minimal")


@dataclass(frozen=True)
class EvalSettings:
    """Configuration needed by validation.

    Args:
        project_root: Absolute project root path.
        checkpoint_path: Trained checkpoint file.
        val_split_path: Validation split CSV path.
        cache_index_path: Cache index CSV path.
        batch_size: Evaluation batch size.
        num_workers: DataLoader worker count.
        device: Requested torch device string.
        output_dir: Directory for metrics/prediction outputs.
        save_predictions: Whether to emit per-sample prediction CSV.
        predictions_filename: Output CSV filename when enabled.
        metrics_filename: Output metrics JSON filename.

    Returns:
        Immutable settings object consumed by `run_validation`.

    Important notes:
        - Paths are resolved before use and validated in `load_eval_settings`.
    """

    project_root: Path
    checkpoint_path: Path
    val_split_path: Path
    cache_index_path: Path
    batch_size: int
    num_workers: int
    device: str
    output_dir: Path
    save_predictions: bool
    predictions_filename: str
    metrics_filename: str


def _as_bool(raw: Any, default: bool = False) -> bool:
    """Parse a loosely-typed bool-like value.

    Args:
        raw: Source value from config.
        default: Fallback when parsing fails.

    Returns:
        Parsed boolean value.

    Important notes:
        - Accepts common truthy/falsey strings.
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
    """Parse an integer with fallback.

    Args:
        raw: Source value from config.
        default: Fallback when parsing fails.

    Returns:
        Integer value.

    Important notes:
        - Bool is coerced to `0/1` for consistency with existing config parsers.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def load_eval_settings(force_reload: bool = True) -> EvalSettings:
    """Load and validate evaluation settings from `configs/train.yaml`.

    Args:
        force_reload: Whether to bypass cached config and reload from disk.

    Returns:
        Validated `EvalSettings`.

    Important notes:
        - Supports fallback wiring:
          `eval.val_split_path -> split.val_path`,
          `eval.cache_index_path -> cache.cache_index_path`,
          `eval.output_dir -> data.output_dir`.
        - Raises on missing required paths or invalid numeric bounds.
    """
    config = load_train_config(force_reload=force_reload)
    if not isinstance(config, dict):
        raise ValueError("Invalid train config.")

    eval_cfg = config.get("eval", {})
    split_cfg = config.get("split", {})
    cache_cfg = config.get("cache", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    if not isinstance(eval_cfg, dict):
        raise ValueError("eval config must be a mapping.")

    checkpoint_path = _resolve_path(eval_cfg.get("checkpoint_path"), PROJECT_ROOT)
    if checkpoint_path is None:
        raise ValueError("eval.checkpoint_path must be configured.")

    val_split_path = _resolve_path(
        eval_cfg.get("val_split_path", split_cfg.get("val_path")),
        PROJECT_ROOT,
    )
    if val_split_path is None:
        raise ValueError("eval.val_split_path or split.val_path must be configured.")

    cache_index_path = _resolve_path(
        eval_cfg.get("cache_index_path", cache_cfg.get("cache_index_path")),
        PROJECT_ROOT,
    )
    if cache_index_path is None:
        raise ValueError("eval.cache_index_path or cache.cache_index_path must be configured.")

    output_dir = _resolve_path(
        eval_cfg.get("output_dir", data_cfg.get("output_dir")),
        PROJECT_ROOT,
    )
    if output_dir is None:
        raise ValueError("eval.output_dir or data.output_dir must be configured.")

    batch_size = _as_int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 16)), 16)
    num_workers = _as_int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 0)), 0)
    if batch_size <= 0:
        raise ValueError(f"eval.batch_size must be positive, got {batch_size!r}")
    if num_workers < 0:
        raise ValueError(f"eval.num_workers must be >= 0, got {num_workers!r}")

    device = str(eval_cfg.get("device", train_cfg.get("device", "cuda"))).strip() or "cuda"
    predictions_filename = str(eval_cfg.get("predictions_filename", "val_predictions.csv")).strip()
    metrics_filename = str(eval_cfg.get("metrics_filename", "val_metrics.json")).strip()
    if not predictions_filename:
        raise ValueError("eval.predictions_filename must not be empty.")
    if not metrics_filename:
        raise ValueError("eval.metrics_filename must not be empty.")

    return EvalSettings(
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint_path,
        val_split_path=val_split_path,
        cache_index_path=cache_index_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        output_dir=output_dir,
        save_predictions=_as_bool(eval_cfg.get("save_predictions", False), False),
        predictions_filename=predictions_filename,
        metrics_filename=metrics_filename,
    )


def _write_predictions(
    path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:
    """Write per-sample validation predictions to CSV.

    Args:
        path: Target CSV file path.
        rows: Iterable of prediction dictionaries.

    Returns:
        None.

    Important notes:
        - Parent directories are created automatically.
        - Schema is fixed for downstream reproducibility.
    """
    fieldnames = [
        "step",
        "batch_index",
        "pred_delta_id",
        "pred_delta_value",
        "target_delta_id",
        "target_delta_value",
        "pred_lane_mask",
        "target_lane_mask",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(settings: EvalSettings) -> Dict[str, Any]:
    """Execute validation over the selected split and cache.

    Args:
        settings: Validated evaluation settings.

    Returns:
        Metrics dict containing loss, counts, provenance paths, and output paths.

    Important notes:
        - Uses frozen vocab from config and enforces checkpoint vocab path match.
        - Does not update model parameters (`eval()` + `torch.no_grad()`).
        - `failed_sample_count` includes unresolved sample ids and sample-pair
          conversion failures.
    """
    val_sample_ids = read_split_sample_ids(settings.val_split_path)
    config = load_train_config(force_reload=False)
    chart_cfg = config.get("chart", {}) if isinstance(config, dict) else {}
    num_lanes = int(chart_cfg.get("num_lanes", 7))
    vocab = load_vocab_from_config(force_reload=True)
    vocab_path_cfg = config.get("vocab", {}).get("path") if isinstance(config, dict) else None
    vocab_path_resolved = _resolve_path(vocab_path_cfg, settings.project_root)
    raw_records, failed_sample_count = collect_cache_samples_for_ids(
        settings.cache_index_path,
        settings.project_root,
        val_sample_ids,
    )
    filtered_records, failed_pairs = filter_valid_cache_sample_records(raw_records, vocab)
    total_failures = failed_sample_count + failed_pairs

    if not filtered_records:
        raise ValueError("No valid validation samples after filtering cache records.")

    filtered_samples = [(record.x, record.y) for record in filtered_records]
    dataset = NextEventDataset(filtered_samples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    first_x, _ = dataset[0]
    grid_stats_dim = int(first_x["grid_stats"].shape[0])

    device = resolve_device(settings.device)
    model = NextEventPredictor(
        delta_vocab_size=vocab.num_tokens,
        grid_stats_dim=grid_stats_dim,
    ).to(device)

    if not settings.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint_path}")
    checkpoint = torch.load(str(settings.checkpoint_path), map_location=device, weights_only=False)
    checkpoint_vocab_path_raw = checkpoint.get("vocab_path")
    checkpoint_vocab_path = _resolve_path(checkpoint_vocab_path_raw, settings.project_root)
    if checkpoint_vocab_path is None:
        raise ValueError("Checkpoint is missing required vocab_path.")
    if vocab_path_resolved is None:
        raise ValueError("Config vocab.path must be configured.")
    if checkpoint_vocab_path.resolve() != vocab_path_resolved.resolve():
        raise ValueError(
            "Vocab path mismatch between config and checkpoint: "
            f"config={vocab_path_resolved} checkpoint={checkpoint_vocab_path}"
        )
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    total_loss = 0.0
    sample_count = 0
    step_count = 0
    prediction_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for step_count, (x, y) in enumerate(loader, start=1):
            x = {k: v.to(device) for k, v in x.items()}
            target_delta = y["delta_tick"].to(device)
            target_lane = y["lane_mask"].to(device)

            logits = model(x)
            losses = compute_next_event_loss(logits, target_delta, target_lane)

            batch_size = int(target_delta.shape[0])
            # Keep a true sample-weighted mean across potentially uneven last batch.
            total_loss += float(losses["total_loss"].item()) * batch_size
            sample_count += batch_size

            if settings.save_predictions:
                pred_delta = logits["delta_tick_logits"].argmax(dim=-1).detach().cpu()
                pred_lane = logits["lane_mask_logits"].argmax(dim=-1).detach().cpu()
                target_delta_cpu = target_delta.detach().cpu()
                target_lane_cpu = target_lane.detach().cpu()
                for i in range(batch_size):
                    pred_delta_id = int(pred_delta[i].item())
                    target_delta_id = int(target_delta_cpu[i].item())
                    prediction_rows.append(
                        {
                            "step": step_count,
                            "batch_index": i,
                            "pred_delta_id": pred_delta_id,
                            "pred_delta_value": vocab.decode_delta_tick(pred_delta_id),
                            "target_delta_id": target_delta_id,
                            "target_delta_value": vocab.decode_delta_tick(target_delta_id),
                            "pred_lane_mask": int(pred_lane[i].item()),
                            "target_lane_mask": int(target_lane_cpu[i].item()),
                        }
                    )

    if sample_count <= 0:
        raise ValueError("Validation produced zero samples.")

    val_loss = total_loss / float(sample_count)
    metrics = {
        "checkpoint_path": canonical_path(settings.checkpoint_path, settings.project_root),
        "val_split_path": canonical_path(settings.val_split_path, settings.project_root),
        "cache_index_path": canonical_path(settings.cache_index_path, settings.project_root),
        "vocab_path": canonical_path(vocab_path_resolved, settings.project_root)
        if vocab_path_resolved is not None
        else "",
        "val_loss": val_loss,
        "sample_count": sample_count,
        "step_count": step_count,
        "failed_sample_count": total_failures,
        "num_lanes": num_lanes,
        "n_mels": int(first_x["audio_window"].shape[0]),
    }

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.output_dir / settings.metrics_filename
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if settings.save_predictions:
        predictions_path = settings.output_dir / settings.predictions_filename
        _write_predictions(predictions_path, prediction_rows)
        metrics["predictions_path"] = canonical_path(predictions_path, settings.project_root)

    metrics["metrics_path"] = canonical_path(metrics_path, settings.project_root)
    return metrics


def _configure_logging() -> None:
    """Configure script-level logging defaults.

    Args:
        None.

    Returns:
        None.

    Important notes:
        - Uses INFO level and timestamped single-line format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """CLI entrypoint for validation.

    Args:
        None.

    Returns:
        None.

    Important notes:
        - Reads all runtime parameters from `configs/train.yaml`.
    """
    _configure_logging()
    settings = load_eval_settings(force_reload=True)
    metrics = run_validation(settings)
    LOGGER.info(
        "Validation done: val_loss=%.6f samples=%d steps=%d failed_samples=%d",
        metrics["val_loss"],
        metrics["sample_count"],
        metrics["step_count"],
        metrics["failed_sample_count"],
    )
    LOGGER.info("Metrics JSON: %s", settings.output_dir / settings.metrics_filename)
    if settings.save_predictions:
        LOGGER.info("Predictions CSV: %s", settings.output_dir / settings.predictions_filename)


if __name__ == "__main__":
    main()

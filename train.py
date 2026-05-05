from __future__ import annotations

"""Training entrypoint and orchestration for next-event model learning.

Purpose:
    Provide the top-level training workflow for the osu!mania 7k next-event task,
    including config loading, artifact snapshotting, data loader construction,
    training/validation epoch loop, checkpointing, and run summary emission.

Input / Output format:
    Input:
        - Runtime config loaded from `configs/train.yaml` via `load_train_config`.
        - Frozen vocab JSON at `vocab.path`.
        - Train/val sample-id split CSVs and cache index/shards.
    Output:
        - Checkpoints: `best.pt`, `last.pt`
        - Logs: `train_log.jsonl`
        - Summary: `metrics.json`
        - Snapshots: `config_snapshot.yaml`, `vocab.json`
        - Return value of `run_training`: metrics summary dictionary.

Pipeline fit:
    Upstream input:
        - Cache and split files prepared by `build_manifest.py`, `scripts/build_train_cache.py`,
          and `scripts/split_dataset.py`.
        - Vocab prepared beforehand and frozen by vocab pipeline.
    Downstream output:
        - Artifacts consumed by evaluation/inference and by resume runs.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("torch is required for training. Install PyTorch first.") from exc

from config_loader import CONFIG_PATH, load_train_config
from model.next_event_model import NextEventPredictor
from scripts.inference_common import canonical_path, resolve_device
from scripts.train_minimal import _resolve_path
from train_checkpoint import (
    build_checkpoint_payload,
    load_checkpoint,
    resolve_checkpoint_paths,
    restore_training_state,
    save_checkpoint,
)
from train_data import build_train_val_loaders, load_train_data_settings
from train_logging import (
    append_jsonl,
    now_utc_iso,
    resolve_train_artifacts,
    write_config_snapshot,
    write_metrics,
    write_vocab_snapshot,
)
from train_runtime import build_scheduler, parse_runtime_settings, run_train_epoch, run_validation_epoch, set_seed
from vocab.delta_tick_vocab import load_vocab, load_vocab_from_config, load_vocab_settings

LOGGER = logging.getLogger("train")
PROJECT_ROOT = Path(__file__).resolve().parent


def _as_bool(raw: Any, default: bool = False) -> bool:
    """Normalize a loosely-typed value into bool.

    Args:
        raw: Candidate value from config or CLI-adjacent sources.
        default: Fallback value when parsing is not possible.

    Returns:
        Parsed boolean value.

    Important notes:
        - Accepts common textual booleans like "true"/"false" and "on"/"off".
        - Empty strings and unknown tokens return `default`.
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

    Important notes:
        - Booleans are converted via `int(bool)` for config compatibility.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _resolve_run_dir(config: Dict[str, Any]) -> Path:
    """Resolve the training run directory from config.

    Args:
        config: Full train config dictionary.

    Returns:
        Absolute run directory path.

    Important notes:
        - Defaults to `./artifacts/train` when `train.output.run_dir` is absent.
    """
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    output_cfg = train_cfg.get("output", {}) if isinstance(train_cfg, dict) else {}
    run_dir_raw = output_cfg.get("run_dir")
    if run_dir_raw is None:
        run_dir_raw = "./artifacts/train"
    run_dir = _resolve_path(run_dir_raw, PROJECT_ROOT)
    if run_dir is None:
        raise ValueError("train.output.run_dir must be configured.")
    return run_dir


def _build_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer for model training.

    Args:
        config: Full train config dictionary.
        model: Model whose parameters will be optimized.

    Returns:
        Configured `torch.optim.Optimizer` instance.

    Important notes:
        - Uses AdamW with `train.learning_rate` and `train.weight_decay`.
    """
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _build_scaler(runtime_amp_enabled: bool) -> Optional[torch.cuda.amp.GradScaler]:
    """Create GradScaler when AMP is active.

    Args:
        runtime_amp_enabled: Whether AMP is enabled for the current runtime.

    Returns:
        `GradScaler` when AMP is enabled; otherwise `None`.
    """
    if runtime_amp_enabled:
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def _copy_vocab_and_verify() -> Path:
    """Resolve and validate the frozen vocab path from config.

    Args:
        None.

    Returns:
        Absolute path to the source vocab JSON.

    Important notes:
        - Fails fast when vocab path is missing/nonexistent.
        - Calls `load_vocab` to ensure format is valid and frozen.
    """
    vocab_settings = load_vocab_settings(force_reload=False)
    vocab_source = vocab_settings.vocab_path
    if vocab_source is None:
        raise ValueError("vocab.path is required.")
    if not vocab_source.exists():
        raise FileNotFoundError(f"vocab.path not found: {vocab_source}")
    # Validate vocab schema before snapshot/copy.
    _ = load_vocab(vocab_source)
    return vocab_source


def _select_resume_path(config: Dict[str, Any], run_dir: Path) -> Optional[Path]:
    """Resolve resume checkpoint path according to `train.resume` config.

    Args:
        config: Full train config dictionary.
        run_dir: Resolved run directory.

    Returns:
        Path to checkpoint for resume, or `None` when resume is disabled.

    Important notes:
        - If resume is enabled and `train.resume.path` is empty,
          defaults to `<run_dir>/checkpoints/last.pt`.
    """
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    resume_cfg = train_cfg.get("resume", {}) if isinstance(train_cfg, dict) else {}
    enabled = _as_bool(resume_cfg.get("enabled", False), False)
    if not enabled:
        return None
    raw_path = resume_cfg.get("path")
    if raw_path is None or not str(raw_path).strip():
        return run_dir / "checkpoints" / "last.pt"
    return _resolve_path(raw_path, PROJECT_ROOT)


def run_training() -> Dict[str, Any]:
    """Run full training/validation workflow and emit train artifacts.

    Args:
        None. Configuration is read from `CONFIG_PATH`.

    Returns:
        Dict[str, Any]: Run summary including paths, counts, final losses,
        best metric, and resume metadata.

    Important notes:
        - Assumes vocab is pre-fitted and frozen; this function never re-fits vocab.
        - Validation runs at the end of every epoch.
        - Always writes `last.pt`; writes `best.pt` when `val_loss` improves.
    """
    config = load_train_config(force_reload=True)
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    seed = _as_int(train_cfg.get("seed", 42), 42)
    set_seed(seed)

    requested_device = str(train_cfg.get("device", "cuda")).strip() or "cuda"
    device = resolve_device(requested_device)
    if requested_device.lower().startswith("cuda") and device.type != "cuda":
        LOGGER.warning("CUDA requested but unavailable; fallback to CPU.")

    runtime = parse_runtime_settings(config, device)
    run_dir = _resolve_run_dir(config)
    artifacts = resolve_train_artifacts(run_dir)
    ckpt_paths = resolve_checkpoint_paths(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_config_snapshot(CONFIG_PATH, artifacts.config_snapshot_path)
    vocab_source = _copy_vocab_and_verify()
    write_vocab_snapshot(vocab_source, artifacts.vocab_snapshot_path)
    vocab = load_vocab_from_config(force_reload=False)

    data_settings = load_train_data_settings(config)
    data_bundle = build_train_val_loaders(data_settings, vocab)

    model = NextEventPredictor(
        delta_vocab_size=vocab.num_tokens,
        grid_stats_dim=data_bundle.grid_stats_dim,
    ).to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = build_scheduler(optimizer, config)
    scaler = _build_scaler(runtime.amp_enabled)

    resume_path = _select_resume_path(config, run_dir)
    resume_strict = _as_bool(train_cfg.get("resume", {}).get("strict", True), True)
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    best_step = 0
    resumed_from = ""
    if resume_path is not None and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, device)
        restored = restore_training_state(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            strict=resume_strict,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_metric = float(restored["best_metric"])
        best_step = int(restored["best_step"])
        resumed_from = str(resume_path)

    append_jsonl(
        artifacts.log_jsonl_path,
        {
            "event": "train_start",
            "time": now_utc_iso(),
            "seed": seed,
            "device": device.type,
            "train_samples": data_bundle.train_sample_count,
            "val_samples": data_bundle.val_sample_count,
            "train_failed_samples": data_bundle.train_failed_sample_count,
            "val_failed_samples": data_bundle.val_failed_sample_count,
            "run_dir": str(run_dir),
            "resumed_from": resumed_from,
        },
    )

    last_train_metrics: Dict[str, Any] = {}
    last_val_metrics: Dict[str, Any] = {}
    stop_requested = False
    epochs_ran = 0

    for epoch in range(start_epoch, runtime.max_epochs + 1):
        epochs_ran += 1
        train_metrics = run_train_epoch(
            model=model,
            loader=data_bundle.train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            runtime=runtime,
            start_global_step=global_step,
        )
        global_step = int(train_metrics["global_step"])
        val_metrics = run_validation_epoch(
            model=model,
            loader=data_bundle.val_loader,
            device=device,
        )
        current_metric = float(val_metrics["val_loss"])
        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric
            best_step = global_step

        train_status = {
            "train_loss": float(train_metrics["train_loss"]),
            "train_delta_loss": float(train_metrics["train_delta_loss"]),
            "train_lane_loss": float(train_metrics["train_lane_loss"]),
            "val_loss": float(val_metrics["val_loss"]),
            "val_delta_loss": float(val_metrics["val_delta_loss"]),
            "val_lane_loss": float(val_metrics["val_lane_loss"]),
            "lr": float(train_metrics["lr"]),
        }
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_step=best_step,
            config=config,
            vocab_path=str(vocab_source),
            train_status=train_status,
        )
        save_checkpoint(payload, ckpt_paths.last_path)
        if is_best:
            save_checkpoint(payload, ckpt_paths.best_path)

        epoch_log = {
            "event": "epoch_end",
            "time": now_utc_iso(),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_metrics["train_loss"]),
            "train_delta_loss": float(train_metrics["train_delta_loss"]),
            "train_lane_loss": float(train_metrics["train_lane_loss"]),
            "val_loss": float(val_metrics["val_loss"]),
            "val_delta_loss": float(val_metrics["val_delta_loss"]),
            "val_lane_loss": float(val_metrics["val_lane_loss"]),
            "train_samples": int(train_metrics["train_samples"]),
            "val_samples": int(val_metrics["val_samples"]),
            "lr": float(train_metrics["lr"]),
            "grad_norm": float(train_metrics["grad_norm"]),
            "best_metric": best_metric,
            "is_best": is_best,
        }
        append_jsonl(artifacts.log_jsonl_path, epoch_log)
        LOGGER.info(
            "epoch=%d step=%d train_loss=%.6f val_loss=%.6f best=%.6f",
            epoch,
            global_step,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            best_metric,
        )

        last_train_metrics = train_metrics
        last_val_metrics = val_metrics
        if bool(train_metrics.get("stop_requested", False)):
            stop_requested = True
            break

    summary = {
        "run_dir": canonical_path(run_dir, PROJECT_ROOT),
        "device": device.type,
        "seed": seed,
        "max_epochs": runtime.max_epochs,
        "epochs_ran": epochs_ran,
        "global_step": global_step,
        "stopped_by_max_steps": stop_requested,
        "best_metric": best_metric,
        "best_step": best_step,
        "best_checkpoint_path": canonical_path(ckpt_paths.best_path, PROJECT_ROOT),
        "last_checkpoint_path": canonical_path(ckpt_paths.last_path, PROJECT_ROOT),
        "metrics_path": canonical_path(artifacts.metrics_path, PROJECT_ROOT),
        "train_log_path": canonical_path(artifacts.log_jsonl_path, PROJECT_ROOT),
        "config_snapshot_path": canonical_path(artifacts.config_snapshot_path, PROJECT_ROOT),
        "vocab_snapshot_path": canonical_path(artifacts.vocab_snapshot_path, PROJECT_ROOT),
        "vocab_source_path": canonical_path(vocab_source, PROJECT_ROOT),
        "vocab_num_tokens": vocab.num_tokens,
        "train_sample_count": data_bundle.train_sample_count,
        "val_sample_count": data_bundle.val_sample_count,
        "train_failed_sample_count": data_bundle.train_failed_sample_count,
        "val_failed_sample_count": data_bundle.val_failed_sample_count,
        "last_train_loss": float(last_train_metrics.get("train_loss", float("nan"))),
        "last_val_loss": float(last_val_metrics.get("val_loss", float("nan"))),
        "resumed_from": resumed_from,
    }
    write_metrics(artifacts.metrics_path, summary)
    append_jsonl(
        artifacts.log_jsonl_path,
        {
            "event": "train_end",
            "time": now_utc_iso(),
            "best_metric": best_metric,
            "best_step": best_step,
            "global_step": global_step,
        },
    )
    return summary


def _configure_logging() -> None:
    """Configure process-wide logging for the train entrypoint.

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """CLI entry for training.

    Args:
        None. Uses command-line parser for compatibility and future options.

    Returns:
        None.

    Important notes:
        - This function delegates real work to `run_training`.
    """
    parser = argparse.ArgumentParser(description="Train next-event predictor.")
    _ = parser.parse_args()
    _configure_logging()
    summary = run_training()
    LOGGER.info("Training completed. best=%.6f step=%d", summary["best_metric"], summary["best_step"])


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()

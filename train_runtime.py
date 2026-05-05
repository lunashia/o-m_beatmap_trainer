from __future__ import annotations

"""Runtime training utilities: seed, schedule, train loop, and val loop.

Purpose:
    Hold low-level runtime logic for epoch-level train/validation execution,
    including AMP autocast, gradient accumulation, clipping, scheduler stepping,
    and scalar metric aggregation.

Input / Output format:
    Input:
        - Parsed config dict and resolved `torch.device`.
        - Model/optimizer/(optional) scheduler and scaler.
        - DataLoaders yielding `(x, y)` where `x` is model input dict and
          `y` contains `delta_tick` and `lane_mask` targets.
    Output:
        - `RuntimeSettings` for loop behavior.
        - Per-epoch metrics dictionaries for training and validation.

Pipeline fit:
    Upstream input:
        - Called by `train.py` after model/data construction.
    Downstream output:
        - Metrics consumed by checkpoint and logging modules.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from model.next_event_model import compute_next_event_loss


@dataclass(frozen=True)
class RuntimeSettings:
    """Normalized runtime controls for train loop execution.

    Fields:
        device: Resolved torch device used for compute.
        max_epochs: Total epochs to run.
        max_steps: Optional hard cap on global steps; 0 means disabled.
        grad_accum_steps: Number of steps to accumulate before optimizer step.
        log_every_steps: Retained for caller-level logging cadence.
        grad_clip_norm: Max-norm threshold; <=0 disables clipping.
        amp_enabled: Whether AMP is active.
        amp_dtype: AMP compute dtype (fp16 or bf16).
    """

    device: torch.device
    max_epochs: int
    max_steps: int
    grad_accum_steps: int
    log_every_steps: int
    grad_clip_norm: float
    amp_enabled: bool
    amp_dtype: torch.dtype


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible training.

    Args:
        seed: Non-negative integer seed value.

    Returns:
        None.

    Important notes:
        - Applies CUDA RNG seeding when CUDA is available.
        - Deterministic CUDA kernels are not forced here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_int(raw: Any, default: int) -> int:
    """Normalize a loosely-typed value into int.

    Args:
        raw: Candidate value.
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


def _as_float(raw: Any, default: float) -> float:
    """Normalize a loosely-typed value into float.

    Args:
        raw: Candidate value.
        default: Fallback float when conversion fails.

    Returns:
        Parsed float value.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _as_bool(raw: Any, default: bool = False) -> bool:
    """Normalize a loosely-typed value into bool.

    Args:
        raw: Candidate value.
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


def parse_runtime_settings(config: Dict[str, Any], device: torch.device) -> RuntimeSettings:
    """Parse runtime controls from config and validate constraints.

    Args:
        config: Full train config dictionary.
        device: Resolved torch device.

    Returns:
        `RuntimeSettings` ready for train loop execution.

    Important notes:
        - AMP is automatically disabled when resolved device is not CUDA.
        - `max_steps=0` means no global-step cap.
    """
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    amp_cfg = train_cfg.get("amp", {}) if isinstance(train_cfg, dict) else {}

    max_epochs = _as_int(train_cfg.get("max_epochs", train_cfg.get("epochs", 1)), 1)
    max_steps = _as_int(train_cfg.get("max_steps", 0), 0)
    grad_accum_steps = _as_int(train_cfg.get("grad_accum_steps", 1), 1)
    log_every_steps = _as_int(train_cfg.get("log_every_steps", 10), 10)
    grad_clip_norm = _as_float(train_cfg.get("grad_clip_norm", 1.0), 1.0)
    amp_enabled_cfg = _as_bool(amp_cfg.get("enabled", device.type == "cuda"), device.type == "cuda")
    amp_dtype_text = str(amp_cfg.get("dtype", "fp16")).strip().lower()
    amp_dtype = torch.float16 if amp_dtype_text == "fp16" else torch.bfloat16

    if max_epochs <= 0:
        raise ValueError(f"train.max_epochs must be positive, got {max_epochs!r}")
    if max_steps < 0:
        raise ValueError(f"train.max_steps must be >=0, got {max_steps!r}")
    if grad_accum_steps <= 0:
        raise ValueError(f"train.grad_accum_steps must be positive, got {grad_accum_steps!r}")
    if log_every_steps <= 0:
        raise ValueError(f"train.log_every_steps must be positive, got {log_every_steps!r}")
    if grad_clip_norm < 0:
        raise ValueError(f"train.grad_clip_norm must be >=0, got {grad_clip_norm!r}")

    return RuntimeSettings(
        device=device,
        max_epochs=max_epochs,
        max_steps=max_steps,
        grad_accum_steps=grad_accum_steps,
        log_every_steps=log_every_steps,
        grad_clip_norm=grad_clip_norm,
        amp_enabled=bool(amp_enabled_cfg and device.type == "cuda"),
        amp_dtype=amp_dtype,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build learning-rate scheduler from config.

    Args:
        optimizer: Optimizer instance whose LR will be scheduled.
        config: Full train config dictionary.

    Returns:
        Optional PyTorch LR scheduler; returns `None` when scheduler is disabled.

    Important notes:
        - Supports `none` and `cosine` currently.
        - `cosine` scheduler steps once per optimizer step in `run_train_epoch`.
    """
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    sched_cfg = train_cfg.get("scheduler", {}) if isinstance(train_cfg, dict) else {}
    name = str(sched_cfg.get("name", "none")).strip().lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        t_max = _as_int(sched_cfg.get("t_max", train_cfg.get("max_epochs", train_cfg.get("epochs", 1))), 1)
        min_lr = _as_float(sched_cfg.get("min_lr", 0.0), 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler name: {name!r}")


def run_validation_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run one full validation epoch.

    Args:
        model: Next-event model in eval mode.
        loader: Validation DataLoader yielding `(x, y)` batches.
        device: Compute device.

    Returns:
        Dictionary with averaged validation losses and sample count:
        `val_loss`, `val_delta_loss`, `val_lane_loss`, `val_samples`.

    Important notes:
        - Uses `torch.no_grad()`; no gradients or optimizer interaction.
        - Raises when loader yields zero samples.
    """
    model.eval()
    total_loss = 0.0
    total_delta = 0.0
    total_lane = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            target_delta = y["delta_tick"].to(device)
            target_lane = y["lane_mask"].to(device)
            logits = model(x)
            losses = compute_next_event_loss(logits, target_delta, target_lane)
            batch_size = int(target_delta.shape[0])
            total_samples += batch_size
            total_loss += float(losses["total_loss"].item()) * batch_size
            total_delta += float(losses["delta_loss"].item()) * batch_size
            total_lane += float(losses["lane_loss"].item()) * batch_size

    if total_samples <= 0:
        raise ValueError("Validation returned zero samples.")

    denom = float(total_samples)
    return {
        "val_loss": total_loss / denom,
        "val_delta_loss": total_delta / denom,
        "val_lane_loss": total_lane / denom,
        "val_samples": float(total_samples),
    }


def run_train_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    runtime: RuntimeSettings,
    start_global_step: int,
) -> Dict[str, Any]:
    """Run one training epoch with optional AMP and grad accumulation.

    Args:
        model: Next-event model to train.
        loader: Training DataLoader yielding `(x, y)` batches.
        optimizer: Optimizer used for parameter updates.
        scheduler: Optional scheduler stepped after each optimizer step.
        scaler: Optional AMP scaler.
        runtime: Parsed runtime controls.
        start_global_step: Global step value before this epoch.

    Returns:
        Dictionary with aggregated train metrics and loop status:
        `global_step`, `train_loss`, `train_delta_loss`, `train_lane_loss`,
        `train_samples`, `lr`, `grad_norm`, `stop_requested`.

    Important notes:
        - Accumulates gradients over `runtime.grad_accum_steps` mini-batches.
        - Applies gradient clipping only when `runtime.grad_clip_norm > 0`.
        - `stop_requested=True` indicates max-step stop condition reached.
    """
    model.train()
    total_loss = 0.0
    total_delta = 0.0
    total_lane = 0.0
    total_samples = 0
    step_in_epoch = 0
    global_step = start_global_step
    stop_requested = False

    optimizer.zero_grad(set_to_none=True)
    for x, y in loader:
        step_in_epoch += 1
        global_step += 1
        x = {k: v.to(runtime.device) for k, v in x.items()}
        target_delta = y["delta_tick"].to(runtime.device)
        target_lane = y["lane_mask"].to(runtime.device)

        autocast_enabled = bool(runtime.amp_enabled and runtime.device.type == "cuda")
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=runtime.amp_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(x)
            losses = compute_next_event_loss(logits, target_delta, target_lane)
            # Scale loss by accumulation factor so total gradient magnitude stays stable.
            loss_to_backprop = losses["total_loss"] / float(runtime.grad_accum_steps)

        if scaler is not None and autocast_enabled:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        should_step = (step_in_epoch % runtime.grad_accum_steps == 0) or (step_in_epoch == len(loader))
        grad_norm = math.nan
        if should_step:
            if scaler is not None and autocast_enabled:
                # Unscale before clipping to apply threshold in true parameter scale.
                scaler.unscale_(optimizer)
            if runtime.grad_clip_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=runtime.grad_clip_norm).item()
                )

            if scaler is not None and autocast_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        batch_size = int(target_delta.shape[0])
        total_samples += batch_size
        total_loss += float(losses["total_loss"].item()) * batch_size
        total_delta += float(losses["delta_loss"].item()) * batch_size
        total_lane += float(losses["lane_loss"].item()) * batch_size

        if runtime.max_steps > 0 and global_step >= runtime.max_steps:
            stop_requested = True
            break

    if total_samples <= 0:
        raise ValueError("Training epoch returned zero samples.")

    denom = float(total_samples)
    lr = float(optimizer.param_groups[0]["lr"])
    return {
        "global_step": global_step,
        "train_loss": total_loss / denom,
        "train_delta_loss": total_delta / denom,
        "train_lane_loss": total_lane / denom,
        "train_samples": total_samples,
        "lr": lr,
        "grad_norm": grad_norm,
        "stop_requested": stop_requested,
    }

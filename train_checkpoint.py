from __future__ import annotations

"""Checkpoint persistence and resume-state restoration helpers.

Purpose:
    Centralize checkpoint path resolution, payload construction, atomic saving,
    loading, and runtime-state restoration for training resume.

Input / Output format:
    Input:
        - Model/optimizer/(optional) scheduler/scaler states.
        - Epoch/step/best-metric bookkeeping values.
        - Optional RNG state.
    Output:
        - Serialized checkpoint `.pt` files.
        - Restored runtime state dictionary for resume:
          `epoch`, `global_step`, `best_metric`, `best_step`.

Pipeline fit:
    Upstream input:
        - Called by `train.py` during end-of-epoch save and startup resume.
    Downstream output:
        - Checkpoints consumed by future training resumes and eval/infer loaders.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from scripts.inference_common import extract_state_dict, normalize_state_dict_keys


@dataclass(frozen=True)
class CheckpointPaths:
    """Canonical checkpoint locations for one training run.

    Fields:
        checkpoint_dir: Parent directory for checkpoint files.
        best_path: Path for best-on-validation checkpoint.
        last_path: Path for latest checkpoint.
    """

    checkpoint_dir: Path
    best_path: Path
    last_path: Path


def resolve_checkpoint_paths(run_dir: Path) -> CheckpointPaths:
    """Resolve run-specific checkpoint paths.

    Args:
        run_dir: Root directory for current training run artifacts.

    Returns:
        `CheckpointPaths` with `checkpoints/best.pt` and `checkpoints/last.pt`.
    """
    checkpoint_dir = run_dir / "checkpoints"
    return CheckpointPaths(
        checkpoint_dir=checkpoint_dir,
        best_path=checkpoint_dir / "best.pt",
        last_path=checkpoint_dir / "last.pt",
    )


def _capture_rng_state() -> Dict[str, Any]:
    """Capture Python/NumPy/Torch RNG states for deterministic resume.

    Args:
        None.

    Returns:
        Mapping containing RNG state blobs.

    Important notes:
        - CUDA RNG states are included only when CUDA is available.
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore Python/NumPy/Torch RNG states.

    Args:
        state: RNG state mapping produced by `_capture_rng_state`.

    Returns:
        None.

    Important notes:
        - Missing keys are ignored to keep backward compatibility.
    """
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_metric: float,
    best_step: int,
    config: Dict[str, Any],
    vocab_path: str,
    train_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct full checkpoint payload for persistence.

    Args:
        model: Trained model instance.
        optimizer: Optimizer instance.
        scheduler: Optional LR scheduler instance.
        scaler: Optional AMP scaler instance.
        epoch: Current epoch index (1-based in caller).
        global_step: Global optimization step.
        best_metric: Best validation metric seen so far.
        best_step: Global step where best metric occurred.
        config: Effective training config dictionary.
        vocab_path: Path to vocab used for this run.
        train_status: Latest compact metrics/status dictionary.

    Returns:
        Serializable checkpoint payload dictionary.

    Important notes:
        - Scheduler/scaler state is included only when corresponding objects exist.
        - RNG state is embedded to support deterministic resume.
    """
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_step": best_step,
        "config": config,
        "vocab_path": vocab_path,
        "train_status": train_status,
        "rng_state": _capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def save_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    """Atomically persist a checkpoint payload.

    Args:
        payload: Checkpoint dictionary to serialize.
        path: Destination checkpoint path.

    Returns:
        None.

    Important notes:
        - Uses temp-file + replace to reduce risk of partial checkpoint files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint payload from disk.

    Args:
        path: Checkpoint file path.
        device: Target `map_location` for torch load.

    Returns:
        Checkpoint payload dictionary.

    Important notes:
        - Raises `FileNotFoundError` when path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(path), map_location=device, weights_only=False)


def restore_training_state(
    *,
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    strict: bool,
) -> Dict[str, Any]:
    """Restore model and runtime states from checkpoint for resume.

    Args:
        checkpoint: Loaded checkpoint payload dictionary.
        model: Model instance to load weights into.
        optimizer: Optimizer instance to restore state for.
        scheduler: Optional scheduler instance.
        scaler: Optional scaler instance.
        strict: Strict flag passed to `model.load_state_dict`.

    Returns:
        Dictionary with resume cursors:
        `epoch`, `global_step`, `best_metric`, `best_step`.

    Important notes:
        - Accepts both plain and `module.`-prefixed state-dict keys.
        - RNG state restoration is best-effort and key-optional.
    """
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=strict)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state)

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_metric": float(checkpoint.get("best_metric", float("inf"))),
        "best_step": int(checkpoint.get("best_step", 0)),
    }

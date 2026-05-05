"""Minimal inference entry for cache-backed next-event prediction.

Purpose:
    Run post-training forward inference on selected cached samples and export
    decoded prediction rows for downstream analysis or generation handoff.

Input / Output format:
    Upstream input:
      - config `infer.*` plus `vocab.path`
      - input split CSV with `sample_id`
      - cache shards with `payload.samples = [(x, y), ...]`
      - checkpoint with model state dict and `vocab_path`
    Output:
      - prediction CSV with per-sample decoded outputs
      - summary JSON with provenance, runtime options, and failure counts

Pipeline fit:
    - Upstream producers:
      training checkpoints, split builder, cache builder, frozen vocab builder.
    - Downstream consumers:
      offline inspection, error analysis, and generation post-processing stages.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("torch is required for inference. Install PyTorch first.") from exc

from config_loader import load_train_config
from model.next_event_model import NextEventPredictor
from scripts.inference_common import (
    CacheSampleRecord,
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

LOGGER = logging.getLogger("infer_minimal")


@dataclass(frozen=True)
class InferSettings:
    """Configuration required for inference execution.

    Args:
        project_root: Absolute project root path.
        checkpoint_path: Model checkpoint path.
        input_split_path: CSV path providing selected `sample_id`s.
        cache_index_path: Cache index CSV path.
        vocab_path: Frozen vocab JSON path expected by checkpoint.
        device: Requested torch device string.
        batch_size: Inference batch size.
        num_workers: DataLoader worker count.
        output_dir: Directory for inference outputs.
        output_filename: Prediction CSV filename.
        summary_filename: Summary JSON filename.
        top_k: Candidate count used when sampling is enabled.
        temperature: Logit scaling factor for sampling mode.
        sampling: Whether to sample delta class instead of argmax.
        max_steps: Optional upper bound on DataLoader steps (0 means unlimited).
        seed: Global random seed for reproducibility.

    Returns:
        Immutable settings object consumed by `run_inference`.

    Important notes:
        - `vocab_path` must match checkpoint metadata to avoid decode mismatch.
    """

    project_root: Path
    checkpoint_path: Path
    input_split_path: Path
    cache_index_path: Path
    vocab_path: Path
    device: str
    batch_size: int
    num_workers: int
    output_dir: Path
    output_filename: str
    summary_filename: str
    top_k: int
    temperature: float
    sampling: bool
    max_steps: int
    seed: int


def _as_bool(raw: Any, default: bool = False) -> bool:
    """Parse a loosely-typed bool-like value.

    Args:
        raw: Source config value.
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
        raw: Source config value.
        default: Fallback when parsing fails.

    Returns:
        Integer value.

    Important notes:
        - Bool is coerced to `0/1` to match project config parsing style.
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
    """Parse a float with fallback.

    Args:
        raw: Source config value.
        default: Fallback when parsing fails.

    Returns:
        Float value.

    Important notes:
        - Bool is coerced to `0.0/1.0` for compatibility.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_infer_settings(force_reload: bool = True) -> InferSettings:
    """Load and validate inference settings from `configs/train.yaml`.

    Args:
        force_reload: Whether to bypass cached config and reload from disk.

    Returns:
        Validated `InferSettings`.

    Important notes:
        - Supports fallback wiring from `eval`/`split`/`cache`/`train` sections.
        - Ensures numeric constraints for batching and sampling parameters.
    """
    config = load_train_config(force_reload=force_reload)
    if not isinstance(config, dict):
        raise ValueError("Invalid train config.")

    infer_cfg = config.get("infer", {})
    split_cfg = config.get("split", {})
    cache_cfg = config.get("cache", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    vocab_cfg = config.get("vocab", {})
    eval_cfg = config.get("eval", {})

    if not isinstance(infer_cfg, dict):
        raise ValueError("infer config must be a mapping.")

    checkpoint_path = _resolve_path(
        infer_cfg.get("checkpoint_path", eval_cfg.get("checkpoint_path")),
        PROJECT_ROOT,
    )
    if checkpoint_path is None:
        raise ValueError("infer.checkpoint_path must be configured.")

    input_split_path = _resolve_path(
        infer_cfg.get("input_split_path", split_cfg.get("val_path")),
        PROJECT_ROOT,
    )
    if input_split_path is None:
        raise ValueError("infer.input_split_path or split.val_path must be configured.")

    cache_index_path = _resolve_path(
        infer_cfg.get("cache_index_path", cache_cfg.get("cache_index_path")),
        PROJECT_ROOT,
    )
    if cache_index_path is None:
        raise ValueError("infer.cache_index_path or cache.cache_index_path must be configured.")

    vocab_path = _resolve_path(vocab_cfg.get("path"), PROJECT_ROOT)
    if vocab_path is None:
        raise ValueError("vocab.path must be configured.")

    output_dir = _resolve_path(
        infer_cfg.get("output_dir", data_cfg.get("output_dir")),
        PROJECT_ROOT,
    )
    if output_dir is None:
        raise ValueError("infer.output_dir or data.output_dir must be configured.")

    batch_size = _as_int(infer_cfg.get("batch_size", train_cfg.get("batch_size", 16)), 16)
    num_workers = _as_int(infer_cfg.get("num_workers", train_cfg.get("num_workers", 0)), 0)
    if batch_size <= 0:
        raise ValueError(f"infer.batch_size must be positive, got {batch_size!r}")
    if num_workers < 0:
        raise ValueError(f"infer.num_workers must be >= 0, got {num_workers!r}")

    top_k = _as_int(infer_cfg.get("top_k", 1), 1)
    if top_k <= 0:
        raise ValueError(f"infer.top_k must be > 0, got {top_k!r}")

    temperature = _as_float(infer_cfg.get("temperature", 1.0), 1.0)
    if temperature <= 0:
        raise ValueError(f"infer.temperature must be > 0, got {temperature!r}")

    max_steps = _as_int(infer_cfg.get("max_steps", 0), 0)
    if max_steps < 0:
        raise ValueError(f"infer.max_steps must be >= 0, got {max_steps!r}")

    seed = _as_int(infer_cfg.get("seed", train_cfg.get("seed", 42)), 42)
    device = str(infer_cfg.get("device", train_cfg.get("device", "cuda"))).strip() or "cuda"
    output_filename = str(infer_cfg.get("output_filename", "infer_predictions.csv")).strip()
    summary_filename = str(infer_cfg.get("summary_filename", "infer_summary.json")).strip()
    if not output_filename:
        raise ValueError("infer.output_filename must not be empty.")
    if not summary_filename:
        raise ValueError("infer.summary_filename must not be empty.")

    return InferSettings(
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint_path,
        input_split_path=input_split_path,
        cache_index_path=cache_index_path,
        vocab_path=vocab_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        output_filename=output_filename,
        summary_filename=summary_filename,
        top_k=top_k,
        temperature=temperature,
        sampling=_as_bool(infer_cfg.get("sampling", False), False),
        max_steps=max_steps,
        seed=seed,
    )


def _select_delta_ids(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    sampling: bool,
) -> torch.Tensor:
    """Select predicted delta class ids from logits.

    Args:
        logits: Delta-head logits tensor `[B, D]`.
        top_k: Candidate count for sampling mode.
        temperature: Logit scaling factor in sampling mode.
        sampling: If false uses argmax; if true samples from top-k candidates.

    Returns:
        Tensor of predicted class ids with shape `[B]`.

    Important notes:
        - `top_k` is clamped to `[1, D]`.
        - Sampling uses multinomial over softmaxed top-k logits.
    """
    if not sampling:
        return logits.argmax(dim=-1)

    scaled = logits / temperature
    vocab_size = int(scaled.shape[-1])
    k = min(max(1, top_k), vocab_size)
    top_values, top_indices = torch.topk(scaled, k=k, dim=-1)
    probs = torch.softmax(top_values, dim=-1)
    sampled_in_top = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return top_indices.gather(dim=-1, index=sampled_in_top.unsqueeze(-1)).squeeze(-1)


def _write_predictions(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write inference rows to CSV.

    Args:
        path: Output CSV path.
        rows: Iterable of prediction dictionaries.

    Returns:
        None.

    Important notes:
        - Parent directories are created automatically.
        - Output schema is fixed for downstream tooling.
    """
    fieldnames = [
        "sample_id",
        "input_source",
        "sample_index_in_record",
        "step",
        "batch_index",
        "pred_delta_id",
        "pred_delta_value",
        "pred_lane_mask",
        "target_delta_id",
        "target_delta_value",
        "target_lane_mask",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_inference(settings: InferSettings) -> Dict[str, Any]:
    """Execute forward inference for selected cache-backed samples.

    Args:
        settings: Validated inference settings.

    Returns:
        Summary dict containing output paths, counts, and provenance metadata.

    Important notes:
        - Uses frozen vocab from config and enforces checkpoint vocab path match.
        - Never updates parameters (`eval()` + `torch.no_grad()`).
        - `max_steps=0` means no step cap.
    """
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    config = load_train_config(force_reload=False)
    chart_cfg = config.get("chart", {}) if isinstance(config, dict) else {}
    num_lanes = int(chart_cfg.get("num_lanes", 7))
    val_ids = read_split_sample_ids(settings.input_split_path)
    vocab = load_vocab_from_config(force_reload=True)
    records, failed_record_samples = collect_cache_samples_for_ids(
        settings.cache_index_path,
        settings.project_root,
        val_ids,
    )
    filtered_records, failed_pairs = filter_valid_cache_sample_records(records, vocab)
    total_failures = failed_record_samples + failed_pairs
    if not filtered_records:
        raise ValueError("No valid samples available for inference.")

    pairs = [(record.x, record.y) for record in filtered_records]
    dataset = NextEventDataset(pairs, vocab)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )

    first_x, _ = dataset[0]
    grid_stats_dim = int(first_x["grid_stats"].shape[0])
    device = resolve_device(settings.device)
    if settings.device.strip().lower().startswith("cuda") and device.type != "cuda":
        LOGGER.warning("CUDA not available, fallback to CPU for inference.")

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
    if checkpoint_vocab_path.resolve() != settings.vocab_path.resolve():
        raise ValueError(
            "Vocab path mismatch between config and checkpoint: "
            f"config={settings.vocab_path} checkpoint={checkpoint_vocab_path}"
        )
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    prediction_rows: List[Dict[str, Any]] = []
    max_steps = settings.max_steps if settings.max_steps > 0 else None
    sample_cursor = 0
    step_count = 0

    with torch.no_grad():
        for step_count, (x, y) in enumerate(loader, start=1):
            if max_steps is not None and step_count > max_steps:
                break
            x = {k: v.to(device) for k, v in x.items()}
            logits = model(x)
            selected_delta = _select_delta_ids(
                logits["delta_tick_logits"],
                top_k=settings.top_k,
                temperature=settings.temperature,
                sampling=settings.sampling,
            ).detach().cpu()
            selected_lane = logits["lane_mask_logits"].argmax(dim=-1).detach().cpu()

            target_delta = y["delta_tick"].detach().cpu()
            target_lane = y["lane_mask"].detach().cpu()
            batch_size = int(selected_delta.shape[0])
            batch_records = filtered_records[sample_cursor : sample_cursor + batch_size]

            for i in range(batch_size):
                delta_id = int(selected_delta[i].item())
                target_delta_id = int(target_delta[i].item())
                record: CacheSampleRecord = batch_records[i]
                prediction_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "input_source": record.input_source,
                        "sample_index_in_record": record.sample_index_in_record,
                        "step": step_count,
                        "batch_index": i,
                        "pred_delta_id": delta_id,
                        "pred_delta_value": vocab.decode_delta_tick(delta_id),
                        "pred_lane_mask": int(selected_lane[i].item()),
                        "target_delta_id": target_delta_id,
                        "target_delta_value": vocab.decode_delta_tick(target_delta_id),
                        "target_lane_mask": int(target_lane[i].item()),
                    }
                )
            # Dataset order is stable (`shuffle=False`), so this cursor maps
            # batch rows back to source record metadata.
            sample_cursor += batch_size

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / settings.output_filename
    summary_path = settings.output_dir / settings.summary_filename
    _write_predictions(output_path, prediction_rows)

    summary = {
        "checkpoint_path": canonical_path(settings.checkpoint_path, settings.project_root),
        "input_split_path": canonical_path(settings.input_split_path, settings.project_root),
        "cache_index_path": canonical_path(settings.cache_index_path, settings.project_root),
        "vocab_path": canonical_path(settings.vocab_path, settings.project_root),
        "output_path": canonical_path(output_path, settings.project_root),
        "output_row_count": len(prediction_rows),
        "step_count": step_count,
        "failed_sample_count": total_failures,
        "sampling": settings.sampling,
        "top_k": settings.top_k,
        "temperature": settings.temperature,
        "max_steps": settings.max_steps,
        "seed": settings.seed,
        "device": device.type,
        "num_lanes": num_lanes,
        "n_mels": int(first_x["audio_window"].shape[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = canonical_path(summary_path, settings.project_root)
    return summary


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
    """CLI entrypoint for inference.

    Args:
        None.

    Returns:
        None.

    Important notes:
        - Reads all runtime parameters from `configs/train.yaml`.
    """
    _configure_logging()
    settings = load_infer_settings(force_reload=True)
    summary = run_inference(settings)
    LOGGER.info(
        "Inference done: rows=%d steps=%d failed_samples=%d",
        summary["output_row_count"],
        summary["step_count"],
        summary["failed_sample_count"],
    )
    LOGGER.info("Predictions CSV: %s", settings.output_dir / settings.output_filename)
    LOGGER.info("Summary JSON: %s", settings.output_dir / settings.summary_filename)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("torch is required for training. Install PyTorch first.") from exc

from config_loader import load_train_config
from model.next_event_model import NextEventPredictor, compute_next_event_loss
from vocab.delta_tick_vocab import DeltaTickVocab, load_vocab_from_config


def _resolve_path(raw_path: Any, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _read_cache_index(cache_index_path: Path) -> List[Dict[str, str]]:
    if not cache_index_path.exists():
        return []
    with cache_index_path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _read_split_sample_ids(split_path: Path) -> set[str]:
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
        raise ValueError(f"No sample_id found in split file: {split_path}")
    return sample_ids


def _is_valid_row(row: Dict[str, Any]) -> bool:
    return str(row.get("valid", "")).strip().lower() == "true"


def _load_shard_records(shard_path: Path) -> List[Dict[str, Any]]:
    suffix = shard_path.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid PT shard payload: {shard_path}")
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"Invalid PT shard records: {shard_path}")
        return records

    if suffix == ".npz":
        with np.load(str(shard_path), allow_pickle=True) as archive:
            if "records" not in archive:
                raise ValueError(f"NPZ shard missing records: {shard_path}")
            records_obj = archive["records"].tolist()
            if isinstance(records_obj, list):
                if records_obj and isinstance(records_obj[0], list):
                    return list(records_obj[0])
                return records_obj
            if isinstance(records_obj, tuple):
                return list(records_obj)
            return []

    raise ValueError(f"Unsupported shard format: {shard_path.suffix!r}")


def _iter_samples_from_cache_index(cache_index_path: Path, project_root: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    rows = [row for row in _read_cache_index(cache_index_path) if _is_valid_row(row)]
    shard_paths: Dict[str, Path] = {}
    for row in rows:
        shard_path = _resolve_path(row.get("shard_path"), project_root)
        if shard_path is None:
            continue
        shard_paths[str(shard_path.resolve()).lower()] = shard_path

    samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for _, shard_path in sorted(shard_paths.items(), key=lambda item: item[0]):
        for record in _load_shard_records(shard_path):
            if not _is_valid_row(record):
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            record_samples = payload.get("samples")
            if not isinstance(record_samples, (list, tuple)):
                continue
            for sample in record_samples:
                if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    samples.append((sample[0], sample[1]))
    return samples


def _iter_samples_from_cache_index_for_split(
    cache_index_path: Path,
    split_path: Path,
    project_root: Path,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], int]:
    sample_ids = _read_split_sample_ids(split_path)
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
        key = str(shard_path.resolve()).lower()
        shard_path_lookup[key] = shard_path
        shard_to_sample_ids.setdefault(key, set()).add(sample_id)

    missing_ids = sample_ids - ids_with_any_row
    failure_sample_ids.update(missing_ids)

    samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
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
            record_samples = payload.get("samples")
            if not isinstance(record_samples, (list, tuple)):
                failure_sample_ids.add(sample_id)
                continue
            for sample in record_samples:
                if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    samples.append((sample[0], sample[1]))

    unresolved = (sample_ids - failure_sample_ids) - seen_record_ids
    failure_sample_ids.update(unresolved)
    return samples, len(failure_sample_ids)


class NextEventDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], vocab: DeltaTickVocab) -> None:
        self.samples = list(samples)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x, y = self.samples[idx]
        past_delta_raw = x.get("past_delta_id", x.get("past_delta_tick"))
        past_lane_raw = x.get("past_lane_mask_id", x.get("past_lane_mask"))
        if past_delta_raw is None or past_lane_raw is None:
            past_events = list(x.get("past_events", []))
            if not past_events:
                raise ValueError("sample is missing non-empty `past_events`.")
            past_delta_raw = [self.vocab.encode_delta_tick(event.get("delta_tick")) for event in past_events]
            past_lane_raw = [int(event.get("lane_mask", 0)) for event in past_events]

        past_delta_id = torch.as_tensor(past_delta_raw, dtype=torch.long)
        past_lane_mask_id = torch.as_tensor(past_lane_raw, dtype=torch.long)
        grid_stats = torch.as_tensor(x["grid_stats"], dtype=torch.float32)
        audio_window = torch.as_tensor(x["audio_window"], dtype=torch.float32)

        target_delta_id = y.get("delta_tick_id")
        if target_delta_id is None:
            target_delta_id = self.vocab.encode_delta_tick(y.get("delta_tick"))
        target_lane_mask = int(y["lane_mask"])

        return (
            {
                "past_delta_id": past_delta_id,
                "past_lane_mask_id": past_lane_mask_id,
                "grid_stats": grid_stats,
                "audio_window": audio_window,
            },
            {
                "delta_tick": torch.tensor(int(target_delta_id), dtype=torch.long),
                "lane_mask": torch.tensor(target_lane_mask, dtype=torch.long),
            },
        )


def _build_loader(
    *,
    cache_index_path: Path,
    split_train_path: Path | None,
    project_root: Path,
    vocab: DeltaTickVocab,
    batch_size: int,
    strict_split: bool,
    max_samples: int | None = None,
) -> DataLoader:
    if strict_split:
        if split_train_path is None:
            raise ValueError("split.train_path is required when strict_split=true.")
        all_samples, failed_sample_count = _iter_samples_from_cache_index_for_split(
            cache_index_path,
            split_train_path,
            project_root,
        )
        if failed_sample_count > 0:
            print(f"warning: failed_or_missing_samples_in_split={failed_sample_count}")
    else:
        all_samples = _iter_samples_from_cache_index(cache_index_path, project_root)

    probe_dataset = NextEventDataset(all_samples, vocab)
    valid_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    failed_pairs = 0
    for idx in range(len(probe_dataset)):
        try:
            _ = probe_dataset[idx]
        except Exception:  # noqa: BLE001
            failed_pairs += 1
            continue
        valid_samples.append(all_samples[idx])
    if failed_pairs > 0:
        print(f"warning: skipped_invalid_sample_pairs={failed_pairs}")

    if max_samples is not None and max_samples > 0:
        valid_samples = valid_samples[:max_samples]
    if not valid_samples:
        raise ValueError(f"No training samples found in cache index: {cache_index_path}")
    dataset = NextEventDataset(valid_samples, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_training(
    epochs: int,
    max_steps: int | None,
    max_samples: int | None,
    *,
    strict_split: bool,
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config = load_train_config(force_reload=True)
    cache_cfg = config.get("cache", {})
    train_cfg = config.get("train", {})
    split_cfg = config.get("split", {})

    cache_index_path = _resolve_path(cache_cfg.get("cache_index_path", "./data/cache/train_cache/cache_index.csv"), PROJECT_ROOT)
    if cache_index_path is None:
        raise ValueError("cache.cache_index_path is required.")
    split_train_path = _resolve_path(split_cfg.get("train_path"), PROJECT_ROOT)

    vocab = load_vocab_from_config(force_reload=True)
    batch_size = int(train_cfg.get("batch_size", 16))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    loader = _build_loader(
        cache_index_path=cache_index_path,
        split_train_path=split_train_path,
        project_root=PROJECT_ROOT,
        vocab=vocab,
        batch_size=batch_size,
        strict_split=strict_split,
        max_samples=max_samples,
    )

    first_x, _ = loader.dataset[0]
    grid_stats_dim = int(first_x["grid_stats"].shape[0])
    delta_vocab_size = vocab.num_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextEventPredictor(
        delta_vocab_size=delta_vocab_size,
        grid_stats_dim=grid_stats_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(
        f"dataset_size={len(loader.dataset)} batch_size={batch_size} "
        f"delta_vocab_size={delta_vocab_size} grid_stats_dim={grid_stats_dim} device={device}"
    )
    model.train()
    global_step = 0
    for epoch in range(1, epochs + 1):
        for x, y in loader:
            global_step += 1
            if max_steps is not None and global_step > max_steps:
                print("Reached max_steps, stopping.")
                return

            x = {k: v.to(device) for k, v in x.items()}
            target_delta = y["delta_tick"].to(device)
            target_lane = y["lane_mask"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            losses = compute_next_event_loss(logits, target_delta, target_lane)
            losses["total_loss"].backward()
            optimizer.step()

            pred_delta = logits["delta_tick_logits"].argmax(dim=-1)
            pred_lane = logits["lane_mask_logits"].argmax(dim=-1)
            print(
                f"epoch={epoch} step={global_step} "
                f"loss={losses['total_loss'].item():.4f} "
                f"delta={losses['delta_loss'].item():.4f} lane={losses['lane_loss'].item():.4f} "
                f"pred=({pred_delta[0].item()},{pred_lane[0].item()}) "
                f"target=({target_delta[0].item()},{target_lane[0].item()})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal training loop for next-event model.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--strict-split", dest="strict_split", action="store_true")
    parser.add_argument("--no-strict-split", dest="strict_split", action="store_false")
    parser.set_defaults(strict_split=True)
    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        max_steps=args.max_steps,
        max_samples=args.max_samples,
        strict_split=bool(args.strict_split),
    )


if __name__ == "__main__":
    main()

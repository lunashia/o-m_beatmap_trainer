"""Build offline training cache shards for osu!mania model training.

Purpose:
    Move expensive preprocessing out of the training loop by converting
    beatmap+audio samples from `manifest.csv` into shard files (`.npz`/`.pt`)
    plus `cache_index.csv`.

Input/Output format:
    Input:
        - Configuration from `configs/train.yaml` via `load_cache_settings`.
        - Manifest rows with columns like:
          `sample_id, osu_path, audio_path, key_count, valid, ...`
    Output:
        - Sharded cache files: `train_cache_shard_00000.{npz|pt}`, ...
        - Index CSV with one row per manifest row:
          `sample_id, shard_path, shard_index, key_count, osu_path, audio_path,
           valid, error`

Pipeline fit:
    Upstream input:
        - `scripts/build_manifest.py` (or equivalent) produces `manifest.csv`.
        - Existing modules provide per-sample transforms:
          `parser.osu_parser`, `quantize.quantizer`, `events.events_builder`,
          `grid.grid_builder`, `audio.mel_extractor`, `dataset_builder`.
    Downstream output:
        - Training/data-loader modules consume cache shards and `cache_index.csv`
          to avoid parse/feature extraction at train time.
"""

from __future__ import annotations

import ast
import csv
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import load_train_config

LOGGER = logging.getLogger("build_train_cache")
CACHE_SCHEMA_VERSION = 1
CACHE_INDEX_COLUMNS = [
    "sample_id",
    "shard_path",
    "shard_index",
    "key_count",
    "osu_path",
    "audio_path",
    "valid",
    "error",
]


@dataclass(frozen=True)
class CacheSettings:
    project_root: Path
    manifest_path: Path
    cache_dir: Path
    cache_index_path: Path
    shard_size: int
    cache_format: str
    overwrite: bool
    num_workers: int
    include_invalid: bool
    skip_failed_samples: bool
    allowed_key_counts: set[int]
    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: Optional[float]
    clip_seconds: Optional[float]
    ticks_per_beat: int
    context_events: int
    audio_window_size: int
    expected_n_mels: int
    num_lanes: int


@dataclass
class ProcessedSample:
    sample_id: str
    key_count: Optional[int]
    osu_path: str
    audio_path: str
    valid: bool
    error: str
    record: Optional[Dict[str, Any]]
    skipped: bool = False


@dataclass
class BuildSummary:
    manifest_rows: int = 0
    eligible_rows: int = 0
    cached_rows: int = 0
    skipped_rows: int = 0
    failed_rows: int = 0
    shards_written: int = 0


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: List[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    if isinstance(raw, (int, float, bool)):
        return [raw]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_int_set(raw: Any) -> set[int]:
    values: set[int] = set()
    for item in _parse_list(raw):
        try:
            values.add(int(item))
        except (TypeError, ValueError):
            continue
    return values


def _as_bool(raw: Any, default: bool = False) -> bool:
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
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(str(raw).strip()))
        except (TypeError, ValueError):
            return default


def _as_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return float(raw)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolve_path(raw_path: Any, base_dir: Path) -> Optional[Path]:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _canonical_path(path: Optional[Path], project_root: Path) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(resolved)


def _row_is_valid(row: Dict[str, Any]) -> bool:
    return str(row.get("valid", "")).strip().lower() == "true"


def _row_key_count(row: Dict[str, Any]) -> Optional[int]:
    value = str(row.get("key_count", "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _row_sample_id(row: Dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    fallback = str(row.get("osu_path", "")).strip()
    return fallback or "unknown"


def _read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def _normalize_grid(grid: List[List[int]], num_lanes: int) -> np.ndarray:
    if not grid:
        return np.zeros((0, num_lanes), dtype=np.uint8)
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape={arr.shape!r}")
    if arr.shape[1] != num_lanes:
        raise ValueError(f"grid must have {num_lanes} lanes, got {arr.shape[1]}")
    return arr


def _build_grid_stats(
    events: Sequence[dict],
    grid: np.ndarray,
    *,
    num_lanes: int,
    ticks_per_beat: int,
) -> np.ndarray:
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be positive, got {num_lanes!r}")
    if ticks_per_beat <= 0:
        raise ValueError(f"ticks_per_beat must be positive, got {ticks_per_beat!r}")
    if not events:
        return np.zeros((0, 10), dtype=np.float32)

    total_notes = int(grid.sum()) if grid.size else 0
    total_notes = max(1, total_notes)
    lane_totals = np.zeros(num_lanes, dtype=np.int64)
    running_delta_sum = 0.0
    stats: List[np.ndarray] = []
    denom_delta = max(1.0, 4.0 * float(ticks_per_beat))
    event_denominator = max(1, len(events) - 1)

    for idx, event in enumerate(events):
        start_tick = event.get("start_tick")
        if isinstance(start_tick, int) and 0 <= start_tick < grid.shape[0]:
            lane_totals += grid[start_tick, :num_lanes]

        delta_tick = event.get("delta_tick")
        if isinstance(delta_tick, int) and delta_tick > 0:
            running_delta_sum += float(delta_tick)

        chord_size = event.get("chord_size")
        if not isinstance(chord_size, int) or chord_size < 0:
            lane_mask = event.get("lane_mask")
            chord_size = int(lane_mask).bit_count() if isinstance(lane_mask, int) else 0

        lane_norms = lane_totals.astype(np.float32) / float(total_notes)
        progress = float(idx) / float(event_denominator)
        avg_delta = (running_delta_sum / max(1, idx)) / denom_delta if idx > 0 else 0.0
        chord_norm = float(chord_size) / float(max(1, num_lanes))

        vector = np.concatenate(
            [
                lane_norms,
                np.asarray([progress, avg_delta, chord_norm], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)
        stats.append(vector)

    return np.stack(stats, axis=0).astype(np.float32, copy=False)


def _make_shard_path(cache_dir: Path, shard_index: int, cache_format: str) -> Path:
    suffix = cache_format.lower().strip()
    return cache_dir / f"train_cache_shard_{shard_index:05d}.{suffix}"


def _write_shard(records: Sequence[Dict[str, Any]], shard_path: Path, cache_format: str) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_name(shard_path.name + ".tmp")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "records": list(records),
    }

    if cache_format == "pt":
        import torch

        torch.save(payload, tmp_path)
    elif cache_format == "npz":
        with tmp_path.open("wb") as fp:
            np.savez_compressed(
                fp,
                schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int32),
                records=np.asarray([list(records)], dtype=object),
            )
    else:
        raise ValueError(f"Unsupported cache_format: {cache_format!r}")

    tmp_path.replace(shard_path)


def _cleanup_existing_outputs(settings: CacheSettings) -> None:
    if settings.cache_index_path.exists():
        settings.cache_index_path.unlink()

    for ext in ("pt", "npz"):
        for path in settings.cache_dir.glob(f"train_cache_shard_*.{ext}"):
            path.unlink()


def _has_existing_outputs(settings: CacheSettings) -> bool:
    if settings.cache_index_path.exists():
        return True
    for ext in ("pt", "npz"):
        if any(settings.cache_dir.glob(f"train_cache_shard_*.{ext}")):
            return True
    return False


def load_cache_settings(force_reload: bool = True) -> CacheSettings:
    """Load and validate cache-building settings from `configs/train.yaml`.

    Args:
        force_reload (bool):
            When `True`, reload config from disk even if cached by
            `config_loader`; otherwise reuse cached config.

    Returns:
        CacheSettings:
            Fully resolved, validated settings object with absolute paths and
            numeric/boolean conversions applied.

    Important notes:
        - Path-like settings are resolved against this project's root.
        - `cache_format` must be `pt` or `npz`.
        - `shard_size` and `num_workers` must be positive.
        - `audio.n_mels` and `dataset.expected_n_mels` must match to keep
          cached features compatible with dataset/model assumptions.
        - Upstream input: `config_loader.load_train_config`.
        - Downstream output: consumed by `build_train_cache`.
    """
    config = load_train_config(force_reload=force_reload)
    project_root = PROJECT_ROOT

    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    manifest_cfg = config.get("manifest", {}) if isinstance(config, dict) else {}
    cache_cfg = config.get("cache", {}) if isinstance(config, dict) else {}
    audio_cfg = config.get("audio", {}) if isinstance(config, dict) else {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    quant_cfg = config.get("quantize", {}) if isinstance(config, dict) else {}
    chart_cfg = config.get("chart", {}) if isinstance(config, dict) else {}

    manifest_path = _resolve_path(
        cache_cfg.get(
            "manifest_path",
            manifest_cfg.get("manifest_path", Path(str(data_cfg.get("cache_dir", "./data/cache"))) / "manifest.csv"),
        ),
        project_root,
    )
    if manifest_path is None:
        raise ValueError("manifest_path must be configured.")

    cache_dir = _resolve_path(
        cache_cfg.get(
            "cache_dir",
            Path(str(data_cfg.get("cache_dir", "./data/cache"))) / "train_cache",
        ),
        project_root,
    )
    if cache_dir is None:
        raise ValueError("cache_dir must be configured.")

    cache_index_path = _resolve_path(
        cache_cfg.get("cache_index_path", cache_dir / "cache_index.csv"),
        project_root,
    )
    if cache_index_path is None:
        raise ValueError("cache_index_path must be configured.")

    cache_format = str(cache_cfg.get("cache_format", "pt")).strip().lower()
    if cache_format not in {"pt", "npz"}:
        raise ValueError(f"cache_format must be 'pt' or 'npz', got {cache_format!r}")

    shard_size = _as_int(cache_cfg.get("shard_size", 5000), 5000)
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size!r}")

    num_workers = _as_int(cache_cfg.get("num_workers", 1), 1)
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers!r}")

    include_invalid = _as_bool(
        cache_cfg.get("include_invalid", manifest_cfg.get("include_invalid", False)),
        False,
    )
    skip_failed_samples = _as_bool(cache_cfg.get("skip_failed_samples", True), True)

    allowed_key_counts = _parse_int_set(
        cache_cfg.get("allowed_key_counts", manifest_cfg.get("allowed_key_counts", "[7]"))
    )
    if not allowed_key_counts:
        allowed_key_counts = {7}

    sample_rate = _as_int(audio_cfg.get("sample_rate", 22050), 22050)
    n_fft = _as_int(audio_cfg.get("n_fft", 2048), 2048)
    win_length = _as_int(audio_cfg.get("win_length", n_fft), n_fft)
    hop_length = _as_int(audio_cfg.get("hop_length", 512), 512)
    n_mels = _as_int(audio_cfg.get("n_mels", 80), 80)
    fmin = float(audio_cfg.get("fmin", 30.0))
    fmax_raw = audio_cfg.get("fmax", None)
    fmax = None if fmax_raw is None else float(fmax_raw)
    clip_seconds = _as_float(audio_cfg.get("clip_seconds", audio_cfg.get("audio_duration")))

    ticks_per_beat = _as_int(quant_cfg.get("ticks_per_beat", 48), 48)
    context_events = _as_int(dataset_cfg.get("context_events", 32), 32)
    audio_window_size = _as_int(dataset_cfg.get("audio_window_size", 64), 64)
    expected_n_mels = _as_int(dataset_cfg.get("expected_n_mels", n_mels), n_mels)
    num_lanes = _as_int(chart_cfg.get("num_lanes", 7), 7)

    if expected_n_mels != n_mels:
        raise ValueError(
            f"audio.n_mels ({n_mels}) must match dataset.expected_n_mels ({expected_n_mels})."
        )

    return CacheSettings(
        project_root=project_root,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        cache_index_path=cache_index_path,
        shard_size=shard_size,
        cache_format=cache_format,
        overwrite=_as_bool(cache_cfg.get("overwrite", False), False),
        num_workers=num_workers,
        include_invalid=include_invalid,
        skip_failed_samples=skip_failed_samples,
        allowed_key_counts=allowed_key_counts,
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        clip_seconds=clip_seconds,
        ticks_per_beat=ticks_per_beat,
        context_events=context_events,
        audio_window_size=audio_window_size,
        expected_n_mels=expected_n_mels,
        num_lanes=num_lanes,
    )


def _build_success_record(
    *,
    sample_id: str,
    key_count: int,
    osu_path: Path,
    audio_path: Path,
    parsed_chart: Dict[str, Any],
    quantized_notes: List[dict],
    events: List[dict],
    grid: np.ndarray,
    grid_stats: np.ndarray,
    mel: np.ndarray,
    samples: List[Any],
) -> Dict[str, Any]:
    normalized_timing_points: List[Dict[str, Any]] = []
    for tp in parsed_chart.get("timing_points", []):
        if not isinstance(tp, dict):
            continue
        beat_length_ms = tp.get("beat_length_ms", tp.get("beat_length"))
        normalized_timing_points.append(
            {
                **tp,
                "beat_length_ms": beat_length_ms,
                "beat_length": beat_length_ms,
            }
        )
    normalized_chart = dict(parsed_chart)
    normalized_chart["timing_points"] = normalized_timing_points

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "key_count": key_count,
        "osu_path": str(osu_path),
        "audio_path": str(audio_path),
        "valid": True,
        "error": "",
        "payload": {
            "parsed_chart": normalized_chart,
            "quantized_notes": quantized_notes,
            "events": events,
            "grid": grid,
            "grid_stats": grid_stats,
            "mel": mel,
            "samples": samples,
        },
    }


def _build_failure_record(
    *,
    sample_id: str,
    key_count: Optional[int],
    osu_path: Optional[Path],
    audio_path: Optional[Path],
    error: str,
) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "key_count": key_count if key_count is not None else "",
        "osu_path": str(osu_path) if osu_path is not None else "",
        "audio_path": str(audio_path) if audio_path is not None else "",
        "valid": False,
        "error": error,
        "payload": None,
    }


def _process_manifest_row(row: Dict[str, Any], settings: CacheSettings) -> ProcessedSample:
    sample_id = _row_sample_id(row)
    manifest_valid = _row_is_valid(row)
    manifest_key_count = _row_key_count(row)
    osu_path = _resolve_path(row.get("osu_path"), settings.project_root)
    manifest_audio_path = _resolve_path(row.get("audio_path"), settings.project_root)

    if not manifest_valid and not settings.include_invalid:
        return ProcessedSample(
            sample_id=sample_id,
            key_count=manifest_key_count,
            osu_path=_canonical_path(osu_path, settings.project_root),
            audio_path=_canonical_path(manifest_audio_path, settings.project_root),
            valid=False,
            error="skipped: manifest valid=false",
            record=None,
            skipped=True,
        )

    try:
        from audio.mel_extractor import extract_mel
        from dataset_builder import build_dataset, make_tick_to_frame
        from events.events_builder import build_events
        from grid.grid_builder import build_grid
        from parser.osu_parser import parse_osu
        from quantize.quantizer import quantize_notes

        if osu_path is None:
            raise ValueError("Missing osu_path in manifest row.")

        parsed = parse_osu(str(osu_path))
        chart_key_count = int(parsed["key_count"])
        if settings.allowed_key_counts and chart_key_count not in settings.allowed_key_counts:
            raise ValueError(
                f"Unsupported key_count={chart_key_count}; expected one of {sorted(settings.allowed_key_counts)}."
            )

        parsed_audio_path = parsed.get("audio_path") or ""
        audio_path = _resolve_path(parsed_audio_path, osu_path.parent) if parsed_audio_path else manifest_audio_path
        if audio_path is None:
            raise ValueError("Missing audio path after parsing osu file.")

        quantized_notes = quantize_notes(
            parsed["timing_points"],
            parsed["hit_objects"],
            ticks_per_beat=settings.ticks_per_beat,
        )
        events = build_events(quantized_notes)
        grid = _normalize_grid(build_grid(quantized_notes), settings.num_lanes)
        grid_stats = _build_grid_stats(
            events,
            grid,
            num_lanes=settings.num_lanes,
            ticks_per_beat=settings.ticks_per_beat,
        )
        mel = extract_mel(
            str(audio_path),
            sample_rate=settings.sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            n_mels=settings.n_mels,
            fmin=settings.fmin,
            fmax=settings.fmax,
        )
        if settings.clip_seconds is not None and settings.clip_seconds > 0:
            max_frames = max(
                1,
                int(math.ceil(settings.clip_seconds * settings.sample_rate / settings.hop_length)),
            )
            mel = mel[:, :max_frames]
        tick_to_frame = make_tick_to_frame(
            parsed["timing_points"],
            settings.ticks_per_beat,
            sample_rate=settings.sample_rate,
            hop_length=settings.hop_length,
        )
        samples = build_dataset(
            events,
            grid_stats,
            mel,
            tick_to_frame,
            context_events=settings.context_events,
            audio_window_size=settings.audio_window_size,
            expected_n_mels=settings.expected_n_mels,
        )

        return ProcessedSample(
            sample_id=sample_id,
            key_count=chart_key_count,
            osu_path=_canonical_path(osu_path, settings.project_root),
            audio_path=_canonical_path(audio_path, settings.project_root),
            valid=True,
            error="",
            record=_build_success_record(
                sample_id=sample_id,
                key_count=chart_key_count,
                osu_path=osu_path,
                audio_path=audio_path,
                parsed_chart=parsed,
                quantized_notes=quantized_notes,
                events=events,
                grid=grid,
                grid_stats=grid_stats,
                mel=mel,
                samples=samples,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if settings.skip_failed_samples:
            return ProcessedSample(
                sample_id=sample_id,
                key_count=manifest_key_count,
                osu_path=_canonical_path(osu_path, settings.project_root),
                audio_path=_canonical_path(manifest_audio_path, settings.project_root),
                valid=False,
                error=error,
                record=None,
                skipped=False,
            )

        return ProcessedSample(
            sample_id=sample_id,
            key_count=manifest_key_count,
            osu_path=_canonical_path(osu_path, settings.project_root),
            audio_path=_canonical_path(manifest_audio_path, settings.project_root),
            valid=False,
            error=error,
            record=_build_failure_record(
                sample_id=sample_id,
                key_count=manifest_key_count,
                osu_path=osu_path,
                audio_path=manifest_audio_path,
                error=error,
            ),
            skipped=False,
        )


def build_train_cache(
    settings: CacheSettings,
    *,
    manifest_rows: Optional[Sequence[Dict[str, Any]]] = None,
    process_row_fn: Callable[[Dict[str, Any], CacheSettings], ProcessedSample] = _process_manifest_row,
) -> BuildSummary:
    """Build sharded training cache files and index rows from manifest input.

    Args:
        settings (CacheSettings):
            Resolved runtime configuration, including output paths, shard size,
            format, filtering flags, and feature parameters.
        manifest_rows (Optional[Sequence[Dict[str, Any]]]):
            Optional in-memory manifest rows. If omitted, rows are loaded from
            `settings.manifest_path`.
        process_row_fn (Callable[[Dict[str, Any], CacheSettings], ProcessedSample]):
            Row processor used per manifest row. Defaults to
            `_process_manifest_row`; can be injected for tests/custom flows.

    Returns:
        BuildSummary:
            Aggregated counters: total/eligible/cached/skipped/failed rows and
            shard count written.

    Important notes:
        - `overwrite=false` raises `FileExistsError` when previous cache output
          exists.
        - Single-row failures do not crash the run; they are represented in the
          index with `valid=false` and an `error` message.
        - Shards are written with bounded size (`settings.shard_size`) and
          index rows point to shard path + shard-local position.
        - Upstream input: manifest rows (`manifest.csv`) and per-row processing
          from parser/quantize/events/grid/audio/dataset modules.
        - Downstream output: cache shard files + `cache_index.csv` for training.
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_index_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.overwrite:
        _cleanup_existing_outputs(settings)
    elif _has_existing_outputs(settings):
        raise FileExistsError(
            f"Cache already exists at {settings.cache_dir} / {settings.cache_index_path}; set cache.overwrite=true to rebuild."
        )

    rows = list(manifest_rows) if manifest_rows is not None else _read_manifest_rows(settings.manifest_path)
    summary = BuildSummary(manifest_rows=len(rows))
    index_rows: List[Dict[str, Any]] = []
    current_records: List[Dict[str, Any]] = []
    current_shard_index = 0

    def current_shard_path() -> Path:
        return _make_shard_path(settings.cache_dir, current_shard_index, settings.cache_format)

    def flush_current_shard() -> None:
        nonlocal current_records, current_shard_index
        if not current_records:
            return
        shard_path = current_shard_path()
        # Flush only when the in-memory shard is complete (or final tail shard).
        _write_shard(current_records, shard_path, settings.cache_format)
        summary.shards_written += 1
        current_records = []
        current_shard_index += 1

    if settings.num_workers > 1 and len(rows) > 1:
        iterator: Iterable[ProcessedSample]
        with ThreadPoolExecutor(max_workers=settings.num_workers) as executor:
            iterator = executor.map(lambda row: process_row_fn(row, settings), rows)
            for result in iterator:
                summary.eligible_rows += 1
                if result.record is None:
                    if result.skipped:
                        summary.skipped_rows += 1
                    else:
                        summary.failed_rows += 1
                    index_rows.append(
                        {
                            "sample_id": result.sample_id,
                            "shard_path": "",
                            "shard_index": "",
                            "key_count": result.key_count if result.key_count is not None else "",
                            "osu_path": result.osu_path,
                            "audio_path": result.audio_path,
                            "valid": str(result.valid).lower(),
                            "error": result.error,
                        }
                    )
                    continue

                if len(current_records) >= settings.shard_size:
                    flush_current_shard()

                shard_path = current_shard_path()
                shard_index = len(current_records)
                current_records.append(result.record)
                summary.cached_rows += 1
                if not result.valid:
                    summary.failed_rows += 1
                index_rows.append(
                    {
                        "sample_id": result.sample_id,
                        "shard_path": _canonical_path(shard_path, settings.project_root),
                        "shard_index": shard_index,
                        "key_count": result.key_count if result.key_count is not None else "",
                        "osu_path": result.osu_path,
                        "audio_path": result.audio_path,
                        "valid": str(result.valid).lower(),
                        "error": result.error,
                    }
                )
    else:
        for row in rows:
            result = process_row_fn(row, settings)
            summary.eligible_rows += 1
            if result.record is None:
                if result.skipped:
                    summary.skipped_rows += 1
                else:
                    summary.failed_rows += 1
                index_rows.append(
                    {
                        "sample_id": result.sample_id,
                        "shard_path": "",
                        "shard_index": "",
                        "key_count": result.key_count if result.key_count is not None else "",
                        "osu_path": result.osu_path,
                        "audio_path": result.audio_path,
                        "valid": str(result.valid).lower(),
                        "error": result.error,
                    }
                )
                continue

            if len(current_records) >= settings.shard_size:
                flush_current_shard()

            shard_path = current_shard_path()
            shard_index = len(current_records)
            current_records.append(result.record)
            summary.cached_rows += 1
            if not result.valid:
                summary.failed_rows += 1
            index_rows.append(
                {
                    "sample_id": result.sample_id,
                    "shard_path": _canonical_path(shard_path, settings.project_root),
                    "shard_index": shard_index,
                    "key_count": result.key_count if result.key_count is not None else "",
                    "osu_path": result.osu_path,
                    "audio_path": result.audio_path,
                    "valid": str(result.valid).lower(),
                    "error": result.error,
                }
            )

    flush_current_shard()

    with settings.cache_index_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CACHE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)

    return summary


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """CLI entry point for offline cache construction.

    Args:
        None.

    Returns:
        None.

    Important notes:
        - This function is orchestration-only: logging setup, settings load,
          cache build invocation, and summary logging.
        - Core transform behavior is implemented in `build_train_cache` and
          `_process_manifest_row` for easier testing.
        - Upstream input: local config file (`configs/train.yaml`).
        - Downstream output: side effects on disk (shards/index) and log lines.
    """
    _configure_logging()
    settings = load_cache_settings(force_reload=True)
    summary = build_train_cache(settings)
    LOGGER.info(
        "Done: manifest=%d eligible=%d cached=%d skipped=%d failed=%d shards=%d index=%s",
        summary.manifest_rows,
        summary.eligible_rows,
        summary.cached_rows,
        summary.skipped_rows,
        summary.failed_rows,
        summary.shards_written,
        settings.cache_index_path,
    )


if __name__ == "__main__":
    main()

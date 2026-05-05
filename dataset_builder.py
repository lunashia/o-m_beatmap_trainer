"""
Build supervised training samples from symbolic events and aligned audio features.

Purpose:
    Convert event-level rhythm/chord data into model-ready `(X, y)` samples for
    next-event prediction tasks.

    Input format:
    - events: sequence of dicts, each containing at least:
        {"start_tick": int, "delta_tick": int, "lane_mask": int, "chord_size": int}
      with `delta_tick > 0` and `lane_mask` in `[0, 127]`.
    - grid_stats: sequence aligned 1:1 with `events`, where each item is a
      fixed-length 1D numeric vector (consistent feature dimension `G`)
    - mel: np.ndarray of shape [n_mels, n_frames]
    - tick_to_frame: callable mapping event tick -> mel frame index

Output format:
    - list of `(X, y)` tuples
    - X:
        {
            "past_events": List[dict] length 32,
            "grid_stats": np.ndarray shape [G], dtype float32
            "audio_window": np.ndarray shape [n_mels, 64]
        }
    - y:
        {"delta_tick": int, "lane_mask": int}

Pipeline role:
    - Upstream input:
        `events` typically come from `src/events/events_builder.py`.
        `mel` typically comes from `src/audio/mel_extractor.py`.
        `grid_stats` typically come from grid/stat feature stages built on top of
        `src/grid/grid_builder.py` outputs.
    - Downstream output:
        consumed by training/data-loader code as supervised examples for sequence
        models that predict timing (`delta_tick`) and lane pattern (`lane_mask`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from config_loader import get_config_value

CONTEXT_EVENTS = 32
AUDIO_WINDOW_SIZE = 64
EXPECTED_N_MELS = 80
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_LENGTH = 512


def make_tick_to_frame(
    timing_points: Sequence[dict],
    ticks_per_beat: int,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
) -> Callable[[int], int]:
    """Build a canonical tick->mel-frame mapper from timing points.

    Timing points may provide either `beat_length` or `beat_length_ms`.
    """
    if sample_rate is None:
        sample_rate = int(get_config_value("audio.sample_rate", DEFAULT_SAMPLE_RATE))
    if hop_length is None:
        hop_length = int(get_config_value("audio.hop_length", DEFAULT_HOP_LENGTH))

    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        raise ValueError(f"ticks_per_beat must be a positive int, got {ticks_per_beat!r}")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive int, got {sample_rate!r}")
    if not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError(f"hop_length must be a positive int, got {hop_length!r}")

    valid_tps: List[tuple[int, float]] = []
    for tp in timing_points:
        if not isinstance(tp, dict):
            continue
        time_ms = tp.get("time_ms")
        beat_length = tp.get("beat_length_ms", tp.get("beat_length"))
        uninherited = tp.get("uninherited")
        if isinstance(uninherited, bool) and not uninherited:
            continue
        if not isinstance(time_ms, int):
            continue
        if not isinstance(beat_length, (int, float)) or beat_length <= 0:
            continue
        valid_tps.append((time_ms, float(beat_length)))

    if not valid_tps:
        raise ValueError("timing_points must include at least one valid uninherited timing point.")

    valid_tps.sort(key=lambda item: item[0])
    segment_base_ticks: List[int] = [0]
    for i in range(1, len(valid_tps)):
        prev_time_ms, prev_beat_length = valid_tps[i - 1]
        curr_time_ms, _ = valid_tps[i]
        seg_ticks = max(
            0,
            round((curr_time_ms - prev_time_ms) / prev_beat_length * ticks_per_beat),
        )
        segment_base_ticks.append(segment_base_ticks[-1] + seg_ticks)

    def tick_to_frame(tick: int) -> int:
        if not isinstance(tick, int):
            raise ValueError(f"tick must be int, got {tick!r}")
        if tick < 0:
            return 0

        tp_idx = 0
        while tp_idx + 1 < len(segment_base_ticks) and segment_base_ticks[tp_idx + 1] <= tick:
            tp_idx += 1

        seg_tick = tick - segment_base_ticks[tp_idx]
        tp_time_ms, tp_beat_length = valid_tps[tp_idx]
        time_ms = tp_time_ms + int(round((seg_tick / ticks_per_beat) * tp_beat_length))
        return max(0, int(round((time_ms / 1000.0) * sample_rate / hop_length)))

    return tick_to_frame


def build_dataset(
    events: Sequence[dict],
    grid_stats: Sequence[Any],
    mel: np.ndarray,
    tick_to_frame: Callable[[int], int],
    *,
    context_events: int | None = None,
    audio_window_size: int | None = None,
    expected_n_mels: int | None = None,
    delta_tick_vocab: Any | None = None,
) -> List[Tuple[Dict[str, Any], Dict[str, int]]]:
    """
    Build supervised training samples from events, grid stats, and mel features.

    Args:
        events (Sequence[dict]):
            Event sequence ordered in time. Each event must provide at least
            `start_tick`, `delta_tick`, and `lane_mask` as integers.
        grid_stats (Sequence[Any]):
            Per-event statistics aligned 1:1 with `events` by index.
            Each item must be 1D numeric and share a common feature dimension.
        mel (np.ndarray):
            Log-mel features with shape `[n_mels, n_frames]`.
        tick_to_frame (Callable[[int], int]):
            Mapping from event tick to mel frame index.

    Returns:
        List[Tuple[Dict[str, Any], Dict[str, int]]]:
            List of `(X, y)` samples where:
            - `X["past_events"]` has exactly 32 events
            - `X["grid_stats"]` is float32 vector `[G]`
            - `X["audio_window"]` has shape `[n_mels, 64]`
            - `y` contains `delta_tick` and `lane_mask` for target event `events[i]`
            - when `delta_tick_vocab` is provided, `X["past_delta_id"]`,
              `X["past_lane_mask_id"]`, and `y["delta_tick_id"]` are also added

    Notes:
        - Samples are created for indices `i` in `[32, len(events) - 1]`.
        - Any sample with an incomplete audio window is skipped.
        - `events` and `grid_stats` must have identical length.
        - `mel` must be 2D; otherwise a `ValueError` is raised.
        - This function assumes event ordering is already finalized upstream.
    """
    if context_events is None:
        context_events = int(get_config_value("dataset.context_events", CONTEXT_EVENTS))
    if audio_window_size is None:
        audio_window_size = int(get_config_value("dataset.audio_window_size", AUDIO_WINDOW_SIZE))
    if expected_n_mels is None:
        expected_n_mels = int(get_config_value("dataset.expected_n_mels", EXPECTED_N_MELS))

    if context_events <= 0:
        raise ValueError(f"context_events must be positive, got {context_events!r}")
    if audio_window_size <= 0:
        raise ValueError(f"audio_window_size must be positive, got {audio_window_size!r}")
    if expected_n_mels <= 0:
        raise ValueError(f"expected_n_mels must be positive, got {expected_n_mels!r}")

    if mel.ndim != 2:
        raise ValueError(f"mel must be 2D [n_mels, n_frames], got shape={mel.shape!r}")
    if int(mel.shape[0]) != expected_n_mels:
        raise ValueError(
            f"mel must have {expected_n_mels} mel bins to match model input, got {mel.shape[0]}"
        )
    if len(events) != len(grid_stats):
        raise ValueError(
            f"events and grid_stats must have same length, got {len(events)} and {len(grid_stats)}"
        )

    n_frames = int(mel.shape[1])
    if n_frames <= 0:
        return []

    normalized_grid_stats: List[np.ndarray] = []
    grid_stats_dim: int | None = None
    for i, stats in enumerate(grid_stats):
        arr = np.asarray(stats, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"grid_stats[{i}] must be 1D, got shape={arr.shape!r}")
        if arr.shape[0] <= 0:
            raise ValueError(f"grid_stats[{i}] must be non-empty, got shape={arr.shape!r}")
        if grid_stats_dim is None:
            grid_stats_dim = int(arr.shape[0])
        elif int(arr.shape[0]) != grid_stats_dim:
            raise ValueError(
                "grid_stats must have a fixed feature dimension across events, "
                f"but got {arr.shape[0]} and {grid_stats_dim}"
            )
        normalized_grid_stats.append(arr)

    samples: List[Tuple[Dict[str, Any], Dict[str, int]]] = []
    half = audio_window_size // 2

    for i in range(context_events, len(events)):
        event = events[i]
        past_events = list(events[i - context_events : i])
        if len(past_events) != context_events:
            continue

        start_tick = event.get("start_tick")
        if not isinstance(start_tick, int):
            continue

        center_frame = tick_to_frame(start_tick)
        if not isinstance(center_frame, int):
            continue

        f0_raw = center_frame - half
        f1_raw = center_frame + half

        # Clamp window bounds to valid mel frame range before slicing.
        f0 = max(0, f0_raw)
        f1 = min(n_frames, f1_raw)

        audio_window = mel[:, f0:f1]
        # Enforce fixed-length audio context for stable model input shape.
        if audio_window.shape[1] != audio_window_size:
            continue

        delta_tick = event.get("delta_tick")
        lane_mask = event.get("lane_mask")
        if not isinstance(delta_tick, int) or delta_tick <= 0:
            continue
        if not isinstance(lane_mask, int) or lane_mask < 0 or lane_mask > 127:
            continue

        x = {
            "past_events": past_events,
            "grid_stats": normalized_grid_stats[i],
            "audio_window": audio_window,
        }
        y = {
            "delta_tick": delta_tick,
            "lane_mask": lane_mask,
        }
        if delta_tick_vocab is not None:
            encode_delta_tick = getattr(delta_tick_vocab, "encode_delta_tick", None)
            if not callable(encode_delta_tick):
                raise TypeError(
                    "delta_tick_vocab must provide an encode_delta_tick(value) method."
                )

            past_delta_id = np.asarray(
                [encode_delta_tick(int(past_event.get("delta_tick", 0))) for past_event in past_events],
                dtype=np.int64,
            )
            past_lane_mask_id = np.asarray(
                [int(past_event.get("lane_mask", 0)) for past_event in past_events],
                dtype=np.int64,
            )
            x["past_delta_id"] = past_delta_id
            x["past_lane_mask_id"] = past_lane_mask_id
            # Backward-compatible aliases for existing callers.
            x["past_delta_tick"] = past_delta_id
            x["past_lane_mask"] = past_lane_mask_id
            y["delta_tick_id"] = int(encode_delta_tick(delta_tick))
        samples.append((x, y))

    return samples

"""Quantize osu!mania hit objects from milliseconds to tick-based notes."""

from __future__ import annotations

from typing import Dict, List, Optional

from config_loader import get_config_value

TICKS_PER_BEAT = 48


def _valid_timing_points(timing_points: List[dict]) -> List[dict]:
    """Keep timing points usable for tick conversion and sort by time."""
    valid = []
    for tp in timing_points:
        time_ms = tp.get("time_ms")
        beat_length = tp.get("beat_length_ms", tp.get("beat_length"))
        uninherited = tp.get("uninherited")
        if not isinstance(time_ms, int):
            continue
        if not isinstance(beat_length, (int, float)):
            continue
        if beat_length <= 0:
            continue
        if isinstance(uninherited, bool) and not uninherited:
            continue
        valid.append({"time_ms": time_ms, "beat_length_ms": float(beat_length)})
    valid.sort(key=lambda x: x["time_ms"])
    return valid


def _to_tick(time_ms: int, seg_base_time_ms: int, beat_length: float, ticks_per_beat: int) -> int:
    tick = round((time_ms - seg_base_time_ms) / beat_length * ticks_per_beat)
    return max(0, tick)


def quantize_notes(
    timing_points: List[dict],
    hit_objects: List[dict],
    ticks_per_beat: Optional[int] = None,
) -> List[dict]:
    """
    Convert hit object times (ms) into quantized tick notes.

    Returns:
        [
            {"start_tick": int, "lane": int, "end_tick": Optional[int]},
            ...
        ]
    """
    if ticks_per_beat is None:
        ticks_per_beat = int(get_config_value("quantize.ticks_per_beat", TICKS_PER_BEAT))
    num_lanes = int(get_config_value("chart.num_lanes", 7))
    if num_lanes <= 0:
        raise ValueError(f"chart.num_lanes must be positive, got {num_lanes!r}")

    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        raise ValueError(f"ticks_per_beat must be a positive int, got {ticks_per_beat!r}")

    valid_tps = _valid_timing_points(timing_points)
    if not valid_tps:
        return []

    notes_in_time = sorted(
        [h for h in hit_objects if isinstance(h.get("time_ms"), int)],
        key=lambda h: h["time_ms"],
    )

    quantized_notes: List[Dict[str, Optional[int]]] = []
    tp_idx = 0
    segment_base_ticks: List[int] = [0]
    for i in range(1, len(valid_tps)):
        prev = valid_tps[i - 1]
        curr = valid_tps[i]
        seg_ticks = _to_tick(
            time_ms=curr["time_ms"],
            seg_base_time_ms=prev["time_ms"],
            beat_length=prev["beat_length_ms"],
            ticks_per_beat=ticks_per_beat,
        )
        segment_base_ticks.append(segment_base_ticks[-1] + seg_ticks)

    for note in notes_in_time:
        note_time = note["time_ms"]
        lane = note.get("lane")
        if not isinstance(lane, int) or lane < 0 or lane >= num_lanes:
            continue

        while tp_idx + 1 < len(valid_tps) and valid_tps[tp_idx + 1]["time_ms"] <= note_time:
            tp_idx += 1

        active_tp = valid_tps[tp_idx]
        start_tick_local = _to_tick(
            time_ms=note_time,
            seg_base_time_ms=active_tp["time_ms"],
            beat_length=active_tp["beat_length_ms"],
            ticks_per_beat=ticks_per_beat,
        )
        start_tick = segment_base_ticks[tp_idx] + start_tick_local

        end_time_ms = note.get("end_time_ms")
        end_tick: Optional[int] = None
        if isinstance(end_time_ms, int):
            end_tick_local = _to_tick(
                time_ms=end_time_ms,
                seg_base_time_ms=active_tp["time_ms"],
                beat_length=active_tp["beat_length_ms"],
                ticks_per_beat=ticks_per_beat,
            )
            end_tick = segment_base_ticks[tp_idx] + end_tick_local

        quantized_notes.append(
            {"start_tick": start_tick, "lane": lane, "end_tick": end_tick}
        )

    quantized_notes.sort(key=lambda n: n["start_tick"])
    return quantized_notes

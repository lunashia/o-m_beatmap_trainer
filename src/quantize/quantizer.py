"""Quantize osu!mania hit objects from milliseconds to tick-based notes."""

from __future__ import annotations

from typing import Dict, List, Optional


TICKS_PER_BEAT = 48


def _valid_timing_points(timing_points: List[dict]) -> List[dict]:
    """Keep timing points usable for tick conversion and sort by time."""
    valid = []
    for tp in timing_points:
        time_ms = tp.get("time_ms")
        beat_length = tp.get("beat_length")
        if not isinstance(time_ms, int):
            continue
        if not isinstance(beat_length, (int, float)):
            continue
        if beat_length <= 0:
            continue
        valid.append({"time_ms": time_ms, "beat_length": float(beat_length)})
    valid.sort(key=lambda x: x["time_ms"])
    return valid


def _to_tick(time_ms: int, tp_time_ms: int, beat_length: float, ticks_per_beat: int) -> int:
    tick = round((time_ms - tp_time_ms) / beat_length * ticks_per_beat)
    return max(0, tick)


def quantize_notes(timing_points: List[dict], hit_objects: List[dict]) -> List[dict]:
    """
    Convert hit object times (ms) into quantized tick notes.

    Returns:
        [
            {"start_tick": int, "lane": int, "end_tick": Optional[int]},
            ...
        ]
    """
    valid_tps = _valid_timing_points(timing_points)
    if not valid_tps:
        return []

    notes_in_time = sorted(
        [h for h in hit_objects if isinstance(h.get("time_ms"), int)],
        key=lambda h: h["time_ms"],
    )

    quantized_notes: List[Dict[str, Optional[int]]] = []
    tp_idx = 0

    for note in notes_in_time:
        note_time = note["time_ms"]
        lane = note.get("lane")
        if not isinstance(lane, int):
            continue

        while tp_idx + 1 < len(valid_tps) and valid_tps[tp_idx + 1]["time_ms"] <= note_time:
            tp_idx += 1

        active_tp = valid_tps[tp_idx]
        start_tick = _to_tick(
            time_ms=note_time,
            tp_time_ms=active_tp["time_ms"],
            beat_length=active_tp["beat_length"],
            ticks_per_beat=TICKS_PER_BEAT,
        )

        end_time_ms = note.get("end_time_ms")
        end_tick: Optional[int] = None
        if isinstance(end_time_ms, int):
            end_tick = _to_tick(
                time_ms=end_time_ms,
                tp_time_ms=active_tp["time_ms"],
                beat_length=active_tp["beat_length"],
                ticks_per_beat=TICKS_PER_BEAT,
            )

        quantized_notes.append(
            {"start_tick": start_tick, "lane": lane, "end_tick": end_tick}
        )

    quantized_notes.sort(key=lambda n: n["start_tick"])
    return quantized_notes


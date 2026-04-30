"""Minimal osu!mania 7k .osu parser for training data extraction."""

from __future__ import annotations

import os
from typing import Dict, List, Optional


# 7-key mania x-coordinate mapping used by osu! stable exports.
LANE_X_TO_INDEX = {
    36: 1,
    109: 2,
    182: 3,
    256: 4,
    329: 5,
    402: 6,
    475: 7,
}


def _split_key_value(line: str) -> Optional[tuple[str, str]]:
    """Return (key, value) for 'key:value' lines, or None if malformed."""
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def _parse_int(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except (TypeError, ValueError):
        return None


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def _parse_timing_point(line: str) -> Optional[dict]:
    """
    Parse one TimingPoints line.

    Format:
    time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None

    time_ms = _parse_int(parts[0])
    beat_length = _parse_float(parts[1])
    if time_ms is None or beat_length is None:
        return None

    meter = _parse_int(parts[2]) if len(parts) > 2 else None
    uninherited_flag = _parse_int(parts[6]) if len(parts) > 6 else 1
    uninherited = uninherited_flag == 1

    return {
        "time_ms": time_ms,
        "beat_length": beat_length,
        "meter": meter,
        "uninherited": uninherited,
    }


def _parse_hit_object(line: str) -> Optional[dict]:
    """
    Parse one HitObjects line for mania 7k.

    We only keep rice notes for this project.
    If the object is LN (type bit 128), we keep only its start as rice
    and force end_time_ms to None.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    x = _parse_int(parts[0])
    time_ms = _parse_int(parts[2])
    hit_type = _parse_int(parts[3])
    if x is None or time_ms is None or hit_type is None:
        return None

    lane = LANE_X_TO_INDEX.get(x)
    if lane is None:
        return None

    # Keep lane+time only for rice training data.
    return {
        "time_ms": time_ms,
        "lane": lane,
        "end_time_ms": None,
    }


def parse_osu(file_path: str) -> Dict[str, object]:
    """
    Convert a .osu file into structured 7k osu!mania training data.

    Returns:
        {
            "audio_path": "...",
            "key_count": 7,
            "timing_points": [...],
            "hit_objects": [
                {"time_ms": 1234, "lane": 2, "end_time_ms": None},
                ...
            ],
        }
    """
    general_audio_filename = ""
    key_count: Optional[int] = None
    timing_points: List[dict] = []
    hit_objects: List[dict] = []
    mode: Optional[int] = None

    current_section: Optional[str] = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                continue

            if current_section == "General":
                kv = _split_key_value(line)
                if not kv:
                    continue
                key, value = kv
                if key == "AudioFilename":
                    general_audio_filename = value
                elif key == "Mode":
                    mode = _parse_int(value)
                continue

            if current_section == "Difficulty":
                kv = _split_key_value(line)
                if not kv:
                    continue
                key, value = kv
                if key == "CircleSize":
                    parsed = _parse_float(value)
                    if parsed is not None:
                        key_count = int(parsed)
                continue

            if current_section == "TimingPoints":
                tp = _parse_timing_point(line)
                if tp is not None:
                    timing_points.append(tp)
                continue

            if current_section == "HitObjects":
                ho = _parse_hit_object(line)
                if ho is not None:
                    hit_objects.append(ho)
                continue

    if mode is not None and mode != 3:
        raise ValueError(f"Unsupported Mode={mode}; expected mania Mode=3.")

    if key_count is None:
        key_count = 7

    if key_count != 7:
        raise ValueError(f"Unsupported CircleSize={key_count}; expected 7k.")

    if general_audio_filename:
        audio_path = os.path.normpath(
            os.path.join(os.path.dirname(file_path), general_audio_filename)
        )
    else:
        audio_path = ""

    chart: Dict[str, object] = {
        "audio_path": audio_path,
        "key_count": key_count,
        "timing_points": timing_points,
        "hit_objects": hit_objects,
    }
    return chart


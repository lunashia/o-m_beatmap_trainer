"""
Parse osu!mania 7k `.osu` files into normalized chart dictionaries.

Purpose:
- Extract a minimal, stable subset of beatmap data required by this project.
- Enforce project assumptions: mania mode, 7 keys, rice-only note output.

Input/Output format:
- Input: path to one `.osu` file.
- Output:
  {
      "audio_path": str,
      "key_count": int,
      "timing_points": [
          {"time_ms": int, "beat_length_ms": float, "beat_length": float, "meter": Optional[int],
           "uninherited": bool},
          ...
      ],
      "hit_objects": [
          {"time_ms": int, "lane": int, "end_time_ms": None},
          ...
      ],
  }

Pipeline integration:
- Upstream input: produced by beatmap discovery/indexing modules that yield
  `.osu` file paths.
- Downstream output: consumed by chart preprocessing / dataset builder modules
  that convert events into model-ready training samples.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from config_loader import get_config_value

# 7-key mania x-coordinate mapping used by osu! stable exports.
DEFAULT_LANE_X_TO_INDEX = {
    36: 0,
    109: 1,
    182: 2,
    256: 3,
    329: 4,
    402: 5,
    475: 6,
}


def _load_lane_mapping() -> dict[int, int]:
    raw_mapping = get_config_value("chart.lane_x_to_index", None)
    if not isinstance(raw_mapping, dict):
        return dict(DEFAULT_LANE_X_TO_INDEX)

    parsed: dict[int, int] = {}
    for raw_k, raw_v in raw_mapping.items():
        try:
            k = int(raw_k)
            v = int(raw_v)
        except (TypeError, ValueError):
            continue
        parsed[k] = v
    return parsed or dict(DEFAULT_LANE_X_TO_INDEX)


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
        "beat_length_ms": beat_length,
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

    lane_map = _load_lane_mapping()
    lane = lane_map.get(x)
    if lane is None:
        return None

    # Type is parsed for validation. Even if the LN bit (128) is set, this parser
    # emits only note starts for rice-only training output.
    # Keep lane+time only for rice training data.
    return {
        "time_ms": time_ms,
        "lane": lane,
        "end_time_ms": None,
    }


def parse_osu(file_path: str) -> Dict[str, object]:
    """
    Convert one `.osu` file into structured 7k osu!mania training data.

    Args:
        file_path (str):
            Filesystem path to a `.osu` beatmap file.

    Returns:
        Dict[str, object]:
            Parsed chart dictionary with:
            - `audio_path` (str): normalized path joined from beatmap directory
              and `[General] -> AudioFilename`; empty string if missing.
            - `key_count` (int): parsed from `[Difficulty] -> CircleSize`.
            - `timing_points` (List[dict]): parsed `[TimingPoints]` entries with
              `time_ms`, `beat_length_ms`, `beat_length`, `meter`, `uninherited`.
            - `hit_objects` (List[dict]): 7k lane/timestamp pairs in rice format:
              `{"time_ms": int, "lane": int, "end_time_ms": None}`.

    Important notes:
        - Only mania is supported. If `Mode` is present and not `3`, a
          `ValueError` is raised.
        - Only 7k charts are supported. If `CircleSize` is missing, defaults to
          `7`; if present but not `7`, a `ValueError` is raised.
        - Malformed lines are skipped safely instead of aborting parse.
        - Irrelevant sections are ignored.
        - LN data is intentionally reduced to start notes only (`end_time_ms=None`)
          for this rice-only project.
        - Upstream input: beatmap path provider modules.
        - Downstream output: training data preparation modules.
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

    expected_mode = int(get_config_value("chart.expected_mode", 3))
    if mode is not None and mode != expected_mode:
        raise ValueError(
            f"Unsupported Mode={mode}; expected mania Mode={expected_mode}."
        )

    expected_key_count = int(get_config_value("chart.key_count", 7))
    if key_count is None:
        key_count = expected_key_count

    if key_count != expected_key_count:
        raise ValueError(
            f"Unsupported CircleSize={key_count}; expected {expected_key_count}k."
        )

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

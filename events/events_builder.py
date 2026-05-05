from __future__ import annotations

"""
Build chord-based press events from quantized osu!mania notes.

Purpose:
- Convert per-note, tick-aligned inputs into per-tick chord events suitable for
  sequence modeling, tokenization, or downstream generation logic.

Input format:
- `quantized_notes: List[dict]`, where each item contains:
  - `start_tick: int`
  - `lane: int` (0..num_lanes-1)
  - `end_tick: Optional[int]` (ignored by this module for press-only events)

Output format:
- `List[dict]`, where each event contains:
  - `start_tick: int`
  - `delta_tick: int` (difference from previous event tick; first is 0)
  - `lane_mask: int` (num_lanes-bit lane bitmap, lane 0 as LSB)
  - `chord_size: int` (unique lane count at `start_tick`)

Pipeline placement:
- Upstream input producer: quantization stage (e.g., `quantizer`), which maps
  parsed hit objects into tick-space notes.
- Downstream consumer: event/token preparation stage (e.g., generated-event
  builder, training/inference sequence pipeline) that consumes compact
  chord-timed events.
"""

from collections import defaultdict
from typing import Dict, List

from config_loader import get_config_value


def build_events(quantized_notes: List[dict]) -> List[dict]:
    """
    Convert quantized notes into chord-based press events.

    Args:
        quantized_notes (List[dict]):
            Quantized note list. Each item is expected to contain:
            - `start_tick` (int): note onset in ticks.
            - `lane` (int): lane index in [0, num_lanes-1].
            - `end_tick` (Optional[int]): accepted but ignored in this function.

    Returns:
        List[dict]:
            Sorted chord events. Each event has:
            - `start_tick` (int): event onset tick.
            - `delta_tick` (int): tick difference from previous event
              (`0` for the first event).
            - `lane_mask` (int): bitmask over configured lanes.
            - `chord_size` (int): number of unique lanes in the chord.

    Notes:
        - Multiple notes at the same tick and lane are deduplicated per event.
        - Events are sorted by `start_tick` regardless of input order.
        - Raises `ValueError` when `start_tick` is not int or lane is outside
          [0, num_lanes-1].
    """
    if not quantized_notes:
        return []
    num_lanes = int(get_config_value("chart.num_lanes", 7))
    if num_lanes <= 0:
        raise ValueError(f"chart.num_lanes must be positive, got {num_lanes!r}")

    grouped_lanes: Dict[int, List[int]] = defaultdict(list)
    for note in quantized_notes:
        start_tick = note.get("start_tick")
        lane = note.get("lane")

        if not isinstance(start_tick, int):
            raise ValueError(f"Invalid start_tick: {start_tick!r}")
        if not isinstance(lane, int) or lane < 0 or lane >= num_lanes:
            raise ValueError(
                f"Invalid lane: {lane!r}. Expected lane in [0, {num_lanes - 1}]."
            )

        grouped_lanes[start_tick].append(lane)

    events: List[dict] = []
    previous_tick = None

    for start_tick in sorted(grouped_lanes.keys()):
        # Deduplicate lanes at the same tick; chord_size is unique-lane count.
        lanes = sorted(set(grouped_lanes[start_tick]))

        lane_mask = 0
        for lane in lanes:
            # 7-bit encoding: lane 0 is LSB, lane 6 is MSB.
            lane_mask |= 1 << lane

        # First event has no predecessor by definition.
        delta_tick = 0 if previous_tick is None else start_tick - previous_tick

        events.append(
            {
                "start_tick": start_tick,
                "delta_tick": delta_tick,
                "lane_mask": lane_mask,
                "chord_size": len(lanes),
            }
        )
        previous_tick = start_tick

    return events


if __name__ == "__main__":
    sample_notes = [
        {"start_tick": 96, "lane": 2},
        {"start_tick": 96, "lane": 4},
        {"start_tick": 108, "lane": 1},
    ]
    print(build_events(sample_notes))

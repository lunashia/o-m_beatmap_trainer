from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


def build_events(quantized_notes: List[dict]) -> List[dict]:
    """
    Convert quantized notes into chord-based press events for 7-key mania.

    Input note schema:
    {
        "start_tick": int,
        "lane": int,
        "end_tick": Optional[int]  # ignored for now
    }

    Output event schema:
    {
        "start_tick": int,
        "delta_tick": int,
        "lane_mask": int,
        "chord_size": int
    }
    """
    if not quantized_notes:
        return []

    grouped_lanes: Dict[int, List[int]] = defaultdict(list)
    for note in quantized_notes:
        start_tick = note.get("start_tick")
        lane = note.get("lane")

        if not isinstance(start_tick, int):
            raise ValueError(f"Invalid start_tick: {start_tick!r}")
        if not isinstance(lane, int) or lane < 0 or lane > 6:
            raise ValueError(f"Invalid lane: {lane!r}. Expected lane in [0, 6].")

        grouped_lanes[start_tick].append(lane)

    events: List[dict] = []
    previous_tick = None

    for start_tick in sorted(grouped_lanes.keys()):
        lanes = sorted(set(grouped_lanes[start_tick]))

        lane_mask = 0
        for lane in lanes:
            lane_mask |= 1 << lane

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

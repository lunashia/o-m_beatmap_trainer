from typing import List, Dict, Any


NUM_LANES = 7


def build_grid(quantized_notes: List[dict]) -> List[List[int]]:
    """Build a binary time-lane grid from quantized notes.

    Args:
        quantized_notes: Note dicts with at least:
            - start_tick: int
            - lane: int (0-based lane index, expected range: 0..6)
            - end_tick: ignored for now

    Returns:
        A 2D list where grid[tick][lane] is 1 if a note exists, else 0.
        Returns [] for empty input or when no valid notes remain after validation.
    """
    if not quantized_notes:
        return []

    valid_notes: List[Dict[str, Any]] = []
    for note in quantized_notes:
        if not isinstance(note, dict):
            continue

        start_tick = note.get("start_tick")
        lane = note.get("lane")

        if not isinstance(start_tick, int) or start_tick < 0:
            continue
        if not isinstance(lane, int) or not (0 <= lane < NUM_LANES):
            continue

        valid_notes.append({"start_tick": start_tick, "lane": lane})

    if not valid_notes:
        return []

    total_ticks = max(note["start_tick"] for note in valid_notes) + 1
    grid = [[0 for _ in range(NUM_LANES)] for _ in range(total_ticks)]

    for note in valid_notes:
        grid[note["start_tick"]][note["lane"]] = 1

    return grid

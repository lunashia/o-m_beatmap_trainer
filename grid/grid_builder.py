"""Convert quantized mania notes into a binary time-lane grid.

Purpose:
    Build a dense `[tick][lane]` matrix for 7-key osu!mania-style data where each
    cell is a binary activation (`1` note present, `0` otherwise).

Input/Output format:
    Input:
        `quantized_notes: List[dict]`, where each item is expected to include:
        - `start_tick: int` (time index on the quantized timeline)
        - `lane: int` (0-based lane index, valid range: 0..6)
        - `end_tick` may exist but is ignored in this rice-only builder.
    Output:
        `List[List[int]]` with shape `[total_ticks, 7]`, where:
        - `total_ticks = max(valid_start_tick) + 1`
        - `grid[tick][lane]` is `1` if a note exists, else `0`.

Pipeline position:
    Upstream input:
        Produced by quantization/event conversion stages (for example
        `events_to_quantized_notes`) that emit per-note `start_tick` + `lane`.
    Downstream output:
        Consumed by dataset/model preparation components (for example
        `dataset_builder`) that require fixed-structure, model-friendly tensors.
"""

from typing import List, Dict, Any

from config_loader import get_config_value

NUM_LANES = 7


def build_grid(quantized_notes: List[dict]) -> List[List[int]]:
    """Build a binary `[tick][lane]` grid from quantized notes.

    Args:
        quantized_notes (List[dict]): Quantized note objects. Each valid note
            must provide:
            - `start_tick` (int): non-negative tick index.
            - `lane` (int): 0-based lane index in `[0, 6]`.
            Optional fields (for example `end_tick`) are accepted but ignored.

    Returns:
        List[List[int]]: Binary grid with shape `[max(start_tick)+1, 7]`.
            `grid[tick][lane] == 1` indicates at least one note at that location.
            Returns `[]` when input is empty or no note passes validation.

    Important notes:
        - Duplicate notes at the same `(tick, lane)` do not accumulate; the cell
          remains `1`.
        - Input ordering is irrelevant.
        - Invalid entries (non-dict note, invalid tick, out-of-range lane) are
          skipped.
    """
    num_lanes = int(get_config_value("chart.num_lanes", NUM_LANES))
    if num_lanes <= 0:
        raise ValueError(f"chart.num_lanes must be positive, got {num_lanes!r}")

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
        if not isinstance(lane, int) or not (0 <= lane < num_lanes):
            continue

        valid_notes.append({"start_tick": start_tick, "lane": lane})

    if not valid_notes:
        return []

    total_ticks = max(note["start_tick"] for note in valid_notes) + 1
    grid = [[0 for _ in range(num_lanes)] for _ in range(total_ticks)]

    for note in valid_notes:
        # Binary occupancy semantics: presence sets the cell to 1 regardless of
        # how many duplicate notes map to the same (tick, lane).
        grid[note["start_tick"]][note["lane"]] = 1

    return grid

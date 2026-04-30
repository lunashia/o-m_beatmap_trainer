import unittest

from src.grid.grid_builder import build_grid


class TestBuildGrid(unittest.TestCase):
    def assert_grid_shape(self, grid, expected_ticks, expected_lanes=7):
        self.assertEqual(len(grid), expected_ticks)
        for row in grid:
            self.assertEqual(len(row), expected_lanes)

    def assert_only_ones_at(self, grid, ones_positions):
        expected = set(ones_positions)
        for tick, row in enumerate(grid):
            for lane, value in enumerate(row):
                if (tick, lane) in expected:
                    self.assertEqual(value, 1, f"Expected 1 at ({tick}, {lane})")
                else:
                    self.assertEqual(value, 0, f"Expected 0 at ({tick}, {lane})")

    def test_single_note(self):
        quantized_notes = [{"start_tick": 96, "lane": 2}]
        grid = build_grid(quantized_notes)

        self.assert_grid_shape(grid, expected_ticks=97, expected_lanes=7)
        self.assert_only_ones_at(grid, ones_positions=[(96, 2)])

    def test_chord_same_tick(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 96, "lane": 4},
        ]
        grid = build_grid(quantized_notes)

        self.assert_grid_shape(grid, expected_ticks=97, expected_lanes=7)
        self.assert_only_ones_at(grid, ones_positions=[(96, 2), (96, 4)])

    def test_multiple_ticks(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 108, "lane": 3},
        ]
        grid = build_grid(quantized_notes)

        self.assert_grid_shape(grid, expected_ticks=109, expected_lanes=7)
        self.assert_only_ones_at(grid, ones_positions=[(96, 2), (108, 3)])

    def test_duplicate_same_lane(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 96, "lane": 2},
        ]
        grid = build_grid(quantized_notes)

        self.assert_grid_shape(grid, expected_ticks=97, expected_lanes=7)
        self.assert_only_ones_at(grid, ones_positions=[(96, 2)])
        self.assertEqual(grid[96][2], 1)

    def test_unsorted_input(self):
        quantized_notes = [
            {"start_tick": 108, "lane": 3},
            {"start_tick": 96, "lane": 2},
        ]
        grid = build_grid(quantized_notes)

        self.assert_grid_shape(grid, expected_ticks=109, expected_lanes=7)
        self.assert_only_ones_at(grid, ones_positions=[(96, 2), (108, 3)])

    def test_empty_input(self):
        quantized_notes = []
        grid = build_grid(quantized_notes)
        self.assertEqual(grid, [])


if __name__ == "__main__":
    unittest.main()

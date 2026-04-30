import unittest

from src.events.events_builder import build_events


class TestBuildEvents(unittest.TestCase):
    def test_single_note(self):
        quantized_notes = [{"start_tick": 96, "lane": 2}]

        events = build_events(quantized_notes)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_tick"], 96)
        self.assertEqual(events[0]["delta_tick"], 0)
        self.assertEqual(events[0]["lane_mask"], 1 << 2)
        self.assertEqual(events[0]["chord_size"], 1)

    def test_chord_same_tick(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 96, "lane": 4},
        ]

        events = build_events(quantized_notes)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_tick"], 96)
        self.assertEqual(events[0]["delta_tick"], 0)
        self.assertEqual(events[0]["lane_mask"], (1 << 2) | (1 << 4))
        self.assertEqual(events[0]["chord_size"], 2)

    def test_multiple_events(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 108, "lane": 1},
        ]

        events = build_events(quantized_notes)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["delta_tick"], 0)
        self.assertEqual(events[1]["delta_tick"], 12)
        self.assertEqual(events[0]["lane_mask"], 1 << 2)
        self.assertEqual(events[1]["lane_mask"], 1 << 1)
        self.assertEqual(events[0]["chord_size"], 1)
        self.assertEqual(events[1]["chord_size"], 1)

    def test_unsorted_input(self):
        quantized_notes = [
            {"start_tick": 108, "lane": 1},
            {"start_tick": 96, "lane": 2},
        ]

        events = build_events(quantized_notes)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start_tick"], 96)
        self.assertEqual(events[1]["start_tick"], 108)
        self.assertEqual(events[0]["delta_tick"], 0)
        self.assertEqual(events[1]["delta_tick"], 12)
        self.assertEqual(events[0]["lane_mask"], 1 << 2)
        self.assertEqual(events[1]["lane_mask"], 1 << 1)
        self.assertEqual(events[0]["chord_size"], 1)
        self.assertEqual(events[1]["chord_size"], 1)

    def test_duplicate_lane_same_tick(self):
        quantized_notes = [
            {"start_tick": 96, "lane": 2},
            {"start_tick": 96, "lane": 2},
        ]

        events = build_events(quantized_notes)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_tick"], 96)
        self.assertEqual(events[0]["delta_tick"], 0)
        self.assertEqual(events[0]["lane_mask"], 1 << 2)
        # Duplicate lanes in a chord should be counted once.
        self.assertEqual(events[0]["chord_size"], 1)


if __name__ == "__main__":
    unittest.main()

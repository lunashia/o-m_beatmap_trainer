import unittest

from src.quantize.quantizer import quantize_notes


class TestQuantizeNotes(unittest.TestCase):
    def test_basic_conversion(self):
        timing_points = [{"time_ms": 0, "beat_length": 500}]
        hit_objects = [{"time_ms": 500, "lane": 4, "end_time_ms": None}]

        notes = quantize_notes(timing_points, hit_objects)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["start_tick"], 48)
        self.assertIsNone(notes[0]["end_tick"])
        self.assertEqual(notes[0]["lane"], 4)

    def test_multiple_notes_same_tick(self):
        timing_points = [{"time_ms": 0, "beat_length": 500}]
        hit_objects = [
            {"time_ms": 500, "lane": 1, "end_time_ms": None},
            {"time_ms": 500, "lane": 7, "end_time_ms": None},
        ]

        notes = quantize_notes(timing_points, hit_objects)

        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]["start_tick"], 48)
        self.assertEqual(notes[1]["start_tick"], 48)
        self.assertEqual(notes[0]["lane"], 1)
        self.assertEqual(notes[1]["lane"], 7)
        self.assertIsNone(notes[0]["end_tick"])
        self.assertIsNone(notes[1]["end_tick"])

    def test_hold_note(self):
        timing_points = [{"time_ms": 0, "beat_length": 500}]
        hit_objects = [{"time_ms": 500, "lane": 3, "end_time_ms": 1000}]

        notes = quantize_notes(timing_points, hit_objects)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["start_tick"], 48)
        self.assertEqual(notes[0]["end_tick"], 96)
        self.assertEqual(notes[0]["lane"], 3)

    def test_bpm_change(self):
        timing_points = [
            {"time_ms": 0, "beat_length": 500},
            {"time_ms": 1000, "beat_length": 250},
        ]
        hit_objects = [
            {"time_ms": 500, "lane": 2, "end_time_ms": None},
            {"time_ms": 1250, "lane": 6, "end_time_ms": None},
        ]

        notes = quantize_notes(timing_points, hit_objects)

        self.assertEqual(len(notes), 2)
        # Before BPM change: (500-0)/500*48 = 48
        self.assertEqual(notes[0]["start_tick"], 48)
        self.assertEqual(notes[0]["lane"], 2)
        self.assertIsNone(notes[0]["end_tick"])
        # After BPM change: (1250-1000)/250*48 = 48
        self.assertEqual(notes[1]["start_tick"], 48)
        self.assertEqual(notes[1]["lane"], 6)
        self.assertIsNone(notes[1]["end_tick"])

    def test_rounding_behavior(self):
        timing_points = [{"time_ms": 0, "beat_length": 500}]
        hit_objects = [{"time_ms": 499, "lane": 5, "end_time_ms": None}]

        notes = quantize_notes(timing_points, hit_objects)

        self.assertEqual(len(notes), 1)
        # round((499/500)*48) = round(47.904) = 48
        self.assertEqual(notes[0]["start_tick"], 48)
        self.assertEqual(notes[0]["lane"], 5)
        self.assertIsNone(notes[0]["end_tick"])


if __name__ == "__main__":
    unittest.main()

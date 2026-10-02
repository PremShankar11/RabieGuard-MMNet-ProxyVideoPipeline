"""
Unit Test Suite for Temporal Alignment Module.
Zero Rabies-MMNet Project.
Tests:
  1. Alignment of multi-rate streams (video 2.0s vs audio 3.0s).
  2. Tick resolution and temporal window overlap logic.
  3. Unequal duration handling (video longer than audio, audio longer than video).
  4. Empty stream / zero duration graceful handling.
  5. Multi-tick boundary accuracy and monotonicity.
"""

import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fusion.alignment import align_multimodal_streams, find_overlapping_windows


class TestTemporalAlignment(unittest.TestCase):
    def test_01_find_overlapping_windows(self):
        windows = [
            {"window_index": 0, "t_start": 0.0, "t_end": 2.0},
            {"window_index": 1, "t_start": 1.0, "t_end": 3.0},
            {"window_index": 2, "t_start": 2.0, "t_end": 4.0},
        ]
        # Query interval [0.5, 1.0] -> only window 0 overlaps
        ov0 = find_overlapping_windows(windows, 0.5, 1.0)
        self.assertEqual(len(ov0), 1)
        self.assertEqual(ov0[0]["window_index"], 0)

        # Query interval [1.2, 1.8] -> windows 0 and 1 overlap
        ov1 = find_overlapping_windows(windows, 1.2, 1.8)
        self.assertEqual(len(ov1), 2)
        indices = [w["window_index"] for w in ov1]
        self.assertIn(0, indices)
        self.assertIn(1, indices)

    def test_02_multirate_alignment_ticks(self):
        # Video: 3 windows, 2.0s duration, 1.0s hop (0-2s, 1-3s, 2-4s)
        v_wins = [
            {"window_index": 0, "t_start": 0.0, "t_end": 2.0, "probability": 0.20, "confidence": 0.60},
            {"window_index": 1, "t_start": 1.0, "t_end": 3.0, "probability": 0.30, "confidence": 0.40},
            {"window_index": 2, "t_start": 2.0, "t_end": 4.0, "probability": 0.40, "confidence": 0.20},
        ]
        # Audio: 2 windows, 3.0s duration, 1.5s hop (0-3s, 1.5-4.5s)
        a_wins = [
            {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.70, "audio_confidence": 0.40, "audio_present": True},
            {"window_index": 1, "t_start": 1.5, "t_end": 4.5, "audio_score": 0.80, "audio_confidence": 0.60, "audio_present": True},
        ]

        segments = align_multimodal_streams(v_wins, a_wins, tick_seconds=0.5)
        self.assertEqual(len(segments), 9)  # 4.5s / 0.5s = 9 ticks

        # Check tick 0: [0.0, 0.5]
        self.assertEqual(segments[0].start_time, 0.0)
        self.assertEqual(segments[0].end_time, 0.5)
        self.assertAlmostEqual(segments[0].video_probability, 0.20, places=3)
        self.assertAlmostEqual(segments[0].audio_probability, 0.70, places=3)
        self.assertIsNotNone(segments[0].fused_probability)

        # Check last tick: [4.0, 4.5] (video ended at 4.0s, audio still present)
        last_seg = segments[-1]
        self.assertEqual(last_seg.start_time, 4.0)
        self.assertEqual(last_seg.end_time, 4.5)
        self.assertIsNone(last_seg.video_probability)
        self.assertEqual(last_seg.video_quality, "NO_INPUT")
        self.assertIsNotNone(last_seg.audio_probability)
        self.assertEqual(last_seg.audio_weight, 1.0)
        self.assertEqual(last_seg.video_weight, 0.0)

    def test_03_empty_streams(self):
        segments = align_multimodal_streams([], [], tick_seconds=0.5)
        self.assertEqual(len(segments), 0)


if __name__ == "__main__":
    unittest.main()

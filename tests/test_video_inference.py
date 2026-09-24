"""
Automated Test Suite for Video V2 Inference Pipeline.
Zero Rabies-MMNet Project.

Verifies:
1. Short video (< 16 frames, temporal padding)
2. Longer video (multiple temporal windows)
3. Multi-dog scene (ambiguity tracking & warning)
4. Corrupted / invalid input (graceful rejection)
5. Video with no detectable canine pose (informative error)
6. Checkpoint integrity & schema conformity (136,257 params, final.pt)
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.video.inference import VideoBehaviorPipeline, InferenceError
from src.video.preprocessing import (
    VideoReadError,
    VideoValidationError,
    validate_video_file,
    get_video_metadata
)


class TestVideoV2Inference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("INITIALIZING VIDEO V2 INFERENCE TEST SUITE")
        print("=" * 60)
        cls.pipeline = VideoBehaviorPipeline()
        cls.raw_video_dir = Path("data/raw/animal_kingdom/video")
        cls.test_out_dir = Path("outputs/inference/tests")
        cls.test_out_dir.mkdir(parents=True, exist_ok=True)

    def test_01_frozen_checkpoint_integrity(self):
        """Verify the loaded model is strictly the frozen V2 checkpoint."""
        ckpt_path = Path(self.pipeline.mamba_classifier.checkpoint_path)
        self.assertTrue(ckpt_path.exists(), "Frozen checkpoint does not exist on disk!")
        self.assertEqual(ckpt_path.name, "final.pt")
        self.assertIn("mamba_behavior_v2", str(ckpt_path))

        meta = self.pipeline.mamba_classifier.metadata
        self.assertEqual(meta["trainable_params"], 136257, "Parameter count mismatch! Expected 136,257")
        self.assertEqual(meta["in_features"], 75, "Expected 75 in_features (72 pose + 3 reliability)")
        self.assertEqual(meta["winner_name"], "exp_d_temporal_tracking")

    def test_02_short_video_with_padding(self):
        """Test on a short development clip (YSWGBGCS: 0.42s, < 16 frames)."""
        video_path = self.raw_video_dir / "YSWGBGCS.mp4"
        self.assertTrue(video_path.exists(), f"Missing test clip: {video_path}")

        result = self.pipeline.process_video(
            video_path=video_path,
            save_json=True,
            output_dir=self.test_out_dir
        )

        # Schema checks
        self.assertIn("behavior", result)
        self.assertIn("probability_agitated", result)
        self.assertIn("behavioral_score", result)
        self.assertIn("score_band", result)
        self.assertIn("quality", result)
        self.assertIn("video", result)
        self.assertIn("temporal", result)
        self.assertIn("model", result)

        # Values within range
        self.assertIn(result["behavior"], ["CALM", "AGITATED"])
        self.assertGreaterEqual(result["probability_agitated"], 0.0)
        self.assertLessEqual(result["probability_agitated"], 1.0)
        self.assertGreaterEqual(result["behavioral_score"], 0)
        self.assertLessEqual(result["behavioral_score"], 100)

        # Padded window behavior
        self.assertGreater(result["quality"]["padded_window_percentage"], 0.0)
        self.assertEqual(result["temporal"]["num_windows"], 1)

    def test_03_longer_video_multiple_windows(self):
        """Test on a longer development clip (AWJEUGCS: ~6.6s, multiple 16-frame windows)."""
        video_path = self.raw_video_dir / "AWJEUGCS.mp4"
        self.assertTrue(video_path.exists(), f"Missing test clip: {video_path}")

        result = self.pipeline.process_video(
            video_path=video_path,
            save_json=True,
            output_dir=self.test_out_dir
        )

        self.assertGreater(result["temporal"]["num_windows"], 1, "Expected multiple temporal windows for long video")
        self.assertEqual(len(result["temporal"]["window_probabilities"]), result["temporal"]["num_windows"])

        for p in result["temporal"]["window_probabilities"]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # Quality metrics populated
        self.assertIn(result["quality"]["overall"], ["GOOD", "FAIR", "LIMITED"])
        self.assertGreater(result["quality"]["valid_frame_percentage"], 50.0)

    def test_04_multi_dog_scene(self):
        """Test on a development clip with documented multi-dog presence (EUKRNXGD: 40 ambiguous frames)."""
        video_path = self.raw_video_dir / "EUKRNXGD.mp4"
        self.assertTrue(video_path.exists(), f"Missing test clip: {video_path}")

        result = self.pipeline.process_video(
            video_path=video_path,
            save_json=True,
            output_dir=self.test_out_dir
        )

        self.assertGreater(result["quality"]["ambiguous_frame_percentage"], 0.0)
        self.assertIn(result["quality"]["ambiguity_level"], ["LOW", "MODERATE", "HIGH"])
        self.assertIn(result["behavior"], ["CALM", "AGITATED"])

    def test_05_corrupted_or_invalid_video(self):
        """Verify pipeline rejects corrupted or non-video files gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"NOT_A_REAL_VIDEO_HEADER_DATA_GARBAGE")
            temp_path = tf.name

        try:
            with self.assertRaises((VideoReadError, VideoValidationError, InferenceError)):
                self.pipeline.process_video(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_06_video_with_no_dogs(self):
        """Verify pipeline handles video with no canine pose detected without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            temp_path = tf.name

        try:
            # Create a 2-second blank video (black frames)
            w, h, fps = 640, 360, 24
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
            for _ in range(48):
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                out.write(blank)
            out.release()

            # Pipeline should raise InferenceError explaining no canine pose was found
            with self.assertRaises(InferenceError) as ctx:
                self.pipeline.process_video(temp_path)

            self.assertIn("No usable dog pose was detected", str(ctx.exception))
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

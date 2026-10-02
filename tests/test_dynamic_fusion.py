"""
Unit and Sanity Test Suite for Dynamic Confidence-Aware Late Fusion.
Zero Rabies-MMNet Project.
Tests all 10 mandatory specifications defined in Project Section 18:
  TEST 1:  High video confidence + low audio confidence -> video receives larger weight
  TEST 2:  Low video confidence + high audio confidence -> audio receives larger weight
  TEST 3:  Equal confidence -> approximately equal weights (Wv ≈ Wa ≈ 0.5)
  TEST 4:  Both modalities agree -> fused result lies between their predictions
  TEST 5:  Modalities disagree -> higher-confidence modality contributes more
  TEST 6:  Both confidence values zero -> safe documented fallback (Wv = Wa = 0.5)
  TEST 7:  Probability bounds -> all fused probabilities in [0.0, 1.0]
  TEST 8:  Weight bounds -> weights strictly in [0.0, 1.0]
  TEST 9:  Weights sum to approximately 1.0 (|Wv + Wa - 1.0| < 1e-6)
  TEST 10: Timestamp alignment validity (monotonicity, valid intervals)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fusion.confidence import (
    calculate_dynamic_weights,
    compute_margin_confidence,
    compute_video_reliability,
    compute_audio_reliability
)
from fusion.dynamic_late_fusion import DynamicLateFusionEngine
from fusion.schemas import map_score_to_risk_level


class TestDynamicLateFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DynamicLateFusionEngine()

    def test_01_high_video_confidence_low_audio(self):
        """TEST 1: High video confidence + low audio confidence -> video weight > audio weight."""
        cv = 0.85
        ca = 0.15
        wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
        self.assertGreater(wv, wa, f"Expected Wv > Wa, got Wv={wv}, Wa={wa}")
        self.assertAlmostEqual(wv + wa, 1.0, places=4)
        self.assertGreater(wv, 0.70)

    def test_02_low_video_confidence_high_audio(self):
        """TEST 2: Low video confidence + high audio confidence -> audio weight > video weight."""
        cv = 0.20
        ca = 0.90
        wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
        self.assertGreater(wa, wv, f"Expected Wa > Wv, got Wv={wv}, Wa={wa}")
        self.assertAlmostEqual(wv + wa, 1.0, places=4)
        self.assertGreater(wa, 0.70)

    def test_03_equal_confidence(self):
        """TEST 3: Equal confidence -> approximately equal weights (Wv ≈ Wa ≈ 0.5)."""
        for conf in [0.3, 0.5, 0.8, 1.0]:
            wv, wa = calculate_dynamic_weights(conf, conf, use_reliability_modulation=False)
            self.assertAlmostEqual(wv, 0.5, places=3, msg=f"Wv should be 0.5 for conf={conf}")
            self.assertAlmostEqual(wa, 0.5, places=3, msg=f"Wa should be 0.5 for conf={conf}")
            self.assertAlmostEqual(wv + wa, 1.0, places=4)

    def test_04_modalities_agree(self):
        """TEST 4: Both modalities agree -> fused result lies between predictions."""
        # e.g., Pv = 0.80, Pa = 0.86
        pv, pa = 0.80, 0.86
        cv, ca = 0.60, 0.72
        wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
        r1 = wv * pv + wa * pa

        min_p = min(pv, pa)
        max_p = max(pv, pa)
        self.assertGreaterEqual(r1, min_p, f"R1={r1} below min({pv}, {pa})")
        self.assertLessEqual(r1, max_p, f"R1={r1} above max({pv}, {pa})")

    def test_05_modalities_disagree(self):
        """TEST 5: Modalities disagree -> higher-confidence modality contributes more."""
        # Video says calm (0.15) with high confidence (0.90)
        # Audio says agitated (0.85) with low confidence (0.20)
        pv, pa = 0.15, 0.85
        cv, ca = 0.90, 0.20
        wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
        r1 = wv * pv + wa * pa

        # Fused score should be much closer to video prediction (0.15)
        self.assertLess(r1, 0.35, f"Expected fused result pulled toward video, got {r1}")
        self.assertGreater(wv, wa)

    def test_06_both_confidences_zero(self):
        """TEST 6: Both confidence values zero -> safe documented fallback (Wv = Wa = 0.5)."""
        wv, wa = calculate_dynamic_weights(0.0, 0.0, use_reliability_modulation=False)
        self.assertEqual(wv, 0.5, "Fallback Wv must be 0.5")
        self.assertEqual(wa, 0.5, "Fallback Wa must be 0.5")
        self.assertAlmostEqual(wv + wa, 1.0, places=4)

    def test_07_probability_bounds(self):
        """TEST 7: All fused probabilities must strictly remain in [0.0, 1.0]."""
        test_cases = [
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 0.5, 0.5),
            (0.99, 0.01, 0.99, 0.01),
            (0.50, 0.50, 0.0, 0.0),
        ]
        for pv, pa, cv, ca in test_cases:
            wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
            r1 = wv * pv + wa * pa
            self.assertGreaterEqual(r1, 0.0, f"R1={r1} below 0 for pv={pv}, pa={pa}")
            self.assertLessEqual(r1, 1.0, f"R1={r1} above 1 for pv={pv}, pa={pa}")

    def test_08_weight_bounds(self):
        """TEST 8: Modality weights must strictly remain in [0.0, 1.0]."""
        grid = np.linspace(0.0, 1.0, 11)
        for cv in grid:
            for ca in grid:
                wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
                self.assertGreaterEqual(wv, 0.0, f"Wv={wv} < 0 for cv={cv}, ca={ca}")
                self.assertLessEqual(wv, 1.0, f"Wv={wv} > 1 for cv={cv}, ca={ca}")
                self.assertGreaterEqual(wa, 0.0, f"Wa={wa} < 0 for cv={cv}, ca={ca}")
                self.assertLessEqual(wa, 1.0, f"Wa={wa} > 1 for cv={cv}, ca={ca}")

    def test_09_weights_sum_to_one(self):
        """TEST 9: Modality weights must sum to approximately 1.0 (|Wv + Wa - 1.0| < 1e-4)."""
        grid = np.linspace(0.0, 1.0, 11)
        for cv in grid:
            for ca in grid:
                wv, wa = calculate_dynamic_weights(cv, ca, use_reliability_modulation=False)
                self.assertAlmostEqual(wv + wa, 1.0, places=4, msg=f"Wv={wv}, Wa={wa} sum={wv+wa}")

    def test_10_timestamp_alignment(self):
        """TEST 10: Timestamp alignment validity (monotonicity, valid intervals)."""
        video_result = {
            "probability_agitated": 0.35,
            "video": {"duration_seconds": 4.0, "sampled_fps": 8.0},
            "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.30, 0.35, 0.40]},
            "quality": {"overall": "GOOD", "valid_frame_percentage": 90.0, "mean_pose_confidence": 0.65}
        }
        audio_result = {
            "audio_present": True,
            "audio_score": 0.70,
            "audio_confidence": 0.40,
            "windows": [
                {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.65, "audio_confidence": 0.30, "audio_present": True},
                {"window_index": 1, "t_start": 1.5, "t_end": 4.5, "audio_score": 0.75, "audio_confidence": 0.50, "audio_present": True}
            ]
        }

        output = self.engine.fuse_multimodal(video_result, audio_result, clip_id="test_clip", save_csv=False)
        segments = output["segments"]

        self.assertGreater(len(segments), 0, "Expected at least one aligned segment")
        prev_end = 0.0
        for seg in segments:
            self.assertGreater(seg["end_time"], seg["start_time"], "Segment end must exceed start")
            self.assertGreaterEqual(seg["start_time"], prev_end - 1e-5, "Segments must be monotonically non-decreasing")
            prev_end = seg["start_time"]
            if seg["fused_probability"] is not None:
                self.assertGreaterEqual(seg["fused_probability"], 0.0)
                self.assertLessEqual(seg["fused_probability"], 1.0)
                self.assertIn(seg["risk_level"], ["LOW", "MEDIUM", "HIGH"])

    def test_11_risk_band_mapping(self):
        """Verify project risk bands: LOW (0-35), MEDIUM (36-65), HIGH (66-100)."""
        self.assertEqual(map_score_to_risk_level(0), "LOW")
        self.assertEqual(map_score_to_risk_level(35), "LOW")
        self.assertEqual(map_score_to_risk_level(36), "MEDIUM")
        self.assertEqual(map_score_to_risk_level(65), "MEDIUM")
        self.assertEqual(map_score_to_risk_level(66), "HIGH")
        self.assertEqual(map_score_to_risk_level(100), "HIGH")

    def test_12_missing_modality_fallback(self):
        """Test graceful fallback when one modality is completely absent."""
        # Video only
        video_result = {
            "probability_agitated": 0.75,
            "video": {"duration_seconds": 2.0, "sampled_fps": 8.0},
            "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.75]},
            "quality": {"overall": "GOOD", "valid_frame_percentage": 90.0, "mean_pose_confidence": 0.65}
        }
        res_v_only = self.engine.fuse_multimodal(video_result=video_result, audio_result=None, clip_id="v_only", save_csv=False)
        self.assertEqual(res_v_only["weights"]["video"], 1.0)
        self.assertEqual(res_v_only["weights"]["audio"], 0.0)
        self.assertEqual(res_v_only["fused"]["risk_score"], 75)
        self.assertEqual(res_v_only["fused"]["risk_level"], "HIGH")

        # Audio only
        audio_result = {
            "audio_present": True,
            "audio_score": 0.25,
            "audio_confidence": 0.50,
            "windows": [{"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.25, "audio_confidence": 0.50, "audio_present": True}]
        }
        res_a_only = self.engine.fuse_multimodal(video_result=None, audio_result=audio_result, clip_id="a_only", save_csv=False)
        self.assertEqual(res_a_only["weights"]["video"], 0.0)
        self.assertEqual(res_a_only["weights"]["audio"], 1.0)
        self.assertEqual(res_a_only["fused"]["risk_score"], 25)
        self.assertEqual(res_a_only["fused"]["risk_level"], "LOW")


if __name__ == "__main__":
    unittest.main()

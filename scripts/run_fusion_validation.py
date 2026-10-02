"""
Synthetic Multimodal Validation Script for Zero Rabies-MMNet.
Audits paired dataset availability and evaluates the Dynamic Confidence Late Fusion engine
across standardized behavioral and sensory condition scenarios.
Generates:
  - outputs/fusion/dynamic_late_fusion_segments.csv
  - outputs/fusion/dynamic_late_fusion_results.csv
  - outputs/fusion/synthetic_multimodal_scenarios.json
"""

import sys
import json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fusion.dynamic_late_fusion import DynamicLateFusionEngine
from fusion.schemas import map_score_to_risk_level


def build_synthetic_scenarios() -> list:
    """
    Construct standardized synthetic paired scenarios reflecting operational canine field challenges.
    """
    scenarios = [
        {
            "clip_id": "SYNTH_01_CALM_RESTING_SILENT",
            "description": "Dog resting/sleeping. Clear pose visibility, silent ambient audio.",
            "expected_outcome": "Video dominates (audio silent), resulting in LOW risk level.",
            "video": {
                "probability_agitated": 0.12,
                "video": {"duration_seconds": 4.0, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.10, 0.12, 0.14]},
                "quality": {
                    "overall": "GOOD",
                    "valid_frame_percentage": 96.0,
                    "mean_pose_confidence": 0.78,
                    "ambiguous_frame_percentage": 0.0
                }
            },
            "audio": {
                "audio_present": False,
                "audio_score": None,
                "audio_confidence": 0.0,
                "audio_reliability": 0.0,
                "energy_db": -58.4,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": None, "audio_confidence": 0.0, "audio_reliability": 0.0, "audio_present": False, "energy_db": -58.0},
                    {"window_index": 1, "t_start": 1.5, "t_end": 4.0, "audio_score": None, "audio_confidence": 0.0, "audio_reliability": 0.0, "audio_present": False, "energy_db": -59.0}
                ]
            }
        },
        {
            "clip_id": "SYNTH_02_HIGH_AGITATION_CONCORDANT",
            "description": "Rapid locomotion/bounding accompanied by repeated barking/growling.",
            "expected_outcome": "Both modalities strongly agree with high confidence -> HIGH risk level.",
            "video": {
                "probability_agitated": 0.88,
                "video": {"duration_seconds": 4.5, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.85, 0.88, 0.92, 0.87]},
                "quality": {
                    "overall": "GOOD",
                    "valid_frame_percentage": 92.0,
                    "mean_pose_confidence": 0.74,
                    "ambiguous_frame_percentage": 5.0
                }
            },
            "audio": {
                "audio_present": True,
                "audio_score": 0.84,
                "audio_confidence": 0.68,
                "audio_reliability": 0.85,
                "energy_db": -22.3,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.82, "audio_confidence": 0.64, "audio_reliability": 0.85, "audio_present": True, "energy_db": -22.0},
                    {"window_index": 1, "t_start": 1.5, "t_end": 4.5, "audio_score": 0.86, "audio_confidence": 0.72, "audio_reliability": 0.85, "audio_present": True, "energy_db": -22.6}
                ]
            }
        },
        {
            "clip_id": "SYNTH_03_OCCLUDED_VIDEO_VOCAL_DISTRESS",
            "description": "Severe visual occlusion / multi-dog ambiguity, but clear acoustic aggressive barking.",
            "expected_outcome": "Dynamic fusion downweights degraded video and trusts audio -> HIGH risk level.",
            "video": {
                "probability_agitated": 0.58,
                "video": {"duration_seconds": 4.5, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.55, 0.60, 0.58]},
                "quality": {
                    "overall": "LIMITED",
                    "valid_frame_percentage": 42.0,
                    "mean_pose_confidence": 0.32,
                    "ambiguous_frame_percentage": 65.0
                }
            },
            "audio": {
                "audio_present": True,
                "audio_score": 0.92,
                "audio_confidence": 0.84,
                "audio_reliability": 0.90,
                "energy_db": -18.5,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.90, "audio_confidence": 0.80, "audio_reliability": 0.90, "audio_present": True, "energy_db": -18.2},
                    {"window_index": 1, "t_start": 1.5, "t_end": 4.5, "audio_score": 0.94, "audio_confidence": 0.88, "audio_reliability": 0.90, "audio_present": True, "energy_db": -18.8}
                ]
            }
        },
        {
            "clip_id": "SYNTH_04_ACOUSTIC_NOISE_CLEAR_MOTION",
            "description": "Acoustic background noise / near-boundary audio, but clear visual running locomotion.",
            "expected_outcome": "Dynamic fusion downweights ambiguous audio and trusts video -> HIGH risk level.",
            "video": {
                "probability_agitated": 0.82,
                "video": {"duration_seconds": 4.0, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.80, 0.82, 0.84]},
                "quality": {
                    "overall": "GOOD",
                    "valid_frame_percentage": 94.0,
                    "mean_pose_confidence": 0.72,
                    "ambiguous_frame_percentage": 0.0
                }
            },
            "audio": {
                "audio_present": True,
                "audio_score": 0.52,
                "audio_confidence": 0.04,
                "audio_reliability": 0.40,
                "energy_db": -38.0,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.51, "audio_confidence": 0.02, "audio_reliability": 0.40, "audio_present": True, "energy_db": -38.0},
                    {"window_index": 1, "t_start": 1.5, "t_end": 4.0, "audio_score": 0.53, "audio_confidence": 0.06, "audio_reliability": 0.40, "audio_present": True, "energy_db": -38.0}
                ]
            }
        },
        {
            "clip_id": "SYNTH_05_TRANSITIONAL_BEHAVIOR",
            "description": "Canine walking/trotting with low-level whining. Transitional arousal state.",
            "expected_outcome": "Fused score reflects transitional behavioral dynamics -> MEDIUM risk level.",
            "video": {
                "probability_agitated": 0.45,
                "video": {"duration_seconds": 4.0, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.42, 0.45, 0.48]},
                "quality": {
                    "overall": "FAIR",
                    "valid_frame_percentage": 78.0,
                    "mean_pose_confidence": 0.54,
                    "ambiguous_frame_percentage": 15.0
                }
            },
            "audio": {
                "audio_present": True,
                "audio_score": 0.55,
                "audio_confidence": 0.10,
                "audio_reliability": 0.55,
                "energy_db": -32.0,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.54, "audio_confidence": 0.08, "audio_reliability": 0.55, "audio_present": True, "energy_db": -32.0},
                    {"window_index": 1, "t_start": 1.5, "t_end": 4.0, "audio_score": 0.56, "audio_confidence": 0.12, "audio_reliability": 0.55, "audio_present": True, "energy_db": -32.0}
                ]
            }
        },
        {
            "clip_id": "SYNTH_06_DISCORDANT_EQUAL_CONFIDENCE",
            "description": "Video indicates calm (0.25, conf 0.50), Audio indicates agitation (0.75, conf 0.50).",
            "expected_outcome": "Equal weighting (Wv=0.5, Wa=0.5) balances predictions -> MEDIUM risk score (50).",
            "video": {
                "probability_agitated": 0.25,
                "video": {"duration_seconds": 3.0, "sampled_fps": 8.0},
                "temporal": {"window_length": 16, "stride": 8, "window_probabilities": [0.25, 0.25]},
                "quality": {
                    "overall": "GOOD",
                    "valid_frame_percentage": 90.0,
                    "mean_pose_confidence": 0.70,
                    "ambiguous_frame_percentage": 0.0
                }
            },
            "audio": {
                "audio_present": True,
                "audio_score": 0.75,
                "audio_confidence": 0.50,
                "audio_reliability": 0.70,
                "energy_db": -25.0,
                "windows": [
                    {"window_index": 0, "t_start": 0.0, "t_end": 3.0, "audio_score": 0.75, "audio_confidence": 0.50, "audio_reliability": 0.70, "audio_present": True, "energy_db": -25.0}
                ]
            }
        }
    ]
    return scenarios


def main():
    print("=" * 70)
    print("ZERO RABIES-MMNET — DYNAMIC LATE FUSION VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Dataset Audit
    raw_v_dir = repo_root / "data" / "raw" / "animal_kingdom" / "video"
    raw_a_dir = repo_root / "Audio pipeline" / "Bark_vs_Nobark"
    videos_found = list(raw_v_dir.glob("*.mp4")) if raw_v_dir.exists() else []
    audios_found = list(raw_a_dir.glob("*.wav")) if raw_a_dir.exists() else []

    print("\n--- Dataset Pairing Audit ---")
    print(f"Raw Animal Kingdom Video Clips Found: {len(videos_found)}")
    print(f"Bioacoustic Audio Clips Found       : {len(audios_found)}")
    print("Genuinely Paired Audio-Video Exists : NO")
    print("Status: Animal Kingdom video and Bioacoustic audio represent independent,")
    print("        unpaired collections. In strict accordance with Project Section 19,")
    print("        no manufactured pairing is fabricated, and multimodal performance")
    print("        metrics are evaluated on rigorous synthetic paired test scenarios.")

    # 2. Initialize Engine
    out_dir = repo_root / "outputs" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing CSV logs for clean regeneration
    for csv_name in ["dynamic_late_fusion_segments.csv", "dynamic_late_fusion_results.csv"]:
        f_p = out_dir / csv_name
        if f_p.exists():
            f_p.unlink()

    engine = DynamicLateFusionEngine(output_dir=out_dir)

    scenarios = build_synthetic_scenarios()
    print(f"\nExecuting {len(scenarios)} Standardized Synthetic Paired Scenarios...")

    results = []
    print("\n" + "-" * 90)
    print(f"{'Clip ID':<35} | {'V-Score':<7} | {'A-Score':<7} | {'W_v':<6} | {'W_a':<6} | {'Fused':<6} | {'Risk Level':<8}")
    print("-" * 90)

    for sc in scenarios:
        res = engine.fuse_multimodal(
            video_result=sc["video"],
            audio_result=sc["audio"],
            clip_id=sc["clip_id"],
            save_csv=True
        )
        results.append({
            "scenario": sc["clip_id"],
            "description": sc["description"],
            "expected_outcome": sc["expected_outcome"],
            "output": res
        })

        v_s = res["video"]["score"] if res["video"]["score"] is not None else "N/A"
        a_s = res["audio"]["score"] if res["audio"]["score"] is not None else "N/A"
        wv = res["weights"]["video"]
        wa = res["weights"]["audio"]
        f_s = res["fused"]["risk_score"]
        r_l = res["fused"]["risk_level"]

        print(f"{sc['clip_id']:<35} | {str(v_s):<7} | {str(a_s):<7} | {wv:<6.2f} | {wa:<6.2f} | {f_s:<6} | {r_l:<8}")

    print("-" * 90)

    # Save detailed scenario log
    scenario_json_path = out_dir / "synthetic_multimodal_scenarios.json"
    with open(scenario_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Automated Reconciliation Audit & Assertion Check
    print("\n--- Artifact Reconciliation & Integrity Audit ---")
    results_csv = out_dir / "dynamic_late_fusion_results.csv"
    segments_csv = out_dir / "dynamic_late_fusion_segments.csv"

    import csv
    with open(results_csv, "r", encoding="utf-8") as f:
        res_rows = list(csv.reader(f))[1:]  # Skip header
    with open(segments_csv, "r", encoding="utf-8") as f:
        seg_rows = list(csv.reader(f))[1:]  # Skip header

    # Verify clip-level uniqueness
    clip_ids = [r[0] for r in res_rows]
    unique_clip_ids = set(clip_ids)
    print(f"Results CSV Data Rows          : {len(res_rows)} (Expected: {len(scenarios)})")
    print(f"Unique Clip IDs in Results CSV : {len(unique_clip_ids)} (Expected: {len(scenarios)})")
    assert len(res_rows) == len(scenarios), f"Row count mismatch in results CSV: {len(res_rows)} != {len(scenarios)}"
    assert len(unique_clip_ids) == len(scenarios), f"Duplicate clip_ids found: {clip_ids}"

    # Verify segment count integrity
    total_json_segments = sum(len(s["output"]["segments"]) for s in results)
    print(f"Segments in Synthetic JSON     : {total_json_segments}")
    print(f"Segments in CSV Log            : {len(seg_rows)}")
    assert len(seg_rows) == total_json_segments, f"Segment count mismatch: CSV has {len(seg_rows)}, JSON has {total_json_segments}"
    assert len(seg_rows) == 49, f"Expected exactly 49 segments across 6 scenarios, got {len(seg_rows)}"

    print("Segment Breakdown per Scenario :")
    for s in results:
        sc_id = s["scenario"]
        sc_segs = len(s["output"]["segments"])
        print(f"  • {sc_id:<36}: {sc_segs} segments")

    print(f"\nSaved synthetic scenarios log: {scenario_json_path}")
    print(f"Saved segment CSV              : {segments_csv}")
    print(f"Saved results CSV              : {results_csv}")
    print("Reconciliation Audit: PASSED (All artifact counts agree, zero duplicates).")


if __name__ == "__main__":
    main()

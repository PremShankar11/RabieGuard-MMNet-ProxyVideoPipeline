"""
Dynamic Confidence-Aware Late Fusion Engine for Zero Rabies-MMNet.
Coordinates frozen Video V2 and Audio V2 branches, performs synchronized temporal alignment,
calculates dynamic confidence weights, evaluates baseline comparisons, and exports CSV reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import csv
import json
import numpy as np

from .schemas import (
    AlignedSegment,
    DynamicLateFusionOutput,
    map_score_to_risk_level
)
from .confidence import (
    compute_margin_confidence,
    compute_video_reliability,
    compute_audio_reliability,
    calculate_dynamic_weights
)
from .alignment import align_multimodal_streams


class DynamicLateFusionEngine:
    """
    Core implementation of Approach 1: Dynamic Confidence-Aware Late Fusion.
    Operates strictly above frozen Video V2 and Audio V2 pipelines.
    """
    def __init__(
        self,
        tick_seconds: float = 0.5,
        default_video_quality: str = "GOOD",
        output_dir: Optional[Union[str, Path]] = None
    ):
        self.tick_seconds = tick_seconds
        self.default_video_quality = default_video_quality
        self.output_dir = Path(output_dir or "outputs/fusion")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fuse_multimodal(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        clip_id: str = "multimodal_clip",
        save_csv: bool = True
    ) -> Dict[str, Any]:
        """
        Fuse unimodal video and audio results into an authoritative multimodal risk assessment.
        """
        # 1. Parse Video Stream
        video_windows = []
        video_quality_overall = "GOOD"
        v_prob_clip = None
        v_conf_clip = 0.0
        v_rel_clip = 0.0
        v_present = False

        if video_result:
            v_present = True
            v_prob_clip = float(video_result.get("probability_agitated", 0.5))
            v_quality_dict = video_result.get("quality", {})
            video_quality_overall = v_quality_dict.get("overall", self.default_video_quality)
            v_rel_clip = compute_video_reliability(v_quality_dict)
            v_conf_clip = compute_margin_confidence(v_prob_clip)

            # Extract window-level video data
            # Check if scored_windows or window_probabilities are provided
            temporal_meta = video_result.get("temporal", {})
            win_probs = temporal_meta.get("window_probabilities", [])
            stride = temporal_meta.get("stride", 8)
            win_len = temporal_meta.get("window_length", 16)
            fps = float(video_result.get("video", {}).get("sampled_fps", 8.0))

            if win_probs:
                for idx, wp in enumerate(win_probs):
                    s_frame = idx * stride
                    e_frame = s_frame + win_len
                    t_s = round(float(s_frame / fps), 3)
                    t_e = round(float(e_frame / fps), 3)
                    w_conf = compute_margin_confidence(wp)
                    video_windows.append({
                        "window_index": idx,
                        "t_start": t_s,
                        "t_end": t_e,
                        "probability": float(wp),
                        "confidence": w_conf,
                        "reliability": v_rel_clip,
                        "quality": video_quality_overall
                    })
            elif v_prob_clip is not None:
                # Single clip window fallback
                dur = float(video_result.get("video", {}).get("duration_seconds", 2.0))
                video_windows.append({
                    "window_index": 0,
                    "t_start": 0.0,
                    "t_end": dur,
                    "probability": v_prob_clip,
                    "confidence": v_conf_clip,
                    "reliability": v_rel_clip,
                    "quality": video_quality_overall
                })

        # 2. Parse Audio Stream
        audio_windows = []
        a_prob_clip = None
        a_conf_clip = 0.0
        a_rel_clip = 0.0
        a_present = False

        if audio_result:
            a_present = bool(audio_result.get("audio_present", False))
            a_score_val = audio_result.get("audio_score")
            a_prob_clip = float(a_score_val) if a_score_val is not None else None
            a_conf_clip = float(audio_result.get("audio_confidence", 0.0))
            a_rel_clip = float(audio_result.get("audio_reliability", 1.0 if a_present else 0.0))

            # Extract window-level audio data
            for w in audio_result.get("windows", []):
                audio_windows.append({
                    "window_index": w.get("window_index", 0),
                    "t_start": float(w.get("t_start", 0.0)),
                    "t_end": float(w.get("t_end", 3.0)),
                    "audio_score": float(w["audio_score"]) if w.get("audio_score") is not None else None,
                    "audio_confidence": float(w.get("audio_confidence", 0.0)),
                    "audio_reliability": float(w.get("audio_reliability", 0.0)),
                    "audio_present": bool(w.get("audio_present", True)),
                    "energy_db": float(w.get("energy_db", -60.0))
                })

        # 3. Synchronized Temporal Alignment
        aligned_segments = align_multimodal_streams(
            video_windows=video_windows,
            audio_windows=audio_windows,
            tick_seconds=self.tick_seconds,
            video_quality_overall=video_quality_overall,
            clip_id=clip_id
        )

        # 4. Clip-Level Dynamic Weights Calculation
        wv, wa = calculate_dynamic_weights(
            video_confidence=v_conf_clip,
            audio_confidence=a_conf_clip,
            video_present=v_present,
            audio_present=a_present,
            video_reliability=v_rel_clip,
            audio_reliability=a_rel_clip
        )

        # 5. Clip-Level Fusion (R1 = Wv * Pv + Wa * Pa)
        if v_prob_clip is not None and a_prob_clip is not None:
            r1 = float(wv * v_prob_clip + wa * a_prob_clip)
        elif v_prob_clip is not None:
            r1 = float(v_prob_clip)
            wv, wa = 1.0, 0.0
        elif a_prob_clip is not None:
            r1 = float(a_prob_clip)
            wv, wa = 0.0, 1.0
        else:
            r1 = 0.0
            wv, wa = 0.5, 0.5

        r1 = max(0.0, min(1.0, r1))
        multimodal_risk_score = int(round(r1 * 100))
        risk_level = map_score_to_risk_level(multimodal_risk_score)

        # 6. Temporal bounds
        t_start = aligned_segments[0].start_time if aligned_segments else 0.0
        t_end = aligned_segments[-1].end_time if aligned_segments else (
            float(video_result.get("video", {}).get("duration_seconds", 0.0)) if video_result else 0.0
        )

        # 7. Baseline Comparisons (Video Only vs Audio Only vs Dynamic Late Fusion)
        v_score = int(round(v_prob_clip * 100)) if v_prob_clip is not None else None
        a_score = int(round(a_prob_clip * 100)) if a_prob_clip is not None else None

        baselines = {
            "video_only": {
                "probability": round(v_prob_clip, 4) if v_prob_clip is not None else None,
                "score": v_score,
                "confidence": round(v_conf_clip, 4),
                "risk_level": map_score_to_risk_level(v_score) if v_score is not None else None,
                "quality": video_quality_overall
            },
            "audio_only": {
                "probability": round(a_prob_clip, 4) if a_prob_clip is not None else None,
                "score": a_score,
                "confidence": round(a_conf_clip, 4),
                "risk_level": map_score_to_risk_level(a_score) if a_score is not None else None,
                "quality": "PRESENT" if a_present else "SILENT"
            },
            "dynamic_late_fusion": {
                "risk_probability": round(r1, 4),
                "risk_score": multimodal_risk_score,
                "risk_level": risk_level,
                "video_weight": round(wv, 4),
                "audio_weight": round(wa, 4)
            }
        }

        # 8. Assemble Authoritative Output (Section 15 schema)
        output = {
            "fusion_method": "dynamic_confidence_late_fusion",
            "clip_id": clip_id,
            "video": {
                "probability": round(v_prob_clip, 4) if v_prob_clip is not None else None,
                "score": v_score,
                "confidence": round(v_conf_clip, 4),
                "quality": video_quality_overall
            },
            "audio": {
                "probability": round(a_prob_clip, 4) if a_prob_clip is not None else None,
                "score": a_score,
                "confidence": round(a_conf_clip, 4)
            },
            "weights": {
                "video": round(wv, 4),
                "audio": round(wa, 4)
            },
            "fused": {
                "risk_probability": round(r1, 4),
                "risk_score": multimodal_risk_score,
                "risk_level": risk_level
            },
            "temporal": {
                "start_time": round(t_start, 2),
                "end_time": round(t_end, 2)
            },
            "baselines": baselines,
            "segments": [s.to_dict() for s in aligned_segments],
            "disclaimer": (
                "Zero Rabies-MMNet is a research prototype for non-invasive behavioral screening / "
                "risk assessment. Outputs represent operational behavioral agitation proxies and "
                "are NOT clinical rabies diagnoses."
            )
        }

        # 9. Save CSV logs if requested
        if save_csv:
            self._append_segment_log(clip_id, aligned_segments)
            self._append_result_log(clip_id, output)

        return output

    def _append_segment_log(self, clip_id: str, segments: List[AlignedSegment]):
        """
        Append aligned segments to outputs/fusion/dynamic_late_fusion_segments.csv.
        Columns match Section 16 specification:
        clip_id, start_time, end_time, video_probability, video_confidence, video_quality,
        audio_probability, audio_confidence, audio_quality, video_weight, audio_weight,
        fused_probability, fused_score, risk_level
        """
        seg_file = self.output_dir / "dynamic_late_fusion_segments.csv"
        headers = [
            "clip_id", "start_time", "end_time", "video_probability",
            "video_confidence", "video_quality", "audio_probability",
            "audio_confidence", "audio_quality", "video_weight",
            "audio_weight", "fused_probability", "fused_score", "risk_level"
        ]
        write_header = not seg_file.exists() or seg_file.stat().st_size == 0

        with open(seg_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            for s in segments:
                writer.writerow([
                    clip_id, s.start_time, s.end_time, s.video_probability,
                    s.video_confidence, s.video_quality, s.audio_probability,
                    s.audio_confidence, s.audio_quality, s.video_weight,
                    s.audio_weight, s.fused_probability, s.fused_score, s.risk_level
                ])

    def _append_result_log(self, clip_id: str, output: Dict[str, Any]):
        """
        Append clip-level fusion results to outputs/fusion/dynamic_late_fusion_results.csv.
        Columns match Section 17 specification:
        clip_id, video_score, video_confidence, audio_score, audio_confidence,
        fused_score, video_weight, audio_weight, final_risk_level
        """
        res_file = self.output_dir / "dynamic_late_fusion_results.csv"
        headers = [
            "clip_id", "video_score", "video_confidence", "audio_score",
            "audio_confidence", "fused_score", "video_weight",
            "audio_weight", "final_risk_level"
        ]
        write_header = not res_file.exists() or res_file.stat().st_size == 0

        with open(res_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow([
                clip_id,
                output["video"]["score"],
                output["video"]["confidence"],
                output["audio"]["score"],
                output["audio"]["confidence"],
                output["fused"]["risk_score"],
                output["weights"]["video"],
                output["weights"]["audio"],
                output["fused"]["risk_level"]
            ])

"""
Temporal Alignment Module for Zero Rabies-MMNet.
Aligns multi-rate temporal streams (Video: 2.0s windows / 1.0s hop; Audio: 3.0s windows / 1.5s hop)
onto a synchronized temporal grid (default 0.5s / 2 Hz ticks) with overlap aggregation.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .schemas import AlignedSegment, map_score_to_risk_level
from .confidence import calculate_dynamic_weights, compute_margin_confidence


def find_overlapping_windows(
    windows: List[Dict[str, Any]],
    t_start: float,
    t_end: float,
    time_key_start: str = "t_start",
    time_key_end: str = "t_end"
) -> List[Dict[str, Any]]:
    """
    Finds all windows that overlap with the interval [t_start, t_end].
    An overlap occurs when window.start < t_end and window.end > t_start.
    """
    overlapping = []
    for w in windows:
        w_s = float(w.get(time_key_start, 0.0))
        w_e = float(w.get(time_key_end, 0.0))
        if w_s < t_end and w_e > t_start:
            overlapping.append(w)
    return overlapping


def align_multimodal_streams(
    video_windows: List[Dict[str, Any]],
    audio_windows: List[Dict[str, Any]],
    tick_seconds: float = 0.5,
    max_duration: Optional[float] = None,
    video_quality_overall: str = "GOOD",
    clip_id: str = "clip"
) -> List[AlignedSegment]:
    """
    Align video and audio window streams onto discrete temporal ticks.
    For each tick:
      1. Aggregates all overlapping video windows.
      2. Aggregates all overlapping audio windows.
      3. Computes modality weights Wv and Wa via dynamic confidence.
      4. Fuses predictions: R1 = Wv * Pv + Wa * Pa.
      5. Maps R1 to 0-100 risk score and LOW / MEDIUM / HIGH risk band.
    """
    # Determine overall duration
    v_end = max((float(w.get("t_end", 0.0)) for w in video_windows), default=0.0)
    a_end = max((float(w.get("t_end", 0.0)) for w in audio_windows), default=0.0)
    end_time = max_duration if max_duration is not None else max(v_end, a_end)

    if end_time <= 0.0:
        return []

    # Generate ticks
    num_ticks = int(np.ceil(end_time / tick_seconds))
    aligned_segments: List[AlignedSegment] = []

    for i in range(num_ticks):
        t0 = round(float(i * tick_seconds), 3)
        t1 = round(float(min((i + 1) * tick_seconds, end_time)), 3)
        if t1 <= t0:
            continue

        # Overlapping video windows
        v_overlaps = find_overlapping_windows(video_windows, t0, t1, "t_start", "t_end")
        # Overlapping audio windows
        a_overlaps = find_overlapping_windows(audio_windows, t0, t1, "t_start", "t_end")

        # Video aggregation for tick
        if v_overlaps:
            v_probs = [float(w["probability"]) for w in v_overlaps if w.get("probability") is not None]
            v_prob = float(np.mean(v_probs)) if v_probs else None
            # Video confidence: can be specified per-window or derived from prediction margin
            v_confs = [float(w.get("confidence", compute_margin_confidence(w.get("probability")))) for w in v_overlaps]
            v_conf = float(np.mean(v_confs)) if v_confs else 0.0
            v_rel = float(np.mean([w.get("reliability", 1.0) for w in v_overlaps]))
            v_present = True
            v_quality = video_quality_overall
        else:
            v_prob = None
            v_conf = 0.0
            v_rel = 0.0
            v_present = False
            v_quality = "NO_INPUT"

        # Audio aggregation for tick
        if a_overlaps:
            a_probs = [float(w["audio_score"]) for w in a_overlaps if w.get("audio_score") is not None and w.get("audio_present", True)]
            a_prob = float(np.mean(a_probs)) if a_probs else None
            a_confs = [float(w.get("audio_confidence", 0.0)) for w in a_overlaps]
            a_conf = float(np.mean(a_confs)) if a_confs else 0.0
            a_rel = float(np.mean([w.get("audio_reliability", 0.0) for w in a_overlaps]))
            a_present = any(w.get("audio_present", True) for w in a_overlaps)
            a_quality = "PRESENT" if a_present else "SILENT"
        else:
            a_prob = None
            a_conf = 0.0
            a_rel = 0.0
            a_present = False
            a_quality = "NO_INPUT"

        # Calculate dynamic weights
        wv, wa = calculate_dynamic_weights(
            video_confidence=v_conf,
            audio_confidence=a_conf,
            video_present=v_present,
            audio_present=a_present,
            video_reliability=v_rel,
            audio_reliability=a_rel
        )

        # Fuse
        if v_prob is not None and a_prob is not None:
            fused_p = float(wv * v_prob + wa * a_prob)
        elif v_prob is not None:
            fused_p = float(v_prob)
            wv = 1.0
            wa = 0.0
        elif a_prob is not None:
            fused_p = float(a_prob)
            wv = 0.0
            wa = 1.0
        else:
            fused_p = None

        if fused_p is not None:
            fused_p = max(0.0, min(1.0, fused_p))
            fused_score = int(round(fused_p * 100))
            risk_level = map_score_to_risk_level(fused_score)
        else:
            fused_score = None
            risk_level = None

        segment = AlignedSegment(
            segment_id=i,
            start_time=t0,
            end_time=t1,
            video_probability=round(v_prob, 4) if v_prob is not None else None,
            video_confidence=round(v_conf, 4),
            video_quality=v_quality,
            audio_probability=round(a_prob, 4) if a_prob is not None else None,
            audio_confidence=round(a_conf, 4),
            audio_quality=a_quality,
            video_weight=round(wv, 4),
            audio_weight=round(wa, 4),
            fused_probability=round(fused_p, 4) if fused_p is not None else None,
            fused_score=fused_score,
            risk_level=risk_level
        )
        aligned_segments.append(segment)

    return aligned_segments

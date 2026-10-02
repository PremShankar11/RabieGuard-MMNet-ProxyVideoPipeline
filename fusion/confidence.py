"""
Confidence and Reliability Calculation Module for Zero Rabies-MMNet.
Distinguishes between prediction probability (P) and observation trustworthiness/confidence (C).
Calculates dynamic modality weights Wv and Wa with strict boundary and zero-sum safety.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def compute_margin_confidence(prob: Optional[float]) -> float:
    """
    Computes distance from the operational decision boundary (0.50), scaled to [0.0, 1.0].
    Confidence = min(1.0, 2 * |prob - 0.5|)
    - 0.50 -> 0.0 (maximum uncertainty / near decision boundary)
    - 0.00 or 1.00 -> 1.0 (high certainty)
    """
    if prob is None or np.isnan(prob):
        return 0.0
    val = min(1.0, max(0.0, 2.0 * abs(float(prob) - 0.5)))
    return round(val, 4)


def compute_video_reliability(quality_data: Optional[Dict[str, Any]] = None) -> float:
    """
    Computes video signal reliability from canine tracking and pose metrics:
      - valid_frame_percentage
      - mean_pose_confidence
      - ambiguous_frame_percentage
    Output in [0.0, 1.0].
    """
    if not quality_data:
        return 0.5

    overall = quality_data.get("overall", "FAIR")
    valid_pct = float(quality_data.get("valid_frame_percentage", 0.0))
    mean_pose_conf = float(quality_data.get("mean_pose_confidence", 0.0))
    amb_pct = float(quality_data.get("ambiguous_frame_percentage", 0.0))

    if valid_pct <= 0.0 or overall == "LIMITED" and valid_pct < 20.0:
        return 0.0

    # Composite reliability: pose completeness * keypoint clarity * ambiguity penalty
    completeness = min(1.0, max(0.0, valid_pct / 100.0))
    clarity = min(1.0, max(0.0, mean_pose_conf))
    amb_factor = max(0.5, 1.0 - (amb_pct / 200.0))

    rel = completeness * clarity * amb_factor
    return round(float(np.clip(rel, 0.0, 1.0)), 4)


def compute_audio_reliability(
    audio_present: bool,
    energy_db: float,
    silence_threshold_db: float = -50.0
) -> float:
    """
    Computes audio signal reliability based on physical signal energy in dBFS.
    If silent (below silence threshold), reliability is 0.0.
    """
    if not audio_present or energy_db <= silence_threshold_db:
        return 0.0

    # Smooth linear ramp from silence (-50 dBFS) to strong signal (-15 dBFS)
    scaled = (energy_db - silence_threshold_db) / 35.0
    rel = float(np.clip(scaled, 0.1, 1.0))
    return round(rel, 4)


def calculate_dynamic_weights(
    video_confidence: float,
    audio_confidence: float,
    video_present: bool = True,
    audio_present: bool = True,
    video_reliability: Optional[float] = None,
    audio_reliability: Optional[float] = None,
    use_reliability_modulation: bool = True
) -> Tuple[float, float]:
    """
    Dynamically calculate modality weights Wv and Wa.
    Formula:
        Wv = Cv / (Cv + Ca)
        Wa = Ca / (Cv + Ca)
    where Wv + Wa = 1.0.

    If use_reliability_modulation is True, effective confidence is:
        Cv_eff = video_confidence * video_reliability
        Ca_eff = audio_confidence * audio_reliability

    Safe documented fallback when (Cv + Ca) == 0:
      - If only video is present: Wv = 1.0, Wa = 0.0
      - If only audio is present: Wv = 0.0, Wa = 1.0
      - If both are present (or neither): Wv = 0.50, Wa = 0.50 (equal split)
    """
    cv = max(0.0, min(1.0, float(video_confidence)))
    ca = max(0.0, min(1.0, float(audio_confidence)))

    if use_reliability_modulation:
        rv = video_reliability if video_reliability is not None else (1.0 if video_present else 0.0)
        ra = audio_reliability if audio_reliability is not None else (1.0 if audio_present else 0.0)
        cv = cv * max(0.0, min(1.0, float(rv)))
        ca = ca * max(0.0, min(1.0, float(ra)))

    # Zero-confidence safe fallback
    total_conf = cv + ca

    if total_conf <= 1e-8:
        if video_present and not audio_present:
            return 1.0, 0.0
        elif audio_present and not video_present:
            return 0.0, 1.0
        else:
            # Documented equal fallback
            return 0.5, 0.5

    wv = cv / total_conf
    wa = ca / total_conf

    # Boundary clamping and normalisation to guarantee Wv + Wa == 1.0
    wv = max(0.0, min(1.0, wv))
    wa = 1.0 - wv

    return round(wv, 4), round(wa, 4)

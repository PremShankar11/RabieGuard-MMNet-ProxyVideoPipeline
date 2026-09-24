"""
Scoring and Quality Assessment Module for Zero Rabies-MMNet Video V2.
Handles 16-frame window slicing, edge padding, reliability weighting,
window probability inference, clip aggregation, behavioral score mapping (0-100),
and pose quality grading (GOOD, FAIR, LIMITED).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from .model import FrozenMambaV2Classifier


def generate_temporal_windows(
    tracked_data: Dict[str, Any],
    window_length: int = 16,
    stride: int = 8
) -> List[Dict[str, Any]]:
    """
    Construct 16-frame temporal windows matching V2 frozen specifications.
    For sequences shorter than window_length, applies edge-repeat temporal padding.
    """
    kpt_norm = tracked_data["keypoints_normalized"]
    bbox_conf = tracked_data["bbox_confidence"]
    valid_mask = tracked_data["valid_mask"]
    ambiguous_mask = tracked_data["ambiguous_mask"]
    N = int(tracked_data["num_frames"])

    if N == 0:
        raise ValueError("Cannot construct temporal windows from 0 frames.")

    if N < window_length:
        starts = [0]
        is_padded = True
    else:
        starts = list(range(0, N - window_length + 1, stride))
        if starts[-1] != N - window_length:
            starts.append(N - window_length)
        is_padded = False

    windows = []
    for s in starts:
        if is_padded:
            w_kpt = np.zeros((window_length, 24, 3), dtype=np.float32)
            w_kpt[:N] = kpt_norm
            w_kpt[N:] = kpt_norm[-1:]

            w_bconf = np.zeros(window_length, dtype=np.float32)
            w_bconf[:N] = bbox_conf
            w_bconf[N:] = bbox_conf[-1:]

            w_valid = np.zeros(window_length, dtype=bool)
            w_valid[:N] = valid_mask

            w_amb = np.zeros(window_length, dtype=bool)
            w_amb[:N] = ambiguous_mask

            pad_mask = np.zeros(window_length, dtype=bool)
            pad_mask[N:] = True
        else:
            e = s + window_length
            w_kpt = kpt_norm[s:e].astype(np.float32)
            w_bconf = bbox_conf[s:e].astype(np.float32)
            w_valid = valid_mask[s:e].astype(bool)
            w_amb = ambiguous_mask[s:e].astype(bool)
            pad_mask = np.zeros(window_length, dtype=bool)

        frame_valid = w_valid & (~pad_mask)
        amb_penalty = np.where(w_amb, 0.8, 1.0)
        reliability = w_bconf * frame_valid.astype(np.float32) * amb_penalty

        feat_72 = w_kpt.reshape(window_length, 72)
        # 3 extra reliability features: [bbox_confidence, frame_valid, ambiguous_mask]
        extra_3 = np.stack([
            w_bconf,
            frame_valid.astype(np.float32),
            w_amb.astype(np.float32)
        ], axis=-1)
        feat_75 = np.concatenate([feat_72, extra_3], axis=-1)

        windows.append({
            "feat_72": feat_72,
            "feat_75": feat_75,
            "reliability": reliability,
            "padding_mask": pad_mask,
            "frame_valid": frame_valid,
            "ambiguous_mask": w_amb,
            "bbox_conf": w_bconf,
            "start_idx": s,
            "is_padded": is_padded
        })

    return windows


def score_temporal_windows(
    classifier: FrozenMambaV2Classifier,
    windows: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run frozen Mamba V2 inference across all temporal windows.
    Returns:
        clip_probability: float in [0.0, 1.0] (mean of window probabilities)
        scored_windows: List of window dicts with 'probability' field
    """
    if not windows:
        raise ValueError("No windows provided for scoring.")

    feat_tensor = torch.tensor(
        np.stack([w["feat_75"] for w in windows]),
        dtype=torch.float32
    )
    rel_tensor = torch.tensor(
        np.stack([w["reliability"] for w in windows]),
        dtype=torch.float32
    )

    probs = classifier.predict_batch(feat_tensor, rel_tensor)

    scored_windows = []
    for i, w in enumerate(windows):
        scored_windows.append({
            "window_index": i,
            "start_frame": int(w["start_idx"]),
            "end_frame": int(w["start_idx"] + len(w["feat_75"])),
            "probability": round(float(probs[i]), 4),
            "is_padded": bool(w["is_padded"]),
            "valid_frames": int(w["frame_valid"].sum()),
            "ambiguous_frames": int(w["ambiguous_mask"].sum()),
        })

    clip_probability = float(np.mean(probs))
    return clip_probability, scored_windows


def compute_behavioral_score(clip_probability: float) -> Dict[str, Any]:
    """
    Maps clip-level AGITATED probability to project operational behavioral score (0-100).
    """
    clamped_prob = max(0.0, min(1.0, float(clip_probability)))
    score = int(round(clamped_prob * 100))

    if score <= 35:
        score_band = "Low Agitation / Calm-like"
        interpretation = "Low observed behavioral arousal under operational scoring scheme."
    elif score <= 65:
        score_band = "Moderate Activity / Transitional"
        interpretation = "Moderate observed behavioral arousal / transitional dynamics under operational scoring scheme."
    else:
        score_band = "High Agitation"
        interpretation = "High observed behavioral arousal under operational scoring scheme."

    behavior = "AGITATED" if clamped_prob >= 0.50 else "CALM"

    return {
        "behavior": behavior,
        "probability_agitated": round(clamped_prob, 4),
        "behavioral_score": score,
        "score_band": score_band,
        "interpretation": interpretation,
    }


def compute_quality_metrics(
    tracked_data: Dict[str, Any],
    windows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute transparent Video/Pose Quality metrics and assign overall quality:
    GOOD, FAIR, or LIMITED.
    """
    total_frames = int(tracked_data["num_frames"])
    valid_mask = tracked_data["valid_mask"]
    amb_mask = tracked_data["ambiguous_mask"]
    kpt_confs = tracked_data["keypoint_confidence"]
    bbox_confs = tracked_data["bbox_confidence"]

    valid_count = int(valid_mask.sum())
    amb_count = int(amb_mask.sum())

    valid_pct = (valid_count / total_frames * 100.0) if total_frames > 0 else 0.0
    amb_pct = (amb_count / total_frames * 100.0) if total_frames > 0 else 0.0

    if valid_count > 0:
        mean_pose_conf = float(np.mean(kpt_confs[valid_mask]))
        mean_bbox_conf = float(np.mean(bbox_confs[valid_mask]))
    else:
        mean_pose_conf = 0.0
        mean_bbox_conf = 0.0

    total_windows = len(windows)
    padded_windows = sum(1 for w in windows if w["is_padded"])
    padded_pct = (padded_windows / total_windows * 100.0) if total_windows > 0 else 0.0

    # Operational quality grading rules
    if valid_pct >= 75.0 and mean_pose_conf >= 0.45 and amb_pct <= 30.0:
        overall = "GOOD"
        quality_note = "High pose confidence and consistent canine tracking across video."
    elif valid_pct >= 50.0 and mean_pose_conf >= 0.30 and amb_pct <= 50.0:
        overall = "FAIR"
        quality_note = "Moderate pose completeness; results are usable but subject to tracking noise."
    else:
        overall = "LIMITED"
        quality_note = "Low pose visibility or severe multi-dog occlusions; behavioral estimate may be unreliable."

    # Multi-dog ambiguity tag
    if amb_pct <= 10.0:
        ambiguity_level = "LOW"
    elif amb_pct <= 35.0:
        ambiguity_level = "MODERATE"
    else:
        ambiguity_level = "HIGH"

    return {
        "overall": overall,
        "quality_note": quality_note,
        "valid_frame_percentage": round(valid_pct, 1),
        "mean_pose_confidence": round(mean_pose_conf, 2),
        "mean_bbox_confidence": round(mean_bbox_conf, 2),
        "ambiguous_frame_percentage": round(amb_pct, 1),
        "ambiguity_level": ambiguity_level,
        "padded_window_percentage": round(padded_pct, 1),
        "total_sampled_frames": total_frames,
        "valid_pose_frames": valid_count,
        "ambiguous_frames": amb_count,
        "total_windows": total_windows,
        "padded_windows": padded_windows,
    }

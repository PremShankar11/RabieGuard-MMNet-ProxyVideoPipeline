"""
End-to-End Reusable Video Inference Pipeline for Zero Rabies-MMNet Video V2.
Orchestrates video preprocessing, pose extraction, temporal tracking,
Mamba V2 scoring, quality evaluation, and transparent research export.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import json
import base64
import numpy as np
import cv2
import torch

from .preprocessing import (
    validate_video_file,
    get_video_metadata,
    sample_video_frames,
    VideoReadError,
    VideoValidationError
)
from .pose import DogPoseEstimator, PoseModelLoadError
from .tracking import TemporalDogTracker, draw_dog_skeleton
from .model import FrozenMambaV2Classifier, ModelLoadError
from .scoring import (
    generate_temporal_windows,
    score_temporal_windows,
    compute_behavioral_score,
    compute_quality_metrics
)


class InferenceError(Exception):
    """Base exception for video inference pipeline failures."""
    pass


class VideoBehaviorPipeline:
    """
    Unified, reusable inference pipeline for Zero Rabies-MMNet Video V2.
    """
    DEFAULT_POSE_CHECKPOINT = "checkpoints/dog_pose/best.pt"
    DEFAULT_MAMBA_CHECKPOINT = "checkpoints/mamba_behavior_v2/final.pt"
    DEFAULT_MANIFEST = "reports/behavior_v2/FROZEN_MODEL_MANIFEST.json"

    def __init__(
        self,
        pose_checkpoint: Optional[str | Path] = None,
        mamba_checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        pose_ckpt = pose_checkpoint or self.DEFAULT_POSE_CHECKPOINT
        mamba_ckpt = mamba_checkpoint or self.DEFAULT_MAMBA_CHECKPOINT

        # Load models once
        self.pose_estimator = DogPoseEstimator(
            checkpoint_path=pose_ckpt,
            device=self.device
        )
        self.mamba_classifier = FrozenMambaV2Classifier(
            checkpoint_path=mamba_ckpt,
            device=self.device
        )

        # Load frozen manifest if available
        manifest_path = Path(self.DEFAULT_MANIFEST)
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def process_video(
        self,
        video_path: str | Path,
        save_json: bool = False,
        output_dir: Optional[str | Path] = None,
        generate_annotated_video: bool = False,
        annotated_video_path: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end canine behavioral screening inference on a video.
        """
        v_path = validate_video_file(video_path)

        # 1. Video metadata & 8 FPS sampling
        sampled_frames, sampling_meta = sample_video_frames(v_path, target_fps=8.0)
        num_sampled = len(sampled_frames)

        if num_sampled == 0:
            raise InferenceError(f"No frames could be sampled from {v_path.name}")

        # 2. Canine Pose Estimation (YOLO11n-Pose)
        raw_detections = self.pose_estimator.predict_frames(sampled_frames)

        # 3. Temporal Dog Tracking & Body-Relative Normalization
        tracker = TemporalDogTracker(
            frame_width=sampling_meta["width"],
            frame_height=sampling_meta["height"]
        )
        tracked_data = tracker.track_sequence(
            raw_detections,
            timestamps=sampling_meta["sample_timestamps"]
        )

        valid_frames_count = int(tracked_data["valid_mask"].sum())
        if valid_frames_count == 0:
            raise InferenceError(
                f"No usable dog pose was detected in this video ({v_path.name}). "
                "Ensure a canine subject is clearly visible."
            )

        # 4. Temporal Window Generation (16-frame windows with edge padding if short)
        windows = generate_temporal_windows(
            tracked_data,
            window_length=16,
            stride=8
        )

        # 5. Frozen Mamba V2 Temporal Scoring
        clip_prob, scored_windows = score_temporal_windows(
            self.mamba_classifier,
            windows
        )

        # 6. Behavioral Score & Operational Interpretation
        score_data = compute_behavioral_score(clip_prob)

        # 7. Video / Pose Quality Assessment
        quality_data = compute_quality_metrics(tracked_data, windows)

        # Multi-dog warning if applicable
        warning_msg = None
        if quality_data["ambiguity_level"] == "HIGH":
            warning_msg = (
                f"Multiple dogs detected in {quality_data['ambiguous_frame_percentage']}% of sampled frames. "
                "Interpretation may be less reliable due to occlusions and potential identity competition."
            )
        elif quality_data["overall"] == "LIMITED":
            warning_msg = (
                f"Video quality is limited ({quality_data['valid_frame_percentage']}% valid pose frames). "
                "Behavioral estimate should be interpreted with caution."
            )

        # Optional annotated video export
        saved_video_path = None
        preview_frames = []
        if generate_annotated_video:
            saved_video_path, preview_frames = self._render_annotated_video(
                sampled_frames,
                tracked_data,
                annotated_video_path or f"outputs/inference/{v_path.stem}_annotated.mp4"
            )

        # Build authoritative output object
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        result: Dict[str, Any] = {
            "behavior": score_data["behavior"],
            "probability_agitated": score_data["probability_agitated"],
            "behavioral_score": score_data["behavioral_score"],
            "score_band": score_data["score_band"],
            "interpretation": score_data["interpretation"],
            "preview_frames": preview_frames,
            "quality": {
                "overall": quality_data["overall"],
                "quality_note": quality_data["quality_note"],
                "valid_frame_percentage": quality_data["valid_frame_percentage"],
                "mean_pose_confidence": quality_data["mean_pose_confidence"],
                "mean_bbox_confidence": quality_data["mean_bbox_confidence"],
                "ambiguous_frame_percentage": quality_data["ambiguous_frame_percentage"],
                "ambiguity_level": quality_data["ambiguity_level"],
                "padded_window_percentage": quality_data["padded_window_percentage"],
            },
            "video": {
                "filename": v_path.name,
                "duration_seconds": sampling_meta["duration_seconds"],
                "source_fps": sampling_meta["source_fps"],
                "sampled_fps": sampling_meta["sampled_fps"],
                "sampled_frames": sampling_meta["sampled_frames"],
                "valid_pose_frames": quality_data["valid_pose_frames"],
                "ambiguous_frames": quality_data["ambiguous_frames"],
                "width": sampling_meta["width"],
                "height": sampling_meta["height"],
            },
            "temporal": {
                "window_length": 16,
                "stride": 8,
                "num_windows": quality_data["total_windows"],
                "padded_windows": quality_data["padded_windows"],
                "window_probabilities": [w["probability"] for w in scored_windows],
            },
            "model": {
                "name": "Zero Rabies-MMNet Video V2",
                "experiment": "exp_d_temporal_tracking",
                "architecture": "Mamba S6",
                "checkpoint": str(Path(self.mamba_classifier.checkpoint_path).name),
                "checkpoint_full": str(self.mamba_classifier.checkpoint_path),
                "pose_checkpoint": str(Path(self.pose_estimator.checkpoint_path).name),
                "device": str(self.device),
                "trainable_parameters": self.mamba_classifier.metadata["trainable_params"],
            },
            "warning": warning_msg,
            "annotated_video_path": saved_video_path,
            "timestamp": timestamp_iso,
            "disclaimer": (
                "Zero Rabies-MMNet is a research prototype for non-invasive behavioral screening / "
                "risk assessment. The video branch estimates operational behavioral arousal from "
                "canine pose dynamics. It is NOT a veterinary diagnostic system and does NOT "
                "determine whether a dog has rabies."
            )
        }

        # Save to JSON if requested
        if save_json:
            out_dir = Path(output_dir or "outputs/inference")
            out_dir.mkdir(parents=True, exist_ok=True)
            clean_ts = timestamp_iso.replace(":", "-").replace(".", "-")
            json_file = out_dir / f"{clean_ts}_{v_path.stem}_result.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            result["exported_json_path"] = str(json_file)

        return result

    def _render_annotated_video(
        self,
        frames: List[np.ndarray],
        tracked_data: Dict[str, Any],
        output_path: str | Path
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Render annotated video with tracked bounding boxes and 24-keypoint skeletons."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not frames:
            return "", []

        h, w = frames[0].shape[:2]
        # H.264 / OpenH264 strictly requires even dimensions
        w = w - (w % 2)
        h = h - (h % 2)

        # Prioritize browser-native H.264 (avc1) for HTML5 video playback
        writer = None
        for codec in ["avc1", "H264", "mp4v"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                w_candidate = cv2.VideoWriter(str(out_p), fourcc, 8.0, (w, h))
                if w_candidate.isOpened():
                    writer = w_candidate
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, 8.0, (w, h))

        timestamps = tracked_data.get("timestamps", [i / 8.0 for i in range(len(frames))])
        kpts_raw = tracked_data["keypoints_raw"]
        kconfs = tracked_data["keypoint_confidence"]
        bboxes = tracked_data["bbox"]
        bconfs = tracked_data["bbox_confidence"]
        amb_mask = tracked_data["ambiguous_mask"]
        valid_mask = tracked_data["valid_mask"]

        preview_frames: List[Dict[str, Any]] = []

        for i, frame in enumerate(frames):
            if valid_mask[i]:
                annotated = draw_dog_skeleton(
                    img=frame,
                    kpts=kpts_raw[i, :, :2],
                    confs=kconfs[i],
                    bbox=bboxes[i],
                    bbox_conf=float(bconfs[i]),
                    is_ambiguous=bool(amb_mask[i]),
                    timestamp_sec=float(timestamps[i])
                )
            else:
                annotated = frame.copy()
                cv2.putText(annotated, "NO DOG DETECTED", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Ensure exact dimension match with writer
            if (annotated.shape[1], annotated.shape[0]) != (w, h):
                annotated = cv2.resize(annotated, (w, h))
            writer.write(annotated)

            # Generate base64 thumbnail for interactive frame inspector
            thumb_w = min(w, 640)
            thumb_h = int(h * (thumb_w / max(1, w)))
            thumb_h = thumb_h - (thumb_h % 2)
            thumb = cv2.resize(annotated, (thumb_w, thumb_h)) if (w != thumb_w or h != thumb_h) else annotated
            ret, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                preview_frames.append({
                    "frame_idx": i,
                    "timestamp": round(float(timestamps[i]), 2),
                    "valid": bool(valid_mask[i]),
                    "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
                })

        writer.release()
        return str(out_p), preview_frames

"""
Video Preprocessing Module for Zero Rabies-MMNet Video V2.
Handles video decoding, metadata extraction, validation, and ~8 FPS temporal frame sampling.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


class VideoReadError(Exception):
    """Raised when video file cannot be opened or decoded."""
    pass


class VideoValidationError(Exception):
    """Raised when video does not meet minimal quality or format requirements."""
    pass


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def validate_video_file(video_path: str | Path) -> Path:
    """
    Validate that video file exists and has a supported video extension.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise VideoValidationError(f"Specified path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video format '{path.suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if path.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty (0 bytes): {path.name}")
    return path


def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
    """
    Extract technical metadata from video without loading full frame content.
    """
    path = validate_video_file(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video file: {path.name}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if actual_fps is None or actual_fps <= 0 or np.isnan(actual_fps):
        actual_fps = 24.0

    duration = total_frames / actual_fps if total_frames > 0 else 0.0

    if total_frames <= 0 or duration <= 0:
        raise VideoReadError(f"Video file '{path.name}' has invalid frame count ({total_frames}) or duration.")

    return {
        "filename": path.name,
        "path": str(path),
        "duration_seconds": round(float(duration), 2),
        "source_fps": round(float(actual_fps), 2),
        "frame_count": int(total_frames),
        "width": int(width),
        "height": int(height),
    }


def sample_video_frames(
    video_path: str | Path,
    target_fps: float = 8.0,
    max_duration_seconds: Optional[float] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Sample frames from video at approximately target_fps (default 8.0 FPS).
    Returns:
        frames: List of BGR numpy arrays (height, width, 3)
        metadata: Dict containing sampling indices, timestamps, and video properties
    """
    path = validate_video_file(video_path)
    meta = get_video_metadata(path)

    actual_fps = meta["source_fps"]
    total_frames = meta["frame_count"]
    duration = meta["duration_seconds"]

    if max_duration_seconds is not None and duration > max_duration_seconds:
        duration = max_duration_seconds
        total_frames = min(total_frames, int(duration * actual_fps))

    num_samples = max(1, int(round(duration * target_fps)))
    sample_timestamps = [i / target_fps for i in range(num_samples)]
    sample_frame_indices = [min(total_frames - 1, int(round(t * actual_fps))) for t in sample_timestamps]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"Failed to open video for frame extraction: {path.name}")

    sampled_frames: List[np.ndarray] = []
    width = meta["width"]
    height = meta["height"]

    for f_idx in sample_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback black frame if a single frame cannot be read
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        sampled_frames.append(frame)
    cap.release()

    if len(sampled_frames) == 0:
        raise VideoReadError(f"Zero frames could be decoded from video: {path.name}")

    sampling_info = {
        **meta,
        "sampled_fps": float(target_fps),
        "sampled_frames": len(sampled_frames),
        "sample_timestamps": sample_timestamps,
        "sample_frame_indices": sample_frame_indices,
    }

    return sampled_frames, sampling_info

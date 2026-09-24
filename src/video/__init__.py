"""
Zero Rabies-MMNet — Video V2 Inference Package.
Productized, reusable inference pipeline for canine behavioral screening.
"""

from .inference import VideoBehaviorPipeline
from .preprocessing import get_video_metadata, sample_video_frames
from .pose import DogPoseEstimator
from .tracking import TemporalDogTracker
from .model import FrozenMambaV2Classifier
from .scoring import (
    generate_temporal_windows,
    score_temporal_windows,
    compute_quality_metrics,
    compute_behavioral_score,
)

__all__ = [
    "VideoBehaviorPipeline",
    "get_video_metadata",
    "sample_video_frames",
    "DogPoseEstimator",
    "TemporalDogTracker",
    "FrozenMambaV2Classifier",
    "generate_temporal_windows",
    "score_temporal_windows",
    "compute_quality_metrics",
    "compute_behavioral_score",
]

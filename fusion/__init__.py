"""
Zero Rabies-MMNet Multimodal Late Fusion Package.
Implements Approach 1: Dynamic Confidence-Aware Late Fusion for Frozen Video V2 and Audio V2.
"""

from .dynamic_late_fusion import DynamicLateFusionEngine
from .confidence import (
    compute_margin_confidence,
    compute_video_reliability,
    compute_audio_reliability,
    calculate_dynamic_weights,
)
from .alignment import align_multimodal_streams
from .schemas import (
    AlignedSegment,
    DynamicLateFusionOutput,
    map_score_to_risk_level,
)

__all__ = [
    "DynamicLateFusionEngine",
    "compute_margin_confidence",
    "compute_video_reliability",
    "compute_audio_reliability",
    "calculate_dynamic_weights",
    "align_multimodal_streams",
    "AlignedSegment",
    "DynamicLateFusionOutput",
    "map_score_to_risk_level",
]

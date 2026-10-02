"""
Data Schemas and Contracts for Zero Rabies-MMNet Multimodal Fusion.
Defines authoritative typing, per-tick aligned segments, and final output structure.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


def map_score_to_risk_level(score: int) -> str:
    """
    Authoritative risk band mapping adhering to frozen video manifest and project specifications:
      - 0 to 35:   LOW    (Low Agitation / Calm-like)
      - 36 to 65:  MEDIUM (Moderate Activity / Transitional)
      - 66 to 100: HIGH   (High Agitation)
    """
    if score <= 35:
        return "LOW"
    elif score <= 65:
        return "MEDIUM"
    else:
        return "HIGH"


@dataclass
class ModalityObservation:
    """Unimodal observation for a temporal window or clip."""
    probability: Optional[float]
    score: Optional[int]
    confidence: float
    quality: str
    is_present: bool = True
    reliability: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignedSegment:
    """
    Represents an aligned temporal segment where video and audio observations coincide.
    Matches schema required for dynamic_late_fusion_segments.csv.
    """
    segment_id: int
    start_time: float
    end_time: float
    video_probability: Optional[float]
    video_confidence: float
    video_quality: str
    audio_probability: Optional[float]
    audio_confidence: float
    audio_quality: str
    video_weight: float
    audio_weight: float
    fused_probability: Optional[float]
    fused_score: Optional[int]
    risk_level: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicLateFusionOutput:
    """
    Authoritative Multimodal Late Fusion Output matching Section 15 contract.
    """
    fusion_method: str = "dynamic_confidence_late_fusion"
    video: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    fused: Dict[str, Any] = field(default_factory=dict)
    temporal: Dict[str, float] = field(default_factory=dict)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    baselines: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Zero Rabies-MMNet is a research prototype for non-invasive behavioral screening / "
        "risk assessment. Outputs represent operational behavioral agitation proxies and "
        "are NOT clinical rabies diagnoses."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

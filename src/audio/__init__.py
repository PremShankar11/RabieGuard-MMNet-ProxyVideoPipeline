"""
Audio module for Zero Rabies-MMNet.
Encapsulates frozen AudioNet v2 inference, audio preprocessing, and calibration,
as well as embedded audio extraction from canine video files.
"""

from .pipeline import AudioBehaviorPipeline
from .model import AudioNet
from .extraction import inspect_video_streams, extract_audio_from_video, get_ffmpeg_path

__all__ = [
    "AudioBehaviorPipeline",
    "AudioNet",
    "inspect_video_streams",
    "extract_audio_from_video",
    "get_ffmpeg_path",
]

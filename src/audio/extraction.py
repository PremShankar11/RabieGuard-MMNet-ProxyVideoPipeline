"""
Audio extraction and stream inspection utilities for Zero Rabies-MMNet.
Extracts 16 kHz mono PCM WAV audio directly from uploaded canine video streams.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def get_ffmpeg_path() -> str:
    """
    Locates the FFmpeg binary.
    Prioritizes system PATH, falls back to imageio_ffmpeg bundled binary.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass

    # Common Windows fallback paths if needed
    for fallback in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.exists(fallback):
            return fallback

    raise FileNotFoundError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")


def inspect_video_streams(video_path: Path) -> Dict[str, Any]:
    """
    Probes video file using FFmpeg to check for presence of video and audio streams,
    codecs, sample rates, channels, and estimated duration.
    """
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [ffmpeg_exe, "-i", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    stderr = proc.stderr

    has_audio = False
    audio_codec = None
    sample_rate = None
    channels = None
    has_video = False
    video_codec = None
    duration_sec = None

    # Parse duration: Duration: 00:00:15.34, start: 0.000000, bitrate: 1234 kb/s
    dur_match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", stderr)
    if dur_match:
        hrs, mins, secs = dur_match.groups()
        duration_sec = round(int(hrs) * 3600 + int(mins) * 60 + float(secs), 2)

    # Parse stream lines
    for line in stderr.splitlines():
        line_clean = line.strip()
        if "Stream #" in line_clean:
            if "Video:" in line_clean:
                has_video = True
                v_match = re.search(r"Video:\s*([a-zA-Z0-9_\-]+)", line_clean)
                if v_match:
                    video_codec = v_match.group(1)
            elif "Audio:" in line_clean:
                has_audio = True
                a_match = re.search(r"Audio:\s*([a-zA-Z0-9_\-]+)", line_clean)
                if a_match:
                    audio_codec = a_match.group(1)
                sr_match = re.search(r"(\d+)\s*Hz", line_clean)
                if sr_match:
                    sample_rate = int(sr_match.group(1))
                ch_match = re.search(r"(mono|stereo|5\.1|7\.1|\d+\s*channels)", line_clean, re.IGNORECASE)
                if ch_match:
                    channels = ch_match.group(1).lower()

    return {
        "has_video": has_video,
        "video_codec": video_codec,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "duration_seconds": duration_sec
    }


def extract_audio_from_video(
    video_path: Path,
    output_wav_path: Path,
    sample_rate: int = 16000
) -> bool:
    """
    Extracts embedded audio from video and converts it directly to 16 kHz mono 16-bit PCM WAV.
    Preserves exact timeline synchronization.

    Returns:
        True if audio was extracted and file exists with > 0 bytes.
        False if video has no audio or extraction fails.
    """
    info = inspect_video_streams(video_path)
    if not info["has_audio"]:
        return False

    ffmpeg_exe = get_ffmpeg_path()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_wav_path)
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    if proc.returncode == 0 and output_wav_path.exists() and output_wav_path.stat().st_size > 44:
        return True
    return False

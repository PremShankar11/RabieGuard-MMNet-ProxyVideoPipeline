"""
Audio Preprocessing and Windowing Module for Zero Rabies-MMNet.
Implements 16 kHz mono resampling, RMS energy dBFS calculation, silence detection,
mel-spectrogram extraction (n_mels=64, hop=256, n_fft=1024), and 3.0s temporal windowing with 1.5s hop.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import soundfile as sf
import librosa


SAMPLE_RATE = 16000
WINDOW_DURATION_SEC = 3.0
HOP_DURATION_SEC = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_SEC)   # 48000 samples
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION_SEC)         # 24000 samples
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
DEFAULT_SILENCE_DB = -50.0


class AudioFileError(Exception):
    """Raised when an audio file cannot be decoded or processed."""
    pass


def load_audio(
    audio_path_or_data: Union[str, Path, np.ndarray],
    target_sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, float]:
    """
    Load an audio file, convert to mono, and resample to target_sr.
    Returns:
        waveform: 1D np.ndarray (float32) normalized to [-1, 1]
        duration_seconds: float
    """
    if isinstance(audio_path_or_data, np.ndarray):
        waveform = audio_path_or_data.astype(np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        duration = float(len(waveform) / target_sr)
        return waveform, duration

    p = Path(audio_path_or_data).resolve()
    if not p.exists():
        raise AudioFileError(f"Audio file not found: {p}")

    try:
        waveform, sr = librosa.load(str(p), sr=target_sr, mono=True)
        waveform = waveform.astype(np.float32)
        duration = float(len(waveform) / target_sr)
        return waveform, duration
    except Exception as e:
        raise AudioFileError(f"Failed to read audio file '{p.name}': {e}") from e


def compute_audio_energy(audio_chunk: np.ndarray) -> Tuple[float, float, bool, float]:
    """
    Compute RMS and dBFS energy for an audio segment.
    Returns:
        rms: float
        energy_db: float in dBFS (20 * log10(rms))
        audio_present: bool (energy_db > DEFAULT_SILENCE_DB)
        reliability: float in [0.0, 1.0]
    """
    rms = float(np.sqrt(np.mean(audio_chunk ** 2) + 1e-9))
    energy_db = float(20.0 * np.log10(rms + 1e-9))
    audio_present = energy_db > DEFAULT_SILENCE_DB

    if not audio_present:
        reliability = 0.0
    else:
        # Scale reliability smoothly from silence threshold (-50 dBFS) to -10 dBFS
        # -50 dBFS -> ~0.1 reliability, >= -15 dBFS -> 1.0 reliability
        rel_scaled = (energy_db - DEFAULT_SILENCE_DB) / 35.0
        reliability = float(np.clip(rel_scaled, 0.1, 1.0))

    return rms, energy_db, audio_present, reliability


def extract_mel_spectrogram(
    audio_chunk: np.ndarray,
    target_samples: int = WINDOW_SAMPLES,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Compute normalized log-power Mel spectrogram matching AudioNet v2 training.
    Expected output shape: (1, 64, 188)
    """
    # Pad or clip to exact target length
    if len(audio_chunk) < target_samples:
        padded = np.pad(audio_chunk, (0, target_samples - len(audio_chunk)), mode='constant')
    else:
        padded = audio_chunk[:target_samples]

    mel = librosa.feature.melspectrogram(
        y=padded,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Z-score normalization matching training code
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    # Shape: (1, 64, 188)
    return mel_norm[np.newaxis, :, :].astype(np.float32)


def generate_audio_windows(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = WINDOW_DURATION_SEC,
    hop_sec: float = HOP_DURATION_SEC
) -> List[Dict[str, Any]]:
    """
    Slice continuous audio into overlapping 3.0s windows with 1.5s hop.
    For sequences shorter than window_sec, a single padded window is generated.
    """
    total_samples = len(waveform)
    win_samples = int(sample_rate * window_sec)
    hop_samples = int(sample_rate * hop_sec)

    if total_samples == 0:
        raise AudioFileError("Cannot generate audio windows from 0-sample waveform.")

    if total_samples <= win_samples:
        starts = [0]
    else:
        starts = list(range(0, total_samples - win_samples + 1, hop_samples))
        # Ensure the trailing tail is covered
        if starts[-1] + win_samples < total_samples:
            starts.append(total_samples - win_samples)

    windows = []
    for i, s in enumerate(starts):
        chunk = waveform[s:s + win_samples]
        t_start = round(float(s / sample_rate), 3)
        t_end = round(float(min(s + win_samples, total_samples) / sample_rate), 3)

        rms, energy_db, present, rel = compute_audio_energy(chunk)
        mel_tensor = extract_mel_spectrogram(chunk, target_samples=win_samples, sr=sample_rate)

        windows.append({
            "window_index": i,
            "t_start": t_start,
            "t_end": t_end,
            "duration": round(t_end - t_start, 3),
            "rms": rms,
            "energy_db": round(energy_db, 2),
            "audio_present": present,
            "audio_reliability": round(rel, 4),
            "mel_feature": mel_tensor,
            "is_padded": len(chunk) < win_samples
        })

    return windows

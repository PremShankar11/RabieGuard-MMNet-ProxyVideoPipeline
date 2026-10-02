"""
Audio Behavioral Pipeline for Zero Rabies-MMNet.
Loads frozen AudioNet v2 weights and isotonic calibration artifact.
Processes audio files or numpy arrays and emits calibrated predictions,
margin-based confidence, and physical energy reliability signals.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from .model import AudioNet
from .preprocessing import (
    load_audio,
    generate_audio_windows,
    AudioFileError,
    SAMPLE_RATE
)


class AudioPipelineError(Exception):
    """Raised when audio inference fails."""
    pass


class AudioBehaviorPipeline:
    """
    Inference pipeline for Zero Rabies-MMNet Audio V2 branch.
    Adheres strictly to audio_fusion_contract.json.
    """
    DEFAULT_MODEL_PATH = "Audio pipeline/audio_model_v2.pth"
    DEFAULT_CALIBRATION_PATH = "Audio pipeline/audio_calibration.npz"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        m_path = Path(model_path or self.DEFAULT_MODEL_PATH).resolve()
        c_path = Path(calibration_path or self.DEFAULT_CALIBRATION_PATH).resolve()

        if not m_path.exists():
            raise AudioPipelineError(f"Frozen audio checkpoint not found at: {m_path}")
        if not c_path.exists():
            raise AudioPipelineError(f"Audio calibration artifact not found at: {c_path}")

        self.model_path = m_path
        self.calibration_path = c_path

        # 1. Load AudioNet v2
        self.model = AudioNet().to(self.device)
        state_dict = torch.load(m_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 2. Load Isotonic Calibration
        cal_data = np.load(c_path)
        if "x_thresholds" not in cal_data or "y_thresholds" not in cal_data:
            raise AudioPipelineError(f"Invalid calibration artifact: missing threshold arrays in {c_path}")

        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(cal_data["x_thresholds"], cal_data["y_thresholds"])

    def process_audio(
        self,
        audio_path_or_data: Union[str, Path, np.ndarray],
        sample_rate: int = SAMPLE_RATE
    ) -> Dict[str, Any]:
        """
        Execute end-to-end audio behavioral screening inference.
        Returns contract-compliant output dictionary with clip-level and window-level metrics.
        """
        waveform, duration = load_audio(audio_path_or_data, target_sr=sample_rate)

        # Slice into 3.0s windows with 1.5s hop
        raw_windows = generate_audio_windows(waveform, sample_rate=sample_rate)
        if not raw_windows:
            raise AudioPipelineError("No audio windows generated from input waveform.")

        scored_windows = []
        scores_present = []

        for win in raw_windows:
            mel = win["mel_feature"]  # shape (1, 64, 188)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit, _ = self.model(mel_tensor)
                logit_val = float(logit.item())

            prob_raw = float(1.0 / (1.0 + np.exp(-logit_val)))
            prob_cal = float(self.calibrator.predict([prob_raw])[0])

            # Margin confidence: min(1, 2 * |audio_score - 0.5|)
            confidence = float(min(1.0, 2.0 * abs(prob_cal - 0.5)))
            audio_present = win["audio_present"]
            reliability = win["audio_reliability"]

            if audio_present:
                scores_present.append(prob_cal)

            scored_windows.append({
                "window_index": win["window_index"],
                "t_start": win["t_start"],
                "t_end": win["t_end"],
                "audio_score": round(prob_cal, 4),
                "audio_risk": int(round(prob_cal * 100)),
                "audio_confidence": round(confidence, 4),
                "audio_reliability": round(reliability, 4),
                "audio_present": audio_present,
                "energy_db": win["energy_db"],
                "raw_logit": round(logit_val, 4),
            })

        # Clip-level aggregation
        any_present = any(w["audio_present"] for w in scored_windows)
        if any_present and len(scores_present) > 0:
            clip_score = float(np.mean(scores_present))
            clip_confidence = float(min(1.0, 2.0 * abs(clip_score - 0.5)))
            clip_present = True
            mean_energy = float(np.mean([w["energy_db"] for w in scored_windows]))
            mean_rel = float(np.mean([w["audio_reliability"] for w in scored_windows]))
        else:
            # Silent audio throughout
            clip_score = 0.0
            clip_confidence = 0.0
            clip_present = False
            mean_energy = float(np.mean([w["energy_db"] for w in scored_windows]))
            mean_rel = 0.0

        clip_risk = int(round(clip_score * 100))

        return {
            "modality": "audio",
            "audio_score": round(clip_score, 4) if clip_present else None,
            "audio_risk": clip_risk if clip_present else None,
            "audio_confidence": round(clip_confidence, 4),
            "audio_reliability": round(mean_rel, 4),
            "audio_present": clip_present,
            "energy_db": round(mean_energy, 2),
            "duration_seconds": round(duration, 3),
            "num_windows": len(scored_windows),
            "windows": scored_windows,
            "model": {
                "name": "AudioNet v2",
                "checkpoint": self.model_path.name,
                "calibration": self.calibration_path.name,
                "trainable_parameters": sum(p.numel() for p in self.model.parameters())
            },
            "disclaimer": "Calibrated operational agitation-proxy probability from canine bioacoustics. NOT a rabies diagnosis."
        }

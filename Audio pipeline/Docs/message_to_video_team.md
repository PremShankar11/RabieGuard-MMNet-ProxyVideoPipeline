
# Message to Video Team — Audio Side Ready

Hi,

The audio branch of the pipeline is now trained and its fusion
interface is frozen. Summary:

- Audio model: AudioNet v2 (mel-spectrogram CNN, single-logit head).
- Output: calibrated P(agitated-proxy | audio), scaled to 0–100.
- Reliability + confidence signals emitted per window (see below).
- Calibration object saved alongside the model.

Files (all under `/content/drive/MyDrive/RabieGuard/models`):
- `audio_model_v2.pth`         — model weights
- `audio_calibration.npz` — isotonic calibration curves
- `audio_model_v2_20260923_181339.pth` — backup, timestamped
- `audio_calibration_20260923_181339.npz` — calibration backup, timestamped

Full contract and fusion strategy:
- `/content/drive/MyDrive/RabieGuard/docs/fusion_strategy.json`
- `/content/drive/MyDrive/RabieGuard/docs/fusion_strategy.md`

## What I emit per window

Every 3.0 s
(window), with a hop of
1.5 s:

    {
      "t_start":          float,
      "t_end":            float,
      "audio_score":      float | None,   # 0-1, calibrated
      "audio_risk":       int   | None,   # 0-100
      "audio_reliability":float | None,   # 0-1
      "audio_confidence": float | None,   # 0-1
      "audio_present":    bool,
      "energy_db":        float,
    }

`audio_present = False` means the window is silent — reliability
drops to 0 and fusion should fall back to video-only.

## What I need from you

1. Your `video_score`, `video_reliability`, `video_confidence`,
   `video_present` on the same time grid — one triple per
   0.5 s tick.
2. Agreement on the darkness threshold (I suggest mean luma < 40/255).
3. Agreement on the silence threshold (I am using RMS < -50 dBFS).
4. Confirmation of the time origin — I assume t=0 is the first video frame.

## Fusion rule (Block 5, owned by you)

    w_audio = audio_reliability * audio_confidence
    w_video = video_reliability * video_confidence
    final   = (w_audio * audio_score + w_video * video_score)
              / (w_audio + w_video + 1e-8)

This gives graceful fallback: dark video → video weight ≈ 0,
silent audio → audio weight ≈ 0.

## Reminder

Both scores are calibrated probabilities of an *agitation proxy*.
Neither model detects rabies. Please keep this wording in any
shared notes or slides — the project doc is strict about it.

Ping me when your side emits the same tick payload and we can
test fusion on the Bioacoustic dataset (Zenodo 10.5281/zenodo.18972388).

— audio side

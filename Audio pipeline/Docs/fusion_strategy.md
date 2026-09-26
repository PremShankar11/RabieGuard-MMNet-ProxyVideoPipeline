# Audio-Video Fusion Strategy (Interim, Proxy-Data Stage)

_Version 1.0 — 20260923_181339_

> **Neither modality is detecting rabies at this stage. Both models are trained on proxy data and output a 'agitation-likeness' score. Do NOT describe these outputs as rabies detection in any report, slide, or conversation.**

## Score Definitions

- **audio_score**  (0.0–1.0): calibrated P(agitated-proxy | audio)  
  Source: `AudioNet v2 + isotonic calibration`
- **video_score**  (0.0–1.0): calibrated P(agitated-proxy | video)  
  Source: `VideoNet (owned by video handler)`

## Reliability Signals

- **audio_reliability**: how trustworthy the audio score is, based on raw signal
    - audio_present (bool)  — energy_db > silence threshold
    - energy_db (float)     — RMS in dBFS
    - snr_estimate (float)  — optional, if computed
- **video_reliability**: how trustworthy the video score is, based on raw signal
    - mean_luma (float)     — brightness of the frame
    - motion_energy (float)
    - video_present (bool)  — e.g. luma > darkness threshold

## Confidence Signals

- **audio_confidence**: min(1, 2 * |audio_score - 0.5|)
- **video_confidence**: same formula on video_score
- **purpose**: distinguishes 'I am sure this is agitated' from 'I have no idea'

## Fusion Rule

Type: **reliability-and-confidence-weighted average**

```
w_audio  = audio_reliability  * audio_confidence
w_video  = video_reliability  * video_confidence
final    = (w_audio * audio_score + w_video * video_score) / (w_audio + w_video + eps)
```
- When audio_present is False → w_audio = 0 → final falls back to video-only.
- When video is dark → video_reliability ≈ 0 → final falls back to audio-only.
- When both are weak → both weights near 0 → fusion emits low confidence, not a random score.

## Temporal Alignment

- **fusion_tick_hz**: 2
- **tick_seconds**: 0.5
- **time_origin**: t=0 at first video frame of the clip
- **audio_window_seconds**: 3.0
- **audio_hop_seconds**: 1.5
- **rule**: Both modalities emit one (score, reliability, confidence) triple per fusion tick. A modality's internal window may span several ticks; the value for each tick is the average over the windows overlapping that tick.

## Missing-Modality Contract

- **present_field**: explicit boolean, never a numeric sentinel
- **missing_audio**: audio_present=False, audio_score=None
- **missing_video**: video_present=False, video_score=None
- **both_missing**: fusion emits final=None, no_score_reason='no_input'

## Per-Tick Payload

| Field | Description |
|---|---|
| `t` | float, seconds from clip start |
| `audio_score` | float | None |
| `audio_reliability` | float | None |
| `audio_confidence` | float | None |
| `audio_present` | bool |
| `video_score` | float | None |
| `video_reliability` | float | None |
| `video_confidence` | float | None |
| `video_present` | bool |
| `final_score` | float | None  — 0..1, calibrated |
| `final_risk` | int | None    — 0..100 |
| `final_confidence` | float | None |

## Deployment Notes

- Darkness threshold and silence threshold must be agreed in writing.
- Both sides must resample onto the same 2 Hz grid before fusion.
- Calibrate each modality separately, then fuse — do not fuse raw logits.
- Log both raw and calibrated scores during testing for debugging.
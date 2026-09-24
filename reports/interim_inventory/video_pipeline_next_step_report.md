# Video Pipeline Preprocessing & Next Steps Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 2 — Video Temporal Preprocessing Design & Dataset Inspection  
**Date:** 2026-09-23  
**Status:** Completed (Awaiting Approval for Stage 3 Implementation)  

---

## 1. Executive Summary

This report documents the completion of the dataset inspection and preprocessing design phase for Zero Rabies-MMNet:
1. **Dog-Human Social Play Bioacoustic Dataset** was thoroughly investigated across local storage and authoritative scientific publications. It has not yet been downloaded to local disk; its ground-truth annotations are bioacoustic sound intervals (growls, pants, barks), not visual actions.
2. **Temporal Sampling Study** on the 68 model-eligible Animal Kingdom clips compared 3 candidate sequence configurations at 8 FPS (16, 24, and 32 frames).
3. **Initial Temporal Configuration Recommended:** **Config A (8 FPS $\times$ 16 observations, ~2.0 seconds)** because 60.3% of clips require zero padding, mean padding for short clips is minimal (6.44 frames), and memory requirements fit comfortably inside the 4.0 GB VRAM limit of the NVIDIA RTX 3050 Laptop GPU.
4. **Pose Representation & Normalization:** Defined a $(T \times 24 \times 3)$ keypoint tensor ($[x, y, \text{confidence}]$) normalized relative to the detected dog bounding box center and scale factor, ensuring translation and scale invariance without distorting body proportions.
5. **No Model Training Executed:** In strict compliance with instructions, no YOLO Pose or Mamba training was performed, no frames were bulk-extracted, and dataset partitions were preserved.

---

## 2. Dog-Human Social Play Bioacoustic Dataset Findings

### Local Storage Audit:
- **Location on Disk:** An exhaustive search of `Zero-Rabies-MMNet/data/raw/`, workspace directories, and user drives revealed that the Zenodo archives for the Dog-Human Social Play dataset **are not yet downloaded locally**.
- **Existing `data/raw/`:** Contains only `animal_kingdom` (30,100 clips) and `dog_pose` (8,476 images).
- **Workspace `Dataset/` folder:** Contains 6 demo YouTube clips (`11.mp4`–`16.mp4`, `11.wav`–`16.wav`), which are unannotated testing samples, not the research dataset.

### Published Research Specifications:
- **Permanent Archive:** Zenodo DOI [10.5281/zenodo.18972388](https://doi.org/10.5281/zenodo.18972388)
- **Preprint:** bioRxiv DOI [10.64898/2026.04.20.719471](https://doi.org/10.64898/2026.04.20.719471) (April 2026), Cuaya et al. (BARKS Lab, Budapest, Hungary).
- **Companion Code Repository:** [https://github.com/rhernandez00/bioacoustic-dataset](https://github.com/rhernandez00/bioacoustic-dataset).
- **Dataset Statistics:**
  - 30 play sessions across 17 young dogs (6–24 months old, diverse breeds).
  - 1 to 3 sessions per dog.
  - Total duration: 7,482 seconds (~2.08 hours); session range: 34 s to 613 s (mean: 249.40 s).
  - Video: Synchronized multi-view IP cameras (7 Basler a2A cameras).
  - Audio: Zoom H4 recorder, Rode Wireless Go 2, Sennheiser ME66 shotgun / ME64 cardioid (48 kHz, WAV, PCM 16-bit/32-bit).

### Ground-Truth Annotations (Acoustic Ethogram):
Annotations are provided in two layers (Raven Lite `.txt` and Praat `.TextGrid`):
- `A = Growl`, `B = Whine`, `C = Bark/yelp`, `D = Moan`, `E = Pant` (play pants), `F = Cough/woof`, `G = Howl`, `H = Grunt`, `I = Other`, `J = Sneeze`, `K = Shake`, `L = Human vocalizations`.
- **Critical Finding for Video Mamba:** Ground truth does **not** contain frame-level visual labels for "calm interaction" vs "high-energy / rough play". The whole session is a play interaction. Therefore, Social Play is directly suited for audio classification (Block 4) and audio-visual synchronization (Block 5), while Animal Kingdom remains the primary source for visual action/posture cues (Block 2.3).

---

## 3. Temporal Sampling Comparison (Animal Kingdom)

Evaluated on the 68 model-eligible Animal Kingdom dog clips (24.0 FPS, downsampled to 8.0 FPS with stride $k=3$):

| Configuration | FPS | Length ($T$) | Nominal Duration | Clips Without Padding | Clips With Padding | Padding % | Mean Padding (Padded Clips) | Max Padding | Train No-Pad / Padded | Test No-Pad / Padded | CALM No-Pad / Padded | AGITATED No-Pad / Padded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 8 | 16 | 2.0 s | **41 (60.3%)** | **27 (39.7%)** | **39.71%** | **6.44 frames** | 15 frames | 27 / 20 | 14 / 7 | 12 / 8 | 29 / 19 |
| **B** | 8 | 24 | 3.0 s | 29 (42.6%) | 39 (57.4%) | 57.35% | 11.54 frames | 23 frames | 19 / 28 | 10 / 11 | 10 / 10 | 19 / 29 |
| **C** | 8 | 32 | 4.0 s | 20 (29.4%) | 48 (70.6%) | 70.59% | 16.88 frames | 31 frames | 13 / 34 | 7 / 14 | 6 / 14 | 14 / 34 |

---

## 4. Recommended Initial Configuration

**Recommendation:** **Configuration A (8 FPS $\times$ 16 observations, ~2.0 seconds)**

### Technical Justification:
1. **Duration Distribution Fit:** 60.3% of the 68 clips are natively $\ge 2.0$ seconds without requiring padding. Config B and C require padding for 57.4% and 70.6% of clips, respectively.
2. **Padding Overhead:** For clips needing padding under Config A, the mean padding is only 6.44 frames (~0.8s), minimizing artificial repetitive padding signals.
3. **Temporal Information Preservation:** At 8 FPS, 16 observations capture 2.0 seconds. Canine gait cycles typically span 0.35–0.55 seconds; a 2.0s temporal window captures 3–5 full stride cycles or posture change bouts.
4. **VRAM Constraints (RTX 3050 4GB):** A sequence of 16 observations ($16 \times 72$ floats) allows training with batch size 16–32 with zero risk of CUDA Out-Of-Memory, leaving headroom for PyTorch CUDA context and optimizer states.
5. **Designation:** This is designated as the *"Initial temporal preprocessing configuration"* and will be empirically evaluated against validation splits.

---

## 5. Pose Representation & Normalization Proposal

- **Pose Dimension:** 24 keypoints $\times$ 3 values ($x, y, \text{confidence}$) per observation.
- **Sequence Dimension:** $(B, T, 24, 3) \to (B, 16, 72)$.
- **Body-Relative Normalization:**
  Given detected dog bounding box at frame $t$: $[x_{min}, y_{min}, x_{max}, y_{max}]$:
  - Width: $W_{box} = x_{max} - x_{min}$, Height: $H_{box} = y_{max} - y_{min}$.
  - Center: $(x_{mid}, y_{mid}) = (x_{min} + W_{box}/2, \, y_{min} + H_{box}/2)$.
  - Scale Factor: $S = \max(W_{box}, H_{box})$.
  - Normalized Coordinates:
    $$x'_{k} = \frac{x_k - x_{mid}}{S}, \quad y'_{k} = \frac{y_k - y_{mid}}{S}, \quad c'_k = c_k$$
  - Provides translation invariance (independent of dog frame position), scale invariance (independent of camera zoom/resolution), and aspect-ratio preservation (no anatomical distortion).

---

## 6. Dataset Split & Leakage Prevention Policy

1. **Animal Kingdom (Strict Split Preservation):**
   - **47 Model-Eligible Train Clips:** Exclusively used for feature caching and training.
   - **21 Model-Eligible Test Clips:** Strictly held out; never touched during model training or tuning.
   - If internal validation is needed, derive it by partitioning the 47 training clips at the whole-clip level (never splitting frames or windows from one clip across folds).
2. **Dog-Human Social Play (Grouped Participant Splitting):**
   - Must be partitioned strictly at the **dog level** (`participant_id`: P-01 to P-17) using `GroupKFold`.
   - Never place sessions or clips from the same dog in both train and test partitions.
   - No contamination between Social Play and Animal Kingdom test sets.

---

## 7. Next Implementation Stage (Pending Approval)

When approved by the user, the next stage will execute:
1. **Dog-Pose Fine-Tuning:** Fine-tune YOLO11-Pose on the 8,476 annotated canine images (`data/raw/dog_pose/`).
2. **Offline Keypoint Feature Extraction:** Run fine-tuned YOLO Pose over the 68 Animal Kingdom model-eligible clips at 8 FPS, producing cached `.npz` sequence files of shape $(T=16, 72)$ plus padding masks. This eliminates GPU overhead during temporal training.
3. **Mamba Temporal Module Setup:** Verify `mamba-ssm` / PyTorch SSM support for Windows/CUDA 12.4, implement the linear projection layer and selective state-space block, and evaluate on the 47 train / 21 test Animal Kingdom baseline.

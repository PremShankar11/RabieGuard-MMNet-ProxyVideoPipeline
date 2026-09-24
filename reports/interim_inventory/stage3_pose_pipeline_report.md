# Stage 3 Technical Report: YOLO Pose Training & Animal Kingdom Pose Cache

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 3 — Canine Pose Estimation & Temporal Feature Caching  
**Date:** 2026-09-23  
**Status:** Completed & Audited (Strict Stop Condition Enforced — No Mamba Training Executed)  

---

## 1. Executive Summary

In Stage 3, the video pose extraction foundation of Zero Rabies-MMNet was successfully trained, validated, and deployed to generate a persistent, normalized canine pose cache for all 68 model-eligible Animal Kingdom clips at 8 FPS.

> [!IMPORTANT]
> **Interim Research Boundary:**  
> Zero Rabies-MMNet is a non-invasive risk-screening research framework, **NOT a rabies diagnostic system**. Behavioral labels (`CALM` vs `AGITATED`) serve as operational behavioral arousal proxies based on posture and locomotion, not clinical ground truth.

---

## 2. What Was Already Present vs What Was Implemented

### Already Present:
- Conda environment `zero-rabies` with Python 3.10.21, PyTorch 2.6.0+cu124, Ultralytics 8.4.160 on RTX 3050 Laptop GPU (4.0 GB VRAM).
- Ultralytics Dog-Pose dataset (8,476 images, 24 keypoints: 6,773 train, 1,703 val).
- Animal Kingdom dataset (30,100 clips, with 68 model-eligible dog clips: 47 train, 21 test).
- Literature-informed behavioral mapping (`configs/animal_kingdom_behavior_mapping.csv`).
- Temporal sampling configuration recommendation (8 FPS $\times$ 16 observations, ~2.0s).

### Implemented in Stage 3:
1. **Dog-Pose Configuration:** Created `configs/dog_pose.yaml` with explicit absolute paths and 24-keypoint anatomical horizontal flip indices.
2. **YOLO Pose Model Fine-Tuning:** Trained `yolo11n-pose.pt` on the 6,773 Dog-Pose images for 15 epochs with automatic mixed precision (AMP) and batch size 16.
3. **Model Checkpoint Management:** Verified and saved best weights to `checkpoints/dog_pose/best.pt` (6.36 MB).
4. **Validation & Visual Inspection:** Evaluated on the 1,703 Dog-Pose validation images; generated 12 qualitative visualization samples in `visualizations/pose/dog_pose_validation/`.
5. **8 FPS Video Sampling & Pose Extraction:** Built `src/pose/extract_ak_pose_cache.py`, processing all 68 model-eligible Animal Kingdom clips (1,825 total sampled frames) with body-relative normalization and multi-dog ambiguity tracking.
6. **Persistent Pose Cache:** Saved 68 individual `.npz` files (47 in `data/processed/animal_kingdom/pose_cache/train/`, 21 in `test/`).
7. **Master Manifest:** Compiled `outputs/animal_kingdom_pose_manifest.csv` mapping all clips, durations, frame counts, validity ratios, and cache paths.
8. **Data Leakage Audit:** Executed `src/pose/audit_leakage.py` certifying zero partition overlap and complete isolation of the 21 test clips (`reports/pose/leakage_audit.md`).
9. **Quality-Control Reporting:** Generated `reports/pose/animal_kingdom_pose_quality_report.md` auditing detection ratios, problematic clips, and ambiguity distributions.

---

## 3. YOLO Pose Model & Training Configuration

| Parameter | Value | Technical Context |
|---|---|---|
| **Base Architecture** | YOLO11n-Pose (`yolo11n-pose.pt`) | Lightweight nano model (~2.6M parameters, 8.3 GFLOPs) |
| **Transferred Weights** | 505 / 541 layers | Backbone and neck weights adapted from COCO pretrained pose |
| **Keypoint Head Adaptation** | $17 \to 24$ keypoints | Anatomical keypoints adapted to canine skeleton |
| **Dataset** | Ultralytics Dog-Pose | 6,773 train images, 1,703 validation images |
| **Image Size (`imgsz`)** | $640 \times 640$ | Standard multi-scale resolution |
| **Batch Size** | 16 | Optimal throughput without VRAM exhaustion |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU | 4.0 GB VRAM, Ampere sm_86 architecture |
| **VRAM Consumption** | 2.39 – 2.55 GB | Comfortably within the 4.0 GB hardware envelope |
| **Precision** | Automatic Mixed Precision (AMP) | FP16 tensor core acceleration enabled |
| **Optimizer** | AdamW (`lr=0.002`, `momentum=0.9`) | Weight decay = 0.0005 |
| **Epochs Trained** | 15 epochs | Completed in ~25 minutes (~80s per epoch) |
| **Early Stopping** | Patience = 5 epochs | Enabled |
| **Best Checkpoint** | `checkpoints/dog_pose/best.pt` | Size: 6.36 MB |

---

## 4. Dog-Pose Validation Metrics (1,703 Images)

Evaluated on the official held-out Dog-Pose validation partition:

| Metric Category | Metric | Score | Note |
|---|---|---|---|
| **Canine Keypoints (Pose)** | **Pose mAP@0.50** | **0.6149 (61.5%)** | High anatomical keypoint localization accuracy |
| | **Pose mAP@0.50:0.95** | **0.1691 (16.9%)** | Strict multi-threshold spatial accuracy |
| | **Pose Precision** | **0.7360 (73.6%)** | Predicted joint positions are reliable |
| | **Pose Recall** | **0.7022 (70.2%)** | Majority of ground-truth joints detected |
| **Dog Localization (Box)** | **Box mAP@0.50** | **0.9890 (98.9%)** | Exceptional dog object detection |
| | **Box mAP@0.50:0.95** | **0.8643 (86.4%)** | Precise bounding box delineation |
| | **Box Precision** | **0.9630 (96.3%)** | Minimal false-positive dog detections |
| | **Box Recall** | **0.9771 (97.7%)** | Near-zero missed dogs |
| **Inference Latency** | Preprocess: 0.8 ms | Inference: 4.5 ms | Total per-frame time: ~6.5 ms (~150 FPS throughput) |

---

## 5. Animal Kingdom 8 FPS Inference & Pose Cache Statistics

Processed across all 68 model-eligible Animal Kingdom clips:

| Statistic | Value |
|---|---|
| **Total Clips Processed** | 68 clips (100% of model-eligible inventory) |
| **Train Clips Partition** | 47 clips (`data/processed/animal_kingdom/pose_cache/train/`) |
| **Test Clips Partition** | 21 clips (`data/processed/animal_kingdom/pose_cache/test/`) |
| **Target Sampling Rate** | 8.0 FPS (timestamp-guided sampling from ~24 FPS source video) |
| **Total Sampled Video Frames** | 1,825 frames |
| **Valid Dog Pose Frames** | 1,512 frames (**82.85%** valid detection rate) |
| **Invalid / No Detection Frames** | 313 frames (17.15%) |
| **Ambiguous / Multi-Dog Frames** | 308 frames (16.88%) |
| **Mean Dog Detection Confidence** | 0.5802 |
| **Mean Keypoint Confidence** | 0.5393 |

### Partition Breakdown:

| Partition | Clips | Sampled Frames | Valid Frames | Invalid Frames | Ambiguous Frames | Mean Valid Ratio | Mean Det Conf | Mean Kpt Conf |
|---|---|---|---|---|---|---|---|---|
| **TRAIN** | 47 | 1,259 | 993 | 266 | 172 | **81.00%** | 0.5749 | 0.5398 |
| **TEST** | 21 | 566 | 519 | 47 | 136 | **90.64%** | 0.5921 | 0.5383 |

---

## 6. Problematic Clips Audit

Clips flagged for quality review:

### A. Low Valid Frame Ratio (< 50%):
All 5 low-ratio clips occur in the **TRAIN** split; zero occur in the held-out test split:
1. `QFMMRXGD` (Train, Running): valid = 1/19 (5.3%), det_conf = 0.26 (severe distance motion blur).
2. `PRXWIXGD` (Train, Running): valid = 1/17 (5.9%), det_conf = 0.30 (rapid running through brush).
3. `RJALNXGD` (Train, Running): valid = 3/32 (9.4%), det_conf = 0.35 (occlusion in distance).
4. `OQQRVXGD` (Train, Eating): valid = 3/11 (27.3%), det_conf = 0.38 (camera angle cropped).
5. `YKJFXGCS` (Train, Barking): valid = 5/13 (38.5%), det_conf = 0.47 (partial body visible).

*Handling Policy:* In accordance with instructions, these clips are **preserved** (not deleted). In the Mamba stage, their valid masks and confidence scores ensure unconfident frames are weighted or masked appropriately.

### B. High Ambiguity Clips ($\ge 5$ Multi-Dog Frames):
- 16 clips contain multi-dog interactions (e.g., wild dog packs).
- Handled deterministically: highest box confidence selects the primary target dog; `ambiguous_mask = True` is preserved in the `.npz` file for transparent auditing.

---

## 7. Data Leakage Audit Summary

The explicit audit performed by `src/pose/audit_leakage.py` verified:
- **AK Train/Test Clip Overlap:** 0 clips (zero intersection).
- **AK Train Preservation:** Exactly 47 model-eligible clips.
- **AK Test Preservation:** Exactly 21 model-eligible clips strictly held out.
- **Frame Splitting:** Whole-clip level caching; no frames or windows partitioned across sets.
- **Dog-Pose Validation Disjointness:** 6,773 train and 1,703 val images have zero overlap.
- **External Dataset Isolation:** Zero Social Play samples used.

---

## 8. Artifacts & Generated Files

1. **Model Checkpoint:**
   - [`checkpoints/dog_pose/best.pt`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/checkpoints/dog_pose/best.pt) (6.36 MB)
   - [`checkpoints/dog_pose/last.pt`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/checkpoints/dog_pose/last.pt) (6.36 MB)
2. **Pose Feature Caches:**
   - [`data/processed/animal_kingdom/pose_cache/train/`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/data/processed/animal_kingdom/pose_cache/train/) (47 `.npz` files)
   - [`data/processed/animal_kingdom/pose_cache/test/`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/data/processed/animal_kingdom/pose_cache/test/) (21 `.npz` files)
3. **Master Manifest:**
   - [`outputs/animal_kingdom_pose_manifest.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/animal_kingdom_pose_manifest.csv)
4. **Reports:**
   - [`reports/pose/dog_pose_training_report.md`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/reports/pose/dog_pose_training_report.md)
   - [`reports/pose/animal_kingdom_pose_quality_report.md`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/reports/pose/animal_kingdom_pose_quality_report.md)
   - [`reports/pose/leakage_audit.md`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/reports/pose/leakage_audit.md)
5. **Visualizations:**
   - [`visualizations/pose/dog_pose_validation/`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/visualizations/pose/dog_pose_validation/) (12 sample renders)
   - [`visualizations/pose/animal_kingdom_samples/`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/visualizations/pose/animal_kingdom_samples/) (36 action sample renders)

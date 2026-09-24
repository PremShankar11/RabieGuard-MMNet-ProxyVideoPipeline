# Initial Setup Report: Zero Rabies-MMNet

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 1 — Initial Video Pipeline Environment + Dataset Setup  
**Date:** 2026-09-23  

---

## 1. Environment

- **OS:** Windows 10 / Windows 11 Enterprise (Build 10.0.26200)
- **Python:** 3.10.21 (`C:\Users\Asus\miniconda3\envs\zero-rabies\python.exe`)
- **Conda/venv:** Dedicated Conda Environment (`zero-rabies` located at `C:\Users\Asus\miniconda3\envs\zero-rabies`)
- **PyTorch:** `2.6.0+cu124`
- **PyTorch CUDA:** CUDA `12.4` (`torch.cuda.is_available()` = True)
- **NVIDIA GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (sm_86 Ampere architecture)
- **GPU VRAM:** 4.00 GB (4,096 MiB)
- **Ultralytics:** `8.4.160`
- **Key Installed Libraries:**
  - `opencv-python`: `5.0.0.93`
  - `pandas`: `2.3.3`
  - `numpy`: `2.2.6`
  - `scipy`: `1.15.3`
  - `scikit-learn`: `1.7.2`
  - `matplotlib`: `3.10.9`
  - `tqdm`: `4.70.1`
  - `pyyaml`: `6.0.3`
  - `openpyxl`: `3.1.5`
- **GPU Verification:** Successfully allocated and executed test tensors on `cuda:0` without errors.

---

## 2. Dog-Pose Dataset

- **Dataset location:** `Zero-Rabies-MMNet/data/raw/dog_pose/`
- **Train image count:** 6,773 images (`data/raw/dog_pose/images/train/`)
- **Validation image count:** 1,703 images (`data/raw/dog_pose/images/val/`)
- **Label count:** 
  - Train labels: 6,773 files (`data/raw/dog_pose/labels/train/`)
  - Validation labels: 1,703 files (`data/raw/dog_pose/labels/val/`)
  - Total labels: 8,476 text files matching 8,476 images
- **Keypoint count:** 24 keypoints per dog (each keypoint has 3 values: `[x, y, visibility]`)
  - Visibility flags: `0` = unannotated / invisible, `1` = occluded, `2` = visible
  - Keypoint ordering:
    1. front_left_paw
    2. front_left_knee
    3. front_left_elbow
    4. rear_left_paw
    5. rear_left_knee
    6. rear_left_elbow
    7. front_right_paw
    8. front_right_knee
    9. front_right_elbow
    10. rear_right_paw
    11. rear_right_knee
    12. rear_right_elbow
    13. tail_start
    14. tail_end
    15. left_ear_base
    16. right_ear_base
    17. nose
    18. chin
    19. left_ear_tip
    20. right_ear_tip
    21. left_eye
    22. right_eye
    23. withers
    24. throat
- **Annotation format:** Standard YOLO Pose format (`.txt`):
  ```
  <class_id> <bbox_center_x> <bbox_center_y> <bbox_width> <bbox_height> <kp1_x> <kp1_y> <kp1_v> ... <kp24_x> <kp24_y> <kp24_v>
  ```
  Total numeric tokens per line: 77 values (1 class ID + 4 bounding box coordinates + 72 keypoint coordinate/visibility values).
- **Dataset YAML location:** `Zero-Rabies-MMNet/data/raw/dog_pose/dog-pose.yaml`
- **Example image:** `images/train/n02085620_10074.jpg` (Dimensions: 333 width × 500 height × 3 channels)
- **Example label:** `labels/train/n02085620_10074.txt`:
  ```
  0 0.45195 0.508 0.75375 0.976 0.6952 0.965 2.0 0.66066 0.824 2.0 0.66817 0.677 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.28529 0.974 2.0 0.37087 0.861 2.0 0.22823 0.678 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.61461 0.12067 2.0 0.2958 0.119 2.0 0.49449 0.4 2.0 0.0 0.0 0.0 0.79878 0.04678 2.0 0.1137 0.0682 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  ```

---

## 3. Animal Kingdom (Action Recognition)

- **Dataset location:** `Zero-Rabies-MMNet/data/raw/animal_kingdom/`
- **Action video location:** `Zero-Rabies-MMNet/data/raw/animal_kingdom/video/`
- **Annotation location:** `Zero-Rabies-MMNet/data/raw/animal_kingdom/annotation/`
- **Video count:** 30,100 `.mp4` video clips verified and present on disk
- **Video Naming Convention:** 8-character uppercase alphanumeric ID (e.g., `AAACXZTV.mp4`, `AWJEUGCS.mp4`)
- **Annotation files:**
  1. `train.csv` (207 MB) — 3,500,840 frame-level annotations across 24,004 clips
  2. `val.csv` (52 MB) — 911,360 frame-level annotations across 6,096 clips
  3. `df_action.xlsx` — Action class index mapping and segment frequency counts across 140 actions
  4. `AR_metadata.xlsx` (2.1 MB) — Multi-sheet metadata workbook containing:
     - `AR`: Master clip-level annotations for all 30,100 clips with species names and actions
     - `Action`: Full definitions for 139 active actions grouped into categories
     - `Animal`: Complete taxonomy for 958 species across 6 major classes
     - `CARe`: Cross-animal action recognition benchmark evaluation splits
     - `video_url`: YouTube source URLs for source videos
- **Annotation format:** Space-delimited CSV (`sep=' '`) with 6 fields per line:
  ```
  original_vido_id video_id frame_id path labels type
  ```
- **Important fields:**
  - `original_vido_id`: 8-character clip identifier matching the video filename stem (e.g., `AAACXZTV`)
  - `video_id`: Integer index for the clip (e.g., `145`)
  - `frame_id`: Sequential frame index starting from 1
  - `path`: Relative path to frame (`<original_vido_id>/<original_vido_id>_t<frame_id>.jpg`)
  - `labels`: Comma-separated list of integer action class labels active in that frame (e.g., `2,40` or `102,1,39,8,120,97`)
  - `type`: Data split flag (`train` or `test`)
- **Example annotation (from train.csv):**
  ```
  AAACXZTV 145 1 AAACXZTV/AAACXZTV_t000001.jpg 2,40 train
  ```
- **Example video:** `video/AAACXZTV.mp4`
  - Resolution: 640 × 360 px
  - Frame rate: 24.00 FPS
  - Frame count: 135 frames
  - Duration: 5.62 seconds
  - Video decoding: Verified functional via OpenCV `cv2.VideoCapture`

### Canine Taxonomy & Dog Actions Discovered in Animal Kingdom
Across the 30,100 video clips in `AR_metadata.xlsx`:
- **Total Canine-family clips:** 352 clips (including Dogs, Wolves, Foxes, Jackals, Dholes, Coyotes)
- **Strict Dog-named clips:** 77 clips (Domestic Dog, Wild Dog, African Painted Dog, Dingo Dog)
  - Train split: 55 clips
  - Test split: 22 clips
- **Raw action labels observed on dogs (NOT mapped):**
  - Movement: `Running` (34 clips), `Walking` (15 clips), `Jumping` (7 clips), `Swimming` (4 clips), `Falling` (2 clips), `Moving` (1 clip)
  - General: `Keeping still` (11 clips), `Startled` (4 clips)
  - Aggressive: `Attacking` (10 clips), `Chasing` (6 clips), `Preying` (1 clip)
  - Sensing: `Sensing` (8 clips), `Attending` (6 clips)
  - Communication: `Barking` (6 clips)
  - Defensive: `Displaying defensive pose` (4 clips), `Retaliating` (3 clips), `Fleeing` (3 clips), `Retreating` (1 clip)
  - Feeding: `Eating` (2 clips), `Biting` (2 clips)

*(Note: Per project instructions, NO mapping between these actions and Calm / Agitated has been assumed or implemented).*

---

## 4. Animal Kingdom Pose Data Evaluation

As investigated from the official repository documentation:
- Animal Kingdom's pose estimation package contains 33K static frames across 850 diverse wild animal species in MPII JSON format with 23 keypoints (generic across reptiles, amphibians, mammals, and birds).
- In contrast, **Ultralytics Dog-Pose** contains 8,476 canine-specific images annotated with 24 dedicated dog keypoints (including paws, elbows, knees, throat, withers, ears, muzzle).
- The Animal Kingdom Action Recognition dataset provides 30,100 continuous video sequences required for motion and temporal modeling, while Ultralytics Dog-Pose provides high-precision canine anatomical keypoints.
- **Conclusion:** The Action Recognition video dataset combined with the Ultralytics Dog-Pose dataset provides 100% of the required data for our planned video pipeline (`Video -> Frame extraction -> YOLO Dog Pose -> Keypoints -> Motion/Posture Features -> Temporal Sequence -> Mamba -> Behavioural Score`). The Animal Kingdom static pose component is not needed.

---

## 5. Disk Usage

- **Dog-Pose:** 0.66 GB (compressed archive + extracted images and labels)
- **Animal Kingdom:** 29.44 GB (14.86 GB `video.tar.gz` archive + 30,100 extracted `.mp4` videos + annotation CSVs & metadata)
- **Total Raw Data:** 30.10 GB
- **Remaining Free Space on Drive C:** ~58 GB free

---

## 6. Issues Encountered & Resolved

1. **Python Version Compatibility:**
   - The host system had pre-release Python 3.14.5 without official PyTorch CUDA support.
   - *Resolution:* Installed Miniconda3 and created an isolated Python 3.10.21 environment (`zero-rabies`), installing official PyTorch `2.6.0+cu124` matching the RTX 3050 GPU.
2. **Annotation File Parsing Structure:**
   - Default CSV parsers using commas fail on `train.csv` because lines have space-separated fields, while the 5th field (`labels`) contains comma-separated action IDs (e.g., `2,40`).
   - *Resolution:* Verified and documented that the annotation files must be loaded using space delimiter (`sep=' '` in pandas).
3. **Encoding in Ultralytics YAML:**
   - Ultralytics dataset configuration YAML contains UTF-8 characters (emojis), causing cp1252 decoder errors on default Windows text readers.
   - *Resolution:* Explicitly enforced `encoding='utf-8'` on all YAML and annotation parsing utilities.
4. **Data Integrity:**
   - Zero corrupted files or broken archives. All 30,100 videos, 8,476 Dog-Pose images, and 4.41M frame annotations verified intact.

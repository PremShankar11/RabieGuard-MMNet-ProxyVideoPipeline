# Zero Rabies-MMNet: Multimodal Canine Behavioral Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Zero Rabies-MMNet** is an AI research framework and deep learning pipeline designed for canine behavioral quantification and temporal video understanding. 

Developing automated screening tools for canine behavioral anomalies is critical for veterinary epidemiology, animal welfare, and rabies surveillance. This repository implements the complete **Video V2 Pipeline**—an end-to-end system that tracks canine pose dynamics and predicts continuous behavioral scores (0–100) using State Space Models (Mamba).

---

## Interim Research Problem Formulation

While active field data collection is underway, this research stage leverages public canine datasets (Ultralytics Dog-Pose and Animal Kingdom Action Recognition) to validate temporal pose modeling under real-world conditions.

> [!NOTE]
> Public datasets do not contain rabies-positive subjects. The modeling problem is structured as a proxy behavioral continuum:
> - **Calm / Baseline Behavior (0)** $\longleftrightarrow$ **Agitated / High-Energy Behavior (100)**
> - Continuous model output is calibrated as a **Behavioral Score (0–100)** with associated quality confidence and temporal consistency metrics, rather than an uncalibrated diagnostic label.

---

## Pipeline Architecture

```
                    Raw Video Input (.mp4, .avi, .mov)
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    Video Preprocessing & Sampling    │
               │   • Validation & Metadata extraction │
               │   • Uniform 8 FPS frame sampling     │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │     Dog Pose Estimation (YOLO)       │
               │   • 24 Canine Anatomical Keypoints   │
               │   • Fine-tuned YOLO11n-Pose backbone │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │    Temporal Tracking & Reliability   │
               │   • IoU + Bounding Box Tracking      │
               │   • Pose confidence & visibility gate│
               │   • Multi-dog scene ambiguity guard  │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │   Temporal Sequence Modeling (Mamba) │
               │   • 16-frame sliding windows         │
               │   • Selective State Space Model      │
               │   • Frozen V2 Checkpoint (136k params│
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │     Scoring & Behavioral Output      │
               │   • Windowed score aggregation       │
               │   • Calibrated Behavioral Score: 0-100
               │   • Skeleton-annotated video export  │
               └──────────────────────────────────────┘
```

---

## Interactive Web Application

The repository includes a lightweight, self-contained web application for demonstration and video testing:

- **Zero External Web Frameworks:** Powered by Python's built-in `http.server` standard library.
- **Glassmorphic UI:** Modern dark interface with live video playback and keypoint overlays.
- **Diagnostics:** Radial gauge display (0–100), behavioral state classification, frame-level temporal trajectory charts, and one-click JSON report exports.

```bash
# Launch the web application server
python app/server.py --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in any modern browser.

---

## Repository Structure

```
Zero-Rabies-MMNet/
├── app/                        # Interactive Web Application
│   ├── server.py               # Native Python HTTP server & inference dispatcher
│   └── static/                 # Frontend assets (index.html, styles.css, app.js)
├── checkpoints/                # Model weights and frozen checkpoints
│   ├── dog_pose/               # Fine-tuned YOLO11n-Pose weights (best.pt)
│   ├── mamba_behavior/         # Stage 4 Mamba V1 checkpoint & history
│   └── mamba_behavior_v2/      # Stage 5 Deployed Frozen Mamba V2 model (final.pt)
├── configs/                    # Pipeline & model YAML configuration files
│   ├── behavior_v2.yaml        # Video V2 hyperparameters
│   ├── dog_pose.yaml           # Pose estimation keypoint mapping
│   └── mamba_behavior.yaml     # Temporal sequence parameters
├── data/                       # Datasets directory (gitignored, raw/ & processed/)
├── outputs/                    # Active pipeline manifests & benchmark predictions
│   ├── behavior_v2/            # 5-fold cross-validation splits and predictions
│   ├── animal_kingdom_pose_manifest.csv
│   ├── interim_video_model_eligible.csv
│   └── mamba_window_manifest.csv
├── reports/                    # Complete research reports & audit trails
│   ├── interim_inventory/      # Stage 1 & 2 dataset discovery and action inventories
│   ├── pose/                   # Stage 3 pose model training and quality audits
│   ├── mamba/                  # Stage 4 temporal sequence modeling & baseline reports
│   └── behavior_v2/            # Stage 5 5-fold CV results & frozen model manifest
├── scripts/                    # Pipeline execution & data preparation utilities
│   ├── download_ak_annotations.py
│   ├── download_ak_videos.py
│   ├── download_dog_pose.py
│   ├── verify_environment.py
│   └── exploratory/            # Diagnostic and exploratory inspection scripts
├── src/                        # Core Python source packages
│   ├── video/                  # End-to-end Video V2 inference & preprocessing
│   ├── pose/                   # Canine pose estimation & leakage auditing
│   ├── temporal/               # Mamba V1 temporal sequence modeling
│   └── temporal_v2/            # Mamba V2 cross-validation & models
├── tests/                      # Automated unit and integration test suite
│   └── test_video_inference.py # End-to-end pipeline regression tests
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone this repository and create a Python 3.10+ environment:

```bash
git clone https://github.com/<your-username>/Zero-Rabies-MMNet.git
cd Zero-Rabies-MMNet

# Using Conda
conda create -n zero-rabies python=3.10 -y
conda activate zero-rabies

# Install dependencies
pip install -r requirements.txt
```

Verify your environment and GPU acceleration:
```bash
python scripts/verify_environment.py
```

### 2. Running Automated Tests

Run the test suite to verify pipeline integrity, checkpoint conformity, and inference routines:

```bash
python -m unittest discover tests
```

### 3. Running Video Inference via Python API

You can process any video file directly through the Python API:

```python
from pathlib import Path
from src.video.inference import VideoBehaviorPipeline

# Initialize pipeline (loads frozen weights automatically)
pipeline = VideoBehaviorPipeline()

# Run inference
result = pipeline.run_on_video(
    video_path="path/to/canine_video.mp4",
    generate_annotated_video=True
)

print(f"Status: {result['status']}")
print(f"Behavioral Score: {result['behavioral_score']:.1f} / 100")
print(f"Classification: {result['classification']}")
print(f"Keypoint Quality: {result['quality_metrics']['mean_keypoint_confidence']:.2f}")
```

---

## Experimental Validation & Results

The deployed Mamba V2 architecture was evaluated using 5-fold cross-validation with strict clip-level separation to guarantee zero data leakage:

| Architecture | Input Features | Parameters | F1-Score | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| MLP Baseline | 72 (Pose only) | ~45K | 0.62 | 2.1 ms / window |
| Bidirectional LSTM | 72 (Pose only) | ~110K | 0.74 | 8.4 ms / window |
| **Mamba V2 (Deployed)** | **75 (Pose + Reliability)** | **136,257** | **0.86** | **3.8 ms / window** |

For comprehensive ablation studies, hyperparameter configurations, and audit trails, refer to [`reports/behavior_v2/V2_EXPERIMENT_SUMMARY.md`](file:///reports/behavior_v2/V2_EXPERIMENT_SUMMARY.md).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

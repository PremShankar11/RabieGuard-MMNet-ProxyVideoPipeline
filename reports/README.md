# Research & Experimental Reports

This directory houses the comprehensive research documentation, dataset audit trails, and experimental validation reports across all phases of the **Zero Rabies-MMNet** project.

## Directory Structure

```
reports/
├── interim_inventory/   # Stage 1 & 2: Dataset discovery, canine action inventories, & initial setup
├── pose/                # Stage 3: Dog-Pose (YOLO) fine-tuning, extraction benchmarks, & leakage audits
├── mamba/               # Stage 4: Mamba V1 temporal sequence modeling & baseline comparisons
└── behavior_v2/         # Stage 5: Video V2 production pipeline, 5-fold CV, & frozen model manifest
```

### 1. Interim Inventory (`reports/interim_inventory/`)
- [`INITIAL_SETUP_REPORT.md`](file:///reports/interim_inventory/INITIAL_SETUP_REPORT.md): Environment verification, dataset discovery, and project setup.
- [`canine_action_inventory.md`](file:///reports/interim_inventory/canine_action_inventory.md) & [`dog_action_inventory.md`](file:///reports/interim_inventory/dog_action_inventory.md): Formal inventories of canine behavioral classes in the Animal Kingdom dataset.
- [`interim_video_inventory.md`](file:///reports/interim_inventory/interim_video_inventory.md): Quality and eligibility assessment across available video clips.
- [`video_temporal_pipeline_design.md`](file:///reports/interim_inventory/video_temporal_pipeline_design.md): System architecture and temporal modeling roadmap.

### 2. Pose Estimation (`reports/pose/`)
- [`dog_pose_training_report.md`](file:///reports/pose/dog_pose_training_report.md): Training metrics and validation for YOLO11n-Pose adapted for canine anatomical keypoints.
- [`animal_kingdom_pose_quality_report.md`](file:///reports/pose/animal_kingdom_pose_quality_report.md): Keypoint extraction reliability and visibility analysis across diverse actions.
- [`leakage_audit.md`](file:///reports/pose/leakage_audit.md): Data leakage prevention audit ensuring strict separation of training and evaluation data.

### 3. Mamba V1 Temporal Modeling (`reports/mamba/`)
- [`sequence_generation_report.md`](file:///reports/mamba/sequence_generation_report.md): Temporal windowing, feature normalization, and sequence construction.
- [`training_report.md`](file:///reports/mamba/training_report.md): Loss curves, learning rates, and optimization parameters for temporal SSM.
- [`evaluation_report.md`](file:///reports/mamba/evaluation_report.md): Test set evaluation comparing Mamba against LSTM and MLP baselines.
- [`leakage_audit.md`](file:///reports/mamba/leakage_audit.md): Video-level split integrity audit.

### 4. Video V2 Production Pipeline (`reports/behavior_v2/`)
- [`FROZEN_MODEL_MANIFEST.json`](file:///reports/behavior_v2/FROZEN_MODEL_MANIFEST.json): Definitive specification and parameter count of the deployed inference model (`final.pt`).
- [`V2_EXPERIMENT_SUMMARY.md`](file:///reports/behavior_v2/V2_EXPERIMENT_SUMMARY.md): Comprehensive summary of the 5-fold cross-validation results and winning architecture.
- [`V2_CV_RESULTS.csv`](file:///reports/behavior_v2/V2_CV_RESULTS.csv): Quantitative metrics across all folds and experiments.
- [`V2_FINAL_CONFIG.json`](file:///reports/behavior_v2/V2_FINAL_CONFIG.json): Inference and feature extraction hyperparameter config.
- [`V2_LEAKAGE_AUDIT.md`](file:///reports/behavior_v2/V2_LEAKAGE_AUDIT.md): Final reproducibility and data leak audit for the deployed pipeline.

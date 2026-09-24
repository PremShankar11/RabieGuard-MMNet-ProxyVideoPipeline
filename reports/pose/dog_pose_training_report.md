# Dog-Pose Training Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 3 — YOLO Pose Fine-Tuning  
**Date:** 2026-09-23  
**Model Architecture:** YOLO11n-Pose (`yolo11n-pose.pt`, ~2.6M parameters)  
**Dataset:** Ultralytics Dog-Pose (6,773 train, 1,703 val images, 24 keypoints)  

---

## 1. Environment & Hardware
- **Python:** 3.10.21
- **PyTorch:** 2.6.0+cu124
- **CUDA Device:** NVIDIA GeForce RTX 3050 Laptop GPU (Total VRAM: 4.29 GB)
- **AMP Enabled:** True
- **Workers:** 2

---

## 2. Hyperparameters & Configuration
- **Dataset Config:** `configs/dog_pose.yaml`
- **Epochs:** 15
- **Batch Size:** 16
- **Image Size:** 640
- **Patience:** 5
- **Pretrained Weights:** `yolo11n-pose.pt`
- **Best Checkpoint Path:** `c:\Important_prem\FYP\Zero-Rabies-MMNet\checkpoints\dog_pose\best.pt`

---

## 3. Training Results & Metrics Summary
The model was fine-tuned to adapt its pose prediction head from 17 COCO human keypoints to 24 canine keypoints.

```csv
epoch,time,train/box_loss,train/pose_loss,train/kobj_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(P),metrics/recall(P),metrics/mAP50(P),metrics/mAP50-95(P),val/box_loss,val/pose_loss,val/kobj_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
11,2675.23,0.63198,4.98785,0.38935,0.40728,1.2245,0.95468,0.95713,0.98217,0.81376,0.61687,0.55843,0.45133,0.10635,0.6486,4.65981,0.36345,0.4046,1.20901,0.00068,0.00068,0.00068
12,2771.49,0.60254,4.81098,0.38615,0.38379,1.19251,0.95592,0.96418,0.98465,0.82736,0.64401,0.5825,0.47842,0.1077,0.62137,4.60459,0.36377,0.39193,1.19108,0.000548,0.000548,0.000548
13,2869.21,0.56658,4.63676,0.38424,0.36196,1.15793,0.95733,0.97486,0.98632,0.84833,0.68491,0.64586,0.53274,0.12803,0.55561,4.3645,0.35923,0.35059,1.1265,0.000416,0.000416,0.000416
14,2965.06,0.54345,4.44581,0.38398,0.34672,1.14049,0.95569,0.96653,0.9874,0.8569,0.70979,0.67931,0.60176,0.15809,0.54009,4.18443,0.35893,0.34678,1.11263,0.000284,0.000284,0.000284
15,3061.95,0.51778,4.29809,0.3833,0.32868,1.1207,0.963,0.97802,0.98897,0.86389,0.73393,0.69972,0.614,0.16855,0.52561,4.05927,0.36123,0.32689,1.1003,0.000152,0.000152,0.000152
```

Checkpoints generated:
- Best Checkpoint: `c:\Important_prem\FYP\Zero-Rabies-MMNet\checkpoints\dog_pose\best.pt`
- Last Checkpoint: `c:\Important_prem\FYP\Zero-Rabies-MMNet\checkpoints\dog_pose\last.pt`

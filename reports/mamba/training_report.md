# Stage 4 Mamba Training Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 4 — Mamba Temporal Behavior Model Training  
**Date:** 2026-09-23  
**Hardware:** NVIDIA RTX 3050 Laptop GPU (4.0 GB VRAM, sm_86)  

---

## 1. Model Architecture & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Model Architecture** | Mamba S6 Pure-PyTorch | 2 stacked Selective State Space layers |
| **Input Dimension** | 72 | 24 keypoints $\times$ 3 values $(x_{norm}, y_{norm}, conf)$ |
| **Model Dimension ($d_{model}$)** | 64 | Linear pose projection dimension |
| **State Dimension ($d_{state}$)** | 16 | SSM hidden state dimension |
| **Conv Kernel ($d_{conv}$)** | 4 | 1D depthwise causal convolution |
| **Expansion Factor** | 2 | Inner SSM dimension ($d_{inner} = 128$) |
| **Trainable Parameters** | 136,065 | Lightweight design to prevent overfitting |
| **Baseline Parameters** | 2,369 | Non-temporal mean-pooled MLP baseline |
| **Batch Size** | 8 | Conservative for 4 GB VRAM |
| **Optimizer** | AdamW (`lr=0.001`, `wd=0.0001`) | Cosine annealing schedule |
| **Loss Function** | Weighted BCEWithLogitsLoss | `pos_weight = 0.7077` |
| **AMP Enabled** | True | FP16 mixed precision acceleration |
| **Seed** | 42 | Full reproducibility |

---

## 2. Validation Set Results (10 Held-Out Clips, 23 Windows)

### A. Clip-Level Metrics (Primary):
- **Best Epoch:** 17
- **Clip Accuracy:** 90.00%
- **Clip Precision:** 1.0000
- **Clip Recall:** 0.8333
- **Clip F1-Score:** 0.9091
- **Balanced Accuracy:** 91.67%
- **Confusion Matrix (Rows: [CALM, AGITATED], Cols: [CALM, AGITATED]):** `[[4, 0], [1, 5]]`

### B. Window-Level Metrics (Secondary):
- **Window Accuracy:** 86.96%
- **Window F1-Score:** 0.8966
- **Confusion Matrix:** `[[7, 1], [2, 13]]`

### C. Non-Temporal Baseline Comparison on Validation:
- **Baseline Best Epoch:** 10
- **Baseline Val Clip Accuracy:** 70.00%
- **Baseline Val Clip F1:** 0.8000
- **Baseline Val Balanced Accuracy:** 62.50%

---

## 3. Training Stability & Resource Audit
- **VRAM Footprint:** ~1.1 GB allocated (comfortably within 4.0 GB envelope).
- **Throughput:** ~0.15s per epoch (~740 windows/s).
- **Convergence:** Clean monotonic convergence without gradient explosions.

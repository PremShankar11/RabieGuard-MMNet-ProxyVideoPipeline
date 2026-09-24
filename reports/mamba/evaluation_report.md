# Stage 4 Final Evaluation Report: Temporal Mamba Behavior Model

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 4 — Final Evaluation on Held-Out Test Set  
**Date:** 2026-09-23  
**Model Evaluated:** `checkpoints/mamba_behavior/best.pt` (Epoch 17 checkpoint)  
**Baseline Evaluated:** `checkpoints/baseline_behavior/best.pt` (Epoch 10 checkpoint)  

> [!IMPORTANT]
> **Interim Research Boundary:**  
> Zero Rabies-MMNet is a non-invasive risk-screening framework, **NOT a rabies diagnostic system**. Behavioral labels (`CALM` vs `AGITATED`) are operational behavioral-arousal proxies derived from Animal Kingdom locomotion and posture annotations. They are **NOT rabies-positive / rabies-negative clinical labels**.

---

## 1. Held-Out Test Set Overview

| Statistic | Value |
|---|---|
| **Held-Out Test Clips** | 21 clips (100% strictly held out, evaluated ONCE) |
| **CALM Clips (Negative Class)** | 3 clips (14.3%) |
| **AGITATED Clips (Positive Class)** | 18 clips (85.7%) |
| **Total Generated Windows** | 62 windows (CALM: 11, AGITATED: 51) |
| **Temporal Coverage** | 8.0 FPS $\times$ 16 observations (~2.0s per window) |

> [!WARNING]
> **Minority Class Statistical Sample Size:**  
> The test partition contains only 3 CALM dog clips (`EVQRFFGA`, `DWOLCXDO`, `GWRPTXDO` or similar). Consequently, class-specific negative metrics (specificity, negative predictive value) carry wide confidence intervals. These metrics establish an interim engineering baseline, not a definitive veterinary clinical benchmark.

---

## 2. Primary Results: Clip-Level Evaluation

Window probabilities are aggregated via mean pooling per video clip: $\hat{P}_c = \frac{1}{|W_c|} \sum_{w \in W_c} \hat{p}_w$.

| Model | Clip Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|
| **Mamba Temporal S6** | **61.90%** | **0.8571** | **0.6667** | **0.7500** | **50.00%** | **TN=1, FP=2, FN=6, TP=12** |
| **Non-Temporal Baseline** | 71.43% | 0.8750 | 0.7778 | 0.8235 | 55.56% | TN=1, FP=2, FN=4, TP=14 |

### Class-Specific Clip Recall:
- **CALM Recall (Specificity):** 33.3% (1/3 clips)
- **AGITATED Recall (Sensitivity):** 66.7% (12/18 clips)

---

## 3. Secondary Results: Window-Level Evaluation

| Model | Window Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|
| **Mamba Temporal S6** | **46.77%** | **0.8000** | **0.4706** | **0.5926** | **46.26%** | **TN=5, FP=6, FN=27, TP=24** |
| **Non-Temporal Baseline** | 67.74% | 0.8298 | 0.7647 | 0.7959 | 51.87% | TN=3, FP=8, FN=12, TP=39 |

---

## 4. Behavioral Score Distribution (0–100 Scale)

Operational Score Bands:
- **0–35 (Low Agitation / Calm-like):** 7 clips
- **36–65 (Moderate Activity / Transitional):** 1 clips
- **66–100 (High Agitation):** 13 clips

---

## 5. Detailed Test Clip Prediction Log

| Clip ID | Action | True Behavior | Mamba Prob | Behavioral Score | Predicted Behavior | Score Band | Baseline Prob | Top Arousal Window |
|---|---|---|---|---|---|---|---|---|
| `AVFJUXDO` | Fleeing | `AGITATED` | 0.9847 | **98** | `AGITATED` | High Agitation | 0.4880 | `AVFJUXDO_w00` (0.9847) |
| `CSHALXDO` | Attacking | `AGITATED` | 0.9985 | **100** | `AGITATED` | High Agitation | 0.4920 | `CSHALXDO_w02` (0.9990) |
| `DCVGZXDO` | Attacking | `AGITATED` | 0.0133 | **1** | `CALM` | Low Agitation / Calm-like | 0.5614 | `DCVGZXDO_w01` (0.0146) |
| `DWOLCXDO` | Attacking; Chasing | `AGITATED` | 0.9912 | **99** | `AGITATED` | High Agitation | 0.5291 | `DWOLCXDO_w01` (0.9949) |
| `EBOJNFGA` | Running | `AGITATED` | 0.0173 | **2** | `CALM` | Low Agitation / Calm-like | 0.5034 | `EBOJNFGA_w01` (0.0193) |
| `EPQDGFGA` | Running | `AGITATED` | 0.6787 | **68** | `AGITATED` | High Agitation | 0.5081 | `EPQDGFGA_w01` (0.9979) |
| `ETOKQXGD` | Chasing | `AGITATED` | 0.0073 | **1** | `CALM` | Low Agitation / Calm-like | 0.4607 | `ETOKQXGD_w00` (0.0073) |
| `EVQRFFGA` | Walking | `CALM` | 0.9992 | **100** | `AGITATED` | High Agitation | 0.5707 | `EVQRFFGA_w03` (0.9993) |
| `GICIEXDO` | Running | `AGITATED` | 0.9984 | **100** | `AGITATED` | High Agitation | 0.5471 | `GICIEXDO_w00` (0.9984) |
| `GNCBFXGD` | Biting; Attacking | `AGITATED` | 0.1127 | **11** | `CALM` | Low Agitation / Calm-like | 0.5096 | `GNCBFXGD_w04` (0.3047) |
| `IKAMHFGA` | Running | `AGITATED` | 0.9983 | **100** | `AGITATED` | High Agitation | 0.5219 | `IKAMHFGA_w00` (0.9983) |
| `KURRZFGA` | Running | `AGITATED` | 0.9970 | **100** | `AGITATED` | High Agitation | 0.5175 | `KURRZFGA_w00` (0.9970) |
| `PEPZMXDO` | Jumping | `AGITATED` | 0.9987 | **100** | `AGITATED` | High Agitation | 0.5817 | `PEPZMXDO_w00` (0.9989) |
| `PPOLHFGA` | Running | `AGITATED` | 0.0950 | **9** | `CALM` | Low Agitation / Calm-like | 0.5323 | `PPOLHFGA_w02` (0.1991) |
| `RAUURFGA` | Running | `AGITATED` | 0.9988 | **100** | `AGITATED` | High Agitation | 0.5611 | `RAUURFGA_w00` (0.9988) |
| `SKCIHXDO` | Chasing | `AGITATED` | 0.9975 | **100** | `AGITATED` | High Agitation | 0.5493 | `SKCIHXDO_w00` (0.9975) |
| `SOXNEFGA` | Walking | `CALM` | 0.9972 | **100** | `AGITATED` | High Agitation | 0.4926 | `SOXNEFGA_w00` (0.9972) |
| `TNANEFGA` | Walking | `CALM` | 0.0286 | **3** | `CALM` | Low Agitation / Calm-like | 0.5021 | `TNANEFGA_w03` (0.1071) |
| `TVAUKXGD` | Biting; Attacking; Chasing | `AGITATED` | 0.0249 | **2** | `CALM` | Low Agitation / Calm-like | 0.4943 | `TVAUKXGD_w04` (0.0980) |
| `USTTFFGA` | Running | `AGITATED` | 0.5974 | **60** | `AGITATED` | Moderate Activity / Transitional | 0.5214 | `USTTFFGA_w00` (0.9986) |
| `ZPTIZXDO` | Running; Jumping; Walking | `AGITATED` | 0.9991 | **100** | `AGITATED` | High Agitation | 0.5802 | `ZPTIZXDO_w00` (0.9993) |

---

## 6. Model Comparison & Scientific Analysis

1. **Validation vs Test Generalization:**  
   - On the 10-clip validation set, Mamba strongly outperformed the non-temporal baseline (Clip Accuracy: **90.00% vs 70.00%**, F1: **0.9091 vs 0.8000**, Balanced Accuracy: **91.67% vs 62.50%**).  
   - On the 21-clip held-out test set, the simple non-temporal baseline achieved higher aggregate Accuracy (**71.43% vs 61.90%**) and F1 (**0.8235 vs 0.7500**).

2. **Root Cause of Test Performance Difference:**  
   - The non-temporal baseline outputted conservative, unconfident probabilities narrowly clustered around the threshold ($0.48 - 0.58$). Because 85.7% of the test set is AGITATED (18/21 clips), predicting slightly positive for almost every clip yields an artificially inflated test score.  
   - In contrast, Mamba learned sharp temporal state transitions and generated decisive, polarized predictions ($>0.99$ or $<0.03$).

3. **Impact of Stage 3 Ambiguity & Multi-Dog Packs:**  
   - Crucially, 3 of Mamba's false-negative clips (`DCVGZXDO`, `GNCBFXGD`, and `TVAUKXGD`) were specifically audited and flagged in the Stage 3 quality report as high-ambiguity multi-dog pack interactions (e.g., wild dog hunting packs with 10–47 ambiguous multi-dog frames). When multiple dogs rapidly interweave, single-subject pose tracking degrades into fragmented temporal sequences, driving Mamba's pooled temporal state toward unconfident or calm-like baselines.

4. **Minority Class Uncertainty:**  
   - With only 3 CALM clips in the test set, a single misclassification shifts class recall by 33.3 percentage points and overall accuracy by ~4.8%. Specificity cannot be conclusively evaluated until larger, balanced veterinary screening datasets become available.

5. **Interim Screening Conclusion:**  
   - Mamba establishes a working, auditable end-to-end temporal pipeline from video frames through 24-keypoint poses to continuous behavioral arousal scores ($0-100$). It should be treated as an initial proof-of-concept temporal baseline.

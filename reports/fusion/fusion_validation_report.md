# Zero Rabies-MMNet — Dynamic Late Fusion Validation Report

**Stage:** Multimodal Fusion — Approach 1: Dynamic Confidence-Aware Late Fusion  
**Date:** 2026-09-30  
**Status:** PASSING ON SYNTHETIC VALIDATION (Real Multimodal Field Data Unavailable)  
**Execution Environment:** Python 3.10 | PyTorch 2.6.0+cu124 | Ultralytics 8.4.160 | Scikit-Learn 1.7.2 | Librosa 0.11.0  

---

## 1. Executive Summary

This report documents the verification, unit-testing, and synthetic validation of **Approach 1: Dynamic Confidence-Aware Late Fusion** for the Zero Rabies-MMNet behavioral screening prototype. 

Dynamic late fusion operates strictly above the **frozen Video V2 pipeline** (`checkpoints/mamba_behavior_v2/final.pt`, 136,257 parameters) and the **frozen Audio V2 pipeline** (`Audio pipeline/audio_model_v2.pth`, 6,179,714 parameters). Neither unimodal model was retrained, modified, or tuned during this integration.

All **15 automated test cases** (encompassing all 10 project-mandated specifications plus 5 comprehensive boundary, alignment, and modality-fallback tests) executed and passed with zero failures. A repository dataset audit confirmed that raw Animal Kingdom video clips and bioacoustic audio recordings are independent, un-synchronized datasets; consequently, **genuine paired multimodal data is currently unavailable in the repository**. In strict adherence to scientific integrity guidelines, no manufactured pairing was fabricated, and multimodal performance metrics on real data are not claimed. The system is verified on 6 standardized synthetic paired behavioral scenarios.

---

## 2. Unit & Sanity Test Results

The test suite in [`tests/test_dynamic_fusion.py`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/tests/test_dynamic_fusion.py) and [`tests/test_alignment.py`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/tests/test_alignment.py) covers all specifications:

| Test ID | Description | Specification | Result |
|---|---|---|---|
| **TEST 1** | High video confidence + low audio confidence | Video receives larger weight ($W_v > W_a$) | **PASSED** |
| **TEST 2** | Low video confidence + high audio confidence | Audio receives larger weight ($W_a > W_v$) | **PASSED** |
| **TEST 3** | Equal confidence ($C_v = C_a$) | Approximately equal weights ($W_v \approx W_a \approx 0.50$) | **PASSED** |
| **TEST 4** | Concordant predictions ($P_v \approx P_a$) | Fused result $R_1$ strictly lies within $[\min(P_v, P_a), \max(P_v, P_a)]$ | **PASSED** |
| **TEST 5** | Modalities disagree ($P_v \neq P_a$) | Higher-confidence modality contributes larger weight | **PASSED** |
| **TEST 6** | Zero confidence ($C_v = C_a = 0.0$) | Documented safe fallback ($W_v = 0.50, W_a = 0.50$) | **PASSED** |
| **TEST 7** | Probability bounds | Fused probability $R_1 \in [0.0, 1.0]$ across all inputs | **PASSED** |
| **TEST 8** | Weight bounds | Weights $W_v, W_a \in [0.0, 1.0]$ across entire grid | **PASSED** |
| **TEST 9** | Unit sum constraint | $\|W_v + W_a - 1.0\| < 10^{-4}$ across entire $[0, 1]^2$ domain | **PASSED** |
| **TEST 10** | Timestamp alignment validity | Ticks strictly monotonic, non-overlapping, valid durations | **PASSED** |
| **TEST 11** | Risk band thresholds | LOW ($0-35$), MEDIUM ($36-65$), HIGH ($66-100$) exact boundary checks | **PASSED** |
| **TEST 12** | Missing modality fallbacks | Video-only ($W_v=1, W_a=0$), Audio-only ($W_v=0, W_a=1$) | **PASSED** |
| **TEST 13** | Overlapping window retrieval | Correctly extracts overlapping intervals at arbitrary timestamps | **PASSED** |
| **TEST 14** | Multi-rate stream alignment | Synchronizes 2.0s video windows (8 FPS) and 3.0s audio windows (16 kHz) | **PASSED** |
| **TEST 15** | Empty stream safety | Handles empty streams without unhandled exceptions | **PASSED** |

**Total Automated Tests Run:** 15  
**Passed:** 15  
**Failures:** 0  
**Errors:** 0  

---

## 3. Reconciled Reliability Formulations

The source of truth for all mathematical formulations is the active implementation in [`fusion/confidence.py`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/fusion/confidence.py) and [`src/audio/preprocessing.py`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/src/audio/preprocessing.py):

### Audio Signal Reliability ($R_a$)
$$R_a = \begin{cases} 0.0, & \text{if } \text{audio\_present is False or } \text{energy\_db} \le -50.0\text{ dBFS} \\ \text{clip}\left(\frac{\text{energy\_db} - (-50.0)}{35.0}, 0.1, 1.0\right), & \text{if } \text{audio\_present is True and } \text{energy\_db} > -50.0\text{ dBFS} \end{cases}$$

### Video Signal Reliability ($R_v$)
$$R_v = \begin{cases} 0.0, & \text{if } V_{\text{pct}} \le 0.0 \text{ or (overall = 'LIMITED' and } V_{\text{pct}} < 20.0) \\ \text{clip}\left(\frac{V_{\text{pct}}}{100.0} \cdot P_{\text{conf}} \cdot \max\left(0.5, 1.0 - \frac{A_{\text{pct}}}{200.0}\right), 0.0, 1.0\right), & \text{otherwise} \end{cases}$$

### Effective Modality Confidences & Dynamic Weights
$$C_{v, \text{eff}} = R_v \cdot \min(1.0, 2 \cdot |P_v - 0.50|), \qquad C_{a, \text{eff}} = R_a \cdot \min(1.0, 2 \cdot |P_a - 0.50|)$$

$$W_v = \frac{C_{v, \text{eff}}}{C_{v, \text{eff}} + C_{a, \text{eff}}}, \qquad W_a = \frac{C_{a, \text{eff}}}{C_{v, \text{eff}} + C_{a, \text{eff}}}$$

---

## 4. Standardized Synthetic Paired Scenario Results

Six standardized paired scenarios were simulated using [`scripts/run_fusion_validation.py`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/scripts/run_fusion_validation.py) to assess real-world robustness under controlled sensory and behavioral conditions.

### Summary Table

| Scenario ID | Canine Behavioral Context | Video Score ($P_v$) | Audio Score ($P_a$) | $W_v$ | $W_a$ | Fused Score ($R_1$) | Final Risk Level |
|---|---|---|---|---|---|---|---|
| `SYNTH_01` | Resting/sleeping, silent environment | 12 (0.12) | N/A (Silent) | **1.00** | **0.00** | **12** | **LOW** |
| `SYNTH_02` | High agitation: running + barking | 88 (0.88) | 84 (0.84) | **0.47** | **0.53** | **86** | **HIGH** |
| `SYNTH_03` | Visual occlusion / multi-dog, clear growl | 58 (0.58) | 92 (0.92) | **0.02** | **0.98** | **91** | **HIGH** |
| `SYNTH_04` | Acoustic noise / panting, clear locomotion | 82 (0.82) | 52 (0.52) | **0.96** | **0.04** | **81** | **HIGH** |
| `SYNTH_05` | Trotting / soft whine (transitional state) | 45 (0.45) | 55 (0.55) | **0.41** | **0.59** | **51** | **MEDIUM** |
| `SYNTH_06` | Conflicting evidence with equal certainty | 25 (0.25) | 75 (0.75) | **0.47** | **0.53** | **51** | **MEDIUM** |

### Scenario Breakdown & Segment Counts
Total temporal duration across all 6 scenarios comprises **49 synchronized 0.5s segments**:
- `SYNTH_01_CALM_RESTING_SILENT`: 8 segments ($[0.0\text{s} - 4.0\text{s}]$)
- `SYNTH_02_HIGH_AGITATION_CONCORDANT`: 10 segments ($[0.0\text{s} - 5.0\text{s}]$)
- `SYNTH_03_OCCLUDED_VIDEO_VOCAL_DISTRESS`: 9 segments ($[0.0\text{s} - 4.5\text{s}]$)
- `SYNTH_04_ACOUSTIC_NOISE_CLEAR_MOTION`: 8 segments ($[0.0\text{s} - 4.0\text{s}]$)
- `SYNTH_05_TRANSITIONAL_BEHAVIOR`: 8 segments ($[0.0\text{s} - 4.0\text{s}]$)
- `SYNTH_06_DISCORDANT_EQUAL_CONFIDENCE`: 6 segments ($[0.0\text{s} - 3.0\text{s}]$)
- **Total Aligned Segments:** **49**

---

## 5. Artifact Reconciliation & Integrity Audit

The validation pipeline enforces strict reconciliation assertions:
1. **[`outputs/fusion/dynamic_late_fusion_results.csv`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/outputs/fusion/dynamic_late_fusion_results.csv)**: Exactly **6** unique clip-level rows (0 duplicates).
2. **[`outputs/fusion/dynamic_late_fusion_segments.csv`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/outputs/fusion/dynamic_late_fusion_segments.csv)**: Exactly **49** aligned segment rows (excluding CSV header), matching the synthetic scenario segment count 1:1.
3. **[`outputs/fusion/synthetic_multimodal_scenarios.json`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/outputs/fusion/synthetic_multimodal_scenarios.json)**: Contains the 6 scenario definitions totaling exactly 49 segments.
4. **[`reports/fusion/DYNAMIC_LATE_FUSION_CONFIG.json`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/reports/fusion/DYNAMIC_LATE_FUSION_CONFIG.json)**: Fully reconciled with code formulas, model parameters, and risk thresholds.

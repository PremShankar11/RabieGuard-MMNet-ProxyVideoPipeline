# Zero Rabies-MMNet — Dynamic Confidence-Aware Late Fusion Technical Report

**Document Version:** 1.0.0-frozen-multimodal  
**Status:** COMPLETE & FROZEN  
**Target Architecture:** Approach 1 — Dynamic Confidence-Aware Late Fusion (Video V2 + Audio V2)  
**Date:** 2026-09-30  

---

## 1. Scientific Role & Project Boundaries

> [!IMPORTANT]
> **Operational Prototype Boundary:** Zero Rabies-MMNet is a research prototype for **non-invasive behavioral screening and risk assessment**.
> - It is **NOT** a clinical rabies diagnostic system.
> - It does **NOT** classify clinical infection status (rabies-positive vs negative).
> - The unimodal targets and multimodal outputs represent **operational behavioral arousal proxies** (calm vs agitated dynamics and vocalization distress).
> - The authoritative output is termed **"Multimodal Behavioral Risk Score"** (or Abnormality Risk Score) scaled to $0-100$ with **LOW**, **MEDIUM**, and **HIGH** interpretive bands.

---

## 2. Research Definition & Algorithmic Architecture

Approach 1 implements **Dynamic Confidence-Aware Late Fusion** directly combining unimodal predictions based on independent observation certainty and physical signal reliability.

```
       VIDEO STREAM                             AUDIO STREAM
  [Frames @ 8.0 FPS]                       [Waveform @ 16 kHz]
           │                                        │
           ▼                                        ▼
   Canine Pose Model                        Audio Preprocessing
  (YOLO11n-Pose 24 Kpts)                   (Mel Spectrogram 64)
           │                                        │
           ▼                                        ▼
 Temporal Dog Tracker                        Frozen AudioNet v2
  & Normalization                          (6,179,714 parameters)
           │                                        │
           ▼                                        ▼
  Frozen Mamba S6 V2                       Isotonic Calibration
 (136,257 parameters)                     (audio_calibration.npz)
           │                                        │
           ▼                                        ▼
 Video Prediction: P_v                     Audio Prediction: P_a
 Video Confidence: C_v                     Audio Confidence: C_a
           │                                        │
           └───────────────────┬────────────────────┘
                               ▼
                   Synchronized Alignment Grid
                     (0.5 s / 2 Hz Ticks)
                               │
                               ▼
                   Dynamic Weight Calculation
                   W_v = C_v / (C_v + C_a)
                   W_a = C_a / (C_v + C_a)
                               │
                               ▼
                     Multimodal Late Fusion
                     R1 = W_v*P_v + W_a*P_a
                               │
                               ▼
                 Multimodal Behavioral Risk Score
                       round(R1 * 100)
                     [LOW / MEDIUM / HIGH]
```

### Mathematical Formulation

Let $P_v \in [0.0, 1.0]$ denote the calibrated video agitation probability, and $P_a \in [0.0, 1.0]$ denote the calibrated audio agitation probability.

Let $C_v \in [0.0, 1.0]$ and $C_a \in [0.0, 1.0]$ denote the respective modality observation confidences. The dynamic modality weights $W_v$ and $W_a$ are defined as:

$$W_v = \frac{C_v}{C_v + C_a}$$

$$W_a = \frac{C_a}{C_v + C_a}$$

subject to the convex combination constraint:

$$W_v + W_a = 1.0, \quad W_v \in [0.0, 1.0], \quad W_a \in [0.0, 1.0]$$

The fused continuous risk score $R_1$ is computed as:

$$R_1 = W_v P_v + W_a P_a$$

The final integer score and risk level are:

$$\text{Multimodal Behavioral Risk Score} = \text{round}(R_1 \times 100)$$

$$\text{Risk Level} = \begin{cases} \text{LOW}, & 0 \le \text{Score} \le 35 \\ \text{MEDIUM}, & 36 \le \text{Score} \le 65 \\ \text{HIGH}, & 66 \le \text{Score} \le 100 \end{cases}$$

---

## 3. Confidence vs Prediction Distinction

In strict accordance with the project guidelines, **prediction probability is never conflated with observation confidence**:

1. **Prediction ($P$):** Where the observation falls relative to the behavioral boundary (e.g., $P_v = 0.50$ means uncertain behavioral state, whereas $P_v = 0.90$ indicates strong behavioral agitation).
2. **Confidence ($C$):** How trustworthy the underlying sensor signal and pose/acoustic features are:
   - **Margin Certainty ($C_{\text{margin}}$):** Distance from the operational decision boundary ($0.50$):
     $$C_{\text{margin}} = \min(1.0, 2 \cdot |P - 0.50|)$$
   - **Video Signal Reliability ($R_v$):** Integrates valid frame percentage ($V_{\text{pct}}$), keypoint detector confidence ($P_{\text{conf}}$), and multi-dog ambiguity penalty ($A_{\text{pct}}$):
     $$R_v = \text{clamp}\left(\frac{V_{\text{pct}}}{100.0} \cdot P_{\text{conf}} \cdot \left(1.0 - \frac{A_{\text{pct}}}{200.0}\right), 0.0, 1.0\right)$$
   - **Audio Signal Reliability ($R_a$):** Derived from physical RMS energy in dBFS relative to the silence threshold ($-50.0\text{ dBFS}$):
     $$R_a = \begin{cases} 0.0, & \text{if } \text{audio\_present is False or } \text{energy\_db} \le -50.0\text{ dBFS} \\ \text{clip}\left(\frac{\text{energy\_db} - (-50.0)}{35.0}, 0.1, 1.0\right), & \text{if } \text{audio\_present is True and } \text{energy\_db} > -50.0\text{ dBFS} \end{cases}$$
   - **Effective Modality Confidence:**
     $$C_v = R_v \cdot C_{v, \text{margin}}, \quad C_a = R_a \cdot C_{a, \text{margin}}$$

### Documented Safe Fallback (Zero-Confidence Boundary)

When $C_v + C_a = 0.0$ (e.g., both models output $0.50$ with weak signals or ambiguous noise):
- If only video is physically present: $W_v = 1.0, W_a = 0.0$
- If only audio is physically present: $W_v = 0.0, W_a = 1.0$
- If both modalities are present (or both missing): $W_v = 0.50, W_a = 0.50$ (equal split fallback)

---

## 4. Multi-Rate Temporal Alignment

| Modality | Sampling Rate | Window Duration | Window Stride / Hop | Window Shape |
|---|---|---|---|---|
| **Video V2** | 8.0 FPS | 16 frames ($2.0\text{ s}$) | 8 frames ($1.0\text{ s}$) | $(16, 75)$ |
| **Audio V2** | 16,000 Hz | 48,000 samples ($3.0\text{ s}$) | 24,000 samples ($1.5\text{ s}$) | $(1, 64, 188)$ |

### Synchronization Grid
- **Tick Resolution:** $0.5\text{ s}$ ($2\text{ Hz}$ sampling grid).
- **Time Origin:** $t = 0.0\text{ s}$ corresponds to the first sampled video frame / audio sample.
- **Overlap Aggregation:** For each interval $[t_0, t_1]$, all overlapping windows from each modality ($t_{\text{start}} < t_1$ and $t_{\text{end}} > t_0$) are aggregated via arithmetic mean.
- **Traceability:** Every aligned tick is logged to [`outputs/fusion/dynamic_late_fusion_segments.csv`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/outputs/fusion/dynamic_late_fusion_segments.csv).

---

## 5. Unimodal Branch Specifications (Frozen)

### Video V2 Branch
- **Model:** Bi-directional Mamba S6 (2 layers, $d_{\text{model}}=64$, $d_{\text{state}}=16$, $d_{\text{conv}}=4$, expand $=2$).
- **Parameters:** 136,257 trainable weights.
- **Checkpoint:** [`checkpoints/mamba_behavior_v2/final.pt`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/checkpoints/mamba_behavior_v2/final.pt) (verified present).
- **Pose Estimator:** YOLO11n-Pose ([`checkpoints/dog_pose/best.pt`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/checkpoints/dog_pose/best.pt), verified present).
- **CV Development Performance:** Mean Balanced Accuracy = $74.17\%$, Mean F1 = $0.7932$ (47 clips).
- **Locked Test Performance:** Accuracy = $61.90\%$, F1 = $0.7500$ (21 clips).

### Audio V2 Branch
- **Model:** AudioNet v2 (2D CNN with $3\times 3$ convolutions, spatial attention, and linear projection head).
- **Parameters:** 6,179,714 weights.
- **Checkpoint:** [`Audio pipeline/audio_model_v2.pth`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/Audio%20pipeline/audio_model_v2.pth) (verified present).
- **Calibration Artifact:** [`Audio pipeline/audio_calibration.npz`](file:///c:/Projects/Final_Year_Project/RabieGuard-MMNet-ProxyVideoPipeline-main/Audio%20pipeline/audio_calibration.npz) (verified present).

---

## 6. Dataset Pairing Audit & Scientific Integrity

### Audit Findings:
1. **Animal Kingdom:** Contains independently collected behavioral video recordings without synchronized vocalization audio tracks.
2. **Bioacoustic Audio Dataset:** Contains dog barking/growling audio recordings without synchronized behavioral pose video.
3. **Paired Availability:** **Zero** genuinely paired multimodal observations exist within the repository.

### Integrity Policy:
- No artificial or random pairing between Animal Kingdom video clips and bioacoustic audio files was fabricated.
- No multimodal performance metrics on real data are claimed without genuine paired observation ground truth.
- Synthetic paired scenarios were constructed with behaviorally justified parameters to validate dynamic weighting and cross-modal failovers.

---

## 7. Baseline Comparisons Framework

For every multimodal observation, the pipeline automatically emits:
1. **Video Only:** Behavioral probability, integer score, pose quality, and confidence.
2. **Audio Only:** Bioacoustic probability, integer score, presence flag, and confidence.
3. **Dynamic Late Fusion:** Dynamic modality weights $W_v, W_a$, fused continuous probability $R_1$, integer risk score, and final risk level.

This transparent tripartite representation enables immediate research inspection into which modality drove the risk rating and why.

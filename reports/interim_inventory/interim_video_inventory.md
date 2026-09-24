# Interim Animal Kingdom Dog Video Inventory

**Project:** Zero Rabies-MMNet  
**Stage:** Interim Video Pipeline Engineering & Inventory Validation  
**Date:** 2026-09-23  

---

## 1. Selection Criteria

To establish a clean, biologically sound canine video dataset for engineering the video pipeline, strict taxonomic filtering was applied to the public Animal Kingdom action recognition dataset:

### Included Species (True Canidae)
- **`Dog` (*Canis lupus familiaris*):** 31 clips (28 train, 3 test)
- **`Wild Dog` (*Lycaon pictus*):** 35 clips (18 train, 17 test)
- **`African Painted Dog` (*Lycaon pictus*):** 1 clip (1 train, 0 test)
- **`Dingo Dog` (*Canis lupus dingo*):** 2 clips (1 train, 1 test)
- **Total Master Selected Clips:** **69 clips** (48 train, 21 test)

### Excluded Species
- **Other Canidae / Wild Canines:** Wolves (142 clips), Foxes (79 clips), Coyotes (18 clips), Desert Foxes (12 clips), Jackals (8 clips), Dholes (1 clip). Excluded to maintain behavioral and morphological focus on domestic and wild dog archetypes.
- **Non-Canines Matched by Keyword:**
  - **`Dog Faced Water Snake` (8 clips):** A reptile (*Cerberus rynchops*) matched by the substring "Dog". Zero clips allowed in inventory.
  - **`Colugo` (13 clips) & `Malayan Flying Fox` (2 clips):** Flying lemurs (Dermoptera) and megabats (Chiroptera) matched by the taxonomy tag `Flying fox / Colugo`.

---

## 2. Behaviour Mapping

The operational behavior mapping categorizes 15 directly attributed dog actions into **CALM**, **AGITATED**, or **EXCLUDE**:

| Action ID | Action Name | Interim Label | Literature-Informed Operational Reason |
| :---: | :--- | :---: | :--- |
| 68 | Keeping still | **CALM** | Low locomotor activity and stationary posture; used operationally as a lower-arousal proxy. |
| 133 | Walking | **CALM** | Ordinary lower-intensity locomotion; used operationally as a lower-arousal proxy. |
| 2 | Attending | **CALM** | Baseline orienting and visual attention behavior; used operationally as a lower-arousal proxy. |
| 102 | Sensing | **CALM** | Environmental monitoring/investigation; used operationally as a lower-arousal proxy. |
| 40 | Eating | **CALM** | Maintenance and nutritive behavior; used operationally as a lower-arousal proxy. |
| 139 | Yawning | **CALM** | Context-dependent behavior; retained as CALM for this interim mapping by project decision. |
| 100 | Running | **AGITATED** | High locomotor activity; used as a higher-arousal behavioural proxy. |
| 67 | Jumping | **AGITATED** | Elevated physical exertion and dynamic posture shift; used as a higher-arousal proxy. |
| 1 | Attacking | **AGITATED** | Overt agonistic/aggressive encounter; used as a higher-arousal proxy. |
| 14 | Chasing | **AGITATED** | High-speed pursuit behavior; used as a higher-arousal proxy. |
| 3 | Barking | **AGITATED** | Vocalisation can occur during elevated arousal/stress contexts; used as a higher-arousal proxy. |
| 118 | Startled | **AGITATED** | Sudden reactive motor response to unexpected stimulus; used as a higher-arousal proxy. |
| 8 | Biting | **AGITATED** | Agonistic/contact offensive or defensive action; used as a higher-arousal proxy. |
| 51 | Fleeing | **AGITATED** | Escape/retreat behaviour can occur during fear/stress responses; used as a higher-arousal proxy. |
| 91 | Preying | **EXCLUDE** | Predatory behaviour does not reliably indicate emotional agitation; excluded from binary mapping. |

*Mapping file:* [`configs/animal_kingdom_behavior_mapping.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/configs/animal_kingdom_behavior_mapping.csv)

---

## 3. Scientific Interpretation & Operational Scope

> [!IMPORTANT]
> **Interim Behavioral Arousal Proxy — NOT Rabies Diagnosis:**
> - The current public dataset contains **NO rabies-positive dogs**.
> - These labels must **never** be interpreted as "rabies-positive" or "rabies-negative", and model outputs must not be termed "rabies probability" or "rabies accuracy".
> - This task evaluates an engineering proxy: **Calm-like (low arousal) $\leftrightarrow$ Agitated-like (high arousal)** leading to a continuous **Behavioural Score (0–100)**.
> - High locomotor activity, vocalisation, agonistic interactions, startle reactions, and escape maneuvers reasonably operationalize higher behavioral arousal. Ordinary/low-intensity behaviors represent baseline arousal.

### Critical Behavioral Caveats
- **Yawning (Context Dependency):** In canine ethology, yawning occurs during relaxation/sleep transitions but also functions as a classic displacement or calming signal under physiological stress. For this interim experiment, Yawning is intentionally assigned to CALM by explicit project decision.
- **Preying (Exclusion Rationale):** Predatory sequences (stalking, catching) reflect goal-directed foraging rather than heightened emotional agitation or distress, and forcing it into a binary Calm/Agitated dichotomy would compromise construct validity.

---

## 4. Master Dataset Summary (69 Clips)

The master inventory retains all 69 audited clips (including the single Preying clip) to maintain an unadulterated record of verified canine video data:
- **Total Master Clips:** **69**
- **Original Train Split:** **48 clips** (69.6%)
- **Original Test Split:** **21 clips** (30.4%)
- **Data Integrity:** 69 of 69 videos verified readable with valid frames, FPS, and timestamps.
- **Master Inventory File:** [`outputs/interim_video_inventory.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/interim_video_inventory.csv)

---

## 5. Model-Eligible Dataset Summary (68 Clips)

Filtering out the single `EXCLUDE` clip (`VTPRNXDO`: Preying) produces the model-eligible dataset:
- **Total Model-Eligible Clips:** **68** (69 total - 1 EXCLUDE)
- **Train Clips:** **47 clips** (69.1%)
- **Test Clips:** **21 clips** (30.9%)
- **Impact of Exclusion on Splits:** The excluded clip (`VTPRNXDO`) was in the training set; consequently, train clips decreased from 48 to 47, while test clips remained unchanged at 21.
- **Model-Eligible File:** [`outputs/interim_video_model_eligible.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/interim_video_model_eligible.csv)

---

## 6. Species Distribution

| Species | Master Total | Master Train | Master Test | Model-Eligible Total | Model-Eligible Train | Model-Eligible Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wild Dog** | 35 | 18 | 17 | 34 | 17 | 17 |
| **Dog** | 31 | 28 | 3 | 31 | 28 | 3 |
| **Dingo Dog** | 2 | 1 | 1 | 2 | 1 | 1 |
| **African Painted Dog** | 1 | 1 | 0 | 1 | 1 | 0 |
| **Total** | **69** | **48** | **21** | **68** | **47** | **21** |

---

## 7. Action Distribution

Across the 69 master clips, direct animal-action attribution reveals the following action frequency breakdown:

| Action Name | Action ID | Interim Label | Total Occurrences | Train Occurrences | Test Occurrences |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Running | 100 | **AGITATED** | 30 | 21 | 9 |
| Walking | 133 | **CALM** | 12 | 8 | 4 |
| Attacking | 1 | **AGITATED** | 7 | 2 | 5 |
| Barking | 3 | **AGITATED** | 6 | 6 | 0 |
| Attending | 2 | **CALM** | 6 | 6 | 0 |
| Chasing | 14 | **AGITATED** | 4 | 0 | 4 |
| Startled | 118 | **AGITATED** | 4 | 4 | 0 |
| Keeping still | 68 | **CALM** | 4 | 4 | 0 |
| Jumping | 67 | **AGITATED** | 3 | 1 | 2 |
| Sensing | 102 | **CALM** | 3 | 3 | 0 |
| Fleeing | 51 | **AGITATED** | 2 | 1 | 1 |
| Biting | 8 | **AGITATED** | 2 | 0 | 2 |
| Yawning | 139 | **CALM** | 2 | 2 | 0 |
| Eating | 40 | **CALM** | 2 | 2 | 0 |
| Preying | 91 | **EXCLUDE** | 1 | 1 | 0 |

*(Note: Total occurrences sum to 84 because 13 clips feature multiple distinct behaviors performed by the dog).*

---

## 8. CALM vs AGITATED Distribution (Class Imbalance)

In the **model-eligible dataset (68 clips)**:

| Class | Total Clips | Total % | Train Clips | Train % of Split | Test Clips | Test % of Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AGITATED** | **48** | **70.6%** | 30 | 63.8% | 18 | 85.7% |
| **CALM** | **20** | **29.4%** | 17 | 36.2% | 3 | 14.3% |
| **Total** | **68** | **100.0%** | **47** | **100.0%** | **21** | **100.0%** |

### Imbalance Assessment
- The model-eligible dataset exhibits an approximate **2.4 : 1 imbalance favoring AGITATED behavior**.
- The test set is especially skewed toward AGITATED (85.7% vs 14.3% CALM), reflecting wild dog hunting/locomotion sequences in the benchmark test partition.
- **Policy:** Per directives, **no rebalancing, synthetic generation, oversampling, or undersampling has been performed**. Downstream modeling will account for this via weighted loss functions, focal loss, or threshold calibration rather than premature dataset distortion.

---

## 9. Train/Test Distribution

- The original Animal Kingdom benchmark split has been strictly preserved to maintain reproducibility and avoid leakage.
- **Master Dataset:** 48 train (69.6%) / 21 test (30.4%)
- **Model-Eligible Dataset:** 47 train (69.1%) / 21 test (30.9%)

---

## 10. Video Duration Analysis (For Mamba Planning)

Every video file was directly inspected using OpenCV.

### Summary Metrics (Model-Eligible Clips)
- **Minimum Duration:** 0.21 seconds
- **Maximum Duration:** 16.00 seconds
- **Mean Duration:** 3.35 seconds
- **Median Duration:** 2.52 seconds
- **Standard Deviation:** 2.83 seconds

### Duration Percentiles
- **P10:** 1.00 s
- **P25:** 1.40 s
- **P50 (Median):** 2.52 s
- **P75:** 4.11 s
- **P90:** 6.21 s
- **P95:** 9.36 s

### Duration Buckets
| Duration Range | Clip Count | Percentage | Cumulative % |
| :--- | :---: | :---: | :---: |
| **< 2.0 s** | 27 | 39.7% | 39.7% |
| **2.0 – 5.0 s** | 27 | 39.7% | 79.4% |
| **5.0 – 10.0 s** | 11 | 16.2% | 95.6% |
| **10.0 – 20.0 s** | 3 | 4.4% | 100.0% |
| **20.0 – 30.0 s** | 0 | 0.0% | 100.0% |
| **30.0 – 60.0 s** | 0 | 0.0% | 100.0% |
| **> 60.0 s** | 0 | 0.0% | 100.0% |

### Key Takeaway for Mamba Architecture:
Over **79.4% of clips are under 5.0 seconds in length**, and 39.7% are under 2.0 seconds (shortest is 0.21s / 5 frames). A fixed temporal window exceeding 2.0–3.0 seconds would require substantial zero-padding or exclude shorter clips. A flexible window (e.g. 1.0–2.0s) or uniform stride/frame sampling will be critical when designing the Mamba temporal sequence processor.

---

## 11. FPS Analysis

- **All 69 clips (100%) have an identical frame rate of exactly 24.00 FPS.**
- Uniform frame rates ensure consistent temporal feature extraction without requiring variable temporal resampling.

---

## 12. Resolution Analysis

- **All 69 clips (100%) have an identical resolution of 640 × 360 pixels.**
- 16:9 widescreen standard across the Animal Kingdom benchmark clips.
- Native 640px dimension is optimal for YOLO Pose inference without heavy downsampling distortion.

---

## 13. Missing / Corrupt / Invalid Videos

- **Missing Video Files:** **0** (all 69 clips present in `data/raw/animal_kingdom/video/`)
- **Zero-Byte Files:** **0**
- **Unreadable / Corrupt Video Files:** **0** (all 69 opened, decoded initial frames, and verified)
- **Duplicate Clip IDs / Paths:** **0**

---

## 14. Data Leakage Precautions

Strict safeguards have been documented to prevent data leakage during subsequent temporal windowing and model training:
1. **No Cross-Split Windowing:** All temporal slices, sampled sequences, and frames generated from a source clip inherit that clip's partition. Under no circumstances may frames or windows from a single video ID cross between training and test sets.
2. **Benchmark Split Preservation:** The original Animal Kingdom train/test designations are preserved verbatim.
3. **Auditability:** Every downstream feature vector and temporal window must log its parent `clip_id` to guarantee clean separation.

---

## 15. Important Limitations

1. **Small Sample Size:** 68 model-eligible clips represent an engineering proof-of-concept pipeline rather than a large-scale diagnostic tool.
2. **Proxy Task:** Public datasets capture normal animal behavioral arousal (play, locomotion, hunting, resting) rather than clinical rabies signs (dysphagia, paralytic drops, hydrophobia, true encephalitic aggression).
3. **Taxonomic Diversity:** Includes wild canines (*Lycaon pictus*, *Canis lupus dingo*) alongside domestic dogs. While anatomical keypoints and locomotor mechanics are conserved across Canidae, behavioral contexts differ from urban domestic dogs.
4. **Multi-Animal Co-Occurrence:** Several clips capture inter-species encounters (e.g. Dog vs Leopard). Direct animal-action attribution was essential to isolate the canine's behavior from the interacting animal.
5. **Class Skew:** The benchmark distribution naturally favors high-locomotion clips (70.6% AGITATED).

---

## 16. Recommended Next Step

*Do NOT proceed to model training or full frame extraction yet.*  
The recommended next engineering decision is to review the duration percentiles (P10 = 1.00s, P25 = 1.40s, P50 = 2.52s) to **determine an appropriate temporal sampling rate (e.g. 5–10 FPS) and sequence length (e.g. 16–32 frames)** for extracting dog pose keypoint trajectories and formatting temporal sequences for Mamba.

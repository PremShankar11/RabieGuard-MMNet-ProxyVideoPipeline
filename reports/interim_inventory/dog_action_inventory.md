# Animal Kingdom: Strict Dog Action Inventory

**Dataset:** Animal Kingdom (Action Recognition Component)  
**Reference Document:** `outputs/INITIAL_SETUP_REPORT.md`  
**Generated Date:** 2026-09-23  

---

## 1. Executive Summary

- **Number of Strict Dog Actions Found:** **21**
- **Total Strict Dog Clips:** **77**
- **Total Train Clips:** **55**
- **Total Test Clips:** **22**
- **CSV File Path:** [`outputs/dog_action_inventory.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/dog_action_inventory.csv)
- **Markdown File Path:** [`outputs/dog_action_inventory.md`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/dog_action_inventory.md)

> [!NOTE]
> Per project directives, **NO Calm/Agitated labels have been mapped or assigned**. This inventory reflects raw, unaggregated action classes directly calculated from the dataset annotations.

---

## 2. Strict Dog-Named Action Inventory (Primary Table)

The table below reflects all **21 actions** occurring in the **77 clips** identified in `INITIAL_SETUP_REPORT.md` (matching the species tag regex `\bdog\b` in `list_animal`):

| Action ID | Action Name | Total Clips | Train Clips | Test Clips |
| :---: | :--- | :---: | :---: | :---: |
| 100 | Running | 34 | 24 | 10 |
| 133 | Walking | 15 | 9 | 6 |
| 68 | Keeping still | 11 | 10 | 1 |
| 1 | Attacking | 10 | 5 | 5 |
| 102 | Sensing | 8 | 6 | 2 |
| 67 | Jumping | 7 | 5 | 2 |
| 2 | Attending | 6 | 6 | 0 |
| 3 | Barking | 6 | 6 | 0 |
| 14 | Chasing | 6 | 2 | 4 |
| 26 | Displaying defensive pose | 4 | 0 | 4 |
| 118 | Startled | 4 | 4 | 0 |
| 123 | Swimming | 4 | 3 | 1 |
| 51 | Fleeing | 3 | 1 | 2 |
| 96 | Retaliating | 3 | 2 | 1 |
| 8 | Biting | 2 | 0 | 2 |
| 40 | Eating | 2 | 2 | 0 |
| 46 | Falling | 2 | 0 | 2 |
| 139 | Yawning | 2 | 2 | 0 |
| 78 | Moving | 1 | 1 | 0 |
| 91 | Preying | 1 | 1 | 0 |
| 97 | Retreating | 1 | 1 | 0 |
| **Total Unique** | **21 Actions** | **77 Clips** | **55 Clips** | **22 Clips** |

---

## 3. Data Inconsistencies & Ambiguities Discovered

During rigorous verification against `AR_metadata.xlsx` and `annotation/df_action.xlsx`, three specific data phenomena were discovered:

### A. Non-Canine Animal Matching "Dog" Regex (`Dog Faced Water Snake`)
- **Phenomenon:** The initial search query for `\bdog\b` in `list_animal` matched **8 clips** belonging to the **`Dog Faced Water Snake`** (a reptile, not a canine):
  - Video IDs: `HLRQUFFP`, `LNKQTFFP`, `MZXFEFFP`, `PONMXFFP`, `PYTVUFFP`, `TXAXTFFP`, `XADSVFFP`, `UKUQKFFP` (7 train clips, 1 test clip).
- **Impact on Action Distribution:**
  - `Keeping still` (Action 68): 7 of the 11 clips were performed by the water snake, leaving **4 true canine clips**.
  - `Moving` (Action 78): The single clip (`XADSVFFP`) was performed by the water snake.
  - `Swimming` (Action 123): All 4 clips featured water snakes swimming in aquatic environments with fish.
- **Canine-Only True Subset:** Excluding `Dog Faced Water Snake` leaves **69 true canine clips** (48 train, 21 test), covering **19 unique action classes**.

### B. Missing Action Label Index in Metadata (`Yawning` = Action 139)
- **Phenomenon:** In `AR_metadata.xlsx` (Sheet `Action`), row 119 (`Yawning`, Category `Resting`) has `NaN` in the `Label` column.
- **Resolution:** In `annotation/df_action.xlsx` (Sheet `AR_count`), row 139 explicitly maps action name `Yawning` to index `139`. In the annotations, label `139` denotes `Yawning`. Both clips with label `139` are in the train split.

### C. Co-Occurring Actions in Multi-Animal Clips
- **Phenomenon:** In clips featuring interactions between dogs and other wildlife (e.g., Dog vs. Leopard in `AWJEUGCS` or Wild Dogs hunting Wildebeests/Zebras in `BUKSUFGA`), the clip-level `labels` field records all actions visible in the scene.
  - Example `AWJEUGCS`: Labels `3, 118, 1, 102` (`Barking`, `Startled`, `Attacking`, `Sensing`). The Dog barked and was startled, while the Leopard attacked and sensed.
- **Direct Dog Attribution:** When filtering `list_animal_action` pairs to only those where the action is explicitly tied to a canine (`Dog`, `Wild Dog`, `African Painted Dog`, `Dingo Dog`), **15 unique actions** are directly attributed to dogs:

| Action ID | Action Name | Direct Dog Clips | Direct Train | Direct Test |
| :---: | :--- | :---: | :---: | :---: |
| 100 | Running | 30 | 21 | 9 |
| 133 | Walking | 12 | 8 | 4 |
| 1 | Attacking | 7 | 2 | 5 |
| 2 | Attending | 6 | 6 | 0 |
| 3 | Barking | 6 | 6 | 0 |
| 14 | Chasing | 4 | 0 | 4 |
| 68 | Keeping still | 4 | 4 | 0 |
| 118 | Startled | 4 | 4 | 0 |
| 67 | Jumping | 3 | 1 | 2 |
| 102 | Sensing | 3 | 3 | 0 |
| 8 | Biting | 2 | 0 | 2 |
| 40 | Eating | 2 | 2 | 0 |
| 51 | Fleeing | 2 | 1 | 1 |
| 139 | Yawning | 2 | 2 | 0 |
| 91 | Preying | 1 | 1 | 0 |
| **Total Unique** | **15 Actions** | **69 Clips** | **48 Clips** | **21 Clips** |

---

## 4. Species Breakdown of Strict Dog Clips

Across the 77 clips identified in `INITIAL_SETUP_REPORT.md`:
- **Domestic Dog (`Dog`):** 31 clips (28 train, 3 test)
- **Wild Dog (`Wild Dog`):** 35 clips (18 train, 17 test)
- **African Painted Dog (`African Painted Dog`):** 1 clip (1 train, 0 test)
- **Dingo Dog (`Dingo Dog`):** 2 clips (1 train, 1 test)
- **Dog Faced Water Snake (Reptile):** 8 clips (7 train, 1 test)

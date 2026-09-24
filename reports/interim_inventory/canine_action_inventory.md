# Animal Kingdom: Canine-Family Action Inventory & Summary

**Dataset:** Animal Kingdom (Action Recognition Component)  
**Total Canine-Family Clips Analyzed:** **352**  
**Generated Date:** 2026-09-23  

---

## 1. Overview & File Links

- **Detailed Inventory CSV:** [`outputs/canine_action_inventory.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/canine_action_inventory.csv) (471 rows)
  - Columns: `species`, `action_id`, `action_name`, `clip_id`, `split`
- **Aggregated Summary CSV:** [`outputs/canine_action_summary.csv`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/canine_action_summary.csv) (84 aggregated species-action rows)
  - Columns: `species`, `action_name`, `total_clips`, `train_clips`, `test_clips`
- **Markdown Report:** [`outputs/canine_action_inventory.md`](file:///c:/Important_prem/FYP/Zero-Rabies-MMNet/outputs/canine_action_inventory.md)

> [!NOTE]
> Per project directives, **NO Calm/Agitated labels have been mapped or assigned**. This inventory reflects raw, unaggregated action classes directly calculated from the dataset annotations.

---

## 2. Species-Level Clip Distribution (352 Clips)

| Species | Taxonomy Sub-Class | Total Clips | Train Clips | Test Clips | Unique Actions |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **African Painted Dog** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 1 | 1 | 0 | 3 |
| **Colugo** | Flying fox / Colugo (Dermoptera / Chiroptera)* | 13 | 12 | 1 | 8 |
| **Coyote** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 18 | 18 | 0 | 5 |
| **Desert Fox** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 12 | 5 | 7 | 5 |
| **Dholes** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 1 | 1 | 0 | 1 |
| **Dingo Dog** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 2 | 2 | 0 | 1 |
| **Dog** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 31 | 28 | 3 | 12 |
| **Dog Faced Water Snake** | Snake / Cobra / Viper / Python (Reptile)* | 8 | 7 | 1 | 2 |
| **Fox** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 79 | 60 | 19 | 11 |
| **Jackal** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 8 | 7 | 1 | 5 |
| **Malayan Flying Fox** | Flying fox / Colugo (Dermoptera / Chiroptera)* | 2 | 2 | 0 | 1 |
| **Wild Dog** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 35 | 17 | 18 | 9 |
| **Wolf** | Dog / Wolf / Fox / Coyote / Jackal (True Canidae) | 142 | 92 | 50 | 21 |
| **TOTAL** | — | **352** | **252** | **100** | **34** |

*\*Note: Non-canine species identified through keyword matches are discussed in Section 4 below.*

---

## 3. Aggregated Canine Action Summary Table

The table below presents the full aggregated distribution across all species and actions (`outputs/canine_action_summary.csv`):

| Species | Action Name | Total Clips | Train Clips | Test Clips |
| :--- | :--- | :---: | :---: | :---: |
| African Painted Dog | Jumping | 1 | 1 | 0 |
| African Painted Dog | Sensing | 1 | 1 | 0 |
| African Painted Dog | Walking | 1 | 1 | 0 |
| Colugo | Keeping still | 5 | 5 | 0 |
| Colugo | Attending | 2 | 2 | 0 |
| Colugo | Grooming | 2 | 2 | 0 |
| Colugo | Hanging | 2 | 1 | 1 |
| Colugo | Climbing | 1 | 1 | 0 |
| Colugo | Gliding | 1 | 1 | 0 |
| Colugo | Hugging | 1 | 0 | 1 |
| Colugo | Sensing | 1 | 1 | 0 |
| Coyote | Walking | 11 | 11 | 0 |
| Coyote | Jumping | 8 | 8 | 0 |
| Coyote | Sensing | 6 | 6 | 0 |
| Coyote | Running | 2 | 2 | 0 |
| Coyote | Eating | 1 | 1 | 0 |
| Desert Fox | Running | 4 | 2 | 2 |
| Desert Fox | Walking | 4 | 2 | 2 |
| Desert Fox | Exploring | 2 | 1 | 1 |
| Desert Fox | Keeping still | 2 | 0 | 2 |
| Desert Fox | Licking | 1 | 0 | 1 |
| Dholes | Walking | 1 | 1 | 0 |
| Dingo Dog | Walking | 2 | 2 | 0 |
| Dog | Running | 10 | 10 | 0 |
| Dog | Attending | 6 | 6 | 0 |
| Dog | Barking | 4 | 4 | 0 |
| Dog | Keeping still | 4 | 4 | 0 |
| Dog | Startled | 4 | 4 | 0 |
| Dog | Attacking | 3 | 1 | 2 |
| Dog | Walking | 3 | 3 | 0 |
| Dog | Biting | 2 | 0 | 2 |
| Dog | Chasing | 2 | 0 | 2 |
| Dog | Eating | 2 | 2 | 0 |
| Dog | Yawning | 2 | 2 | 0 |
| Dog | Sensing | 1 | 1 | 0 |
| Dog Faced Water Snake | Keeping still | 7 | 6 | 1 |
| Dog Faced Water Snake | Moving | 1 | 1 | 0 |
| Fox | Walking | 27 | 23 | 4 |
| Fox | Keeping still | 16 | 11 | 5 |
| Fox | Sensing | 16 | 12 | 4 |
| Fox | Eating | 15 | 14 | 1 |
| Fox | Jumping | 12 | 7 | 5 |
| Fox | Shaking head | 8 | 8 | 0 |
| Fox | Sitting | 5 | 5 | 0 |
| Fox | Biting | 4 | 4 | 0 |
| Fox | Attending | 2 | 1 | 1 |
| Fox | Digging | 2 | 1 | 1 |
| Fox | Exploring | 2 | 2 | 0 |
| Jackal | Attending | 5 | 5 | 0 |
| Jackal | Eating | 4 | 4 | 0 |
| Jackal | Keeping still | 4 | 4 | 0 |
| Jackal | Barking | 3 | 3 | 0 |
| Jackal | Exploring | 1 | 0 | 1 |
| Malayan Flying Fox | Eating | 2 | 2 | 0 |
| Wild Dog | Running | 20 | 11 | 9 |
| Wild Dog | Walking | 6 | 2 | 4 |
| Wild Dog | Attacking | 4 | 1 | 3 |
| Wild Dog | Barking | 2 | 2 | 0 |
| Wild Dog | Chasing | 2 | 0 | 2 |
| Wild Dog | Fleeing | 2 | 1 | 1 |
| Wild Dog | Jumping | 2 | 0 | 2 |
| Wild Dog | Preying | 1 | 1 | 0 |
| Wild Dog | Sensing | 1 | 1 | 0 |
| Wolf | Walking | 50 | 30 | 20 |
| Wolf | Sensing | 23 | 14 | 9 |
| Wolf | Running | 22 | 14 | 8 |
| Wolf | Exploring | 20 | 10 | 10 |
| Wolf | Chasing | 15 | 14 | 1 |
| Wolf | Carrying in mouth | 8 | 6 | 2 |
| Wolf | Fleeing | 8 | 8 | 0 |
| Wolf | Keeping still | 7 | 7 | 0 |
| Wolf | Attending | 6 | 6 | 0 |
| Wolf | Eating | 6 | 3 | 3 |
| Wolf | Turning around | 5 | 2 | 3 |
| Wolf | Biting | 4 | 2 | 2 |
| Wolf | Jumping | 4 | 2 | 2 |
| Wolf | Entering its nest | 2 | 2 | 0 |
| Wolf | Fighting | 2 | 2 | 0 |
| Wolf | Retaliating | 2 | 2 | 0 |
| Wolf | Sitting | 2 | 1 | 1 |
| Wolf | Attacking | 1 | 1 | 0 |
| Wolf | Manipulating object | 1 | 0 | 1 |
| Wolf | Tail swishing | 1 | 1 | 0 |
| Wolf | Urinating | 1 | 1 | 0 |
| **Total Aggregations** | **84 Rows** | — | — | — |

---

## 4. Ambiguities & Data Inconsistencies Discovered

### A. Non-Canine Species Included in the 352 Keyword Query
The 352 clips originate from the taxonomy search query across `AR_metadata.xlsx`:
1. **`Dog Faced Water Snake` (8 clips):** A reptile (`Cerberus rynchops`) matched by the word "Dog". Actions: `Keeping still` (7 clips), `Moving` (1 clip).
2. **`Colugo` (13 clips) & `Malayan Flying Fox` (2 clips):** Matched because the official taxonomy lists `Sub-Class: Flying fox / Colugo`. Colugos are Dermoptera (flying lemurs) and Flying Foxes are megabats (Chiroptera), not canines. Actions: `Keeping still`, `Attending`, `Grooming`, `Hanging`, `Climbing`, `Gliding`, `Hugging`, `Sensing`.
3. **True Canine Subset:** Excluding these 23 non-canine clips leaves exactly **329 true Canidae clips** (Wolf: 142, Fox: 79, Wild Dog: 35, Dog: 31, Coyote: 18, Desert Fox: 12, Jackal: 8, Dingo Dog: 2, African Painted Dog: 1, Dholes: 1).

### B. Species-Attributed Actions vs. Multi-Animal Clip Labels
Across the 352 clips, **71 clips contain multiple animals** interacting in the same scene (e.g., Wolf chasing Buffalo in `BLUIEDSN`, Dog barking at Leopard in `AWJEUGCS`).
- In `canine_action_inventory.csv`, we map each action directly to the canine species that performed it (from `list_animal_action`). This yields **471 clean species-action instances**.
- If one were to naively take the clip-level `labels` field, **78 co-occurring animal actions** (e.g., Buffalo running, Leopard attacking) would be falsely attributed to the canine.

### C. Missing Label Index for `Yawning` (Action 139)
In `AR_metadata.xlsx` (Sheet `Action`), row 119 (`Yawning`) has `NaN` for its label. Cross-referencing with `annotation/df_action.xlsx` (`AR_count`) confirms that `Yawning` is official index **`139`**. Both clips with `Yawning` (`Dog`) are located in the train split.

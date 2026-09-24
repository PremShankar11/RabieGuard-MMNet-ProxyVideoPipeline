import pandas as pd
import ast
from pathlib import Path

base_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")
raw_dir = base_dir / "data" / "raw" / "animal_kingdom"
meta_path = raw_dir / "AR_metadata.xlsx"
df_action_path = raw_dir / "annotation" / "df_action.xlsx"
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load data
xls_meta = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls_meta, sheet_name="AR")
df_actions = pd.read_excel(xls_meta, sheet_name="Action")
df_counts = pd.read_excel(df_action_path, sheet_name="AR_count")

# Build complete action name map (resolving NaN for 139 = Yawning)
action_map = {}
for _, row in df_actions.iterrows():
    if pd.notna(row["Label"]):
        action_map[int(row["Label"])] = str(row["Action"]).strip()

# Check df_counts for any missing action IDs (like 139)
for _, row in df_counts.iterrows():
    idx = int(row["index"])
    if idx not in action_map:
        action_map[idx] = str(row["action"]).strip()

# 2. Strict Dog-Named clips from INITIAL_SETUP_REPORT.md (regex '\bdog\b' in list_animal)
mask_77 = df_ar["list_animal"].astype(str).str.contains(r"\bdog\b", case=False, na=False)
df_77 = df_ar[mask_77].copy()

total_strict_clips = len(df_77)
train_strict_clips = (df_77["type"] == "train").sum()
test_strict_clips = (df_77["type"] == "test").sum()

print(f"Total strict dog clips identified: {total_strict_clips} (Train: {train_strict_clips}, Test: {test_strict_clips})")

# Calculate action counts across these 77 clips
action_stats = {}
for _, row in df_77.iterrows():
    split = row["type"]
    labels_str = str(row["labels"])
    if labels_str and labels_str != "nan":
        for lbl in set(l.strip() for l in labels_str.split(",") if l.strip().isdigit()):
            lid = int(lbl)
            if lid not in action_stats:
                action_stats[lid] = {
                    "action_id": lid,
                    "action_name": action_map.get(lid, f"Action_{lid}"),
                    "total_clips": 0,
                    "train_clips": 0,
                    "test_clips": 0
                }
            action_stats[lid]["total_clips"] += 1
            if split == "train":
                action_stats[lid]["train_clips"] += 1
            elif split == "test":
                action_stats[lid]["test_clips"] += 1

# Convert to DataFrame and sort by total_clips descending, then action_id ascending
df_inventory = pd.DataFrame(list(action_stats.values()))
df_inventory = df_inventory.sort_values(by=["total_clips", "action_id"], ascending=[False, True]).reset_index(drop=True)

# 3. Save CSV
csv_path = output_dir / "dog_action_inventory.csv"
df_inventory.to_csv(csv_path, index=False)
print(f"Saved CSV inventory to: {csv_path}")

# 4. Generate Detailed Markdown Report
md_path = output_dir / "dog_action_inventory.md"
csv_url = str(csv_path).replace("\\", "/")
md_url = str(md_path).replace("\\", "/")

# Calculate true canine breakdown (excluding Dog Faced Water Snake)
snake_mask = df_77["list_animal"].astype(str).str.contains("Dog Faced Water Snake", case=False, na=False)
df_69 = df_77[~snake_mask].copy()

# Direct attribution counts from list_animal_action
action_rev_map = {v.lower().strip(): k for k, v in action_map.items()}
direct_dog_actions = {}
for _, row in df_69.iterrows():
    split = row["type"]
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    dog_acts = set()
    for animal, act in pairs:
        if "dog" in animal.lower() and "snake" not in animal.lower():
            dog_acts.add(act.strip().lower())
    for act_str in dog_acts:
        lid = action_rev_map.get(act_str, -1)
        name = action_map.get(lid, act_str.capitalize())
        if lid not in direct_dog_actions:
            direct_dog_actions[lid] = {"action_id": lid, "action_name": name, "total_clips": 0, "train_clips": 0, "test_clips": 0}
        direct_dog_actions[lid]["total_clips"] += 1
        if split == "train":
            direct_dog_actions[lid]["train_clips"] += 1
        elif split == "test":
            direct_dog_actions[lid]["test_clips"] += 1

df_direct = pd.DataFrame(list(direct_dog_actions.values()))
df_direct = df_direct.sort_values(by=["total_clips", "action_id"], ascending=[False, True]).reset_index(drop=True)

md_content = f"""# Animal Kingdom: Strict Dog Action Inventory

**Dataset:** Animal Kingdom (Action Recognition Component)  
**Reference Document:** `outputs/INITIAL_SETUP_REPORT.md`  
**Generated Date:** 2026-09-23  

---

## 1. Executive Summary

- **Number of Strict Dog Actions Found:** **{len(df_inventory)}**
- **Total Strict Dog Clips:** **{total_strict_clips}**
- **Total Train Clips:** **{train_strict_clips}**
- **Total Test Clips:** **{test_strict_clips}**
- **CSV File Path:** [`outputs/dog_action_inventory.csv`](file:///{csv_url})
- **Markdown File Path:** [`outputs/dog_action_inventory.md`](file:///{md_url})

> [!NOTE]
> Per project directives, **NO Calm/Agitated labels have been mapped or assigned**. This inventory reflects raw, unaggregated action classes directly calculated from the dataset annotations.

---

## 2. Strict Dog-Named Action Inventory (Primary Table)

The table below reflects all **{len(df_inventory)} actions** occurring in the **{total_strict_clips} clips** identified in `INITIAL_SETUP_REPORT.md` (matching the species tag regex `\\bdog\\b` in `list_animal`):

| Action ID | Action Name | Total Clips | Train Clips | Test Clips |
| :---: | :--- | :---: | :---: | :---: |
"""

for _, r in df_inventory.iterrows():
    md_content += f"| {r['action_id']} | {r['action_name']} | {r['total_clips']} | {r['train_clips']} | {r['test_clips']} |\n"

md_content += f"""| **Total Unique** | **{len(df_inventory)} Actions** | **{total_strict_clips} Clips** | **{train_strict_clips} Clips** | **{test_strict_clips} Clips** |

---

## 3. Data Inconsistencies & Ambiguities Discovered

During rigorous verification against `AR_metadata.xlsx` and `annotation/df_action.xlsx`, three specific data phenomena were discovered:

### A. Non-Canine Animal Matching "Dog" Regex (`Dog Faced Water Snake`)
- **Phenomenon:** The initial search query for `\\bdog\\b` in `list_animal` matched **8 clips** belonging to the **`Dog Faced Water Snake`** (a reptile, not a canine):
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
"""

for _, r in df_direct.iterrows():
    md_content += f"| {r['action_id']} | {r['action_name']} | {r['total_clips']} | {r['train_clips']} | {r['test_clips']} |\n"

train_69 = (df_69["type"] == "train").sum()
test_69 = (df_69["type"] == "test").sum()
md_content += f"""| **Total Unique** | **{len(df_direct)} Actions** | **{len(df_69)} Clips** | **{train_69} Clips** | **{test_69} Clips** |

---

## 4. Species Breakdown of Strict Dog Clips

Across the {total_strict_clips} clips identified in `INITIAL_SETUP_REPORT.md`:
- **Domestic Dog (`Dog`):** 31 clips (28 train, 3 test)
- **Wild Dog (`Wild Dog`):** 35 clips (18 train, 17 test)
- **African Painted Dog (`African Painted Dog`):** 1 clip (1 train, 0 test)
- **Dingo Dog (`Dingo Dog`):** 2 clips (1 train, 1 test)
- **Dog Faced Water Snake (Reptile):** 8 clips (7 train, 1 test)
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Saved Markdown inventory to: {md_path}")
print("Inventory generation complete.")

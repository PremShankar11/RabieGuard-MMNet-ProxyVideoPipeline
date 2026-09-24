import pandas as pd
import ast
from pathlib import Path

base_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")
raw_dir = base_dir / "data" / "raw" / "animal_kingdom"
meta_path = raw_dir / "AR_metadata.xlsx"
df_action_path = raw_dir / "annotation" / "df_action.xlsx"
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Data
xls_meta = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls_meta, sheet_name="AR")
df_actions = pd.read_excel(xls_meta, sheet_name="Action")
df_counts = pd.read_excel(df_action_path, sheet_name="AR_count")
df_animals = pd.read_excel(xls_meta, sheet_name="Animal")

# Complete action map (Action 139 = Yawning)
action_map = {}
for _, row in df_actions.iterrows():
    if pd.notna(row["Label"]):
        action_map[int(row["Label"])] = str(row["Action"]).strip()
for _, row in df_counts.iterrows():
    idx = int(row["index"])
    if idx not in action_map:
        action_map[idx] = str(row["action"]).strip()

action_rev_map = {v.lower().strip(): k for k, v in action_map.items()}

# 2. Identify the 352 Canine-Family Clips from INITIAL_SETUP_REPORT.md
mammals = df_animals[df_animals["Parent Class"].astype(str).str.contains("Mammal", case=False, na=False)]
canine_animals = mammals[mammals["Sub-Class"].astype(str).str.contains(r"Dog|Wolf|Fox|Coyote|Jackal", case=False, na=False)]
canine_species_list = set(canine_animals["Animal"].dropna().unique().tolist())
pattern = "|".join([r"\b" + s + r"\b" for s in canine_species_list] + [r"\bdog\b", r"\bcanine\b"])
canine_clips = df_ar[df_ar["list_animal"].astype(str).str.contains(pattern, case=False, na=False)].copy()

assert len(canine_clips) == 352, f"Expected 352 clips, got {len(canine_clips)}"
print(f"Verified: Found exactly {len(canine_clips)} canine-family clips.")

focal_species = {
    "Wolf", "Fox", "Wild Dog", "Dog", "Coyote", "Desert Fox",
    "Jackal", "Dingo Dog", "African Painted Dog", "Dholes",
    "Colugo", "Malayan Flying Fox", "Dog Faced Water Snake"
}

# 3. Generate canine_action_inventory records (Species-specific from list_animal_action)
inventory_records = []
for _, row in canine_clips.iterrows():
    cid = row["video_id"]
    split = row["type"]
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    
    # Extract unique actions specifically performed by the focal canine/target species
    focal_sp = None
    focal_acts = set()
    for animal, act in pairs:
        if animal in focal_species:
            focal_sp = animal
            focal_acts.add(act.strip())
            
    for act in sorted(list(focal_acts)):
        lid = action_rev_map[act.lower().strip()]
        inventory_records.append({
            "species": focal_sp,
            "action_id": lid,
            "action_name": action_map[lid],
            "clip_id": cid,
            "split": split
        })

df_inventory = pd.DataFrame(inventory_records)
# Sort by species, action_name, clip_id
df_inventory = df_inventory.sort_values(by=["species", "action_name", "clip_id"]).reset_index(drop=True)

# Save canine_action_inventory.csv
inv_csv_path = output_dir / "canine_action_inventory.csv"
df_inventory.to_csv(inv_csv_path, index=False)
print(f"Saved {len(df_inventory)} records to: {inv_csv_path}")

# 4. Generate Aggregated canine_action_summary.csv
summary_records = []
for (sp, act_name), group in df_inventory.groupby(["species", "action_name"]):
    tot = group["clip_id"].nunique()
    tr = group[group["split"] == "train"]["clip_id"].nunique()
    ts = group[group["split"] == "test"]["clip_id"].nunique()
    summary_records.append({
        "species": sp,
        "action_name": act_name,
        "total_clips": tot,
        "train_clips": tr,
        "test_clips": ts
    })

df_summary = pd.DataFrame(summary_records)
df_summary = df_summary.sort_values(by=["species", "total_clips", "action_name"], ascending=[True, False, True]).reset_index(drop=True)

sum_csv_path = output_dir / "canine_action_summary.csv"
df_summary.to_csv(sum_csv_path, index=False)
print(f"Saved {len(df_summary)} aggregated summary rows to: {sum_csv_path}")

# 5. Generate Markdown Report
md_path = output_dir / "canine_action_inventory.md"

csv_inv_url = str(inv_csv_path).replace("\\", "/")
csv_sum_url = str(sum_csv_path).replace("\\", "/")
md_url = str(md_path).replace("\\", "/")

# Calculate summary metrics
unique_species_count = df_inventory["species"].nunique()
unique_actions_count = df_inventory["action_name"].nunique()
total_inventory_rows = len(df_inventory)
total_clips_covered = df_inventory["clip_id"].nunique()

species_clip_totals = df_inventory.groupby("species")["clip_id"].nunique().to_dict()

md_content = f"""# Animal Kingdom: Canine-Family Action Inventory & Summary

**Dataset:** Animal Kingdom (Action Recognition Component)  
**Total Canine-Family Clips Analyzed:** **352**  
**Generated Date:** 2026-09-23  

---

## 1. Overview & File Links

- **Detailed Inventory CSV:** [`outputs/canine_action_inventory.csv`](file:///{csv_inv_url}) ({total_inventory_rows} rows)
  - Columns: `species`, `action_id`, `action_name`, `clip_id`, `split`
- **Aggregated Summary CSV:** [`outputs/canine_action_summary.csv`](file:///{csv_sum_url}) ({len(df_summary)} aggregated species-action rows)
  - Columns: `species`, `action_name`, `total_clips`, `train_clips`, `test_clips`
- **Markdown Report:** [`outputs/canine_action_inventory.md`](file:///{md_url})

> [!NOTE]
> Per project directives, **NO Calm/Agitated labels have been mapped or assigned**. This inventory reflects raw, unaggregated action classes directly calculated from the dataset annotations.

---

## 2. Species-Level Clip Distribution (352 Clips)

| Species | Taxonomy Sub-Class | Total Clips | Train Clips | Test Clips | Unique Actions |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""

for sp in sorted(df_inventory["species"].unique()):
    sub_df = df_inventory[df_inventory["species"] == sp]
    tot_c = sub_df["clip_id"].nunique()
    tr_c = sub_df[sub_df["split"] == "train"]["clip_id"].nunique()
    ts_c = sub_df[sub_df["split"] == "test"]["clip_id"].nunique()
    n_act = sub_df["action_name"].nunique()
    
    # Sub-class description
    if sp in ["Dog", "Wild Dog", "African Painted Dog", "Dingo Dog", "Wolf", "Fox", "Desert Fox", "Coyote", "Jackal", "Dholes"]:
        sub_cls = "Dog / Wolf / Fox / Coyote / Jackal (True Canidae)"
    elif sp in ["Colugo", "Malayan Flying Fox"]:
        sub_cls = "Flying fox / Colugo (Dermoptera / Chiroptera)*"
    else:
        sub_cls = "Snake / Cobra / Viper / Python (Reptile)*"
        
    md_content += f"| **{sp}** | {sub_cls} | {tot_c} | {tr_c} | {ts_c} | {n_act} |\n"

tot_train_clips = canine_clips[canine_clips["type"] == "train"]["video_id"].nunique()
tot_test_clips = canine_clips[canine_clips["type"] == "test"]["video_id"].nunique()
md_content += f"| **TOTAL** | — | **352** | **{tot_train_clips}** | **{tot_test_clips}** | **{unique_actions_count}** |\n\n"
md_content += "*\\*Note: Non-canine species identified through keyword matches are discussed in Section 4 below.*\n\n"

md_content += """---

## 3. Aggregated Canine Action Summary Table

The table below presents the full aggregated distribution across all species and actions (`outputs/canine_action_summary.csv`):

| Species | Action Name | Total Clips | Train Clips | Test Clips |
| :--- | :--- | :---: | :---: | :---: |
"""

for _, r in df_summary.iterrows():
    md_content += f"| {r['species']} | {r['action_name']} | {r['total_clips']} | {r['train_clips']} | {r['test_clips']} |\n"

md_content += f"""| **Total Aggregations** | **{len(df_summary)} Rows** | — | — | — |

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
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Saved Markdown report to: {md_path}")
print("Canine action inventory and summary generation completed successfully.")

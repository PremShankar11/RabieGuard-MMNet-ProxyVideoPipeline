import pandas as pd
import ast
from pathlib import Path

meta_path = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\AR_metadata.xlsx")
xls = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls, sheet_name="AR")
df_actions = pd.read_excel(xls, sheet_name="Action")
df_counts = pd.read_excel(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\annotation\df_action.xlsx", sheet_name="AR_count")
df_animals = pd.read_excel(xls, sheet_name="Animal")

action_map = {}
for _, row in df_actions.iterrows():
    if pd.notna(row["Label"]):
        action_map[int(row["Label"])] = str(row["Action"]).strip()
for _, row in df_counts.iterrows():
    idx = int(row["index"])
    if idx not in action_map:
        action_map[idx] = str(row["action"]).strip()
action_rev_map = {v.lower().strip(): k for k, v in action_map.items()}

mammals = df_animals[df_animals["Parent Class"].astype(str).str.contains("Mammal", case=False, na=False)]
canine_animals = mammals[mammals["Sub-Class"].astype(str).str.contains(r"Dog|Wolf|Fox|Coyote|Jackal", case=False, na=False)]
canine_species_list = set(canine_animals["Animal"].dropna().unique().tolist())
pattern = "|".join([r"\b" + s + r"\b" for s in canine_species_list] + [r"\bdog\b", r"\bcanine\b"])
canine_clips = df_ar[df_ar["list_animal"].astype(str).str.contains(pattern, case=False, na=False)].copy()

focal_species = {
    "Wolf", "Fox", "Wild Dog", "Dog", "Coyote", "Desert Fox",
    "Jackal", "Dingo Dog", "African Painted Dog", "Dholes",
    "Colugo", "Malayan Flying Fox", "Dog Faced Water Snake"
}

# Method 1: Target animal only from list_animal_action
records_m1 = []
for _, row in canine_clips.iterrows():
    cid = row["video_id"]
    split = row["type"]
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    focal_acts = set()
    focal_sp = None
    for a, act in pairs:
        if a in focal_species:
            focal_sp = a
            focal_acts.add(act.strip())
    for act in sorted(list(focal_acts)):
        lid = action_rev_map[act.lower().strip()]
        records_m1.append({
            "species": focal_sp,
            "action_id": lid,
            "action_name": action_map[lid],
            "clip_id": cid,
            "split": split
        })

# Method 2: Clip-level labels for the focal species
records_m2 = []
for _, row in canine_clips.iterrows():
    cid = row["video_id"]
    split = row["type"]
    animals = ast.literal_eval(str(row["list_animal"]))
    focal_sp = [a for a in animals if a in focal_species][0]
    labels_str = str(row["labels"])
    lbls = set(int(l.strip()) for l in labels_str.split(",") if l.strip().isdigit())
    for lid in sorted(list(lbls)):
        records_m2.append({
            "species": focal_sp,
            "action_id": lid,
            "action_name": action_map[lid],
            "clip_id": cid,
            "split": split
        })

print(f"Method 1 (species-specific actions from list_animal_action): {len(records_m1)} rows across {len(set(r['clip_id'] for r in records_m1))} clips")
print(f"Method 2 (clip-level labels): {len(records_m2)} rows across {len(set(r['clip_id'] for r in records_m2))} clips")

# Difference check
m1_set = set((r["species"], r["action_id"], r["clip_id"]) for r in records_m1)
m2_set = set((r["species"], r["action_id"], r["clip_id"]) for r in records_m2)
diff = m2_set - m1_set
print(f"\nNumber of entries in M2 that belong to other co-occurring animals: {len(diff)}")
sample_diff = list(diff)[:5]
for sp, lid, cid in sample_diff:
    print(f"  Clip {cid}: Species {sp} attributed with Action [{lid}] '{action_map[lid]}' in M2, but in M1 that action belongs to another animal in the scene!")

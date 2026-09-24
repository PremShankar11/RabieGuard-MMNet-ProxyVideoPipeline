import pandas as pd
import ast
from pathlib import Path

meta_path = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\AR_metadata.xlsx")
xls = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls, sheet_name="AR")
df_actions = pd.read_excel(xls, sheet_name="Action").dropna(subset=["Label"])
action_map = dict(zip(df_actions["Label"].astype(int), df_actions["Action"].str.strip()))
action_rev_map = {v.lower().strip(): k for k, v in action_map.items()}

print("=" * 70)
print("COMPARING PERSPECTIVES OF DOG ACTION INVENTORIES")
print("=" * 70)

# Perspective 1: The 77 clips identified in INITIAL_SETUP_REPORT.md (\bdog\b in list_animal)
mask_77 = df_ar["list_animal"].astype(str).str.contains(r"\bdog\b", case=False, na=False)
df_77 = df_ar[mask_77].copy()

action_counts_77 = {}
for _, row in df_77.iterrows():
    split = row["type"]
    labels = str(row["labels"]).split(",")
    for lbl in set(l.strip() for l in labels if l.strip().isdigit()):
        lid = int(lbl)
        if lid not in action_counts_77:
            action_counts_77[lid] = {"total": 0, "train": 0, "test": 0}
        action_counts_77[lid]["total"] += 1
        if split == "train":
            action_counts_77[lid]["train"] += 1
        elif split == "test":
            action_counts_77[lid]["test"] += 1

print("\n--- Perspective 1: 77 CLIPS (from INITIAL_SETUP_REPORT.md matching '\\bdog\\b') ---")
print(f"Total clips: {len(df_77)} (Train: {(df_77['type']=='train').sum()}, Test: {(df_77['type']=='test').sum()})")
print(f"Total unique actions: {len(action_counts_77)}")
for lid, counts in sorted(action_counts_77.items(), key=lambda x: x[1]["total"], reverse=True):
    name = action_map.get(lid, f"Action_{lid}")
    print(f"  [{lid:3d}] {name:26s} | Total: {counts['total']:2d} | Train: {counts['train']:2d} | Test: {counts['test']:2d}")

# Perspective 2: 69 TRUE CANINE CLIPS (Excluding Dog Faced Water Snake)
snake_mask = df_77["list_animal"].astype(str).str.contains("Dog Faced Water Snake", case=False, na=False)
df_69 = df_77[~snake_mask].copy()

action_counts_69 = {}
for _, row in df_69.iterrows():
    split = row["type"]
    labels = str(row["labels"]).split(",")
    for lbl in set(l.strip() for l in labels if l.strip().isdigit()):
        lid = int(lbl)
        if lid not in action_counts_69:
            action_counts_69[lid] = {"total": 0, "train": 0, "test": 0}
        action_counts_69[lid]["total"] += 1
        if split == "train":
            action_counts_69[lid]["train"] += 1
        elif split == "test":
            action_counts_69[lid]["test"] += 1

print("\n--- Perspective 2: 69 TRUE CANINE CLIPS (Excluding 'Dog Faced Water Snake') ---")
print(f"Total clips: {len(df_69)} (Train: {(df_69['type']=='train').sum()}, Test: {(df_69['type']=='test').sum()})")
print(f"Total unique actions: {len(action_counts_69)}")
for lid, counts in sorted(action_counts_69.items(), key=lambda x: x[1]["total"], reverse=True):
    name = action_map.get(lid, f"Action_{lid}")
    print(f"  [{lid:3d}] {name:26s} | Total: {counts['total']:2d} | Train: {counts['train']:2d} | Test: {counts['test']:2d}")

# Perspective 3: ONLY ACTIONS DIRECTLY ATTRIBUTED TO DOG in list_animal_action
action_counts_direct = {}
for _, row in df_69.iterrows():
    split = row["type"]
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    dog_actions_in_clip = set()
    for animal, act in pairs:
        if "dog" in animal.lower() and "snake" not in animal.lower():
            dog_actions_in_clip.add(act.strip().lower())
    for act_lower in dog_actions_in_clip:
        lid = action_rev_map.get(act_lower, -1)
        name = action_map.get(lid, act_lower.capitalize())
        if lid not in action_counts_direct:
            action_counts_direct[lid] = {"name": name, "total": 0, "train": 0, "test": 0}
        action_counts_direct[lid]["total"] += 1
        if split == "train":
            action_counts_direct[lid]["train"] += 1
        elif split == "test":
            action_counts_direct[lid]["test"] += 1

print("\n--- Perspective 3: STRICT DOG ATTRIBUTIONS (From list_animal_action pairs) ---")
print(f"Total unique actions: {len(action_counts_direct)}")
for lid, counts in sorted(action_counts_direct.items(), key=lambda x: x[1]["total"], reverse=True):
    print(f"  [{lid:3d}] {counts['name']:26s} | Total: {counts['total']:2d} | Train: {counts['train']:2d} | Test: {counts['test']:2d}")

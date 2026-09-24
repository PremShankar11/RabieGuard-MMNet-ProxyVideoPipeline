import pandas as pd
import ast
from pathlib import Path

meta_path = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\AR_metadata.xlsx")
xls = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls, sheet_name="AR")
df_actions = pd.read_excel(xls, sheet_name="Action")
df_counts = pd.read_excel(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\annotation\df_action.xlsx", sheet_name="AR_count")
df_animals = pd.read_excel(xls, sheet_name="Animal")

# Complete action map
action_map = {}
for _, row in df_actions.iterrows():
    if pd.notna(row["Label"]):
        action_map[int(row["Label"])] = str(row["Action"]).strip()
for _, row in df_counts.iterrows():
    idx = int(row["index"])
    if idx not in action_map:
        action_map[idx] = str(row["action"]).strip()

action_rev_map = {v.lower().strip(): k for k, v in action_map.items()}

# 352 clips
mammals = df_animals[df_animals["Parent Class"].astype(str).str.contains("Mammal", case=False, na=False)]
canine_animals = mammals[mammals["Sub-Class"].astype(str).str.contains(r"Dog|Wolf|Fox|Coyote|Jackal", case=False, na=False)]
canine_species_list = set(canine_animals["Animal"].dropna().unique().tolist())
pattern = "|".join([r"\b" + s + r"\b" for s in canine_species_list] + [r"\bdog\b", r"\bcanine\b"])
canine_clips = df_ar[df_ar["list_animal"].astype(str).str.contains(pattern, case=False, na=False)].copy()

print(f"Total clips matching pattern: {len(canine_clips)}")

# Count clips per animal in list_animal
species_clip_count = {}
for _, row in canine_clips.iterrows():
    try:
        animals = ast.literal_eval(str(row["list_animal"]))
        for a in set(animals):
            species_clip_count[a] = species_clip_count.get(a, 0) + 1
    except:
        pass

print("\nClips per species in the 352 clips:")
for sp, cnt in sorted(species_clip_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {sp:25s}: {cnt:3d} clips")

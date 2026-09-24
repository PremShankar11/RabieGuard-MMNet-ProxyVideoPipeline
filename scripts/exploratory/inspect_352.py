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

# 352 clips query
mammals = df_animals[df_animals["Parent Class"].astype(str).str.contains("Mammal", case=False, na=False)]
canine_animals = mammals[mammals["Sub-Class"].astype(str).str.contains(r"Dog|Wolf|Fox|Coyote|Jackal", case=False, na=False)]
canine_species_list = set(canine_animals["Animal"].dropna().unique().tolist())

pattern = "|".join([r"\b" + s + r"\b" for s in canine_species_list] + [r"\bdog\b", r"\bcanine\b"])
canine_clips = df_ar[df_ar["list_animal"].astype(str).str.contains(pattern, case=False, na=False)].copy()
print(f"Total canine-family clips: {len(canine_clips)}")

# Check canine species that appear in the 352 clips
canine_species_in_data = set()
for a in canine_clips["list_animal"]:
    try:
        lst = ast.literal_eval(str(a))
        for item in lst:
            # Check if this item is one of our canine species or contains dog/wolf/fox/coyote/jackal
            canine_species_in_data.add(item)
    except:
        pass

print("\nAll species in list_animal of the 352 clips:")
for sp in sorted(list(canine_species_in_data)):
    print(" ", sp)

# Check list_animal_action structure for each clip
sample_count = 0
multi_animal_clips = 0
for idx, row in canine_clips.iterrows():
    try:
        pairs = ast.literal_eval(str(row["list_animal_action"]))
        species_in_clip = set(p[0] for p in pairs)
        if len(species_in_clip) > 1:
            multi_animal_clips += 1
            if sample_count < 3:
                print(f"\nMulti-animal clip {row['video_id']}:")
                print("  Animals:", row["list_animal"])
                print("  Pairs:", pairs)
                print("  Labels:", row["labels"])
                sample_count += 1
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")

print(f"\nTotal multi-animal clips among the 352: {multi_animal_clips}")

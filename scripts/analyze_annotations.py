import pandas as pd
from pathlib import Path

annot_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom\annotation")
raw_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet\data\raw\animal_kingdom")

train_path = annot_dir / "train.csv"
val_path = annot_dir / "val.csv"
meta_path = raw_dir / "AR_metadata.xlsx"
df_action_path = annot_dir / "df_action.xlsx"

print("=" * 60)
print("DEEP ANALYSIS OF ANIMAL KINGDOM ANNOTATIONS")
print("=" * 60)

xls_meta = pd.ExcelFile(meta_path)
df_ar = pd.read_excel(xls_meta, sheet_name='AR')
df_actions = pd.read_excel(xls_meta, sheet_name='Action').dropna(subset=['Label'])
df_animals = pd.read_excel(xls_meta, sheet_name='Animal')
df_counts = pd.read_excel(df_action_path, sheet_name='AR_count')

print(f"Total video clips in AR metadata: {len(df_ar)}")
print(f"Total action definitions: {len(df_actions)}")
print(f"Total animal definitions: {len(df_animals)}")

# Action map
action_map = dict(zip(df_actions['Label'].astype(int), df_actions['Action']))
action_cat_map = dict(zip(df_actions['Label'].astype(int), df_actions['Category']))

# Find canine-related clips in AR metadata
# Let's inspect all unique animal names in df_animals under Mammal
mammals = df_animals[df_animals['Parent Class'].astype(str).str.contains('Mammal', case=False, na=False)]
canine_animals = mammals[mammals['Sub-Class'].astype(str).str.contains(r'Dog|Wolf|Fox|Coyote|Jackal', case=False, na=False)]
print(f"\nCanine species defined in taxonomy: {len(canine_animals)}")
print(canine_animals[['Animal', 'Sub-Class']].to_string())

# Filter clips in df_ar with canine species
canine_species_list = canine_animals['Animal'].dropna().unique().tolist()
# Also add general 'Dog'
pattern = '|'.join([r'\b' + s + r'\b' for s in canine_species_list] + [r'\bdog\b', r'\bcanine\b'])
canine_clips = df_ar[df_ar['list_animal'].astype(str).str.contains(pattern, case=False, na=False)]
strict_dog_clips = df_ar[df_ar['list_animal'].astype(str).str.contains(r'\bdog\b', case=False, na=False)]

print(f"\nTotal Canine-family clips (Dogs, Wolves, Foxes, Jackals, Dholes): {len(canine_clips)}")
print(f"Specifically 'Dog' named clips (Domestic Dog, Wild Dog, African Painted Dog, Dingo Dog): {len(strict_dog_clips)}")
print(f"Strict Dog clips by split: {strict_dog_clips['type'].value_counts().to_dict()}")

# Summarize action labels observed across strict Dog clips
dog_action_counts = {}
for _, row in strict_dog_clips.iterrows():
    labels_str = str(row['labels'])
    if labels_str and labels_str != 'nan':
        for lbl in str(labels_str).split(','):
            lbl_clean = lbl.strip()
            if lbl_clean.isdigit():
                lid = int(lbl_clean)
                act_name = action_map.get(lid, f"Action_{lid}")
                cat_name = action_cat_map.get(lid, "Unknown")
                dog_action_counts[(lid, act_name, cat_name)] = dog_action_counts.get((lid, act_name, cat_name), 0) + 1

print("\nAction labels observed across strict Dog clips (Category | Action [ID] -> Count):")
for (lid, act, cat), count in sorted(dog_action_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - [{lid:3d}] {cat:16s} | {act:24s} -> {count} clips")

# 2. Check train.csv and val.csv frame-level counts (using sep=' ')
print("\n" + "=" * 60)
print("FRAME-LEVEL ANNOTATIONS (train.csv & val.csv)")
print("=" * 60)
df_train = pd.read_csv(train_path, sep=' ')
print(f"Train frames count: {len(df_train):,}")
print(f"Train unique video clips: {df_train['original_vido_id'].nunique():,}")

df_val = pd.read_csv(val_path, sep=' ')
print(f"Val frames count: {len(df_val):,}")
print(f"Val unique video clips: {df_val['original_vido_id'].nunique():,}")

total_frames = len(df_train) + len(df_val)
total_clips = df_train['original_vido_id'].nunique() + df_val['original_vido_id'].nunique()
print(f"\nTOTAL Frame annotations: {total_frames:,}")
print(f"TOTAL Unique video clips in CSVs: {total_clips:,}")

print("\nSample train frame:")
print(df_train.head(1).to_dict(orient='records')[0])
print("\nSample val frame:")
print(df_val.head(1).to_dict(orient='records')[0])

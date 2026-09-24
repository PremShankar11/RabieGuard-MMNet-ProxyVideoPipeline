import pandas as pd
import ast
from pathlib import Path

base_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")
meta_path = base_dir / "data" / "raw" / "animal_kingdom" / "AR_metadata.xlsx"
df_ar = pd.read_excel(meta_path, sheet_name="AR")
true_dog_species = {"Dog", "Wild Dog", "African Painted Dog", "Dingo Dog"}

dog_clips = df_ar[df_ar["list_animal"].apply(lambda x: any(d in str(x) for d in true_dog_species) and "Dog Faced Water Snake" not in str(x))].copy()

for idx, row in dog_clips.iterrows():
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    dog_acts = [act for a, act in pairs if a in true_dog_species]
    if len(set(dog_acts)) > 1:
        print(f"Clip: {row['video_id']} ({row['type']})")
        print(f"  Dog Actions: {dog_acts}")
        print(f"  All Pairs: {pairs}")
        print(f"  Labels: {row['labels']}")

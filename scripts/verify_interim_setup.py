import pandas as pd
from pathlib import Path

base_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")
mapping_path = base_dir / "configs" / "animal_kingdom_behavior_mapping.csv"
master_path = base_dir / "outputs" / "interim_video_inventory.csv"
model_path = base_dir / "outputs" / "interim_video_model_eligible.csv"
md_path = base_dir / "outputs" / "interim_video_inventory.md"

print("=" * 60)
print("RUNNING PROGRAMMATIC VERIFICATION")
print("=" * 60)

# 1. Verify mapping CSV
df_map = pd.read_csv(mapping_path)
assert len(df_map) == 15, f"Expected 15 actions, got {len(df_map)}"
assert set(df_map["interim_label"].unique()) == {"CALM", "AGITATED", "EXCLUDE"}
assert df_map[df_map["action_name"] == "Preying"]["interim_label"].values[0] == "EXCLUDE"
assert df_map[df_map["action_name"] == "Yawning"]["interim_label"].values[0] == "CALM"
assert df_map[df_map["action_name"] == "Barking"]["interim_label"].values[0] == "AGITATED"
print("[PASS] Behavior mapping verified (15 actions, correct label assignments).")

# 2. Verify Master CSV
df_master = pd.read_csv(master_path)
assert len(df_master) == 69, f"Expected 69 master clips, got {len(df_master)}"
assert (df_master["split"] == "train").sum() == 48, f"Expected 48 train clips, got {(df_master['split'] == 'train').sum()}"
assert (df_master["split"] == "test").sum() == 21, f"Expected 21 test clips, got {(df_master['split'] == 'test').sum()}"

allowed_species = {"Dog", "Wild Dog", "African Painted Dog", "Dingo Dog"}
assert set(df_master["species"].unique()).issubset(allowed_species), f"Invalid species: {set(df_master['species'].unique())}"
assert "Dog Faced Water Snake" not in df_master["species"].values
assert not df_master["species"].str.contains("Snake").any()
print("[PASS] Master inventory verified (69 clips, 48 train / 21 test, exact allowed species).")

# Verify video metadata validity
assert (df_master["duration_seconds"] > 0).all(), "Invalid duration found"
assert (df_master["fps"] > 0).all(), "Invalid fps found"
assert (df_master["frame_count"] > 0).all(), "Invalid frame_count found"
assert (df_master["width"] > 0).all(), "Invalid width found"
assert (df_master["height"] > 0).all(), "Invalid height found"
print("[PASS] Video file metadata verified (all valid dimensions, fps, frames, durations).")

# 3. Verify Model-Eligible CSV
df_model = pd.read_csv(model_path)
assert len(df_model) == 68, f"Expected 68 model-eligible clips, got {len(df_model)}"
assert (df_model["split"] == "train").sum() == 47, f"Expected 47 train clips, got {(df_model['split'] == 'train').sum()}"
assert (df_model["split"] == "test").sum() == 21, f"Expected 21 test clips, got {(df_model['split'] == 'test').sum()}"
assert "EXCLUDE" not in df_model["interim_label"].values
assert set(df_model["interim_label"].unique()) == {"CALM", "AGITATED"}
print("[PASS] Model-eligible inventory verified (68 clips, 47 train / 21 test, CALM/AGITATED only).")

# 4. Verify Markdown file existence
assert md_path.exists(), "Markdown report missing"
assert md_path.stat().st_size > 1000, "Markdown report too small"
print("[PASS] Human-readable Markdown report verified.")

print("\nALL PROGRAMMATIC CHECKS PASSED SUCCESSFULLY.")
print("=" * 60)

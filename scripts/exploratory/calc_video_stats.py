import pandas as pd
import numpy as np
import ast
import cv2
from pathlib import Path

base_dir = Path(r"c:\Important_prem\FYP\Zero-Rabies-MMNet")
video_dir = base_dir / "data" / "raw" / "animal_kingdom" / "video"
meta_path = base_dir / "data" / "raw" / "animal_kingdom" / "AR_metadata.xlsx"

df_ar = pd.read_excel(meta_path, sheet_name="AR")
true_dog_species = {"Dog", "Wild Dog", "African Painted Dog", "Dingo Dog"}

calm_actions = {"keeping still", "walking", "attending", "sensing", "eating", "yawning"}
agitated_actions = {"running", "jumping", "attacking", "chasing", "barking", "startled", "biting", "fleeing"}
exclude_actions = {"preying"}

dog_rows = []
for idx, row in df_ar.iterrows():
    if any(d in str(row["list_animal"]) for d in true_dog_species):
        if "Dog Faced Water Snake" not in str(row["list_animal"]):
            dog_rows.append(row)

video_data = []
for row in dog_rows:
    vid = row["video_id"]
    split = row["type"]
    vpath = video_dir / f"{vid}.mp4"
    
    # Species
    animals = ast.literal_eval(str(row["list_animal"]))
    sp = [a for a in animals if a in true_dog_species][0]
    
    # Dog actions
    pairs = ast.literal_eval(str(row["list_animal_action"]))
    dog_acts = [act.strip() for a, act in pairs if a in true_dog_species]
    # Deduplicate while preserving order
    seen = set()
    dog_acts_unique = [x for x in dog_acts if not (x.lower() in seen or seen.add(x.lower()))]
    
    # Label
    acts_lower = set(x.lower() for x in dog_acts_unique)
    if any(a in exclude_actions for a in acts_lower):
        lbl = "EXCLUDE"
    elif any(a in agitated_actions for a in acts_lower):
        lbl = "AGITATED"
    elif all(a in calm_actions for a in acts_lower):
        lbl = "CALM"
    else:
        lbl = "UNKNOWN"
        
    cap = cv2.VideoCapture(str(vpath))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = fc / fps if fps > 0 else 0.0
    cap.release()
    
    video_data.append({
        "clip_id": vid,
        "species": sp,
        "actions": dog_acts_unique,
        "interim_label": lbl,
        "split": split,
        "video_path": str(vpath),
        "duration_seconds": dur,
        "fps": fps,
        "frame_count": fc,
        "width": w,
        "height": h
    })

df_all = pd.DataFrame(video_data)
df_model = df_all[df_all["interim_label"] != "EXCLUDE"].copy()

print("=" * 60)
print("STATISTICS SUMMARY")
print("=" * 60)
print(f"Master clips: {len(df_all)}")
print(f"Model-eligible clips: {len(df_model)}")

# Duration stats for Model-eligible
durations = df_model["duration_seconds"].values
print("\nDuration Stats (Model-Eligible):")
print(f"  Min:    {np.min(durations):.2f} s")
print(f"  Max:    {np.max(durations):.2f} s")
print(f"  Mean:   {np.mean(durations):.2f} s")
print(f"  Median: {np.median(durations):.2f} s")
print(f"  Std:    {np.std(durations):.2f} s")

# Percentiles
p10, p25, p50, p75, p90, p95 = np.percentile(durations, [10, 25, 50, 75, 90, 95])
print("\nDuration Percentiles:")
print(f"  P10: {p10:.2f} s")
print(f"  P25: {p25:.2f} s")
print(f"  P50: {p50:.2f} s")
print(f"  P75: {p75:.2f} s")
print(f"  P90: {p90:.2f} s")
print(f"  P95: {p95:.2f} s")

# Buckets
buckets = {
    "< 2s": np.sum(durations < 2.0),
    "2–5s": np.sum((durations >= 2.0) & (durations < 5.0)),
    "5–10s": np.sum((durations >= 5.0) & (durations < 10.0)),
    "10–20s": np.sum((durations >= 10.0) & (durations < 20.0)),
    "20–30s": np.sum((durations >= 20.0) & (durations < 30.0)),
    "30–60s": np.sum((durations >= 30.0) & (durations <= 60.0)),
    "> 60s": np.sum(durations > 60.0)
}
print("\nDuration Buckets (Model-Eligible):")
for b, count in buckets.items():
    print(f"  {b:10s}: {count:2d} clips ({count/len(df_model)*100:5.1f}%)")

# FPS and Resolution
print("\nFPS Distribution (Model-Eligible):")
print(df_model["fps"].value_counts().to_dict())

print("\nResolution Distribution (Model-Eligible):")
df_model["resolution"] = df_model["width"].astype(str) + "x" + df_model["height"].astype(str)
print(df_model["resolution"].value_counts().to_dict())

# Frame count stats
fc = df_model["frame_count"].values
print("\nFrame Count Stats (Model-Eligible):")
print(f"  Min:    {np.min(fc)}")
print(f"  Max:    {np.max(fc)}")
print(f"  Mean:   {np.mean(fc):.1f}")
print(f"  Median: {np.median(fc):.1f}")

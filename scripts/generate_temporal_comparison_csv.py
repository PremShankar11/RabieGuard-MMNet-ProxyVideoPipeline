import pandas as pd
import numpy as np
from pathlib import Path

# Load model-eligible Animal Kingdom clips
df = pd.read_csv("outputs/interim_video_model_eligible.csv")

# 8 FPS from 24 FPS video means stride of 3: available observations = frame_count // 3
df["avail_obs"] = df["frame_count"] // 3

configs = [
    {"cfg": "A", "fps": 8, "seq_len": 16, "nom_sec": 2.0},
    {"cfg": "B", "fps": 8, "seq_len": 24, "nom_sec": 3.0},
    {"cfg": "C", "fps": 8, "seq_len": 32, "nom_sec": 4.0},
]

rows = []
for c in configs:
    T = c["seq_len"]
    avail = df["avail_obs"]
    no_pad = (avail >= T)
    pad = (avail < T)
    padding_needed = np.maximum(0, T - avail)
    
    total_clips = len(df)
    n_no_pad = int(no_pad.sum())
    n_pad = int(pad.sum())
    pct_pad = round((n_pad / total_clips) * 100.0, 2)
    mean_pad = round(float(padding_needed[pad].mean()), 2) if n_pad > 0 else 0.0
    max_pad = int(padding_needed.max())
    
    # Coverage by split
    train_total = int((df["split"] == "train").sum())
    train_no_pad = int((no_pad & (df["split"] == "train")).sum())
    train_pad = int((pad & (df["split"] == "train")).sum())
    
    test_total = int((df["split"] == "test").sum())
    test_no_pad = int((no_pad & (df["split"] == "test")).sum())
    test_pad = int((pad & (df["split"] == "test")).sum())
    
    # Coverage by interim label
    calm_total = int((df["interim_label"] == "CALM").sum())
    calm_no_pad = int((no_pad & (df["interim_label"] == "CALM")).sum())
    calm_pad = int((pad & (df["interim_label"] == "CALM")).sum())
    
    agit_total = int((df["interim_label"] == "AGITATED").sum())
    agit_no_pad = int((no_pad & (df["interim_label"] == "AGITATED")).sum())
    agit_pad = int((pad & (df["interim_label"] == "AGITATED")).sum())
    
    rows.append({
        "configuration": c["cfg"],
        "sampling_fps": c["fps"],
        "sequence_length": c["seq_len"],
        "nominal_seconds": c["nom_sec"],
        "total_model_eligible_clips": total_clips,
        "clips_without_padding": n_no_pad,
        "clips_with_padding": n_pad,
        "padding_percentage": pct_pad,
        "mean_padding_frames": mean_pad,
        "max_padding_frames": max_pad,
        "train_clips": train_total,
        "train_without_padding": train_no_pad,
        "train_with_padding": train_pad,
        "test_clips": test_total,
        "test_without_padding": test_no_pad,
        "test_with_padding": test_pad,
        "calm_clips": calm_total,
        "calm_without_padding": calm_no_pad,
        "calm_with_padding": calm_pad,
        "agitated_clips": agit_total,
        "agitated_without_padding": agit_no_pad,
        "agitated_with_padding": agit_pad,
    })

res_df = pd.DataFrame(rows)
out_csv = Path("outputs/temporal_sampling_comparison.csv")
res_df.to_csv(out_csv, index=False)
print("Saved comparison CSV to:", out_csv)
print(res_df.to_string())

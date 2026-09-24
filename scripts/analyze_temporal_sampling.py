import pandas as pd
import numpy as np
from pathlib import Path

# Load model-eligible clips
csv_path = Path("outputs/interim_video_model_eligible.csv")
df = pd.read_csv(csv_path)

print(f"Total model-eligible clips: {len(df)}")
print(f"Split distribution:\n{df['split'].value_counts()}")
print(f"Interim label distribution:\n{df['interim_label'].value_counts()}")

# Configurations:
# A: 8 FPS, 16 obs, nominal 2.0 s
# B: 8 FPS, 24 obs, nominal 3.0 s
# C: 8 FPS, 32 obs, nominal 4.0 s

configs = [
    {"name": "A", "fps": 8, "T": 16, "nom_sec": 2.0},
    {"name": "B", "fps": 8, "T": 24, "nom_sec": 3.0},
    {"name": "C", "fps": 8, "T": 32, "nom_sec": 4.0},
]

# When sampling at 8 FPS from 24 FPS video:
# Video frame rate is 24.0 FPS.
# Downsampling factor = 24 / 8 = 3.
# Method 1: Discrete frame count. Available observations = frame_count // 3
# Method 2: Duration in seconds. Available observations = int(duration_seconds * 8)
# Let's inspect both.

df['obs_m1'] = df['frame_count'] // 3
df['obs_m2'] = (df['duration_seconds'] * 8).astype(int)

diff = (df['obs_m1'] != df['obs_m2']).sum()
print(f"Discrepancies between frame_count // 3 and int(duration * 8): {diff}")

for cfg in configs:
    T = cfg['T']
    name = cfg['name']
    
    # Using frame_count // 3 (exact discrete frames available at 8 FPS from 24 FPS)
    avail = df['frame_count'] // 3
    long_enough = avail >= T
    requires_padding = avail < T
    padding_needed = np.maximum(0, T - avail)
    
    num_total = len(df)
    num_no_pad = long_enough.sum()
    num_pad = requires_padding.sum()
    pct_pad = (num_pad / num_total) * 100.0
    mean_pad = padding_needed[requires_padding].mean() if num_pad > 0 else 0.0
    max_pad = padding_needed.max()
    
    # Train / Test coverage
    train_total = (df['split'] == 'train').sum()
    train_no_pad = (long_enough & (df['split'] == 'train')).sum()
    train_pad = (requires_padding & (df['split'] == 'train')).sum()
    
    test_total = (df['split'] == 'test').sum()
    test_no_pad = (long_enough & (df['split'] == 'test')).sum()
    test_pad = (requires_padding & (df['split'] == 'test')).sum()
    
    # Calm / Agitated coverage
    calm_total = (df['interim_label'] == 'CALM').sum()
    calm_no_pad = (long_enough & (df['interim_label'] == 'CALM')).sum()
    calm_pad = (requires_padding & (df['interim_label'] == 'CALM')).sum()
    
    agit_total = (df['interim_label'] == 'AGITATED').sum()
    agit_no_pad = (long_enough & (df['interim_label'] == 'AGITATED')).sum()
    agit_pad = (requires_padding & (df['interim_label'] == 'AGITATED')).sum()
    
    print(f"\n================ Configuration {name}: 8 FPS x {T} obs (~{cfg['nom_sec']}s) ================")
    print(f"Total clips: {num_total}")
    print(f"Without padding: {num_no_pad} ({num_no_pad/num_total*100:.1f}%)")
    print(f"With padding: {num_pad} ({pct_pad:.1f}%)")
    print(f"Mean padding (over padded clips): {mean_pad:.2f} frames")
    print(f"Max padding: {max_pad} frames")
    print(f"Train coverage: {train_no_pad}/{train_total} without padding ({train_no_pad/train_total*100:.1f}%), {train_pad} padded")
    print(f"Test coverage: {test_no_pad}/{test_total} without padding ({test_no_pad/test_total*100:.1f}%), {test_pad} padded")
    print(f"CALM coverage: {calm_no_pad}/{calm_total} without padding ({calm_no_pad/calm_total*100:.1f}%), {calm_pad} padded")
    print(f"AGITATED coverage: {agit_no_pad}/{agit_total} without padding ({agit_no_pad/agit_total*100:.1f}%), {agit_pad} padded")

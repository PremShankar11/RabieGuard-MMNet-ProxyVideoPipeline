"""
Dataset Construction and 5-Fold Stratified Split for Video V2 Development.
Zero Rabies-MMNet Project.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def construct_windows_for_clip(cache_data, window_length, stride):
    kpt_norm = cache_data["keypoints_normalized"]
    bbox_conf = cache_data["bbox_confidence"]
    valid_mask = cache_data["valid_mask"]
    ambiguous_mask = cache_data["ambiguous_mask"]
    timestamps = cache_data["timestamps"]
    N = int(cache_data["num_frames"])

    if N < window_length:
        starts = [0]
        is_padded = True
    else:
        starts = list(range(0, N - window_length + 1, stride))
        if starts[-1] != N - window_length:
            starts.append(N - window_length)
        is_padded = False

    windows = []
    for s in starts:
        if is_padded:
            w_kpt = np.zeros((window_length, 24, 3), dtype=np.float32)
            w_kpt[:N] = kpt_norm
            w_kpt[N:] = kpt_norm[-1:]

            w_bconf = np.zeros(window_length, dtype=np.float32)
            w_bconf[:N] = bbox_conf
            w_bconf[N:] = bbox_conf[-1:]

            w_valid = np.zeros(window_length, dtype=bool)
            w_valid[:N] = valid_mask

            w_amb = np.zeros(window_length, dtype=bool)
            w_amb[:N] = ambiguous_mask

            pad_mask = np.zeros(window_length, dtype=bool)
            pad_mask[N:] = True
        else:
            e = s + window_length
            w_kpt = kpt_norm[s:e].astype(np.float32)
            w_bconf = bbox_conf[s:e].astype(np.float32)
            w_valid = valid_mask[s:e].astype(bool)
            w_amb = ambiguous_mask[s:e].astype(bool)
            pad_mask = np.zeros(window_length, dtype=bool)

        frame_valid = w_valid & (~pad_mask)
        amb_penalty = np.where(w_amb, 0.8, 1.0)
        reliability = w_bconf * frame_valid.astype(np.float32) * amb_penalty

        feat_72 = w_kpt.reshape(window_length, 72)
        # 3 extra features: [bbox_confidence, frame_valid, ambiguous_mask]
        extra_3 = np.stack([
            w_bconf,
            frame_valid.astype(np.float32),
            w_amb.astype(np.float32)
        ], axis=-1) # (T, 3)
        feat_75 = np.concatenate([feat_72, extra_3], axis=-1) # (T, 75)

        windows.append({
            "feat_72": feat_72,
            "feat_75": feat_75,
            "reliability": reliability,
            "padding_mask": pad_mask,
            "frame_valid": frame_valid,
            "ambiguous_mask": w_amb,
            "bbox_conf": w_bconf,
            "start_idx": s,
            "is_padded": is_padded
        })
    return windows


def build_v2_cv_dataset():
    print("=" * 60)
    print("STAGE 4 (V2): 5-FOLD CV DATASET CONSTRUCTION")
    print("=" * 60)

    # 1. Load authoritative manifest
    manifest = pd.read_csv("outputs/animal_kingdom_pose_manifest.csv")
    dev_clips_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    test_clips_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    assert len(dev_clips_df) == 47, f"Expected 47 development clips, got {len(dev_clips_df)}"
    assert len(test_clips_df) == 21, f"Expected 21 test clips, got {len(test_clips_df)}"

    # 2. Stratified 5-Fold Split on 47 Development Clips
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dev_clips_df["cv_fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_clips_df, dev_clips_df["behavior_label"])):
        dev_clips_df.loc[val_idx, "cv_fold"] = fold_idx

    # Combine into outputs/behavior_v2/v2_cv_splits.csv
    test_clips_df["cv_fold"] = -1 # Locked test set
    combined_splits = pd.concat([dev_clips_df, test_clips_df], ignore_index=True)

    out_splits_dir = Path("outputs/behavior_v2")
    out_splits_dir.mkdir(parents=True, exist_ok=True)
    splits_csv = out_splits_dir / "v2_cv_splits.csv"
    combined_splits[["clip_id", "split", "cv_fold", "behavior_label", "behavior_name", "action"]].to_csv(splits_csv, index=False)
    print(f"Saved 5-Fold CV split metadata to: {splits_csv}")

    # Summary of folds
    for f in range(5):
        f_sub = dev_clips_df[dev_clips_df["cv_fold"] == f]
        c_cnt = (f_sub["behavior_name"] == "CALM").sum()
        a_cnt = (f_sub["behavior_name"] == "AGITATED").sum()
        print(f"  Fold {f}: Val={len(f_sub)} clips (CALM={c_cnt}, AGITATED={a_cnt}) | Train={len(dev_clips_df)-len(f_sub)} clips")

    # 3. Generate Window Datasets for T = 16, 24, 32
    v2_cache_root = Path("data/processed/animal_kingdom/v2_cache")
    v2_cache_root.mkdir(parents=True, exist_ok=True)
    v1_pose_root = Path("data/processed/animal_kingdom/pose_cache")

    configs_to_build = [
        (16, 8, "w16_s8"),
        (24, 12, "w24_s12"),
        (32, 16, "w32_s16")
    ]

    for T, stride, tag in configs_to_build:
        print(f"\nConstructing windows for T={T}, stride={stride} ({tag})...")
        records = []
        for _, row in combined_splits.iterrows():
            clip_id = row["clip_id"]
            split = row["split"]
            fold = int(row["cv_fold"])
            label = int(row["behavior_label"])
            b_name = row["behavior_name"]
            action = row["action"]

            cache_file = v1_pose_root / split / f"{clip_id}.npz"
            with np.load(cache_file, allow_pickle=True) as d:
                cache_data = {k: d[k] for k in d.keys()}

            wins = construct_windows_for_clip(cache_data, T, stride)
            for w_idx, w in enumerate(wins):
                records.append({
                    "window_id": f"{clip_id}_w{w_idx:02d}",
                    "clip_id": clip_id,
                    "split": split,
                    "cv_fold": fold,
                    "label": label,
                    "behavior_name": b_name,
                    "action": action,
                    **w
                })

        # Save compressed NPZ
        out_npz = v2_cache_root / f"dataset_{tag}.npz"
        np.savez_compressed(
            out_npz,
            window_ids=np.array([r["window_id"] for r in records]),
            clip_ids=np.array([r["clip_id"] for r in records]),
            splits=np.array([r["split"] for r in records]),
            cv_folds=np.array([r["cv_fold"] for r in records], dtype=np.int32),
            labels=np.array([r["label"] for r in records], dtype=np.int64),
            behavior_names=np.array([r["behavior_name"] for r in records]),
            action_names=np.array([r["action"] for r in records]),
            feat_72=np.stack([r["feat_72"] for r in records], axis=0),
            feat_75=np.stack([r["feat_75"] for r in records], axis=0),
            reliability=np.stack([r["reliability"] for r in records], axis=0),
            padding_mask=np.stack([r["padding_mask"] for r in records], axis=0),
            frame_valid=np.stack([r["frame_valid"] for r in records], axis=0),
            ambiguous_mask=np.stack([r["ambiguous_mask"] for r in records], axis=0),
            bbox_conf=np.stack([r["bbox_conf"] for r in records], axis=0),
            is_padded=np.array([r["is_padded"] for r in records], dtype=bool)
        )
        print(f"  Saved {len(records)} total windows to: {out_npz}")

    print("\nV2 CV Dataset Construction Complete.")
    return splits_csv

if __name__ == "__main__":
    build_v2_cv_dataset()

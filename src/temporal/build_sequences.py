"""
Sequence and Window Construction for Stage 4: Temporal Mamba Behavior Model.
Zero Rabies-MMNet Project.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def build_sequences():
    print("=" * 60)
    print("STAGE 4: SEQUENCE & WINDOW CONSTRUCTION")
    print("=" * 60)

    # 1. Read authoritative manifest
    manifest_path = Path("outputs/animal_kingdom_pose_manifest.csv")
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    df_manifest = pd.read_csv(manifest_path)
    print(f"Read manifest with {len(df_manifest)} clips.")

    # 2. Split 47 official train clips into 37 train / 10 val (stratified)
    train_clips_df = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)
    test_clips_df = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    assert len(train_clips_df) == 47, f"Expected 47 train clips, found {len(train_clips_df)}"
    assert len(test_clips_df) == 21, f"Expected 21 test clips, found {len(test_clips_df)}"

    sss = StratifiedShuffleSplit(n_splits=1, test_size=10, random_state=42)
    train_idx, val_idx = next(sss.split(train_clips_df, train_clips_df["behavior_label"]))

    sub_train_df = train_clips_df.iloc[train_idx].copy()
    val_df = train_clips_df.iloc[val_idx].copy()

    sub_train_df["mamba_split"] = "train"
    val_df["mamba_split"] = "val"
    test_clips_df["mamba_split"] = "test"

    combined_split_df = pd.concat([sub_train_df, val_df, test_clips_df], ignore_index=True)

    # Write outputs/mamba_train_val_split.csv
    split_out_csv = Path("outputs/mamba_train_val_split.csv")
    split_export = combined_split_df[["clip_id", "mamba_split", "behavior_label", "behavior_name", "action"]].copy()
    split_export.rename(columns={"mamba_split": "split"}, inplace=True)
    split_export.to_csv(split_out_csv, index=False)
    print(f"Saved split metadata to: {split_out_csv}")
    print(f"  - Train: {len(sub_train_df)} clips (CALM: {(sub_train_df['behavior_name']=='CALM').sum()}, AGITATED: {(sub_train_df['behavior_name']=='AGITATED').sum()})")
    print(f"  - Val:   {len(val_df)} clips (CALM: {(val_df['behavior_name']=='CALM').sum()}, AGITATED: {(val_df['behavior_name']=='AGITATED').sum()})")
    print(f"  - Test:  {len(test_clips_df)} clips (CALM: {(test_clips_df['behavior_name']=='CALM').sum()}, AGITATED: {(test_clips_df['behavior_name']=='AGITATED').sum()})")

    # Map clip to mamba_split
    clip_to_split = dict(zip(split_export["clip_id"], split_export["split"]))

    # 3. Construct 16-frame windows
    cache_root = Path("data/processed/animal_kingdom/pose_cache")
    window_length = 16
    stride = 8

    windows_data = []
    manifest_records = []

    for _, row in combined_split_df.iterrows():
        clip_id = row["clip_id"]
        original_split = row["split"] # 'train' or 'test' for directory lookup
        mamba_split = clip_to_split[clip_id]
        label = int(row["behavior_label"])
        behavior_name = row["behavior_name"]
        action_name = row["action"]

        cache_file = cache_root / original_split / f"{clip_id}.npz"
        assert cache_file.exists(), f"Missing cache file: {cache_file}"

        with np.load(cache_file, allow_pickle=True) as d:
            kpt_norm = d["keypoints_normalized"] # (N, 24, 3)
            kpt_raw = d["keypoints_raw"]         # (N, 24, 3)
            kpt_conf = d["keypoint_confidence"]  # (N, 24)
            bbox = d["bbox"]                     # (N, 4)
            bbox_conf = d["bbox_confidence"]     # (N,)
            frame_indices = d["frame_indices"]   # (N,)
            timestamps = d["timestamps"]         # (N,)
            valid_mask = d["valid_mask"]         # (N,)
            ambiguous_mask = d["ambiguous_mask"] # (N,)
            N = int(d["num_frames"])

        # Determine window slice indices
        if N < window_length:
            # Edge-replication padding
            starts = [0]
            is_padded_clip = True
        else:
            starts = list(range(0, N - window_length + 1, stride))
            if starts[-1] != N - window_length:
                starts.append(N - window_length)
            is_padded_clip = False

        for win_idx, s in enumerate(starts):
            window_id = f"{clip_id}_w{win_idx:02d}"
            
            if is_padded_clip:
                # Replicate the last observation
                win_kpt_norm = np.zeros((window_length, 24, 3), dtype=np.float32)
                win_kpt_norm[:N] = kpt_norm
                win_kpt_norm[N:] = kpt_norm[-1:] # edge replication

                win_bbox_conf = np.zeros(window_length, dtype=np.float32)
                win_bbox_conf[:N] = bbox_conf
                win_bbox_conf[N:] = bbox_conf[-1:]

                win_valid_mask = np.zeros(window_length, dtype=bool)
                win_valid_mask[:N] = valid_mask
                win_valid_mask[N:] = False # padded observations are not true detections

                win_ambiguous_mask = np.zeros(window_length, dtype=bool)
                win_ambiguous_mask[:N] = ambiguous_mask
                win_ambiguous_mask[N:] = False

                padding_mask = np.zeros(window_length, dtype=bool)
                padding_mask[N:] = True

                win_timestamps = np.zeros(window_length, dtype=np.float32)
                win_timestamps[:N] = timestamps
                # Extend timestamps by 0.125s (8 FPS)
                last_t = timestamps[-1] if len(timestamps) > 0 else 0.0
                for p_idx in range(N, window_length):
                    win_timestamps[p_idx] = last_t + (p_idx - N + 1) * 0.125
            else:
                e = s + window_length
                win_kpt_norm = kpt_norm[s:e].astype(np.float32)
                win_bbox_conf = bbox_conf[s:e].astype(np.float32)
                win_valid_mask = valid_mask[s:e].astype(bool)
                win_ambiguous_mask = ambiguous_mask[s:e].astype(bool)
                padding_mask = np.zeros(window_length, dtype=bool)
                win_timestamps = timestamps[s:e].astype(np.float32)

            # Frame reliability: frame_valid AND NOT padding_mask
            frame_valid = win_valid_mask & (~padding_mask)
            
            # Reliability weight per frame for masked temporal pooling
            # Modulated by detection confidence and ambiguity penalty
            amb_factor = np.where(win_ambiguous_mask, 0.8, 1.0)
            reliability = win_bbox_conf * frame_valid.astype(np.float32) * amb_factor

            # Flatten (16, 24, 3) -> (16, 72)
            features_72 = win_kpt_norm.reshape(window_length, 72)

            windows_data.append({
                "window_id": window_id,
                "clip_id": clip_id,
                "split": mamba_split,
                "behavior_label": label,
                "behavior_name": behavior_name,
                "action_name": action_name,
                "features_72": features_72,
                "padding_mask": padding_mask,
                "valid_mask": win_valid_mask,
                "ambiguous_mask": win_ambiguous_mask,
                "frame_valid": frame_valid,
                "reliability": reliability,
                "timestamps": win_timestamps,
                "start_idx": s,
                "num_original_frames": N
            })

            manifest_records.append({
                "window_id": window_id,
                "clip_id": clip_id,
                "split": mamba_split,
                "behavior_label": label,
                "behavior_name": behavior_name,
                "action_name": action_name,
                "start_frame": s,
                "end_frame": s + (N if is_padded_clip else window_length),
                "num_original_clip_frames": N,
                "is_padded": is_padded_clip,
                "num_padded_frames": int(padding_mask.sum()),
                "num_valid_frames": int(frame_valid.sum()),
                "num_ambiguous_frames": int(win_ambiguous_mask.sum()),
                "mean_reliability": round(float(reliability.mean()), 4)
            })

    # Save outputs/mamba_window_manifest.csv
    manifest_df = pd.DataFrame(manifest_records)
    window_manifest_csv = Path("outputs/mamba_window_manifest.csv")
    manifest_df.to_csv(window_manifest_csv, index=False)
    print(f"\nSaved window manifest to: {window_manifest_csv}")

    # Save persistent tensor cache
    cache_out_dir = Path("data/processed/animal_kingdom/mamba_cache")
    cache_out_dir.mkdir(parents=True, exist_ok=True)
    cache_out_file = cache_out_dir / "mamba_windows_dataset.npz"

    np.savez_compressed(
        cache_out_file,
        window_ids=np.array([w["window_id"] for w in windows_data]),
        clip_ids=np.array([w["clip_id"] for w in windows_data]),
        splits=np.array([w["split"] for w in windows_data]),
        labels=np.array([w["behavior_label"] for w in windows_data], dtype=np.int64),
        behavior_names=np.array([w["behavior_name"] for w in windows_data]),
        action_names=np.array([w["action_name"] for w in windows_data]),
        features_72=np.stack([w["features_72"] for w in windows_data], axis=0),
        padding_mask=np.stack([w["padding_mask"] for w in windows_data], axis=0),
        valid_mask=np.stack([w["valid_mask"] for w in windows_data], axis=0),
        ambiguous_mask=np.stack([w["ambiguous_mask"] for w in windows_data], axis=0),
        frame_valid=np.stack([w["frame_valid"] for w in windows_data], axis=0),
        reliability=np.stack([w["reliability"] for w in windows_data], axis=0),
        timestamps=np.stack([w["timestamps"] for w in windows_data], axis=0)
    )
    print(f"Saved persistent window dataset cache to: {cache_out_file}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("WINDOW GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Windows Generated: {len(windows_data)}")
    for s in ["train", "val", "test"]:
        sub = manifest_df[manifest_df["split"] == s]
        c_count = (sub["behavior_name"] == "CALM").sum()
        a_count = (sub["behavior_name"] == "AGITATED").sum()
        pad_count = (sub["is_padded"]).sum()
        print(f"Split {s.upper():5s} | Clips: {len(sub['clip_id'].unique()):2d} | Windows: {len(sub):3d} (CALM: {c_count:2d}, AGITATED: {a_count:3d}) | Padded Windows: {pad_count:2d}")

    # Generate Sequence Generation Report
    report_dir = Path("reports/mamba")
    report_dir.mkdir(parents=True, exist_ok=True)
    seq_report_path = report_dir / "sequence_generation_report.md"

    report_lines = [
        "# Stage 4 Sequence Generation Report",
        "",
        "**Project:** Zero Rabies-MMNet  ",
        "**Stage:** Stage 4 — Temporal Sequence & Window Construction  ",
        "**Date:** 2026-09-23  ",
        "",
        "---",
        "",
        "## 1. Partition Breakdown (Clip & Window Level)",
        "",
        "| Partition | Clips | Total Windows | CALM Windows | AGITATED Windows | Padded Windows | Total Valid Frames | Total Invalid Frames | Total Ambiguous Frames |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for s in ["train", "val", "test"]:
        sub = manifest_df[manifest_df["split"] == s]
        clips_cnt = len(sub["clip_id"].unique())
        win_cnt = len(sub)
        c_cnt = (sub["behavior_name"] == "CALM").sum()
        a_cnt = (sub["behavior_name"] == "AGITATED").sum()
        pad_cnt = int(sub["is_padded"].sum())
        val_f = int(sub["num_valid_frames"].sum())
        amb_f = int(sub["num_ambiguous_frames"].sum())
        inv_f = win_cnt * window_length - val_f
        report_lines.append(
            f"| **{s.upper()}** | {clips_cnt} | {win_cnt} | {c_cnt} | {a_cnt} | {pad_cnt} | {val_f:,} | {inv_f:,} | {amb_f:,} |"
        )

    tot_sub = manifest_df
    report_lines.append(
        f"| **TOTAL** | {len(tot_sub['clip_id'].unique())} | {len(tot_sub)} | {(tot_sub['behavior_name']=='CALM').sum()} | {(tot_sub['behavior_name']=='AGITATED').sum()} | {int(tot_sub['is_padded'].sum())} | {int(tot_sub['num_valid_frames'].sum()):,} | {len(tot_sub)*window_length - int(tot_sub['num_valid_frames'].sum()):,} | {int(tot_sub['num_ambiguous_frames'].sum()):,} |"
    )

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Sequence Construction Specifications",
        "- **Sampling Rate:** 8.0 FPS (~125 ms temporal delta per step)",
        "- **Sequence Length ($T$):** 16 observations (~2.0 seconds receptive field)",
        "- **Frame Feature Vector:** 24 normalized keypoint coordinates $(x_{norm}, y_{norm}, conf_{kpt}) = 72$ dimensions",
        "- **Window Stride:** 8 frames (50% temporal overlap for clips with $N \\ge 16$ frames)",
        "- **Short Clips Handling ($N < 16$ frames):** Edge-replication padding to $T=16$ with explicit boolean `padding_mask`",
        "- **Reliability Weighting:** $\\text{reliability}_t = \\text{bbox\\_conf}_t \\times (\\text{valid\\_mask}_t \\land \\neg \\text{padding\\_mask}_t) \\times (0.8 \\text{ if ambiguous else } 1.0)$",
        "",
        "---",
        "",
        "## 3. Clip Partition Allocation",
        "",
        "### Training Clips (37 clips):",
        ", ".join([f"`{c}`" for c in sorted(sub_train_df["clip_id"].tolist())]),
        "",
        "### Validation Clips (10 clips):",
        ", ".join([f"`{c}`" for c in sorted(val_df["clip_id"].tolist())]),
        "",
        "### Held-Out Test Clips (21 clips):",
        ", ".join([f"`{c}`" for c in sorted(test_clips_df["clip_id"].tolist())]),
        "",
        "---",
        "",
        "## 4. Integrity Assertion",
        "- Zero clip overlap between Train, Validation, and Test partitions.",
        "- No temporal window spans across multiple source video clips.",
        "- Official Animal Kingdom 21 test clips strictly held out.",
        ""
    ])

    seq_report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Generated sequence generation report: {seq_report_path}")

if __name__ == "__main__":
    build_sequences()

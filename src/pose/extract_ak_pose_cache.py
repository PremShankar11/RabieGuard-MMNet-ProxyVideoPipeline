import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 24 Keypoint connections for canine visualization
SKELETON_PAIRS = [
    (1, 2), (2, 3), (3, 23),       # Front Left leg -> withers
    (7, 8), (8, 9), (9, 23),       # Front Right leg -> withers
    (4, 5), (5, 6), (6, 13),       # Rear Left leg -> tail start (pelvis)
    (10, 11), (11, 12), (12, 13),  # Rear Right leg -> tail start (pelvis)
    (23, 13),                      # Spine: withers <-> tail start
    (13, 14),                      # Tail: start <-> end
    (23, 24),                      # Neck: withers <-> throat
    (24, 18),                      # Throat <-> chin
    (18, 17),                      # Chin <-> nose
    (17, 21), (17, 22),            # Nose <-> eyes
    (21, 15), (22, 16),            # Eyes <-> ear bases
    (15, 19), (16, 20),            # Ear bases <-> ear tips
]

def draw_dog_skeleton(img, kpts, confs, bbox=None, bbox_conf=None, is_ambiguous=False, conf_thresh=0.25):
    """Draw bounding box and 24-keypoint skeleton on image."""
    canvas = img.copy()

    # Draw Bounding Box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_color = (0, 165, 255) if is_ambiguous else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        label = f"Dog {bbox_conf:.2f}" if bbox_conf is not None else "Dog"
        if is_ambiguous:
            label += " [MULTI-DOG]"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Draw Skeleton Lines
    for p1, p2 in SKELETON_PAIRS:
        idx1, idx2 = p1 - 1, p2 - 1
        if idx1 < len(kpts) and idx2 < len(kpts):
            c1, c2 = confs[idx1], confs[idx2]
            if c1 >= conf_thresh and c2 >= conf_thresh:
                pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                cv2.line(canvas, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

    # Draw Keypoint Circles
    for i, (pt, c) in enumerate(zip(kpts, confs)):
        if c >= conf_thresh:
            px, py = int(pt[0]), int(pt[1])
            if i in [14, 15, 16, 17, 18, 19, 20, 21, 23]:
                color = (255, 0, 255)
            elif i in [12, 13, 22]:
                color = (0, 255, 255)
            else:
                color = (255, 255, 0)
            cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)

    return canvas

def main():
    parser = argparse.ArgumentParser(description="Extract 8 FPS YOLO Pose cache for Animal Kingdom clips")
    parser.add_argument("--weights", type=str, default="checkpoints/dog_pose/best.pt", help="YOLO Pose model weights")
    parser.add_argument("--input_csv", type=str, default="outputs/interim_video_model_eligible.csv", help="Eligible clips CSV")
    parser.add_argument("--target_fps", type=float, default=8.0, help="Target sampling rate (default: 8.0 FPS)")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--save_visuals", action="store_true", default=True, help="Save qualitative visual samples")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    assert weights_path.exists(), f"Model weights not found: {weights_path}"

    df_eligible = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df_eligible)} model-eligible clips from {args.input_csv}")
    print(f"Splits: {dict(df_eligible['split'].value_counts())}")

    # Output directories
    cache_root = Path("data/processed/animal_kingdom/pose_cache")
    train_cache_dir = cache_root / "train"
    test_cache_dir = cache_root / "test"
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = Path("visualizations/pose/animal_kingdom_samples")
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading YOLO Pose model from {weights_path}...")
    model = YOLO(str(weights_path))

    manifest_rows = []
    actions_visualized = set()

    for idx, row in df_eligible.iterrows():
        clip_id = str(row["clip_id"]).strip()
        split = str(row["split"]).strip()
        action_name = str(row["action_name"]).strip()
        interim_label = str(row["interim_label"]).strip()
        video_rel_path = str(row["video_path"]).strip()

        video_path = Path(video_rel_path)
        if not video_path.exists():
            video_path = Path("c:/Important_prem/FYP/Zero-Rabies-MMNet") / video_rel_path
        assert video_path.exists(), f"Video file not found: {video_path}"

        # Open video and inspect properties
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Cannot open video: {video_path}"
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / actual_fps if actual_fps > 0 else 0.0

        # Calculate 8 FPS sampled frame indices based on timestamps
        num_samples = max(1, int(round(duration * args.target_fps)))
        sample_timestamps = [i / args.target_fps for i in range(num_samples)]
        sample_frame_indices = [min(total_frames - 1, int(round(t * actual_fps))) for t in sample_timestamps]

        # Extract sampled frames
        sampled_frames = []
        for f_idx in sample_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # If read fails, insert black frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            sampled_frames.append(frame)
        cap.release()

        # Run YOLO Pose inference on sampled frames
        res_list = model.predict(source=sampled_frames, conf=args.conf_thresh, imgsz=640, verbose=False)

        N = len(sampled_frames)
        keypoints_raw = np.zeros((N, 24, 3), dtype=np.float32)
        keypoints_normalized = np.zeros((N, 24, 3), dtype=np.float32)
        keypoint_confidence = np.zeros((N, 24), dtype=np.float32)
        bbox = np.zeros((N, 4), dtype=np.float32)
        bbox_confidence = np.zeros((N,), dtype=np.float32)
        valid_mask = np.zeros((N,), dtype=bool)
        ambiguous_mask = np.zeros((N,), dtype=bool)
        num_detections = np.zeros((N,), dtype=np.int32)

        # Should we save visual samples for this clip?
        save_vis = args.save_visuals and (action_name not in actions_visualized or len(actions_visualized) < 12)
        vis_frames = []

        for i, res in enumerate(res_list):
            if res.boxes is not None and len(res.boxes) > 0:
                det_count = len(res.boxes)
                num_detections[i] = det_count

                boxes = res.boxes.xyxy.cpu().numpy()
                bconfs = res.boxes.conf.cpu().numpy()
                kpts_all = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((det_count, 24, 2))
                kconfs_all = res.keypoints.conf.cpu().numpy() if (res.keypoints is not None and res.keypoints.conf is not None) else np.zeros((det_count, 24))

                # Primary target selection: highest box confidence
                best_det_idx = int(np.argmax(bconfs))
                valid_mask[i] = True
                ambiguous_mask[i] = (det_count > 1)

                best_box = boxes[best_det_idx]
                best_bconf = float(bconfs[best_det_idx])
                best_kpts = kpts_all[best_det_idx]
                best_kconfs = kconfs_all[best_det_idx]

                bbox[i] = best_box
                bbox_confidence[i] = best_bconf
                keypoint_confidence[i] = best_kconfs
                keypoints_raw[i, :, :2] = best_kpts
                keypoints_raw[i, :, 2] = best_kconfs

                # Body-relative normalization
                x1, y1, x2, y2 = best_box
                w_box = max(1.0, x2 - x1)
                h_box = max(1.0, y2 - y1)
                x_mid = x1 + w_box / 2.0
                y_mid = y1 + h_box / 2.0
                scale = max(w_box, h_box)

                keypoints_normalized[i, :, 0] = (best_kpts[:, 0] - x_mid) / scale
                keypoints_normalized[i, :, 1] = (best_kpts[:, 1] - y_mid) / scale
                keypoints_normalized[i, :, 2] = best_kconfs

                if save_vis:
                    annotated = draw_dog_skeleton(
                        sampled_frames[i],
                        best_kpts,
                        best_kconfs,
                        bbox=best_box,
                        bbox_conf=best_bconf,
                        is_ambiguous=ambiguous_mask[i]
                    )
                    vis_frames.append(annotated)
            else:
                num_detections[i] = 0
                valid_mask[i] = False
                ambiguous_mask[i] = False
                if save_vis:
                    canvas = sampled_frames[i].copy()
                    cv2.putText(canvas, "No dog detected", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    vis_frames.append(canvas)

        # Save visual sample video/gif/strip if selected
        if save_vis and vis_frames:
            actions_visualized.add(action_name)
            # Save visual image grid (middle frame or 3 key frames)
            mid_idx = len(vis_frames) // 2
            out_img_path = vis_dir / f"{split}_{clip_id}_{action_name.replace(' ', '_')}.png"
            cv2.imwrite(str(out_img_path), vis_frames[mid_idx])

        # Behavior label convention: CALM = 0, AGITATED = 1
        behavior_int = 0 if interim_label == "CALM" else 1

        # Save persistent NPZ cache file
        target_dir = train_cache_dir if split == "train" else test_cache_dir
        cache_file = target_dir / f"{clip_id}.npz"

        np.savez_compressed(
            cache_file,
            keypoints_raw=keypoints_raw,
            keypoints_normalized=keypoints_normalized,
            keypoint_confidence=keypoint_confidence,
            bbox=bbox,
            bbox_confidence=bbox_confidence,
            frame_indices=np.array(sample_frame_indices, dtype=np.int32),
            timestamps=np.array(sample_timestamps, dtype=np.float32),
            valid_mask=valid_mask,
            ambiguous_mask=ambiguous_mask,
            num_detections=num_detections,
            fps=np.float32(actual_fps),
            sampled_fps=np.float32(args.target_fps),
            source_video=str(video_rel_path),
            split=split,
            behavior_label=behavior_int,
            behavior_name=interim_label,
            action_name=action_name,
            num_frames=N
        )

        valid_count = int(valid_mask.sum())
        invalid_count = N - valid_count
        ambiguous_count = int(ambiguous_mask.sum())
        valid_ratio = valid_count / N if N > 0 else 0.0

        mean_det_conf = float(bbox_confidence[valid_mask].mean()) if valid_count > 0 else 0.0
        mean_kpt_conf = float(keypoint_confidence[valid_mask].mean()) if valid_count > 0 else 0.0

        manifest_rows.append({
            "clip_id": clip_id,
            "source_path": video_rel_path,
            "split": split,
            "action": action_name,
            "behavior_label": behavior_int,
            "behavior_name": interim_label,
            "duration": round(duration, 2),
            "fps": round(actual_fps, 2),
            "sampled_fps": args.target_fps,
            "number_of_sampled_frames": N,
            "valid_pose_frames": valid_count,
            "invalid_pose_frames": invalid_count,
            "ambiguous_frames": ambiguous_count,
            "mean_detection_confidence": round(mean_det_conf, 4),
            "mean_keypoint_confidence": round(mean_kpt_conf, 4),
            "valid_frame_ratio": round(valid_ratio, 4),
            "cache_path": cache_file.as_posix()
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_eligible):
            print(f"Processed {idx+1}/{len(df_eligible)} clips (Latest: {clip_id}, valid ratio: {valid_ratio*100:.1f}%)")

    # Save master manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = Path("outputs/animal_kingdom_pose_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nMaster manifest saved to: {manifest_csv}")
    print(f"Total train caches created: {len(list(train_cache_dir.glob('*.npz')))} (Expected: 47)")
    print(f"Total test caches created:  {len(list(test_cache_dir.glob('*.npz')))} (Expected: 21)")
    print(f"Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()

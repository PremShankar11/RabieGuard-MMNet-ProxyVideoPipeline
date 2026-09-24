"""
Multi-Dog Temporal Identity-Consistency Tracker for Video V2 (Experiment D).
Zero Rabies-MMNet Project.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ultralytics import YOLO

def box_iou(box1, box2):
    """Compute IoU between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-4, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-4, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def box_center_dist(box1, box2, diag):
    """Normalized distance between centers of two boxes."""
    c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0])
    c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0])
    dist = np.linalg.norm(c1 - c2)
    return dist / max(1.0, diag)

def extract_temporally_consistent_poses(
    dev_clips,
    yolo_model_path="checkpoints/dog_pose/best.pt",
    output_dir="data/processed/animal_kingdom/v2_cache/pose_consistent/train"
):
    """
    Extract temporally consistent canine pose tracks for development clips.
    Avoids identity hops in multi-dog scenes via IoU & center-proximity track persistence.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT D: EXTRACTING TEMPORALLY CONSISTENT POSES")
    print("=" * 60)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(yolo_model_path)
    manifest = pd.read_csv("outputs/animal_kingdom_pose_manifest.csv")
    manifest_map = {r["clip_id"]: r for _, r in manifest.iterrows()}

    processed_count = 0
    multi_dog_enhanced = 0

    for clip_id in dev_clips:
        dst_file = out_path / f"{clip_id}.npz"
        if dst_file.exists():
            processed_count += 1
            continue

        meta = manifest_map[clip_id]
        video_path = Path(meta["source_path"])
        if not video_path.exists():
            video_path = Path("data/raw/animal_kingdom/video") / f"{clip_id}.mp4"

        cap = cv2.VideoCapture(str(video_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / actual_fps if actual_fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        diag = np.sqrt(width ** 2 + height ** 2)

        target_fps = 8.0
        num_samples = max(1, int(round(duration * target_fps)))
        sample_timestamps = [i / target_fps for i in range(num_samples)]
        sample_frame_indices = [min(total_frames - 1, int(round(t * actual_fps))) for t in sample_timestamps]

        sampled_frames = []
        for f_idx in sample_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            sampled_frames.append(frame)
        cap.release()

        # Run YOLO Pose on sampled frames
        res_list = model.predict(source=sampled_frames, conf=0.25, imgsz=640, verbose=False)

        N = len(sampled_frames)
        keypoints_raw = np.zeros((N, 24, 3), dtype=np.float32)
        keypoints_normalized = np.zeros((N, 24, 3), dtype=np.float32)
        keypoint_confidence = np.zeros((N, 24), dtype=np.float32)
        bbox = np.zeros((N, 4), dtype=np.float32)
        bbox_confidence = np.zeros((N,), dtype=np.float32)
        valid_mask = np.zeros((N,), dtype=bool)
        ambiguous_mask = np.zeros((N,), dtype=bool)
        num_detections = np.zeros((N,), dtype=np.int32)

        prev_box = None
        has_multi = False

        for i, res in enumerate(res_list):
            if res.boxes is not None and len(res.boxes) > 0:
                det_count = len(res.boxes)
                num_detections[i] = det_count
                boxes = res.boxes.xyxy.cpu().numpy()
                bconfs = res.boxes.conf.cpu().numpy()
                kpts_all = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((det_count, 24, 2))
                kconfs_all = res.keypoints.conf.cpu().numpy() if (res.keypoints is not None and res.keypoints.conf is not None) else np.zeros((det_count, 24))

                valid_mask[i] = True
                ambiguous_mask[i] = (det_count > 1)
                if det_count > 1:
                    has_multi = True

                # Selection strategy: Temporal consistency vs confidence
                if prev_box is None or det_count == 1:
                    best_idx = int(np.argmax(bconfs))
                else:
                    # Score each candidate by IoU and distance to previous tracked box
                    scores = []
                    for c_idx, cand_b in enumerate(boxes):
                        iou = box_iou(cand_b, prev_box)
                        dist = box_center_dist(cand_b, prev_box, diag)
                        # Proximity score + confidence bonus
                        score = iou * 1.5 + (1.0 - min(1.0, dist)) * 0.8 + float(bconfs[c_idx]) * 0.5
                        scores.append(score)
                    best_idx = int(np.argmax(scores))

                best_box = boxes[best_idx]
                best_bconf = float(bconfs[best_idx])
                best_kpts = kpts_all[best_idx]
                best_kconfs = kconfs_all[best_idx]

                prev_box = best_box
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
            else:
                prev_box = None
                valid_mask[i] = False
                ambiguous_mask[i] = False

        if has_multi:
            multi_dog_enhanced += 1

        np.savez_compressed(
            dst_file,
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
            sampled_fps=np.float32(target_fps),
            source_video=str(video_path),
            split=meta["split"],
            behavior_label=int(meta["behavior_label"]),
            behavior_name=meta["behavior_name"],
            action_name=meta["action"],
            num_frames=N
        )
        processed_count += 1

    print(f"Temporally consistent poses cached: {processed_count} clips ({multi_dog_enhanced} multi-dog clips tracked)")
    return out_path

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 24 Keypoint connections for canine anatomical skeleton visualization
# Keypoints (1-indexed):
# 1-FL_paw, 2-FL_knee, 3-FL_elbow, 4-RL_paw, 5-RL_knee, 6-RL_elbow
# 7-FR_paw, 8-FR_knee, 9-FR_elbow, 10-RR_paw, 11-RR_knee, 12-RR_elbow
# 13-tail_start, 14-tail_end, 15-L_ear_base, 16-R_ear_base, 17-nose, 18-chin
# 19-L_ear_tip, 20-R_ear_tip, 21-L_eye, 22-R_eye, 23-withers, 24-throat
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

def draw_dog_skeleton(img, kpts, confs, bbox=None, bbox_conf=None, conf_thresh=0.25):
    """Draw bounding box and 24-keypoint skeleton on image."""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # Draw Bounding Box
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if bbox_conf is not None:
            label = f"Dog {bbox_conf:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            # Head/face: Magenta; Legs: Cyan; Spine/Tail: Yellow
            if i in [14, 15, 16, 17, 18, 19, 20, 21, 23]:
                color = (255, 0, 255)
            elif i in [12, 13, 22]:
                color = (0, 255, 255)
            else:
                color = (255, 255, 0)
            cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11n-Pose on Dog-Pose validation set")
    parser.add_argument("--weights", type=str, default="checkpoints/dog_pose/best.pt", help="Model weights path")
    parser.add_argument("--config", type=str, default="configs/dog_pose.yaml", help="Dataset config YAML")
    parser.add_argument("--vis_count", type=int, default=12, help="Number of qualitative visualization samples")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    assert weights_path.exists(), f"Weights not found: {weights_path}"

    vis_dir = Path("visualizations/pose/dog_pose_validation")
    vis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/pose")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EVALUATING YOLO POSE ON DOG-POSE VALIDATION SET")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Config: {args.config}")

    model = YOLO(str(weights_path))

    # Quantitative Validation
    print("\nRunning Ultralytics validation on val split...")
    metrics = model.val(data=args.config, split="val", batch=16, imgsz=640, device=0 if torch.cuda.is_available() else "cpu")

    # Extract metrics
    pose_map50 = float(metrics.pose.map50)
    pose_map = float(metrics.pose.map)
    pose_p = float(metrics.pose.mp)
    pose_r = float(metrics.pose.mr)

    box_map50 = float(metrics.box.map50)
    box_map = float(metrics.box.map)
    box_p = float(metrics.box.mp)
    box_r = float(metrics.box.mr)

    print("\n--- VALIDATION METRICS ---")
    print(f"Pose mAP@0.50:      {pose_map50:.4f}")
    print(f"Pose mAP@0.50:0.95: {pose_map:.4f}")
    print(f"Pose Precision:     {pose_p:.4f}")
    print(f"Pose Recall:        {pose_r:.4f}")
    print(f"Box mAP@0.50:       {box_map50:.4f}")
    print(f"Box mAP@0.50:0.95:  {box_map:.4f}")
    print(f"Box Precision:      {box_p:.4f}")
    print(f"Box Recall:         {box_r:.4f}")

    # Generate Qualitative Visualizations
    val_img_dir = Path("data/raw/dog_pose/images/val")
    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    # Select evenly spaced images for broad coverage
    step = max(1, len(val_images) // args.vis_count)
    selected_images = [val_images[i * step] for i in range(args.vis_count)]

    print(f"\nGenerating {len(selected_images)} qualitative visualization samples...")
    for idx, img_path in enumerate(selected_images):
        results = model.predict(source=str(img_path), conf=0.25, imgsz=640, verbose=False)
        res = results[0]
        orig_img = res.orig_img

        if res.keypoints is not None and len(res.keypoints) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            bconfs = res.boxes.conf.cpu().numpy()
            kpts_all = res.keypoints.xy.cpu().numpy()
            kconfs_all = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones((len(boxes), 24))

            canvas = orig_img.copy()
            for b_idx in range(len(boxes)):
                canvas = draw_dog_skeleton(
                    canvas,
                    kpts_all[b_idx],
                    kconfs_all[b_idx],
                    bbox=boxes[b_idx],
                    bbox_conf=bconfs[b_idx],
                    conf_thresh=0.25
                )
        else:
            canvas = orig_img.copy()
            cv2.putText(canvas, "No dog pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out_path = vis_dir / f"val_sample_{idx+1:02d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  Saved sample {idx+1}/{len(selected_images)}: {out_path.name}")

    print(f"\nAll qualitative samples saved to: {vis_dir}")
    print("Evaluation script finished successfully.")

if __name__ == "__main__":
    main()

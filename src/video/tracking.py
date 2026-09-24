"""
Temporal Consistency Canine Tracking Module for Zero Rabies-MMNet Video V2.
Preserves identity of primary dog across video frames in single- and multi-dog scenes,
applies body-relative normalization, and computes ambiguity statistics.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


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


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-4, float(box1[2] - box1[0]) * float(box1[3] - box1[1]))
    area2 = max(1e-4, float(box2[2] - box2[0]) * float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def box_center_dist(box1: np.ndarray, box2: np.ndarray, diag: float) -> float:
    """Compute normalized Euclidean distance between centers of two boxes."""
    c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0])
    c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0])
    dist = float(np.linalg.norm(c1 - c2))
    return dist / max(1.0, float(diag))


def draw_dog_skeleton(
    img: np.ndarray,
    kpts: np.ndarray,
    confs: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    bbox_conf: Optional[float] = None,
    is_ambiguous: bool = False,
    conf_thresh: float = 0.25,
    timestamp_sec: Optional[float] = None
) -> np.ndarray:
    """Draw canine bounding box, 24-keypoint anatomical skeleton, and telemetry overlay."""
    canvas = img.copy()

    # Draw Bounding Box
    if bbox is not None and len(bbox) == 4 and np.any(bbox > 0):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_color = (0, 165, 255) if is_ambiguous else (0, 255, 0) # Orange if multi-dog, Green if primary
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        label = f"Tracked Dog {bbox_conf:.2f}" if bbox_conf is not None else "Tracked Dog"
        if is_ambiguous:
            label += " [MULTI-DOG]"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Draw Skeleton Lines
    if kpts is not None and len(kpts) == 24:
        for p1, p2 in SKELETON_PAIRS:
            idx1, idx2 = p1 - 1, p2 - 1
            if idx1 < len(kpts) and idx2 < len(kpts):
                c1, c2 = confs[idx1], confs[idx2]
                if c1 >= conf_thresh and c2 >= conf_thresh:
                    pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                    pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                    cv2.line(canvas, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

        # Draw Keypoints
        for i, (pt, c) in enumerate(zip(kpts, confs)):
            if c >= conf_thresh:
                px, py = int(pt[0]), int(pt[1])
                color = (255, 0, 255) if i in [14, 15, 16, 17, 18, 19, 20, 21, 23] else (0, 255, 255)
                cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)

    # Draw Timestamp badge if provided
    if timestamp_sec is not None:
        ts_text = f"t = {timestamp_sec:.2f}s"
        cv2.putText(canvas, ts_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(canvas, ts_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return canvas


class TemporalDogTracker:
    """
    Temporal consistency tracker for canine pose sequences.
    Tracks a persistent dog identity across frames and outputs body-relative normalized features.
    """
    def __init__(self, frame_width: int = 640, frame_height: int = 360):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.diag = float(np.sqrt(frame_width ** 2 + frame_height ** 2))

    def track_sequence(
        self,
        frame_detections: List[Dict[str, Any]],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Track canine detections across all sampled frames.
        Returns:
            Dictionary containing:
                keypoints_raw: (N, 24, 3) [x, y, conf]
                keypoints_normalized: (N, 24, 3) [norm_x, norm_y, conf]
                keypoint_confidence: (N, 24)
                bbox: (N, 4)
                bbox_confidence: (N,)
                valid_mask: (N,) bool
                ambiguous_mask: (N,) bool
                num_detections: (N,) int
                num_frames: int
        """
        N = len(frame_detections)
        keypoints_raw = np.zeros((N, 24, 3), dtype=np.float32)
        keypoints_normalized = np.zeros((N, 24, 3), dtype=np.float32)
        keypoint_confidence = np.zeros((N, 24), dtype=np.float32)
        bbox = np.zeros((N, 4), dtype=np.float32)
        bbox_confidence = np.zeros((N,), dtype=np.float32)
        valid_mask = np.zeros((N,), dtype=bool)
        ambiguous_mask = np.zeros((N,), dtype=bool)
        num_detections = np.zeros((N,), dtype=np.int32)

        prev_box = None

        for i, det in enumerate(frame_detections):
            det_count = det.get("num_detections", 0)
            num_detections[i] = det_count

            if det_count > 0:
                boxes = det["boxes"]
                bconfs = det["box_confidences"]
                kpts_all = det["keypoints_xy"]
                kconfs_all = det["keypoint_confidences"]

                valid_mask[i] = True
                ambiguous_mask[i] = (det_count > 1)

                # Selection strategy: Temporal consistency vs confidence
                if prev_box is None or det_count == 1:
                    best_idx = int(np.argmax(bconfs))
                else:
                    scores = []
                    for c_idx, cand_b in enumerate(boxes):
                        iou = box_iou(cand_b, prev_box)
                        dist = box_center_dist(cand_b, prev_box, self.diag)
                        # V2 continuity scoring: IoU bonus + proximity bonus + confidence bonus
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
                w_box = max(1.0, float(x2 - x1))
                h_box = max(1.0, float(y2 - y1))
                x_mid = float(x1 + w_box / 2.0)
                y_mid = float(y1 + h_box / 2.0)
                scale = max(w_box, h_box)

                norm_x = (best_kpts[:, 0] - x_mid) / scale
                norm_y = (best_kpts[:, 1] - y_mid) / scale
                # Clip normalized coordinates to safe physiological range [-3.0, 3.0]
                norm_x = np.clip(norm_x, -3.0, 3.0)
                norm_y = np.clip(norm_y, -3.0, 3.0)

                keypoints_normalized[i, :, 0] = norm_x
                keypoints_normalized[i, :, 1] = norm_y
                keypoints_normalized[i, :, 2] = best_kconfs
            else:
                prev_box = None
                valid_mask[i] = False
                ambiguous_mask[i] = False

        return {
            "keypoints_raw": keypoints_raw,
            "keypoints_normalized": keypoints_normalized,
            "keypoint_confidence": keypoint_confidence,
            "bbox": bbox,
            "bbox_confidence": bbox_confidence,
            "valid_mask": valid_mask,
            "ambiguous_mask": ambiguous_mask,
            "num_detections": num_detections,
            "num_frames": N,
            "timestamps": np.array(timestamps if timestamps is not None else [i / 8.0 for i in range(N)], dtype=np.float32)
        }

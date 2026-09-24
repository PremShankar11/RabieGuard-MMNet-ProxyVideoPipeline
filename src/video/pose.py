"""
Dog Pose Estimation Module for Zero Rabies-MMNet Video V2.
Encapsulates YOLO11n-Pose inference for 24 canine anatomical keypoints.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO


class PoseModelLoadError(Exception):
    """Raised when YOLO11n-Pose weights cannot be loaded."""
    pass


class DogPoseEstimator:
    """
    Inference wrapper for canine YOLO11n-Pose checkpoint.
    Extracts 24 anatomical keypoints per detected dog.
    """
    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/dog_pose/best.pt",
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        imgsz: int = 640
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise PoseModelLoadError(f"Dog pose checkpoint not found at: {self.checkpoint_path}")

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(str(self.checkpoint_path))
            return model
        except Exception as e:
            raise PoseModelLoadError(f"Failed to load YOLO pose model: {e}") from e

    def predict_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Run canine pose estimation on sampled frames.
        Returns a list of dicts (one per frame) containing all detected dogs:
        {
            "num_detections": int,
            "boxes": np.ndarray (K, 4) [x1, y1, x2, y2],
            "box_confidences": np.ndarray (K,),
            "keypoints_xy": np.ndarray (K, 24, 2),
            "keypoint_confidences": np.ndarray (K, 24)
        }
        """
        if not frames:
            return []

        results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            preds = self.model.predict(
                source=batch,
                conf=self.conf_thresh,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            for res in preds:
                if res.boxes is not None and len(res.boxes) > 0:
                    det_count = len(res.boxes)
                    boxes = res.boxes.xyxy.cpu().numpy()
                    bconfs = res.boxes.conf.cpu().numpy()

                    if res.keypoints is not None and res.keypoints.xy is not None:
                        kpts = res.keypoints.xy.cpu().numpy()
                    else:
                        kpts = np.zeros((det_count, 24, 2), dtype=np.float32)

                    if res.keypoints is not None and res.keypoints.conf is not None:
                        kconfs = res.keypoints.conf.cpu().numpy()
                    else:
                        kconfs = np.zeros((det_count, 24), dtype=np.float32)

                    results.append({
                        "num_detections": det_count,
                        "boxes": boxes,
                        "box_confidences": bconfs,
                        "keypoints_xy": kpts,
                        "keypoint_confidences": kconfs,
                    })
                else:
                    results.append({
                        "num_detections": 0,
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "box_confidences": np.zeros((0,), dtype=np.float32),
                        "keypoints_xy": np.zeros((0, 24, 2), dtype=np.float32),
                        "keypoint_confidences": np.zeros((0, 24), dtype=np.float32),
                    })

        return results

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection2D:
    bbox_xyxy: np.ndarray
    class_name: str
    confidence: float
    ground_point: Tuple[float, float]


class VehicleDetector2D:
    """Simple YOLOv8n wrapper for 2D vehicle detection."""

    VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model = YOLO(model_name)
        self.model.to("cpu")

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        results = self.model(frame, stream=False, conf=0.35, imgsz=640)[0]
        detections: List[Detection2D] = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = self.model.names.get(cls_id, "")
            if cls_name not in self.VEHICLE_CLASSES:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            bbox_xyxy = np.array([x1, y1, x2, y2], dtype=float)
            ground_point = (float((x1 + x2) / 2.0), float(y2))
            detections.append(
                Detection2D(
                    bbox_xyxy=bbox_xyxy,
                    class_name=cls_name,
                    confidence=conf,
                    ground_point=ground_point,
                )
            )
        return detections

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import numpy as np

from src.detection2d.yolo_detector import Detection2D


@dataclass
class Box3D:
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.l, self.w, self.h, self.yaw]

    def __iter__(self):
        yield from self.to_list()

    def __getitem__(self, idx):
        return self.to_list()[idx]


class MonocularLifter:
    """Estimate a simple 3D box from a 2D vehicle detection using a flat ground plane."""

    DIM_PRIORS = {
        "car": (4.0, 1.8, 1.5),
        "bus": (10.5, 2.5, 3.2),
        "truck": (6.0, 2.2, 2.4),
        "motorcycle": (1.8, 0.8, 1.5),
    }

    def __init__(
        self,
        camera_height_m: float,
        tilt_deg: float,
        focal_px: float,
        principal_point: Tuple[float, float],
    ) -> None:
        self.camera_height_m = float(camera_height_m)
        self.tilt_deg = float(tilt_deg)
        self.focal_px = float(focal_px)
        self.principal_point = tuple(principal_point)
        self.tilt_rad = math.radians(self.tilt_deg)

    def lift(self, detection2d: Detection2D) -> Box3D:
        px, py = detection2d.ground_point
        cx, cy = self.principal_point
        # Real pinhole/flat-ground projection:
        # distance = camera_height / tan(tilt_angle + atan((pixel_y - principal_point_y) / focal_px))
        # The pixel's horizontal offset sets the lateral position in the ground plane.
        angle_y = math.atan2(py - cy, self.focal_px)
        angle_total = self.tilt_rad + angle_y
        ground_distance = self.camera_height_m / math.tan(angle_total)
        x = ground_distance
        y = -(px - cx) * ground_distance / self.focal_px
        z = 0.0
        l, w, h = self.DIM_PRIORS.get(detection2d.class_name.lower(), (4.0, 1.8, 1.5))
        # Match the existing tracker interface: [x, y, z, l, w, h, yaw]
        yaw = 0.0
        return Box3D(x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw)

    def to_bbox_3d(self, detection2d: Detection2D) -> list[float]:
        box = self.lift(detection2d)
        return box.to_list()

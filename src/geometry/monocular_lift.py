from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

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
        l, w, h = self.DIM_PRIORS.get(detection2d.class_name.lower(), (4.0, 1.8, 1.5))
        if self.tilt_deg > 45.0:
            x1, y1, x2, y2 = detection2d.bbox_xyxy
            px = (x1 + x2) / 2.0
            py = (y1 + y2) / 2.0
            z = -h / 2.0
        else:
            px, py = detection2d.ground_point
            z = 0.0

        cx, cy = self.principal_point
        # Real pinhole/flat-ground projection:
        # distance = camera_height / tan(tilt_angle + atan((pixel_y - principal_point_y) / focal_px))
        # The pixel's horizontal offset sets the lateral position in the ground plane.
        angle_y = math.atan2(py - cy, self.focal_px)
        angle_total = self.tilt_rad + angle_y
        # Safeguard: prevent total angle from going to 0 or negative (approaching or crossing the horizon)
        angle_total = max(0.05, angle_total)
        ground_distance = self.camera_height_m / math.tan(angle_total)
        x = ground_distance
        y = -(px - cx) * ground_distance / self.focal_px
        # Match the existing tracker interface: [x, y, z, l, w, h, yaw]
        yaw = 0.0

        return Box3D(x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw)

    def to_bbox_3d(self, detection2d: Detection2D) -> list[float]:
        box = self.lift(detection2d)
        return box.to_list()

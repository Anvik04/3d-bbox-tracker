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
        "car": (3.6, 1.6, 1.3),
        "bus": (9.0, 2.4, 2.8),
        "truck": (5.0, 2.0, 2.0),
        "motorcycle": (1.6, 0.7, 1.2),
    }

    def __init__(
        self,
        camera_height_m: float,
        tilt_deg: float,
        focal_px: float,
        principal_point: Tuple[float, float],
        use_camera_space: bool = False,
    ) -> None:
        self.camera_height_m = float(camera_height_m)
        self.tilt_deg = float(tilt_deg)
        self.focal_px = float(focal_px)
        self.principal_point = tuple(principal_point)
        self.tilt_rad = math.radians(self.tilt_deg)
        self.use_camera_space = use_camera_space

    def lift(self, detection2d: Detection2D) -> Box3D:
        l, w, h = self.DIM_PRIORS.get(detection2d.class_name.lower(), (3.6, 1.6, 1.3))

        if self.use_camera_space:
            x1, y1, x2, y2 = detection2d.bbox_xyxy
            h_box = max(y2 - y1, 1.0)

            # Depth (Z) estimation using height of 2D bounding box and focal length
            z_c = (self.focal_px * h) / h_box
            z_c = max(0.5, z_c)

            # X and Y estimation in camera coordinates
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            cx, cy = self.principal_point

            x_c = ((u_center - cx) * z_c) / self.focal_px
            y_c = ((v_center - cy) * z_c) / self.focal_px

            # Coordinate layout for Box3D to match tracking: x=X_c, y=Y_c, z=Z_c
            return Box3D(x=x_c, y=y_c, z=z_c, l=l, w=w, h=h, yaw=0.0)

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

        # Adaptive perspective safeguard: clamp minimum distance based on camera height
        # to prevent close-range infinite stretches
        min_dist = self.camera_height_m * 1.5
        ground_distance = max(min_dist, ground_distance)

        x = ground_distance
        y = -(px - cx) * ground_distance / self.focal_px
        # Match the existing tracker interface: [x, y, z, l, w, h, yaw]
        yaw = 0.0

        return Box3D(x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw)

    def to_bbox_3d(self, detection2d: Detection2D) -> list[float]:
        box = self.lift(detection2d)
        return box.to_list()

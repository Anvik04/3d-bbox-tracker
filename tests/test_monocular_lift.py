import math

import numpy as np

from src.detection2d.yolo_detector import Detection2D
from src.geometry.monocular_lift import MonocularLifter


def test_monocular_lift_ground_point_sanity():
    lifter = MonocularLifter(
        camera_height_m=1.6,
        tilt_deg=10.0,
        focal_px=600.0,
        principal_point=(320.0, 240.0),
    )

    cases = [
        ((320.0, 480.0), 1.6 / math.tan(math.radians(10.0) + math.atan2(240.0, 600.0))),
        ((320.0, 360.0), 1.6 / math.tan(math.radians(10.0) + math.atan2(120.0, 600.0))),
        ((400.0, 480.0), 1.6 / math.tan(math.radians(10.0) + math.atan2(240.0, 600.0))),
    ]

    for (px, py), expected_distance in cases:
        detection = Detection2D(
            bbox_xyxy=np.array([100.0, 180.0, 220.0, 260.0]),
            class_name="car",
            confidence=0.9,
            ground_point=(px, py),
        )
        box3d = lifter.lift(detection)
        assert math.isclose(box3d[0], expected_distance, rel_tol=1e-3, abs_tol=1e-3)
        assert math.isclose(box3d[2], 0.0, abs_tol=1e-9)
        assert box3d[3] > 2.0  # length prior
        assert box3d[4] > 1.0  # width prior
        assert box3d[5] > 1.0  # height prior

def test_monocular_lift_low_tilt():
    lifter = MonocularLifter(
        camera_height_m=2.0, tilt_deg=10.0, focal_px=600.0, principal_point=(320.0, 240.0)
    )
    det = Detection2D(
        bbox_xyxy=np.array([100.0, 100.0, 200.0, 200.0]),
        class_name="car",
        confidence=0.9,
        ground_point=(150.0, 200.0),
    )
    box3d = lifter.lift(det)
    # tilt < 45, so yaw is 0.0
    assert math.isclose(box3d.yaw, 0.0, abs_tol=1e-9)

def test_monocular_lift_high_tilt():
    lifter = MonocularLifter(
        camera_height_m=50.0, tilt_deg=90.0, focal_px=600.0, principal_point=(320.0, 240.0)
    )
    det = Detection2D(
        bbox_xyxy=np.array([100.0, 100.0, 200.0, 200.0]),
        class_name="car",
        confidence=0.9,
        ground_point=(150.0, 200.0),
    )
    box3d = lifter.lift(det)
    # tilt > 45, centroid is used, but yaw is 0.0 pre-tracking
    assert math.isclose(box3d.yaw, 0.0, abs_tol=1e-9)

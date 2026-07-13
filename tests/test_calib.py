import os
import tempfile
import numpy as np
from src.data.calib import Calibration
from src.data.synth_fixtures import CALIB_TEXT


def test_calibration_projection():
    # Write a temporary calibration file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(CALIB_TEXT)
        temp_calib_path = f.name

    try:
        # Load calibration
        calib = Calibration(temp_calib_path)

        # Test point in LiDAR coordinates: 10m forward, 2m left, 0m height
        pts_lidar = np.array([[10.0, 2.0, 0.0]])

        # Project to Camera Rectified
        pts_cam = calib.lidar_to_cam(pts_lidar)
        assert pts_cam.shape == (1, 3)

        # Invert project back to LiDAR
        pts_lidar_recon = calib.cam_to_lidar(pts_cam)
        assert pts_lidar_recon.shape == (1, 3)
        np.testing.assert_allclose(pts_lidar, pts_lidar_recon, atol=1e-5)

        # Project to Image
        pts_img, depths = calib.cam_to_img(pts_cam)
        assert pts_img.shape == (1, 2)
        assert depths.shape == (1,)
        # Depth should be positive for forward point
        assert depths[0] > 0

        # Project directly LiDAR to Image
        pts_img_direct, depths_direct = calib.lidar_to_img(pts_lidar)
        np.testing.assert_allclose(pts_img, pts_img_direct, atol=1e-5)
        np.testing.assert_allclose(depths, depths_direct, atol=1e-5)

    finally:
        os.remove(temp_calib_path)

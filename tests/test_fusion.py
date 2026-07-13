import os
import torch
import numpy as np
from src.data.calib import Calibration
from src.data.synth_fixtures import CALIB_TEXT
from src.models.fusion import CalibratedFusion
from src.models.image_backbone import ImageBackbone


def test_fusion_module():
    # Setup temporary calib file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(CALIB_TEXT)
        temp_calib_path = f.name

    try:
        calib = Calibration(temp_calib_path)

        # Image dimensions and features
        img_h, img_w = 375, 1242
        backbone = ImageBackbone(in_channels=3, out_channels=32)
        dummy_img = torch.randn((1, 3, img_h, img_w))
        img_feats = backbone(dummy_img)
        assert img_feats.shape == (1, 32, 47, 156)

        # LiDAR BEV features
        lidar_bev = torch.randn((1, 64, 128, 192))

        # Fusion module
        fusion = CalibratedFusion(
            x_range=(0.0, 48.0),
            y_range=(-16.0, 16.0),
            z_range=(-3.0, 1.0),
            voxel_size=(0.25, 0.25),
        )

        # Run fusion
        fused_bev = fusion(
            lidar_bev=lidar_bev,
            img_features=img_feats,
            calib=calib,
            img_shape=(img_h, img_w),
        )

        # Combined features should be 64 (LiDAR) + 32 (Image) = 96
        assert fused_bev.shape == (1, 96, 128, 192)

    finally:
        os.remove(temp_calib_path)

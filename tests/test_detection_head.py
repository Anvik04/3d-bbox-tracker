import os
import torch
import numpy as np
from src.data.calib import Calibration
from src.data.synth_fixtures import CALIB_TEXT
from src.models.detector import CameraLiDARDetector


def test_detector_end_to_end():
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(CALIB_TEXT)
        temp_calib_path = f.name

    try:
        calib = Calibration(temp_calib_path)

        # Instantiate detector
        detector = CameraLiDARDetector()
        # Ensure zero predictions with random weights by setting negative bias
        torch.nn.init.constant_(detector.detection_head.cls_head.bias, -10.0)
        torch.nn.init.constant_(detector.detection_head.cls_head.weight, 0.0)

        # Dummy inputs
        points = torch.randn((100, 4))
        # Map points to valid range to get non-empty voxelization
        points[:, 0] = torch.clamp(points[:, 0] * 10 + 20, 0.0, 48.0)
        points[:, 1] = torch.clamp(points[:, 1] * 5, -16.0, 16.0)
        points[:, 2] = torch.clamp(points[:, 2] - 1.0, -3.0, 1.0)
        points[:, 3] = torch.rand(100)

        image = torch.randn((3, 375, 1242))

        # Forward
        cls_logits, reg_preds = detector(points, image, calib)

        assert cls_logits.shape == (1, 1, 128, 192)
        assert reg_preds.shape == (1, 8, 128, 192)

        # Decode empty detections
        boxes, scores, classes = detector.decode_predictions(
            cls_logits, reg_preds, score_threshold=0.99
        )
        assert len(boxes) == 0

        # Inject a highly confident detection map and test decoding
        cls_logits_fake = torch.full((1, 1, 128, 192), -10.0)
        cls_logits_fake[0, 0, 50, 50] = 10.0  # highly confident point

        reg_preds_fake = torch.zeros((1, 8, 128, 192))
        # dx, dy, dz, dl, dw, dh, sin, cos
        reg_preds_fake[0, :, 50, 50] = torch.tensor(
            [0.1, -0.2, 0.0, 0.1, 0.2, 0.3, 0.0, 1.0]
        )

        boxes, scores, classes = detector.decode_predictions(
            cls_logits_fake, reg_preds_fake, score_threshold=0.5
        )

        assert len(boxes) == 1
        assert len(scores) == 1
        assert scores[0] > 0.9
        # Check coordinates:
        # col=50, row=50.
        # cx = 50 * 0.25 + 0.0 + 0.125 = 12.625
        # cy = 50 * 0.25 - 16.0 + 0.125 = -3.375
        # cz = -1.0
        # x_pred = cx + 0.1 = 12.725
        # y_pred = cy - 0.2 = -3.575
        # z_pred = cz + 0.0 = -1.0
        np.testing.assert_allclose(boxes[0][0], 12.625 + 0.1, atol=1e-5)
        np.testing.assert_allclose(boxes[0][1], -3.375 - 0.2, atol=1e-5)
        np.testing.assert_allclose(boxes[0][2], -1.0, atol=1e-5)
        assert boxes[0][6] == 0.0  # yaw = arctan2(0.0, 1.0)

    finally:
        os.remove(temp_calib_path)

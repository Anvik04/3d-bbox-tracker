import os
import torch
from src.data.kitti_dataset import KITTIDataset
from src.data.calib import Calibration


def test_kitti_dataset():
    # Find dataset relative to the test location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(test_dir, "..", "data", "fixtures"))

    # Instantiate dataset
    dataset = KITTIDataset(data_dir=data_dir)

    # Check length
    assert len(dataset) == 10

    # Retrieve first item
    sample = dataset[0]

    # Verify keys
    assert "points" in sample
    assert "image" in sample
    assert "calib" in sample
    assert "gt_boxes_3d" in sample
    assert "gt_names" in sample
    assert "file_id" in sample

    # Verify shapes and types
    assert isinstance(sample["points"], torch.Tensor)
    assert sample["points"].ndim == 2
    assert sample["points"].shape[1] == 4  # (x, y, z, intensity)

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].ndim == 3
    assert sample["image"].shape[0] == 3  # channels
    assert sample["image"].shape[1] == 375
    assert sample["image"].shape[2] == 1242

    assert isinstance(sample["calib"], Calibration)

    assert isinstance(sample["gt_boxes_3d"], torch.Tensor)
    assert sample["gt_boxes_3d"].ndim == 2
    assert sample["gt_boxes_3d"].shape[1] == 7  # (x, y, z, l, w, h, yaw)

    # In our fixtures, we simulated 2 cars per frame
    assert sample["gt_boxes_3d"].shape[0] == 2
    assert len(sample["gt_names"]) == 2
    assert sample["gt_names"] == ["Car", "Car"]

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.calib import Calibration


class KITTIDataset(Dataset):
    """
    Dataset loader for KITTI-formatted data.
    """

    def __init__(self, data_dir, split="fixtures", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.velodyne_dir = os.path.join(data_dir, "velodyne")
        self.image_dir = os.path.join(data_dir, "image_2")
        self.calib_dir = os.path.join(data_dir, "calib")
        self.label_dir = os.path.join(data_dir, "label_2")

        # Get list of file prefixes
        self.file_ids = sorted(
            [os.path.splitext(f)[0] for f in os.listdir(self.image_dir) if f.endswith(".png")]
        )

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        # 1. Load point cloud
        bin_path = os.path.join(self.velodyne_dir, f"{file_id}.bin")
        # KITTI points are float32 (x, y, z, intensity)
        pts_lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # 2. Load image
        img_path = os.path.join(self.image_dir, f"{file_id}.png")
        img = Image.open(img_path).convert("RGB")
        # Convert image to float32 tensor in range [0, 1], shape (3, H, W)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # 3. Load calibration
        calib_path = os.path.join(self.calib_dir, f"{file_id}.txt")
        calib = Calibration(calib_path)

        # 4. Load label file (if exists)
        gt_boxes = []
        gt_names = []
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ")
                    obj_type = parts[0]
                    if obj_type == "DontCare":
                        continue

                    # KITTI label format:
                    # type, truncated, occluded, alpha, bbox_2d (4), dims (h, w, l), loc_cam (x, y, z), ry_cam
                    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                    tx, ty, tz = float(parts[11]), float(parts[12]), float(parts[13])
                    ry = float(parts[14])

                    # 3D location in camera coordinate system is the bottom face center of the box.
                    # Convert bottom center in camera to LiDAR:
                    bottom_center_cam = np.array([[tx, ty, tz]])
                    bottom_center_lidar = calib.cam_to_lidar(bottom_center_cam)[0]

                    # The center of the 3D box in LiDAR:
                    # LiDAR z-axis is up, camera y-axis is down.
                    # bottom_center_lidar z is (center_z - h/2) in LiDAR.
                    # So center_z_lidar = bottom_center_lidar[2] + h/2.
                    cx = bottom_center_lidar[0]
                    cy = bottom_center_lidar[1]
                    cz = bottom_center_lidar[2] + h / 2.0

                    # Heading conversion:
                    # Camera heading vector when pointing in direction ry: [cos(ry), 0, -sin(ry)]
                    heading_cam = np.array([[np.cos(ry), 0.0, -np.sin(ry)]])
                    # Difference of projected points gives the heading direction vector in LiDAR
                    origin_lidar = calib.cam_to_lidar(np.array([[0.0, 0.0, 0.0]]))
                    heading_lidar_raw = calib.cam_to_lidar(heading_cam) - origin_lidar
                    yaw_lidar = np.arctan2(heading_lidar_raw[0, 1], heading_lidar_raw[0, 0])

                    gt_boxes.append([cx, cy, cz, l, w, h, yaw_lidar])
                    gt_names.append(obj_type)

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 else np.zeros((0, 7), dtype=np.float32)

        sample = {
            "points": torch.tensor(pts_lidar, dtype=torch.float32),
            "image": img_tensor,
            "calib": calib,
            "gt_boxes_3d": torch.tensor(gt_boxes, dtype=torch.float32),
            "gt_names": gt_names,
            "file_id": file_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon

from src.models.pillar_encoder import PillarEncoder
from src.models.image_backbone import ImageBackbone
from src.models.fusion import CalibratedFusion
from src.models.detection_head import DetectionHead


def box2d_to_polygon(x, y, l, w, yaw):
    """
    Computes a shapely Polygon for a 2D oriented bounding box.
    """
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    # Corners in local coordinate system
    dx = np.array([l / 2.0, l / 2.0, -l / 2.0, -l / 2.0])
    dy = np.array([w / 2.0, -w / 2.0, -w / 2.0, w / 2.0])
    # Rotate and translate
    rx = dx * cos_y - dy * sin_y + x
    ry = dx * sin_y + dy * cos_y + y
    return Polygon(np.column_stack((rx, ry)))


def bev_iou_numpy(box1, box2):
    """
    Calculates 2D BEV IoU between two boxes: [x, y, z, l, w, h, yaw].
    """
    poly1 = box2d_to_polygon(box1[0], box1[1], box1[3], box1[4], box1[6])
    poly2 = box2d_to_polygon(box2[0], box2[1], box2[3], box2[4], box2[6])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / max(union, 1e-6)
    except Exception:
        return 0.0


def rotated_nms(boxes, scores, iou_threshold=0.1):
    """
    Performs Rotated BEV NMS on decoded 3D boxes.
    Args:
        boxes: (N, 7) list or numpy array [x, y, z, l, w, h, yaw]
        scores: (N,) scores
        iou_threshold: float, IoU threshold for overlap suppression
    Returns:
        keep_indices: list of indices to keep
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(int(idx))
        if len(order) == 1:
            break

        # Compute IoUs with the rest
        ious = []
        for rest_idx in order[1:]:
            ious.append(bev_iou_numpy(boxes[idx], boxes[rest_idx]))

        ious = np.array(ious)
        # Keep boxes with IoU less than the threshold
        order = order[1:][ious < iou_threshold]

    return keep


class CameraLiDARDetector(nn.Module):
    """
    End-to-end 3D detector fusing camera images and LiDAR point clouds.
    """

    def __init__(
        self,
        x_range=(0.0, 48.0),
        y_range=(-16.0, 16.0),
        z_range=(-3.0, 1.0),
        voxel_size=(0.25, 0.25),
        max_points_per_pillar=20,
        l_mean=4.0,
        w_mean=1.6,
        h_mean=1.5,
    ):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.vx, self.vy = voxel_size

        self.nx = int(round((self.x_max - self.x_min) / self.vx))
        self.ny = int(round((self.y_max - self.y_min) / self.vy))

        self.l_mean = l_mean
        self.w_mean = w_mean
        self.h_mean = h_mean

        # Modules
        self.pillar_encoder = PillarEncoder(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            voxel_size=voxel_size,
            max_points_per_pillar=max_points_per_pillar,
            out_channels=64,
        )

        self.image_backbone = ImageBackbone(in_channels=3, out_channels=32)

        self.fusion = CalibratedFusion(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            voxel_size=voxel_size,
        )

        # Fused features will have 64 (LiDAR) + 32 (Image) = 96 channels
        self.detection_head = DetectionHead(in_channels=96, num_classes=1)

    def forward(self, points, image, calib):
        """
        Runs the forward pass.
        Args:
            points: (N, 4) tensor of point cloud
            image: (3, H_img, W_img) tensor of image
            calib: Calibration object
        Returns:
            cls_logits: (1, 1, H_bev, W_bev)
            reg_preds: (1, 8, H_bev, W_bev)
        """
        # 1. LiDAR branch
        lidar_bev = self.pillar_encoder(points)  # (1, 64, H_bev, W_bev)

        # 2. Image branch
        # Add batch dimension to image: (1, 3, H_img, W_img)
        img_input = image.unsqueeze(0)
        img_feats = self.image_backbone(img_input)  # (1, 32, H_feat, W_feat)

        # 3. Fusion
        img_shape = (image.shape[1], image.shape[2])
        fused_bev = self.fusion(
            lidar_bev, img_feats, calib, img_shape
        )  # (1, 96, H_bev, W_bev)

        # 4. Head
        cls_logits, reg_preds = self.detection_head(fused_bev)

        return cls_logits, reg_preds

    def decode_predictions(
        self, cls_logits, reg_preds, score_threshold=0.3, nms_threshold=0.1
    ):
        """
        Decodes classification and regression maps into a list of 3D bounding boxes.
        Returns:
            boxes: list of [x, y, z, l, w, h, yaw] (LiDAR frame)
            scores: list of float scores
            classes: list of int class IDs
        """
        device = cls_logits.device
        scores_map = torch.sigmoid(cls_logits[0, 0])  # (H_bev, W_bev)

        # Filter by threshold
        mask = scores_map > score_threshold
        y_idxs, x_idxs = torch.where(mask)

        if len(x_idxs) == 0:
            return [], [], []

        # Get values
        scores = scores_map[mask].detach().cpu().numpy()
        reg_vals = reg_preds[0, :, y_idxs, x_idxs].detach().cpu().numpy()  # (8, M)
        y_idxs = y_idxs.cpu().numpy()
        x_idxs = x_idxs.cpu().numpy()

        decoded_boxes = []
        for i in range(len(x_idxs)):
            col = x_idxs[i]
            row = y_idxs[i]

            # Voxel center in world coordinates
            cx = col * self.vx + self.x_min + self.vx / 2.0
            cy = row * self.vy + self.y_min + self.vy / 2.0
            cz = (self.z_min + self.z_max) / 2.0

            dx, dy, dz, dl, dw, dh, sin_yaw, cos_yaw = reg_vals[:, i]

            x = cx + dx
            y = cy + dy
            z = cz + dz

            # Avoid huge values by bounding dimensions
            l = self.l_mean * np.exp(np.clip(dl, -2.0, 2.0))
            w = self.w_mean * np.exp(np.clip(dw, -2.0, 2.0))
            h = self.h_mean * np.exp(np.clip(dh, -2.0, 2.0))

            yaw = np.arctan2(sin_yaw, cos_yaw)

            decoded_boxes.append([x, y, z, l, w, h, yaw])

        decoded_boxes = np.array(decoded_boxes)

        # Run NMS
        keep_idxs = rotated_nms(
            decoded_boxes, scores, iou_threshold=nms_threshold
        )

        if len(keep_idxs) == 0:
            return [], [], []

        return (
            decoded_boxes[keep_idxs].tolist(),
            scores[keep_idxs].tolist(),
            [0] * len(keep_idxs),
        )

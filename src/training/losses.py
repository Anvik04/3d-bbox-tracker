import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraLiDARLoss(nn.Module):
    """
    Combined classification (Focal Loss) and 3D bounding box regression (Smooth L1) loss.
    """

    def __init__(
        self,
        x_range=(0.0, 48.0),
        y_range=(-16.0, 16.0),
        z_range=(-3.0, 1.0),
        voxel_size=(0.25, 0.25),
        l_mean=4.0,
        w_mean=1.6,
        h_mean=1.5,
        alpha=0.25,
        gamma=2.0,
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

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_logits, reg_preds, gt_boxes_3d):
        """
        Args:
            cls_logits: (1, 1, H, W) PyTorch tensor
            reg_preds: (1, 8, H, W) PyTorch tensor
            gt_boxes_3d: (M, 7) or (1, M, 7) PyTorch tensor
        Returns:
            total_loss: torch.Tensor
            cls_loss: torch.Tensor
            reg_loss: torch.Tensor
        """
        device = cls_logits.device
        H, W = cls_logits.shape[2:]

        # Create targets
        cls_target = torch.zeros((H, W), device=device)
        reg_target = torch.zeros((8, H, W), device=device)
        reg_mask = torch.zeros((H, W), device=device, dtype=torch.bool)

        if gt_boxes_3d.ndim == 3:
            if gt_boxes_3d.shape[0] > 0:
                gt_boxes_3d = gt_boxes_3d[0]
            else:
                gt_boxes_3d = torch.zeros((0, 7), device=device)

        for box in gt_boxes_3d:
            tx, ty, tz, tl, tw, th, tyaw = box

            # Check bounds
            if not (self.x_min <= tx < self.x_max and self.y_min <= ty < self.y_max):
                continue

            col = int((tx - self.x_min) / self.vx)
            row = int((ty - self.y_min) / self.vy)

            col = max(0, min(W - 1, col))
            row = max(0, min(H - 1, row))

            cls_target[row, col] = 1.0
            reg_mask[row, col] = True

            # Voxel center in world coordinates
            cx = col * self.vx + self.x_min + self.vx / 2.0
            cy = row * self.vy + self.y_min + self.vy / 2.0
            cz = (self.z_min + self.z_max) / 2.0

            dx = tx - cx
            dy = ty - cy
            dz = tz - cz
            dl = torch.log(tl / self.l_mean)
            dw = torch.log(tw / self.w_mean)
            dh = torch.log(th / self.h_mean)
            sin_yaw = torch.sin(tyaw)
            cos_yaw = torch.cos(tyaw)

            reg_target[:, row, col] = torch.stack(
                [dx, dy, dz, dl, dw, dh, sin_yaw, cos_yaw]
            )

        # Classification loss: Focal Loss
        pred_probs = torch.sigmoid(cls_logits[0, 0])
        eps = 1e-8
        focal_loss = -self.alpha * (
            (1 - pred_probs) ** self.gamma
        ) * cls_target * torch.log(pred_probs + eps) - (1 - self.alpha) * (
            pred_probs**self.gamma
        ) * (
            1 - cls_target
        ) * torch.log(
            1 - pred_probs + eps
        )
        cls_loss = focal_loss.mean()

        # Regression loss: Smooth L1 Loss (only on cell locations containing center points)
        if reg_mask.any():
            pred_pos = reg_preds[0, :, reg_mask].t()  # (num_pos, 8)
            target_pos = reg_target[:, reg_mask].t()  # (num_pos, 8)
            reg_loss = F.smooth_l1_loss(pred_pos, target_pos, reduction="mean")
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = cls_loss + 2.0 * reg_loss
        return total_loss, cls_loss, reg_loss

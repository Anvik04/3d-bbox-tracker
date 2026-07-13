import torch.nn as nn


class DetectionHead(nn.Module):
    """
    3D object detection head that regresses 3D bounding boxes and classifies objects from BEV.
    """

    def __init__(self, in_channels=96, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extraction block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # Classification branch (objectness/score)
        self.cls_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Regression branch: dx, dy, dz, dl, dw, dh, sin(yaw), cos(yaw) -> 8 channels
        self.reg_head = nn.Conv2d(64, 8, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: BEV feature map of shape (B, C, H_bev, W_bev)
        Returns:
            cls_logits: (B, num_classes, H_bev, W_bev)
            reg_preds: (B, 8, H_bev, W_bev)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        cls_logits = self.cls_head(x)
        reg_preds = self.reg_head(x)

        return cls_logits, reg_preds

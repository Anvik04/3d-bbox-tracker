import torch
import torch.nn as nn


class ImageBackbone(nn.Module):
    """
    Lightweight CNN image backbone that extracts feature maps from camera images.
    """

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # Simple conv network downsampling by 8 using stride=2 convs
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        # Final refinement block (keeps resolution same)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H_img, W_img)
        Returns:
            features: (B, out_channels, H_img//8, W_img//8)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedFusion(nn.Module):
    """
    Fuses camera image feature maps into LiDAR BEV feature maps using calibration.
    """

    def __init__(
        self,
        x_range=(0.0, 48.0),
        y_range=(-16.0, 16.0),
        z_range=(-3.0, 1.0),
        voxel_size=(0.25, 0.25),
    ):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.vx, self.vy = voxel_size

        self.nx = int(round((self.x_max - self.x_min) / self.vx))
        self.ny = int(round((self.y_max - self.y_min) / self.vy))

        # Precompute the 3D center points for all grid cells in BEV
        # xs: (nx,), ys: (ny,)
        # We will dynamically create the grid in forward pass to handle device matching,
        # but let's define a helper for it.

    def get_bev_centers(self, device):
        xs = torch.linspace(
            self.x_min + self.vx / 2.0,
            self.x_max - self.vx / 2.0,
            self.nx,
            device=device,
        )
        ys = torch.linspace(
            self.y_min + self.vy / 2.0,
            self.y_max - self.vy / 2.0,
            self.ny,
            device=device,
        )
        # grid_y, grid_x: (ny, nx)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Assume objects are mostly on the ground plane, z center is average of z range
        grid_z = torch.ones_like(grid_x) * ((self.z_min + self.z_max) / 2.0)
        # pts_lidar: (ny, nx, 3) -> flattened to (ny * nx, 3)
        pts_lidar = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return pts_lidar.view(-1, 3)

    def forward(self, lidar_bev, img_features, calib, img_shape):
        """
        Args:
            lidar_bev: (1, C_lidar, H_bev, W_bev)
            img_features: (1, C_img, H_feat, W_feat)
            calib: Calibration object for the current frame
            img_shape: Tuple of (H_img, W_img)
        Returns:
            fused_bev: (1, C_lidar + C_img, H_bev, W_bev)
        """
        device = lidar_bev.device
        C_img, H_feat, W_feat = img_features.shape[1:]
        H_img, W_img = img_shape

        # 1. Get 3D grid centers in LiDAR coordinates (ny * nx, 3)
        pts_lidar = self.get_bev_centers(device)
        num_pts = pts_lidar.shape[0]

        # 2. Extract calibration matrices and move to device
        P2 = torch.tensor(calib.P2, dtype=torch.float32, device=device)  # (3, 4)
        R0 = torch.tensor(calib.R0_rect, dtype=torch.float32, device=device)  # (4, 4)
        Tr = torch.tensor(calib.Tr_velo_to_cam, dtype=torch.float32, device=device)  # (4, 4)

        # 3. Project points: pts_lidar -> pts_cam
        pts_hom = torch.cat([pts_lidar, torch.ones((num_pts, 1), device=device)], dim=1)  # (M, 4)
        T_velo_to_cam_rect = R0 @ Tr  # (4, 4)
        pts_cam = pts_hom @ T_velo_to_cam_rect.t()  # (M, 4)

        # 4. Project points: pts_cam -> pts_img
        pts_img_hom = pts_cam @ P2.t()  # (M, 3)
        depth = pts_img_hom[:, 2]

        depth_safe = torch.clamp(depth, min=1e-6)
        u = pts_img_hom[:, 0] / depth_safe
        v = pts_img_hom[:, 1] / depth_safe

        # 5. Normalize pixel coordinates to [-1, 1] for grid_sample
        # grid_sample expects coordinates normalized: x in [-1, 1] (width), y in [-1, 1] (height)
        u_norm = 2.0 * u / (W_img - 1) - 1.0
        v_norm = 2.0 * v / (H_img - 1) - 1.0

        # Zero out features for points that are out of bounds or behind the camera
        mask_behind = depth < 0.1
        u_norm[mask_behind] = -2.0
        v_norm[mask_behind] = -2.0

        # grid shape for grid_sample: (batch_size, H_grid, W_grid, 2)
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, self.ny, self.nx, 2)

        # 6. Sample features from the image feature map
        # img_features: (1, C_img, H_feat, W_feat)
        # grid: (1, ny, nx, 2)
        # Returns: (1, C_img, ny, nx)
        sampled_img_features = F.grid_sample(
            img_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # 7. Concatenate image and LiDAR features
        fused_bev = torch.cat([lidar_bev, sampled_img_features], dim=1)

        return fused_bev

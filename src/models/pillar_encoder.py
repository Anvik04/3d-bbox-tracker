import torch
import torch.nn as nn


class PillarEncoder(nn.Module):
    """
    PointPillars-style encoder that converts point clouds into BEV pseudo-images.
    """

    def __init__(
        self,
        x_range=(0.0, 48.0),
        y_range=(-16.0, 16.0),
        z_range=(-3.0, 1.0),
        voxel_size=(0.25, 0.25),
        max_points_per_pillar=20,
        out_channels=64,
    ):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.vx, self.vy = voxel_size
        self.max_pts = max_points_per_pillar
        self.out_channels = out_channels

        self.nx = int(round((self.x_max - self.x_min) / self.vx))
        self.ny = int(round((self.y_max - self.y_min) / self.vy))

        # PointNet-style layer: 9 input features (x, y, z, i, x-xc, y-yc, z-zc, x-xp, y-yp)
        self.linear = nn.Linear(9, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, points):
        """
        Args:
            points: (N, 4) tensor [x, y, z, intensity] on CPU/GPU
        Returns:
            pseudo_image: (1, C, H, W) tensor where H = ny, W = nx
        """
        device = points.device
        if len(points) == 0:
            return torch.zeros((1, self.out_channels, self.ny, self.nx), device=device)

        # 1. Filter points outside range
        mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] < self.z_max)
        )
        pts = points[mask]
        if len(pts) == 0:
            return torch.zeros((1, self.out_channels, self.ny, self.nx), device=device)

        # 2. Compute voxel coordinates
        coords_x = torch.clamp(
            ((pts[:, 0] - self.x_min) / self.vx).long(), 0, self.nx - 1
        )
        coords_y = torch.clamp(
            ((pts[:, 1] - self.y_min) / self.vy).long(), 0, self.ny - 1
        )
        pillar_idx = coords_y * self.nx + coords_x

        # 3. Vectorized grouping of points to pillars
        unique_pillars, inverse_indices = torch.unique(pillar_idx, return_inverse=True)

        # Sort points by their pillar index to group them together
        sorted_inv, perm = torch.sort(inverse_indices)
        pts_sorted = pts[perm]
        coords_x_sorted = coords_x[perm]
        coords_y_sorted = coords_y[perm]

        # Calculate local indices (0 to max_pts-1) for each point in its pillar
        ranks = torch.arange(len(pts), device=device)
        # Find start indices of each unique pillar group
        change_mask = torch.cat(
            [torch.tensor([True], device=device), sorted_inv[1:] != sorted_inv[:-1]]
        )
        group_starts = ranks[change_mask]
        group_starts_expanded = torch.gather(group_starts, 0, sorted_inv)
        local_idx = ranks - group_starts_expanded

        # Filter out points exceeding max_points_per_pillar
        keep_mask = local_idx < self.max_pts
        pts_filtered = pts_sorted[keep_mask]
        sorted_inv_filtered = sorted_inv[keep_mask]
        local_idx_filtered = local_idx[keep_mask]
        coords_x_filtered = coords_x_sorted[keep_mask]
        coords_y_filtered = coords_y_sorted[keep_mask]

        # 4. Compute cluster features (x-xc, y-yc, z-zc)
        # Sum coordinate values for each unique pillar
        counts = torch.zeros(len(unique_pillars), device=device)
        counts.scatter_add_(
            0,
            sorted_inv_filtered,
            torch.ones_like(sorted_inv_filtered, dtype=torch.float32),
        )
        counts = torch.clamp(counts, min=1.0)  # avoid div by zero

        sums = torch.zeros((len(unique_pillars), 3), device=device)
        sums.scatter_add_(
            0, sorted_inv_filtered.unsqueeze(1).repeat(1, 3), pts_filtered[:, :3]
        )
        means = sums / counts.unsqueeze(1)

        # 5. Compute center features (x-xp, y-yp)
        xp = coords_x_filtered.float() * self.vx + self.x_min + self.vx / 2.0
        yp = coords_y_filtered.float() * self.vy + self.y_min + self.vy / 2.0

        # Construct final 9-dimensional features
        f_pts = pts_filtered  # [x, y, z, intensity]
        f_cluster = pts_filtered[:, :3] - means[sorted_inv_filtered]
        f_center = torch.stack(
            [pts_filtered[:, 0] - xp, pts_filtered[:, 1] - yp], dim=1
        )

        features = torch.cat([f_pts, f_cluster, f_center], dim=1)

        # 6. Scatter features into dense (num_unique, max_pts, 9)
        pillar_features = torch.zeros(
            (len(unique_pillars), self.max_pts, 9), device=device
        )
        pillar_features[sorted_inv_filtered, local_idx_filtered] = features

        # 7. Apply PointNet-style network
        # Reshape to (num_unique * max_pts, 9) for linear layer
        x = self.linear(pillar_features.view(-1, 9))  # (M * N, C)
        # Reshape to (M, C, N) for BN
        x = x.view(-1, self.max_pts, self.out_channels).transpose(1, 2)
        x = self.bn(x)
        x = self.relu(x)

        # Max pool along point dimension N
        x, _ = torch.max(x, dim=2)  # (num_unique, C)

        # 8. Scatter back to 2D pseudo-image
        grid_x = unique_pillars % self.nx
        grid_y = unique_pillars // self.nx

        pseudo_image = torch.zeros((self.out_channels, self.ny, self.nx), device=device)
        # Scatter: transpose x to (C, num_unique)
        pseudo_image[:, grid_y, grid_x] = x.t()

        # Add batch dimension: (1, C, H, W)
        return pseudo_image.unsqueeze(0)

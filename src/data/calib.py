import numpy as np


class Calibration:
    """
    Handles KITTI calibration matrices and projection math.
    """

    def __init__(self, calib_file_path):
        """
        Loads calibration matrices from a text file.
        KITTI calibration contains matrices P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo.
        """
        calib = {}
        with open(calib_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                val = np.fromstring(val, sep=" ")
                calib[key] = val

        # Camera 2 (left color camera) projection matrix (3x4)
        self.P2 = calib["P2"].reshape(3, 4)

        # Rectification matrix (3x3), augmented to 4x4
        R0 = calib["R0_rect"].reshape(3, 3)
        self.R0_rect = np.eye(4)
        self.R0_rect[:3, :3] = R0

        # LiDAR to Camera translation/rotation matrix (3x4), augmented to 4x4
        Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
        self.Tr_velo_to_cam = np.eye(4)
        self.Tr_velo_to_cam[:3, :4] = Tr

    def lidar_to_cam(self, pts_lidar):
        """
        Projects points from LiDAR coordinate system to Camera rectified coordinate system.
        Args:
            pts_lidar: (N, 3) numpy array of LiDAR coordinates [x, y, z]
        Returns:
            pts_cam: (N, 3) numpy array of rectified Camera coordinates [x, y, z]
        """
        n = pts_lidar.shape[0]
        pts_lidar_hom = np.hstack((pts_lidar, np.ones((n, 1))))
        # T_cam = R0_rect * Tr_velo_to_cam * T_lidar
        pts_cam_hom = pts_lidar_hom @ (self.R0_rect @ self.Tr_velo_to_cam).T
        return pts_cam_hom[:, :3]

    def cam_to_img(self, pts_cam):
        """
        Projects points from Camera rectified coordinate system to 2D image coordinates.
        Args:
            pts_cam: (N, 3) numpy array of Camera coordinates
        Returns:
            pts_img: (N, 2) numpy array of pixel coordinates [u, v]
            depths: (N,) numpy array of depths (z-coordinate in camera space)
        """
        n = pts_cam.shape[0]
        pts_cam_hom = np.hstack((pts_cam, np.ones((n, 1))))
        pts_img_hom = pts_cam_hom @ self.P2.T

        # Avoid division by zero
        depths = pts_img_hom[:, 2]
        depths = np.where(depths == 0, 1e-6, depths)

        u = pts_img_hom[:, 0] / depths
        v = pts_img_hom[:, 1] / depths

        return np.column_stack((u, v)), depths

    def lidar_to_img(self, pts_lidar):
        """
        Projects LiDAR points directly to 2D image coordinates.
        Args:
            pts_lidar: (N, 3) numpy array of LiDAR coordinates
        Returns:
            pts_img: (N, 2) pixel coordinates
            depths: (N,) depths in camera space
        """
        pts_cam = self.lidar_to_cam(pts_lidar)
        return self.cam_to_img(pts_cam)

    def cam_to_lidar(self, pts_cam):
        """
        Projects camera rectified coordinates back to LiDAR coordinates.
        """
        n = pts_cam.shape[0]
        pts_cam_hom = np.hstack((pts_cam, np.ones((n, 1))))
        R0_Tr_inv = np.linalg.inv(self.R0_rect @ self.Tr_velo_to_cam)
        pts_lidar_hom = pts_cam_hom @ R0_Tr_inv.T
        return pts_lidar_hom[:, :3]

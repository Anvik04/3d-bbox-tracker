import numpy as np


class Kalman3D:
    """
    3D constant-velocity Kalman Filter for tracking objects in 3D space.
    State: [x, y, z, yaw, vx, vy, vz]^T
    Measurement: [x, y, z, yaw]^T
    """

    def __init__(self, init_pos, init_yaw, dt=0.1, R_scale=1.0):
        self.dt = dt

        # State vector: [x, y, z, yaw, vx, vy, vz]
        self.x = np.zeros((7, 1))
        self.x[:3, 0] = init_pos
        self.x[3, 0] = init_yaw

        # State transition matrix F
        self.F = np.eye(7)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        # Measurement matrix H
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # State covariance P
        self.P = np.eye(7) * 1.0
        # High uncertainty for initial velocities
        self.P[4:, 4:] *= 10.0

        # Process noise covariance Q
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.1  # higher process noise for velocity

        # Measurement noise covariance R (scaled up for noisy monocular boxes)
        self.R = np.eye(4) * 0.1 * R_scale
        self.R[3, 3] = 0.2 * R_scale  # higher measurement noise for yaw

    def predict(self):
        """
        Predicts the next state.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # Normalize predicted yaw
        self.x[3, 0] = (self.x[3, 0] + np.pi) % (2 * np.pi) - np.pi
        return self.x

    def update(self, meas_pos, meas_yaw):
        """
        Updates the state with a new measurement.
        """
        z = np.zeros((4, 1))
        z[:3, 0] = meas_pos
        z[3, 0] = meas_yaw

        y = z - np.dot(self.H, self.x)

        # Normalize yaw residual
        y[3, 0] = (y[3, 0] + np.pi) % (2 * np.pi) - np.pi

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(7) - np.dot(K, self.H), self.P)

        # Normalize updated yaw
        self.x[3, 0] = (self.x[3, 0] + np.pi) % (2 * np.pi) - np.pi
        return self.x

    def get_state(self):
        """
        Returns (pos, yaw, vel).
        """
        pos = self.x[:3, 0]
        yaw = self.x[3, 0]
        vel = self.x[4:, 0]
        return pos, yaw, vel

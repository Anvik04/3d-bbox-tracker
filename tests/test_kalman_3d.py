import numpy as np

from src.tracking.kalman_3d import Kalman3D


def test_kalman_filter():
    dt = 0.1
    kf = Kalman3D(init_pos=[10.0, -2.0, 1.0], init_yaw=0.5, dt=dt)

    # State check
    pos, yaw, vel = kf.get_state()
    np.testing.assert_allclose(pos, [10.0, -2.0, 1.0])
    assert yaw == 0.5
    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0])

    # Predict
    kf.predict()
    pos, yaw, vel = kf.get_state()
    # Velocity is 0, so position should not change much (except process noise influence on state is zero mean)
    np.testing.assert_allclose(pos, [10.0, -2.0, 1.0])

    # Update with some movement
    # Move forward along X by 1.0m (implying a velocity of ~10m/s if dt=0.1)
    kf.update(meas_pos=[11.0, -2.0, 1.0], meas_yaw=0.5)
    pos, yaw, vel = kf.get_state()

    # The filter should adjust position towards the measurement
    assert pos[0] > 10.0
    # Velocity along X should now be positive (estimated from transition)
    assert vel[0] > 0.0

    # Test yaw normalization round-trip in residual
    # If yaw rotates across the boundary, e.g. from pi-0.1 to -pi+0.1, difference is small
    kf_yaw = Kalman3D(init_pos=[0.0, 0.0, 0.0], init_yaw=np.pi - 0.05, dt=dt)
    # Update with measurement just past pi (which wraps to -pi + 0.05)
    kf_yaw.update(meas_pos=[0.0, 0.0, 0.0], meas_yaw=-np.pi + 0.05)
    _, yaw_val, _ = kf_yaw.get_state()
    # The new yaw should still be close to the boundary, not average of positive/negative far apart
    assert np.abs(yaw_val) > 3.0

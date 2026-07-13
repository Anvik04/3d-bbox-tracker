import os
import numpy as np
from PIL import Image, ImageDraw

# Realistic KITTI calibration text
CALIB_TEXT = """P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.798500000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485057000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199933000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729940000000e-03
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999128000000e-01 1.009329000000e-02 -8.511930000000e-03 -1.011030000000e-02 9.999406000000e-01 -4.037100000000e-03 8.491680000000e-03 4.100110000000e-03 9.999554000000e-01 -8.086750000000e-01 -3.195590000000e-01 -7.997231000000e-01
"""


def generate_all_fixtures(output_dir, num_frames=10):
    """
    Generates dummy KITTI data in output_dir.
    """
    os.makedirs(os.path.join(output_dir, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "image_2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "calib"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "label_2"), exist_ok=True)

    # Let's save a calibration file template (same for all frames)
    calib_lines = CALIB_TEXT.strip().split("\n")
    calib_dict = {}
    for line in calib_lines:
        key, val = line.split(":", 1)
        calib_dict[key] = np.fromstring(val, sep=" ")

    P2 = calib_dict["P2"].reshape(3, 4)
    R0 = calib_dict["R0_rect"].reshape(3, 3)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = R0
    Tr = calib_dict["Tr_velo_to_cam"].reshape(3, 4)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = Tr

    # Define 2 moving cars
    # Car 1: moves along X in LiDAR (moving forward, y=-2.0, z=-1.0)
    # Car 2: moves along X in LiDAR (moving backward, y=3.0, z=-1.0)
    car1_dims = [1.5, 1.6, 4.0]  # h, w, l
    car2_dims = [1.6, 1.7, 4.2]  # h, w, l

    for frame_idx in range(num_frames):
        # Positions in LiDAR coordinates [x, y, z]
        c1_x = 10.0 + frame_idx * 1.5
        c1_y = -2.0
        c1_z = -1.0
        c1_yaw = 0.1  # slight angle

        c2_x = 25.0 - frame_idx * 0.8
        c2_y = 3.0
        c2_z = -1.0
        c2_yaw = np.pi - 0.1  # facing opposite direction

        cars = [
            {"pos": [c1_x, c1_y, c1_z], "dims": car1_dims, "yaw": c1_yaw, "id": 1},
            {"pos": [c2_x, c2_y, c2_z], "dims": car2_dims, "yaw": c2_yaw, "id": 2},
        ]

        # 1. Generate LiDAR point cloud
        points = []

        # Ground plane points (Z around -1.7)
        num_ground = 8000
        gx = np.random.uniform(2.0, 45.0, num_ground)
        gy = np.random.uniform(-10.0, 10.0, num_ground)
        gz = np.random.uniform(-1.75, -1.65, num_ground)
        gi = np.random.uniform(0.1, 0.5, num_ground)
        points.append(np.column_stack((gx, gy, gz, gi)))

        # Background points (some pillars/walls on sides)
        num_bg = 2000
        bx = np.random.uniform(2.0, 45.0, num_bg)
        by = np.random.choice([-8.0, 8.0], num_bg) + np.random.normal(0, 0.2, num_bg)
        bz = np.random.uniform(-1.7, 2.0, num_bg)
        bi = np.random.uniform(0.0, 0.3, num_bg)
        points.append(np.column_stack((bx, by, bz, bi)))

        # Generate points on the surfaces of each car
        for car in cars:
            h, w, l = car["dims"]
            cx, cy, cz = car["pos"]
            cyaw = car["yaw"]

            # Generate grid of points on the box surface
            num_car_pts = 400
            # Local box coordinates (length/x, width/y, height/z)
            lx = np.random.uniform(-l / 2, l / 2, num_car_pts)
            ly = np.random.uniform(-w / 2, w / 2, num_car_pts)
            lz = np.random.uniform(-h / 2, h / 2, num_car_pts)

            # Rotate to LiDAR coordinates
            rot_mat = np.array(
                [[np.cos(cyaw), -np.sin(cyaw), 0],
                 [np.sin(cyaw), np.cos(cyaw), 0],
                 [0, 0, 1]]
            )
            local_pts = np.column_stack((lx, ly, lz))
            rot_pts = local_pts @ rot_mat.T
            rot_pts[:, 0] += cx
            rot_pts[:, 1] += cy
            rot_pts[:, 2] += cz

            ci = np.random.uniform(0.6, 1.0, num_car_pts)
            points.append(np.column_stack((rot_pts, ci)))

        pc = np.vstack(points).astype(np.float32)

        # Save binary velodyne file
        bin_path = os.path.join(output_dir, "velodyne", f"{frame_idx:06d}.bin")
        pc.tofile(bin_path)

        # 2. Save Calibration file
        calib_path = os.path.join(output_dir, "calib", f"{frame_idx:06d}.txt")
        with open(calib_path, "w") as f:
            f.write(CALIB_TEXT)

        # 3. Create dummy image and draw simulated projections
        img_w, img_h = 1242, 375
        # Create a gradient image
        img = Image.new("RGB", (img_w, img_h), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Draw a horizontal line for the horizon
        draw.line([(0, img_h // 2), (img_w, img_h // 2)], fill=(60, 60, 70), width=2)

        labels = []
        for car in cars:
            # Let's project car center to camera coordinates
            h, w, l = car["dims"]
            cx, cy, cz = car["pos"]
            cyaw = car["yaw"]

            # Compute box corner coordinates in LiDAR frame to project them
            dx = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            dy = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            dz = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

            corners_lidar = np.column_stack((dx, dy, dz))
            # Rotate
            rot_mat = np.array(
                [[np.cos(cyaw), -np.sin(cyaw), 0],
                 [np.sin(cyaw), np.cos(cyaw), 0],
                 [0, 0, 1]]
            )
            corners_lidar = corners_lidar @ rot_mat.T
            corners_lidar[:, 0] += cx
            corners_lidar[:, 1] += cy
            corners_lidar[:, 2] += cz

            # Convert corners to Camera frame
            corners_hom = np.hstack((corners_lidar, np.ones((8, 1))))
            corners_cam = corners_hom @ (R0_rect @ Tr_velo_to_cam).T
            corners_cam = corners_cam[:, :3]

            # Project corners to image
            corners_img_hom = np.hstack((corners_cam, np.ones((8, 1)))) @ P2.T
            u = corners_img_hom[:, 0] / corners_img_hom[:, 2]
            v = corners_img_hom[:, 1] / corners_img_hom[:, 2]

            # 2D Bounding Box in image
            u_min, u_max = np.min(u), np.max(u)
            v_min, v_max = np.min(v), np.max(v)
            # Clip to image bounds
            u_min = max(0.0, min(img_w - 1, u_min))
            u_max = max(0.0, min(img_w - 1, u_max))
            v_min = max(0.0, min(img_h - 1, v_min))
            v_max = max(0.0, min(img_h - 1, v_max))

            # Draw a bounding box on the image to make it visually correlated
            draw.rectangle([u_min, v_min, u_max, v_max], outline=(0, 255, 100), width=2)
            draw.text((u_min + 2, v_min + 2), f"Car-{car['id']}", fill=(255, 255, 255))

            # Camera 3D center coordinate
            # Note: in KITTI label, the location x,y,z is the center of the bottom face of the 3D box, in camera coordinate system!
            # Let's verify: Yes, KITTI location (x, y, z) is the 3D box center of the bottom face.
            # So bottom face center in LiDAR: (cx, cy, cz - h/2)
            bottom_center_lidar = np.array([[cx, cy, cz - h/2, 1.0]])
            bottom_center_cam = (bottom_center_lidar @ (R0_rect @ Tr_velo_to_cam).T)[0, :3]

            # ry (yaw in camera coords):
            # KITTI ry is the rotation around the camera Y-axis.
            # Let's map LiDAR yaw to Camera ry.
            # LiDAR: x-forward, y-left. Camera: x-right, z-forward, y-down.
            # A heading of 0 in LiDAR (pointing along +x) maps to pointing along camera +z.
            # So, ry = -yaw - pi/2 (modulo 2pi). Let's approximate:
            # Let's compute standard projection.
            # If yaw=0: points along +x. Camera: points along +z. Rotation ry = 0.
            # Let's set ry = -cyaw. Let's make it standard.
            # KITTI formula: ry = -cyaw - pi/2 (approx, we will do a clean conversion or consistent representation).
            ry = -cyaw - np.pi / 2
            # normalize ry to [-pi, pi]
            ry = (ry + np.pi) % (2 * np.pi) - np.pi

            # Alpha: observation angle: alpha = ry - arctan2(x, z)
            alpha = ry - np.arctan2(bottom_center_cam[0], bottom_center_cam[2])
            alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

            labels.append(
                f"Car 0.00 0 {alpha:.2f} {u_min:.2f} {v_min:.2f} {u_max:.2f} {v_max:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} {bottom_center_cam[0]:.2f} {bottom_center_cam[1]:.2f} "
                f"{bottom_center_cam[2]:.2f} {ry:.2f}"
            )

        # Save image
        img_path = os.path.join(output_dir, "image_2", f"{frame_idx:06d}.png")
        img.save(img_path)

        # Save label txt
        label_path = os.path.join(output_dir, "label_2", f"{frame_idx:06d}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

    print(f"Generated {num_frames} frames of synthetic KITTI data in {output_dir}")

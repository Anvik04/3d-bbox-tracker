import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def get_3d_box_corners(box):
    """
    Computes the 8 corners of a 3D box [x, y, z, l, w, h, yaw] in LiDAR coordinates.
    Returns: (8, 3) numpy array
    """
    x, y, z, l, w, h, yaw = box
    # Corners in local box coordinates
    dx = np.array(
        [l / 2.0, l / 2.0, -l / 2.0, -l / 2.0, l / 2.0, l / 2.0, -l / 2.0, -l / 2.0]
    )
    dy = np.array(
        [w / 2.0, -w / 2.0, -w / 2.0, w / 2.0, w / 2.0, -w / 2.0, -w / 2.0, w / 2.0]
    )
    dz = np.array(
        [h / 2.0, h / 2.0, h / 2.0, h / 2.0, -h / 2.0, -h / 2.0, -h / 2.0, -h / 2.0]
    )

    corners = np.column_stack((dx, dy, dz))
    # Rotation matrix around Z-axis
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    corners = corners @ rot_mat.T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners


def draw_projected_boxes_2d(
    image, calib, boxes, track_ids=None, velocities=None, colors=None
):
    """
    Projects 3D bounding boxes to 2D image plane and draws wireframe boxes.
    image: numpy array HxWx3 (RGB) or PIL Image
    calib: Calibration object
    boxes: list of [x, y, z, l, w, h, yaw] in LiDAR coords
    """
    if isinstance(image, Image.Image):
        img = np.array(image).copy()
    else:
        img = image.copy()

    # Draw line connections for a 3D box
    # Top face: 0-1, 1-2, 2-3, 3-0
    # Bottom face: 4-5, 5-6, 6-7, 7-4
    # Vertical lines: 0-4, 1-5, 2-6, 3-7
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # top
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # bottom
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # pillars
    ]

    for idx, box in enumerate(boxes):
        corners = get_3d_box_corners(box)
        # Project corners to image
        pts_img, depths = calib.lidar_to_img(corners)

        # Skip if all corners are behind camera
        if np.all(depths < 0.1):
            continue

        pts_img = pts_img.astype(int)

        # Determine color
        color = (0, 255, 100)  # default green
        if colors is not None and idx < len(colors):
            color = colors[idx]
        elif track_ids is not None:
            # Color based on track ID
            tid = track_ids[idx]
            np.random.seed(tid)
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        # Draw box edges
        for start, end in connections:
            cv2.line(img, tuple(pts_img[start]), tuple(pts_img[end]), color, 2)

        # Annotate
        label = ""
        if track_ids is not None and idx < len(track_ids):
            label += f"ID: {track_ids[idx]} "
        if velocities is not None and idx < len(velocities):
            vel = velocities[idx]
            v_norm = np.linalg.norm(vel)
            label += f"V: {v_norm:.1f}m/s"

        if label:
            # Draw text near top-front corner (corner 0)
            u, v = pts_img[0]
            # Clip to image bounds
            u = max(10, min(img.shape[1] - 100, u))
            v = max(20, min(img.shape[0] - 10, v))
            cv2.putText(
                img,
                label,
                (u, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return Image.fromarray(img)


def draw_scene_3d(points, boxes, track_ids=None, output_path=None):
    """
    Renders 3D point cloud and bounding boxes using Open3D (or Matplotlib fallback in headless).
    """
    pts_np = (
        points[:, :3].cpu().numpy()
        if hasattr(points, "cpu")
        else np.array(points)[:, :3]
    )

    # Generate box line sets
    box_lines = []
    box_colors = []
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for idx, box in enumerate(boxes):
        corners = get_3d_box_corners(box)
        box_lines.append(corners)

        color = [0.0, 1.0, 0.0]  # green
        if track_ids is not None:
            tid = track_ids[idx]
            np.random.seed(tid)
            color = (np.random.randint(50, 255, 3) / 255.0).tolist()
        box_colors.append(color)

    # Headless fallback plotting using matplotlib
    if not OPEN3D_AVAILABLE or output_path is not None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Subsample point cloud to avoid huge plot size
        sub_idx = np.random.choice(len(pts_np), min(len(pts_np), 2000), replace=False)
        ax.scatter(
            pts_np[sub_idx, 0],
            pts_np[sub_idx, 1],
            pts_np[sub_idx, 2],
            s=1,
            c="gray",
            alpha=0.5,
        )

        for corners, color in zip(box_lines, box_colors):
            # Draw edges
            for start, end in connections:
                ax.plot(
                    [corners[start, 0], corners[end, 0]],
                    [corners[start, 1], corners[end, 1]],
                    [corners[start, 2], corners[end, 2]],
                    color=color,
                    linewidth=2,
                )

        ax.set_xlabel("X (Forward)")
        ax.set_ylabel("Y (Left)")
        ax.set_zlabel("Z (Up)")
        ax.set_xlim(0, 48)
        ax.set_ylim(-16, 16)
        ax.set_zlim(-3, 3)
        ax.set_title("3D Bounding Box Tracking")

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved 3D BEV plot to {output_path}")
        else:
            plt.show()
        return

    # Open3D Interactive Window
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)

        geometries = [pcd]

        for corners, color in zip(box_lines, box_colors):
            lines = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
            geometries.append(line_set)

        o3d.visualization.draw_geometries(geometries)
    except Exception as e:
        print(f"Open3D visualizer failed to initialize (GUI unavailable): {e}")
        # Run matplotlib fallback
        if output_path is None:
            output_path = "outputs/viz_3d_fallback.png"
        draw_scene_3d(points, boxes, track_ids, output_path=output_path)

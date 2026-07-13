import os
import sys
import torch
import numpy as np
from PIL import Image

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.kitti_dataset import KITTIDataset
from src.models.detector import CameraLiDARDetector
from src.tracking.tracker import AB3DMOTTracker
from src.viz.visualize import draw_projected_boxes_2d, draw_scene_3d, get_3d_box_corners


def convert_lidar_to_kitti_camera(box_lidar, calib):
    """
    Converts a 3D box in LiDAR coordinates [x, y, z, l, w, h, yaw] to
    KITTI camera coordinates format: [dims (h,w,l), loc (x,y,z), ry]
    """
    x, y, z, l, w, h, yaw = box_lidar

    # Bottom center in LiDAR
    bottom_center_lidar = np.array([[x, y, z - h / 2.0]])
    bottom_center_cam = calib.lidar_to_cam(bottom_center_lidar)[0]

    # Convert yaw from LiDAR to Camera ry
    ry = -yaw - np.pi / 2.0
    ry = (ry + np.pi) % (2 * np.pi) - np.pi

    # Alpha: observation angle
    alpha = ry - np.arctan2(bottom_center_cam[0], bottom_center_cam[2])
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

    return h, w, l, bottom_center_cam, ry, alpha


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data", "fixtures")
    checkpoint_path = os.path.join(repo_root, "checkpoints", "detector.pt")

    # Outputs
    predict_dir = os.path.join(repo_root, "outputs", "predict")
    viz_2d_dir = os.path.join(repo_root, "outputs", "viz", "image_2")
    viz_3d_dir = os.path.join(repo_root, "outputs", "viz", "3d")

    os.makedirs(predict_dir, exist_ok=True)
    os.makedirs(viz_2d_dir, exist_ok=True)
    os.makedirs(viz_3d_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = KITTIDataset(data_dir=data_dir)

    print("Initializing detector and tracker...")
    detector = CameraLiDARDetector()
    tracker = AB3DMOTTracker(max_age=3, min_hits=2, dt=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = detector.to(device)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        detector.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint found. Running with untrained model weights.")

    detector.eval()

    print("Running inference loop...")
    for idx, sample in enumerate(dataset):
        file_id = sample["file_id"]
        points = sample["points"].to(device)
        image = sample["image"].to(device)
        calib = sample["calib"]

        # Run forward pass
        with torch.no_grad():
            cls_logits, reg_preds = detector(points, image, calib)

            # We use a lower threshold for decoding during inference to ensure some detections
            # on our miniature synthetic fixtures.
            boxes, scores, classes = detector.decode_predictions(
                cls_logits,
                reg_preds,
                score_threshold=0.15,
                nms_threshold=0.1,
            )

        # Update tracker
        active_tracks = tracker.update(boxes, scores, classes)

        # Detections for visualization / output labels
        pred_labels = []
        viz_boxes = []
        track_ids = []
        velocities = []

        for track in active_tracks:
            tid = track["track_id"]
            box_lidar = track["bbox_3d"]
            vel = track["velocity"]
            score = track["score"]

            viz_boxes.append(box_lidar)
            track_ids.append(tid)
            velocities.append(vel)

            # Convert to KITTI camera format
            h, w, l, loc_cam, ry, alpha = convert_lidar_to_kitti_camera(
                box_lidar, calib
            )

            # Get 2D projected bounding box for label (left, top, right, bottom)
            corners_lidar = get_3d_box_corners(box_lidar)
            corners_img, _ = calib.lidar_to_img(corners_lidar)
            u_min, u_max = np.min(corners_img[:, 0]), np.max(corners_img[:, 0])
            v_min, v_max = np.min(corners_img[:, 1]), np.max(corners_img[:, 1])

            # Clip bounding box
            img_w, img_h = 1242, 375
            u_min = max(0.0, min(img_w - 1, u_min))
            u_max = max(0.0, min(img_w - 1, u_max))
            v_min = max(0.0, min(img_h - 1, v_min))
            v_max = max(0.0, min(img_h - 1, v_max))

            pred_labels.append(
                f"Car 0.00 0 {alpha:.2f} {u_min:.2f} {v_min:.2f} {u_max:.2f} {v_max:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} {loc_cam[0]:.2f} {loc_cam[1]:.2f} {loc_cam[2]:.2f} "
                f"{ry:.2f} {score:.4f}"
            )

        # Write predictions file
        pred_file_path = os.path.join(predict_dir, f"{file_id}.txt")
        with open(pred_file_path, "w") as f:
            f.write("\n".join(pred_labels))

        # Save 2D Visualization
        # Convert image tensor back to PIL Image
        img_pil = Image.fromarray(
            (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        )
        img_viz = draw_projected_boxes_2d(
            img_pil, calib, viz_boxes, track_ids, velocities
        )
        viz_2d_path = os.path.join(viz_2d_dir, f"{file_id}.png")
        img_viz.save(viz_2d_path)

        # Save 3D BEV Plot Fallback (Matplotlib)
        viz_3d_path = os.path.join(viz_3d_dir, f"{file_id}.png")
        draw_scene_3d(points, viz_boxes, track_ids=track_ids, output_path=viz_3d_path)

        print(
            f"Processed frame {file_id}: found {len(active_tracks)} active tracks"
        )

    print("Inference completed!")
    print(f"KITTI label predictions dumped in {predict_dir}")
    print(f"2D visual overlays saved in {viz_2d_dir}")
    print(f"3D bird's-eye-view plots saved in {viz_3d_dir}")


if __name__ == "__main__":
    main()

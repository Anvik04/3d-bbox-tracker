# Earlier motion-based prototype; superseded by scripts/run_mono_demo.py.
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

# Ensure project root is in Python path to import tracker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.tracker import AB3DMOTTracker


def main():
    # 1. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam successfully opened.")
    print("Press 'q' in the window to quit.")

    # 2. Initialize Pre-trained 2D Object Detector (SSDLite MobileNetV3)
    print("Loading lightweight SSDLite detector...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detector_model = ssdlite320_mobilenet_v3_large(weights=weights)
    detector_model.eval()

    COCO_CLASSES = weights.meta["categories"]
    print("SSDLite model loaded successfully.")

    # 3. Initialize AB3DMOT Tracker
    # max_age = 5: holds track for up to 5 missing frames to eliminate flickering
    # min_hits = 1: quick activation for responsive user feedback
    tracker = AB3DMOTTracker(max_age=5, min_hits=1, iou_threshold=0.01, dt=0.033)

    focal_length = 600.0

    # 3D Bounding Box connection indexes
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # top face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # bottom face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # pillars
    ]

    # Map COCO classes to standard physical 3D dimensions [length, width, height] in meters
    # Length is along X, Width is along Z (depth), Height is along Y
    CLASS_DIMENSIONS = {
        "person": [0.6, 0.6, 1.7],
        "car": [4.0, 1.7, 1.5],
        "truck": [6.0, 2.0, 2.5],
        "bus": [10.0, 2.5, 3.0],
        "dog": [0.8, 0.4, 0.6],
        "cat": [0.5, 0.3, 0.4],
        "sports ball": [0.25, 0.25, 0.25],
        "bicycle": [1.6, 0.5, 1.0],
        "motorcycle": [1.8, 0.6, 1.1],
        "remote": [0.2, 0.15, 0.1],  # Charger/remote size parameters
    }

    # Map COCO category names to custom display names (e.g. mapping "remote" to "charger")
    CLASS_RENAME_OVERRIDES = {
        "remote": "CHARGER",
        "cell phone": "PHONE",
        "sports ball": "BALL",
    }

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Mirror frame for intuitive user interaction
        frame = cv2.flip(frame, 1)
        h_img, w_img = frame.shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0

        # 4. Compute Grayscale Frame Difference (Motion Mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        frame_delta = cv2.absdiff(prev_gray, gray)
        _, motion_mask = cv2.threshold(frame_delta, 12, 255, cv2.THRESH_BINARY)
        prev_gray = gray

        # 5. Perform 2D Object Detection
        # Transform frame to tensor format expected by PyTorch
        img_tensor = T.functional.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            predictions = detector_model([img_tensor])[0]

        boxes2d = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        dets_3d = []
        det_scores = []
        det_classes = []

        # 6. Filter Detections by Motion and Score
        for i in range(len(boxes2d)):
            score = scores[i]
            if score < 0.45:
                continue

            xmin, ymin, xmax, ymax = boxes2d[i].astype(int)
            # Clip bounds
            xmin, xmax = np.clip([xmin, xmax], 0, w_img - 1)
            ymin, ymax = np.clip([ymin, ymax], 0, h_img - 1)

            # Crop the motion mask inside the object bounding box
            obj_motion_crop = motion_mask[ymin:ymax, xmin:xmax]
            if obj_motion_crop.size == 0:
                continue

            # Calculate density of moving pixels (percentage of active pixels)
            motion_density = np.mean(obj_motion_crop) / 255.0

            # Ignore static objects: must have at least 2.5% motion density inside the box
            if motion_density < 0.025:
                continue

            # Get class name
            class_id = labels[i]
            class_name = (
                COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
            )

            # Determine standard dimensions for this class
            dim = CLASS_DIMENSIONS.get(class_name, [0.8, 0.8, 0.8])  # default [l, w, h]
            l, w, h = dim

            # 7. Estimate 3D Coordinates
            # Z (depth) estimated using focal length and standard height
            h_box = ymax - ymin
            z_c = (focal_length * h) / max(h_box, 1)
            z_c = np.clip(z_c, 0.3, 5.0)

            # Compute X_c and Y_c in camera frame
            u_center = (xmin + xmax) / 2.0
            v_center = (ymin + ymax) / 2.0
            x_c = ((u_center - cx) * z_c) / focal_length
            y_c = ((v_center - cy) * z_c) / focal_length

            # Store detection in [x, y, z, l, w, h, yaw] format for tracker
            # (Assume yaw is aligned with camera heading = 0.0)
            dets_3d.append([x_c, y_c, z_c, l, w, h, 0.0])
            det_scores.append(score)
            det_classes.append(class_id)

        # 8. Update Kalman Tracker
        active_tracks = tracker.update(dets_3d, det_scores, det_classes)

        # 9. Render 3D Cuboids
        for track in active_tracks:
            tid = track["track_id"]
            box_3d = track["bbox_3d"]  # [x, y, z, l, w, h, yaw]
            class_id = track["class_id"]

            x_c, y_c, z_c, l, w, h, yaw = box_3d
            class_name = (
                COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "object"
            )
            class_name = CLASS_RENAME_OVERRIDES.get(class_name, class_name)
            class_name_upper = class_name.upper()

            # Compute 8 corners in camera space (X: right, Y: down, Z: forward)
            # Top face (Y = -h/2), Bottom face (Y = h/2)
            dx = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            dy = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
            dz = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

            corners_cam = np.column_stack((dx, dy, dz))

            # Apply yaw rotation around the Y-axis (vertical) in camera frame
            if abs(yaw) > 0.001:
                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)
                rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
                corners_cam = corners_cam @ rot_y.T

            # Translate by 3D center
            corners_cam[:, 0] += x_c
            corners_cam[:, 1] += y_c
            corners_cam[:, 2] += z_c

            # Project corners to image space
            pts_img = np.zeros((8, 2), dtype=int)
            for j in range(8):
                pt_z = max(corners_cam[j, 2], 0.1)
                pts_img[j, 0] = int((corners_cam[j, 0] * focal_length) / pt_z + cx)
                pts_img[j, 1] = int((corners_cam[j, 1] * focal_length) / pt_z + cy)

            # Draw 3D wireframe box (in Red)
            for start, end in connections:
                pt1 = (
                    np.clip(pts_img[start][0], 0, w_img - 1),
                    np.clip(pts_img[start][1], 0, h_img - 1),
                )
                pt2 = (
                    np.clip(pts_img[end][0], 0, w_img - 1),
                    np.clip(pts_img[end][1], 0, h_img - 1),
                )
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Draw contact point (green dot) at bottom center of the box
            bottom_z = max(z_c, 0.1)
            u_bottom = int((x_c * focal_length) / bottom_z + cx)
            v_bottom = int(((y_c + h / 2) * focal_length) / bottom_z + cy)
            u_bottom = np.clip(u_bottom, 0, w_img - 1)
            v_bottom = np.clip(v_bottom, 0, h_img - 1)
            cv2.circle(frame, (u_bottom, v_bottom), 6, (0, 255, 0), -1)

            # Overlay class name and tracking ID stable label
            label = f"{class_name_upper} #{tid} | {z_c:.1f}m"
            u_text = int(np.clip(pts_img[0][0], 10, w_img - 150))
            v_text = int(np.clip(pts_img[0][1] - 10, 20, h_img - 10))
            cv2.putText(
                frame,
                label,
                (u_text, v_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Show visualization window
        cv2.imshow("Real-Time 3D Object Detection & Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

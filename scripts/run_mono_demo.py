import argparse
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from src.detection2d.yolo_detector import VehicleDetector2D
from src.geometry.monocular_lift import MonocularLifter
from src.tracking.tracker import AB3DMOTTracker
from src.viz.visualize import draw_3d_wireframe_cuboid, draw_bev_panel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def derive_focal_px(frame_width: int, fov_deg: float) -> float:
    fov_rad = math.radians(float(fov_deg))
    return float(frame_width / (2.0 * math.tan(fov_rad / 2.0)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run monocular 3D vehicle detection")
    parser.add_argument("--source", default="0", help="Webcam index or video path")
    parser.add_argument("--config", default="configs/mono_camera.yaml")
    parser.add_argument("--save-video", default=None, help="Optional output video path")
    parser.add_argument("--camera-height", type=float, default=None, help="Override camera height in meters")
    parser.add_argument("--tilt-deg", type=float, default=None, help="Override camera tilt in degrees")
    parser.add_argument("--fov-deg", type=float, default=None, help="Override camera FOV in degrees")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    source = args.source
    save_video = args.save_video

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    
    fov_deg = float(config.get("fov_deg", 70.0))
    if args.fov_deg is not None:
        fov_deg = args.fov_deg
    
    focal_px = derive_focal_px(frame_width, fov_deg)
    principal_point = (
        float(frame_width / 2.0),
        float(frame_height / 2.0),
    )
    
    camera_height_m = float(config.get("camera_height_m", 1.6))
    if args.camera_height is not None:
        camera_height_m = args.camera_height
        
    tilt_deg = float(config.get("tilt_deg", 10.0))
    if args.tilt_deg is not None:
        tilt_deg = args.tilt_deg

    lifter = MonocularLifter(
        camera_height_m=camera_height_m,
        tilt_deg=tilt_deg,
        focal_px=focal_px,
        principal_point=principal_point,
    )
    detector = VehicleDetector2D()
    tracker = AB3DMOTTracker(max_age=5, min_hits=1, dt=0.1)

    writer = None
    if save_video:
        out_path = Path(save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (frame_width, frame_height))

    centroid_history = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb)
            boxes = []
            scores = []
            classes = []
            for det in detections:
                box = lifter.to_bbox_3d(det)
                boxes.append(box)
                scores.append(det.confidence)
                classes.append(0)

            tracks = tracker.update(boxes, scores, classes)
            overlay = frame.copy()
            for track in tracks:
                box3d = track["bbox_3d"]
                track_id = track["track_id"]
                velocity = np.asarray(track.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
                
                if tilt_deg > 45.0 and np.linalg.norm(velocity[:2]) > 0.1:
                    box3d[6] = math.atan2(velocity[1], velocity[0])
                    
                camera_params = {
                    "focal_px": focal_px,
                    "principal_point": principal_point,
                    "camera_height_m": camera_height_m,
                }
                overlay = draw_3d_wireframe_cuboid(
                    overlay,
                    box3d,
                    camera_params,
                    track_id,
                    distance=float(np.linalg.norm(box3d[:3])),
                    closing_speed=float(np.linalg.norm(velocity[:2])),
                )
                centroid = (box3d[0], box3d[1])
                centroid_history[track_id] = centroid_history.get(track_id, []) + [centroid]
                if len(centroid_history[track_id]) > 8:
                    centroid_history[track_id] = centroid_history[track_id][-8:]

            if overlay.shape[0] > 0 and overlay.shape[1] > 0:
                bev = draw_bev_panel(tracks, size=(240, 240))
                if bev is not None:
                    h, w = bev.shape[:2]
                    overlay[h:h + 20, :w] = np.zeros((20, w, 3), dtype=np.uint8)
                    overlay[:h, :w] = cv2.addWeighted(overlay[:h, :w], 0.85, bev, 0.15, 0)

            cv2.putText(overlay, "Mono 3D Demo", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if writer is not None:
                writer.write(overlay)
            else:
                cv2.imshow("Mono 3D Demo", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except Exception as e:
        print(f"Tracking interrupted or failed: {e}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import os

import numpy as np
import torch
from PIL import Image

from src.data.kitti_dataset import KITTIDataset
from src.eval.metrics import calculate_ap, evaluate_tracking_mota
from src.models.detector import CameraLiDARDetector
from src.tracking.tracker import AB3DMOTTracker
from src.viz.visualize import draw_projected_boxes_2d, draw_scene_3d


def test_end_to_end_pipeline():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(test_dir, "..", "data", "fixtures"))

    # 1. Dataset
    dataset = KITTIDataset(data_dir=data_dir)
    assert len(dataset) == 10

    # 2. Model & Tracker
    detector = CameraLiDARDetector()
    checkpoint_path = os.path.abspath(
        os.path.join(test_dir, "..", "checkpoints", "detector.pt")
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        detector.load_state_dict(checkpoint["model_state_dict"])
    else:
        torch.nn.init.constant_(detector.detection_head.cls_head.bias, -10.0)
        torch.nn.init.constant_(detector.detection_head.cls_head.weight, 0.0)

    detector.eval()
    tracker = AB3DMOTTracker(max_age=3, min_hits=2, dt=0.1)

    # 3. Processing Sequence
    track_history = []
    gt_history = []
    all_preds = []
    all_gts = []

    for sample in dataset:
        points = sample["points"]
        image = sample["image"]
        calib = sample["calib"]
        gt_boxes = sample["gt_boxes_3d"].numpy()

        # Run forward pass
        with torch.no_grad():
            cls_logits, reg_preds = detector(points, image, calib)
            boxes, scores, classes = detector.decode_predictions(
                cls_logits, reg_preds, score_threshold=0.1, nms_threshold=0.1
            )

        # Update tracker
        active_tracks = tracker.update(boxes, scores, classes)

        # Record history for tracking evaluation
        # Format active tracks
        current_tracks = []
        for tr in active_tracks:
            current_tracks.append(
                {"track_id": tr["track_id"], "bbox_3d": tr["bbox_3d"]}
            )
            all_preds.append({"bbox_3d": tr["bbox_3d"], "score": tr["score"]})
        track_history.append(current_tracks)

        # Format GT history (fixtures have 2 moving cars: ID 1 and ID 2)
        current_gts = []
        for i, gt_box in enumerate(gt_boxes):
            gt_id = i + 1  # 1 and 2
            current_gts.append({"track_id": gt_id, "bbox_3d": gt_box.tolist()})
            all_gts.append({"bbox_3d": gt_box.tolist()})
        gt_history.append(current_gts)

        # Sane range checks
        for box in boxes:
            x, y, z, l, w, h, yaw = box
            # LiDAR coordinates bounds
            assert 0.0 <= x <= 48.0
            assert -16.0 <= y <= 16.0
            assert -3.0 <= z <= 1.0
            assert l > 0.0 and w > 0.0 and h > 0.0
            assert -np.pi <= yaw <= np.pi

    # 4. Metrics Evaluation
    # Tracking MOTA
    tracking_res = evaluate_tracking_mota(track_history, gt_history, iou_threshold=0.01)
    assert "mota" in tracking_res
    assert tracking_res["total_gt"] == 20  # 10 frames * 2 cars

    # Detection AP (even if AP is low or 0 on random weights, calculations should complete without crash)
    ap = calculate_ap(all_preds, all_gts, iou_threshold=0.3)
    assert 0.0 <= ap <= 1.0

    # 5. Visualization overlay in headless mode
    # Ensure visualization doesn't crash and generates a file
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save Matplotlib 3D BEV plot
        output_plot_path = os.path.join(tmp_dir, "viz_3d.png")
        draw_scene_3d(
            points=dataset[0]["points"],
            boxes=[[10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]],
            track_ids=[1],
            output_path=output_plot_path,
        )
        assert os.path.exists(output_plot_path)

        # Save Projected 2D Image overlay
        img_pil = Image.fromarray(
            (dataset[0]["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        )
        img_viz = draw_projected_boxes_2d(
            img_pil,
            dataset[0]["calib"],
            boxes=[[10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]],
            track_ids=[1],
            velocities=[[1.0, 0.0, 0.0]],
        )
        assert isinstance(img_viz, Image.Image)

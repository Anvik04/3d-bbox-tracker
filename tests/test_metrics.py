import numpy as np

from src.eval.metrics import calculate_ap, evaluate_tracking_mota, iou_3d


def test_iou_3d():
    # Identical boxes
    box1 = [10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]
    assert np.isclose(iou_3d(box1, box1), 1.0)

    # Disjoint boxes
    box2 = [20.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]
    assert iou_3d(box1, box2) == 0.0

    # Partially overlapping boxes (e.g. half shifted in x)
    # Box1 x: [8, 12], y: [-2.8, -1.2], z: [0.25, 1.75]. Vol = 4 * 1.6 * 1.5 = 9.6
    # Box3 is shifted by 2m in X -> x: 12.0
    # Overlap in X: [10, 12] (width 2m).
    # Overlap in Y: [-2.8, -1.2] (width 1.6m).
    # Overlap in Z: [0.25, 1.75] (height 1.5m).
    # Inter volume: 2 * 1.6 * 1.5 = 4.8
    # Union volume = 9.6 + 9.6 - 4.8 = 14.4
    # IoU = 4.8 / 14.4 = 1/3 = 0.33333
    box3 = [12.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]
    assert np.isclose(iou_3d(box1, box3), 1.0 / 3.0)


def test_calculate_ap():
    gts = [{"bbox_3d": [10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]}]
    preds = [
        {"bbox_3d": [10.1, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0], "score": 0.95},
        {"bbox_3d": [20.0, 2.0, 1.0, 4.0, 1.6, 1.5, 0.0], "score": 0.8},
    ]

    ap = calculate_ap(preds, gts, iou_threshold=0.5)
    # The first prediction matches, the second is a false positive.
    # AP should be 1.0 because the single GT is detected by the highest scoring prediction.
    assert ap == 1.0


def test_mota():
    # 2 frames
    gt_history = [
        [{"track_id": 1, "bbox_3d": [10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]}],
        [{"track_id": 1, "bbox_3d": [11.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]}],
    ]
    # Track 1 has correct ID, Track 2 has wrong ID in second frame (ID switch)
    track_history = [
        [{"track_id": 1, "bbox_3d": [10.1, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]}],
        [{"track_id": 2, "bbox_3d": [11.1, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]}],
    ]

    res = evaluate_tracking_mota(track_history, gt_history, iou_threshold=0.3)
    # Total GT = 2, FP = 0, FN = 0, IDSW = 1 (switch from 1 to 2)
    # MOTA = 1 - (0 + 0 + 1)/2 = 0.5
    assert res["mota"] == 0.5
    assert res["idsw"] == 1
    assert res["fn"] == 0
    assert res["fp"] == 0

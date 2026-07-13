import numpy as np
from src.models.detector import box2d_to_polygon


def iou_3d(box1, box2):
    """
    Computes 3D IoU between two rotated 3D bounding boxes.
    Box format: [x, y, z, l, w, h, yaw]
    """
    # 1. Height overlap
    h1_min = box1[2] - box1[5] / 2.0
    h1_max = box1[2] + box1[5] / 2.0
    h2_min = box2[2] - box2[5] / 2.0
    h2_max = box2[2] + box2[5] / 2.0

    h_overlap = max(0.0, min(h1_max, h2_max) - max(h1_min, h2_min))
    if h_overlap <= 0.0:
        return 0.0

    # 2. BEV overlap (2D Rotated IoU)
    poly1 = box2d_to_polygon(box1[0], box1[1], box1[3], box1[4], box1[6])
    poly2 = box2d_to_polygon(box2[0], box2[1], box2[3], box2[4], box2[6])

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    try:
        bev_inter = poly1.intersection(poly2).area
    except Exception:
        return 0.0

    vol_inter = bev_inter * h_overlap
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]

    vol_union = vol1 + vol2 - vol_inter
    return vol_inter / max(vol_union, 1e-6)


def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculates Average Precision (AP) for a set of predictions and ground truths.
    predictions: list of dicts with 'bbox_3d' and 'score'
    ground_truths: list of dicts with 'bbox_3d'
    """
    if len(predictions) == 0:
        return 0.0
    if len(ground_truths) == 0:
        return 0.0

    # Sort predictions by score descending
    preds_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)
    num_preds = len(preds_sorted)
    num_gts = len(ground_truths)

    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)
    gt_matched = np.zeros(num_gts, dtype=bool)

    for p_idx, pred in enumerate(preds_sorted):
        best_iou = -1.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            iou = iou_3d(pred["bbox_3d"], gt["bbox_3d"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            if not gt_matched[best_gt_idx]:
                tp[p_idx] = 1.0
                gt_matched[best_gt_idx] = True
            else:
                fp[p_idx] = 1.0
        else:
            fp[p_idx] = 1.0

    # Cumulative sum
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / num_gts
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

    # 11-point interpolation or standard precision integration
    ap = 0.0
    # Append boundaries
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def evaluate_tracking_mota(track_history, gt_history, iou_threshold=0.1):
    """
    Computes MOTA-style metrics for a sequence.
    track_history: list of lists, each containing active tracks in frame t:
                   [{'track_id': id, 'bbox_3d': bbox}, ...]
    gt_history: list of lists, each containing GT targets in frame t:
                [{'track_id': id, 'bbox_3d': bbox}, ...]
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    # Keep track of previous matches to identify ID switches
    prev_match_map = {}  # gt_id -> track_id

    for t in range(len(gt_history)):
        gts = gt_history[t]
        tracks = track_history[t]

        total_gt += len(gts)

        # Cost matrix: shape (num_gts, num_tracks)
        if len(gts) > 0 and len(tracks) > 0:
            iou_matrix = np.zeros((len(gts), len(tracks)))
            for g_idx, gt in enumerate(gts):
                for t_idx, tr in enumerate(tracks):
                    iou_matrix[g_idx, t_idx] = iou_3d(gt["bbox_3d"], tr["bbox_3d"])

            # Greedy match or optimal match
            matched_gt_indices = []
            matched_tr_indices = []
            temp_iou = iou_matrix.copy()

            while np.max(temp_iou) >= iou_threshold:
                g_max, t_max = np.unravel_index(np.argmax(temp_iou), temp_iou.shape)
                if temp_iou[g_max, t_max] < iou_threshold:
                    break
                matched_gt_indices.append(g_max)
                matched_tr_indices.append(t_max)
                # Suppress matched row and col
                temp_iou[g_max, :] = -1.0
                temp_iou[:, t_max] = -1.0

            # Count TP, FP, FN, IDSW
            matched_gts = set(matched_gt_indices)
            matched_trs = set(matched_tr_indices)

            fp = len(tracks) - len(matched_trs)
            fn = len(gts) - len(matched_gts)

            total_fp += fp
            total_fn += fn

            # ID Switches
            for g_idx, t_idx in zip(matched_gt_indices, matched_tr_indices):
                gt_id = gts[g_idx]["track_id"]
                tr_id = tracks[t_idx]["track_id"]

                if gt_id in prev_match_map:
                    if prev_match_map[gt_id] != tr_id:
                        total_idsw += 1
                prev_match_map[gt_id] = tr_id
        else:
            total_fn += len(gts)
            total_fp += len(tracks)

    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1.0)

    return {
        "mota": mota,
        "fp": total_fp,
        "fn": total_fn,
        "idsw": total_idsw,
        "total_gt": total_gt,
    }

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman_3d import Kalman3D
from src.models.detector import bev_iou_numpy


class Track:
    """
    Represents an individual tracked object.
    """

    def __init__(self, track_id, bbox_3d, class_id, score, dt=0.1):
        self.track_id = track_id
        self.class_id = class_id
        self.score = score

        # bbox_3d: [x, y, z, l, w, h, yaw]
        pos = bbox_3d[:3]
        yaw = bbox_3d[6]
        self.dims = bbox_3d[3:6]  # [l, w, h]

        self.kf = Kalman3D(pos, yaw, dt=dt)

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"  # 'tentative', 'confirmed'

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_3d, score):
        pos = bbox_3d[:3]
        yaw = bbox_3d[6]
        self.dims = bbox_3d[3:6]
        self.score = score

        self.kf.update(pos, yaw)
        self.hits += 1
        self.time_since_update = 0

    def get_state(self):
        """
        Returns bbox_3d: [x, y, z, l, w, h, yaw] and velocity: [vx, vy, vz].
        """
        pos, yaw, vel = self.kf.get_state()
        bbox_3d = [pos[0], pos[1], pos[2], self.dims[0], self.dims[1], self.dims[2], yaw]
        return bbox_3d, vel


class AB3DMOTTracker:
    """
    3D Multi-Object Tracker based on 3D Kalman Filter and Hungarian Association.
    """

    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.1, dt=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt

        self.tracks = []
        self.next_track_id = 1

    def update(self, detections, scores, classes):
        """
        Updates tracks with new detections.
        Args:
            detections: list of [x, y, z, l, w, h, yaw]
            scores: list of float scores
            classes: list of int class IDs
        Returns:
            active_tracks: list of dicts containing track_id, bbox_3d, velocity, class_id, score
        """
        # 1. Predict state of current tracks
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # 2. Match detections to tracks
        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.zeros((num_tracks, num_dets))

            for t_idx, track in enumerate(self.tracks):
                track_bbox, _ = track.get_state()
                for d_idx, det in enumerate(detections):
                    # Cost is based on BEV IoU
                    iou = bev_iou_numpy(np.array(track_bbox), np.array(det))
                    cost = 1.0 - iou

                    # Fallback to center distance if IoU is near zero
                    if iou < 0.01:
                        dist = np.linalg.norm(np.array(track_bbox[:2]) - np.array(det[:2]))
                        # Limit distance to reasonable range (e.g. 4.0 meters)
                        cost = 1.0 + (dist / 4.0)

                    cost_matrix[t_idx, d_idx] = cost

            # Solve Hungarian matching
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r_idx, c_idx in zip(row_indices, col_indices):
                cost = cost_matrix[r_idx, c_idx]
                # If cost is too high (e.g. distance > 4.0m and iou=0, so cost > 2.0), reject match
                if cost > 2.0:
                    continue

                self.tracks[r_idx].update(detections[c_idx], scores[c_idx])
                matched_tracks.add(r_idx)
                matched_dets.add(c_idx)
        else:
            matched_tracks = set()
            matched_dets = set()

        # 3. Handle unmatched tracks (death)
        unmatched_tracks = set(range(num_tracks)) - matched_tracks
        for t_idx in sorted(list(unmatched_tracks), reverse=True):
            track = self.tracks[t_idx]
            if track.time_since_update >= self.max_age:
                self.tracks.pop(t_idx)

        # 4. Handle unmatched detections (birth)
        unmatched_dets = set(range(num_dets)) - matched_dets
        for d_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                bbox_3d=detections[d_idx],
                class_id=classes[d_idx],
                score=scores[d_idx],
                dt=self.dt,
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        # 5. Output active and confirmed tracks
        active_tracks = []
        for track in self.tracks:
            # Promote to confirmed if it meets hits threshold
            if track.state == "tentative" and track.hits >= self.min_hits:
                track.state = "confirmed"

            if track.state == "confirmed" and track.time_since_update == 0:
                bbox_3d, velocity = track.get_state()
                active_tracks.append(
                    {
                        "track_id": track.track_id,
                        "bbox_3d": bbox_3d,
                        "velocity": velocity.tolist(),
                        "class_id": track.class_id,
                        "score": track.score,
                    }
                )

        return active_tracks

import numpy as np
from src.tracking.tracker import AB3DMOTTracker


def test_tracker_lifecycle():
    # Instantiate tracker
    tracker = AB3DMOTTracker(max_age=2, min_hits=2, dt=0.1)

    # Frame 1: Detection of 1 car
    dets = [[10.0, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]]
    scores = [0.9]
    classes = [0]

    tracks = tracker.update(dets, scores, classes)
    # tentative track is not confirmed yet (hits=1, min_hits=2)
    assert len(tracks) == 0

    # Frame 2: Match detection of same car moving slightly
    dets = [[10.1, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]]
    scores = [0.9]
    classes = [0]

    tracks = tracker.update(dets, scores, classes)
    # Now confirmed (hits=2)
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1
    np.testing.assert_allclose(tracks[0]["bbox_3d"][0], 10.1, atol=0.2)
    assert tracks[0]["class_id"] == 0

    # Frame 3: Keep tracking
    dets = [[10.2, -2.0, 1.0, 4.0, 1.6, 1.5, 0.0]]
    scores = [0.9]
    classes = [0]
    tracks = tracker.update(dets, scores, classes)
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1

    # Frame 4: Missed detection (unmatched)
    tracks = tracker.update([], [], [])
    assert len(tracks) == 0  # not active/updated in current frame

    # Frame 5: Missed detection again -> should die (since max_age=2, time_since_update becomes 2)
    tracks = tracker.update([], [], [])
    assert len(tracker.tracks) == 0  # track is deleted

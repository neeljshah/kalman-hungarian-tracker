"""pytest -- run: python -m pytest tests/test_tracker.py -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kh_tracker.kalman import KalmanFilter6D  # noqa: E402
from kh_tracker.tracker import MultiObjectTracker, hungarian_assign  # noqa: E402


def _box_at(cx, cy, w=20.0, h=20.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_kalman_converges_on_constant_velocity():
    """Feeding a true constant-velocity sequence should make predictions
    converge to the true next position (prediction error shrinks over time)."""
    vx, vy = 5.0, 3.0
    kf = KalmanFilter6D(_box_at(0, 0))
    errors = []
    cx, cy = 0.0, 0.0
    for i in range(1, 20):
        cx, cy = i * vx, i * vy
        pred = kf.predict()
        pred_cx, pred_cy = pred[0], pred[1]
        errors.append(abs(pred_cx - cx) + abs(pred_cy - cy))
        kf.update(_box_at(cx, cy))
    # error in the last few steps should be much smaller than the first step
    assert errors[-1] < errors[0]
    assert errors[-1] < 1.0


def test_hungarian_beats_greedy_on_crafted_ambiguity():
    """3-track/3-detection cost matrix where greedy's row-order-dependent
    choice picks a total-cost-suboptimal assignment but Hungarian finds the
    global optimum."""
    # Track 0 slightly prefers det 0 (cost 0.1) but det 0 is det 1's only
    # good option, greedy taking track0->det0 first forces a bad leftover
    # assignment; Hungarian looks at the whole matrix.
    cost = np.array([
        [0.10, 0.11, 0.90],
        [0.10, 0.90, 0.90],
        [0.90, 0.90, 0.10],
    ])

    def greedy(cost):
        used = set()
        pairs = []
        for r in range(cost.shape[0]):
            best_c, best_v = -1, float("inf")
            for c in range(cost.shape[1]):
                if c not in used and cost[r, c] < best_v:
                    best_v, best_c = cost[r, c], c
            if best_c >= 0:
                pairs.append((r, best_c))
                used.add(best_c)
        return pairs

    def total(pairs):
        return sum(cost[r, c] for r, c in pairs)

    greedy_pairs = greedy(cost)
    hungarian_pairs = hungarian_assign(cost)

    assert total(hungarian_pairs) <= total(greedy_pairs)
    # This specific matrix is constructed so greedy is strictly worse:
    # greedy: (0,0)=.10 (1,?) det0 taken -> (1,1)=.90 wait det1 free? recompute
    assert total(hungarian_pairs) < total(greedy_pairs)


def test_lifecycle_tentative_confirmed_lost():
    tracker = MultiObjectTracker(max_age=3, min_hits=2)
    # frame 1: new detection -> tentative
    tracks = tracker.step([_box_at(0, 0)])
    assert tracks[0].state == "tentative"

    # frame 2: matched again -> confirmed (hits reaches min_hits)
    tracks = tracker.step([_box_at(4, 0)])
    assert tracks[0].state == "confirmed"

    # frames with no detections -> track goes to lost, then evicted after max_age
    for _ in range(3):
        tracks = tracker.step([])
    assert all(t.state != "confirmed" for t in tracker.tracks)
    # after max_age misses it should be evicted entirely
    tracker.step([])
    assert len(tracker.tracks) == 0


def test_dropout_tolerance_survives_max_age_minus_one_misses():
    tracker = MultiObjectTracker(max_age=5, min_hits=2)
    tracker.step([_box_at(0, 0)])
    tracker.step([_box_at(4, 0)])  # confirmed
    track_id = tracker.tracks[0].track_id

    for _ in range(4):  # max_age - 1 misses
        tracker.step([])
    assert any(t.track_id == track_id for t in tracker.tracks), \
        "track should survive max_age-1 consecutive misses"

    for _ in range(2):  # push age past max_age (eviction is age > max_age)
        tracker.step([])
    assert not any(t.track_id == track_id for t in tracker.tracks)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

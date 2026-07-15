"""
tracker.py -- multi-object tracking by detection: Kalman prediction +
globally-optimal (Hungarian) assignment over a blended IoU/appearance cost.

Pipeline per frame:
  1. Every existing track predicts its next box (Kalman.predict()).
  2. Build a cost matrix [n_tracks x n_detections]; cost = blend of
     (1 - IoU) and appearance distance, gated by a max-cost threshold.
  3. Solve the assignment with scipy.optimize.linear_sum_assignment
     (Hungarian / Jonker-Volgenant). Falls back to a greedy row-by-row
     assignment if scipy is not installed.
  4. Matched tracks: Kalman.update() with the new box, lifecycle advances.
  5. Unmatched tracks: age up; evicted after `max_age` consecutive misses.
  6. Unmatched detections: spawn new tentative tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .kalman import KalmanFilter6D

BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # ponytail: greedy fallback only kicks in without scipy
    _HAS_SCIPY = False


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def appearance_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """L2 distance in [0, ~1.4] on L2-normalized embeddings. 0.5 (neutral) if unknown."""
    if a is None or b is None:
        return 0.5
    return float(np.linalg.norm(a - b))


def hungarian_assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Row/col pairs minimizing total cost. Globally optimal via scipy, else greedy."""
    if cost.size == 0:
        return []
    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    used: set[int] = set()
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


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter6D
    appearance: Optional[np.ndarray] = None
    state: str = "tentative"        # tentative -> confirmed -> lost
    hits: int = 0                   # consecutive matched frames
    age: int = 0                    # consecutive missed frames
    predicted_bbox: BBox = field(default=None)  # set each step() by predict()

    @property
    def bbox(self) -> BBox:
        return self.kf.bbox_xyxy()


class MultiObjectTracker:
    """Tracks 2D boxes across frames. Detections in, Track list out."""

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        cost_gate: float = 0.7,
        appearance_weight: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.cost_gate = cost_gate
        self.appearance_weight = appearance_weight
        self.tracks: list[Track] = []
        self._next_id = 0
        self.id_switches = 0  # incremented externally by evaluation code if desired

    def step(
        self,
        detections: list[BBox],
        appearances: Optional[list[np.ndarray]] = None,
    ) -> list[Track]:
        appearances = appearances or [None] * len(detections)

        # 1. predict every active track forward
        for t in self.tracks:
            t.kf.predict()
            t.predicted_bbox = t.kf.bbox_xyxy()

        # 2. blended cost matrix
        n_t, n_d = len(self.tracks), len(detections)
        cost = np.ones((n_t, n_d), dtype=np.float64)
        for ti, t in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou_cost = 1.0 - iou(t.predicted_bbox, det)
                app_cost = appearance_distance(t.appearance, appearances[di])
                w = self.appearance_weight
                cost[ti, di] = (1 - w) * iou_cost + w * app_cost

        # 3. Hungarian (or greedy) assignment, gated
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for ti, di in hungarian_assign(cost):
            if cost[ti, di] <= self.cost_gate:
                matched_t.add(ti)
                matched_d.add(di)
                t = self.tracks[ti]
                t.kf.update(detections[di])
                if appearances[di] is not None:
                    t.appearance = appearances[di]
                t.hits += 1
                t.age = 0
                if t.state == "tentative" and t.hits >= self.min_hits:
                    t.state = "confirmed"
                elif t.state == "lost":
                    t.state = "confirmed"

        # 4. unmatched tracks age up / get evicted
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.age += 1
                t.hits = 0
                if t.state != "tentative":
                    t.state = "lost"
                if t.age > self.max_age:
                    t.state = "dead"
        self.tracks = [t for t in self.tracks if t.state != "dead"]

        # 5. unmatched detections spawn new tentative tracks
        for di in range(n_d):
            if di not in matched_d:
                new_t = Track(
                    track_id=self._next_id,
                    kf=KalmanFilter6D(detections[di]),
                    appearance=appearances[di],
                    hits=1,
                )
                self._next_id += 1
                self.tracks.append(new_t)

        return self.tracks

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == "confirmed"]

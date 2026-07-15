"""
kalman.py -- 6D constant-velocity Kalman filter, built from scratch with numpy.

State vector  x = [cx, cy, vx, vy, w, h]^T          (6,)
Measurement   z = [cx, cy, w, h]^T                   (4,)  -- a detected box's
                                                              center + size;
                                                              velocity is NOT
                                                              observed directly,
                                                              only inferred by
                                                              the filter over time.

Motion model (constant velocity, dt=1 frame):
    cx' = cx + vx
    cy' = cy + vy
    vx' = vx
    vy' = vy
    w'  = w
    h'  = h

  F = [[1, 0, 1, 0, 0, 0],
       [0, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1]]

Measurement model (we observe position + size, not velocity):
  H = [[1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1]]

Q (process noise) and R (measurement noise) are diagonal -- a simplification
(ponytail: real systems tune per-axis noise; a single scalar per matrix is
enough to demonstrate the algorithm and is what the source system used too).
"""
from __future__ import annotations

import numpy as np

State = np.ndarray  # shape (6,)


class KalmanFilter6D:
    """Constant-velocity Kalman filter over [cx, cy, vx, vy, w, h]."""

    def __init__(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        process_noise: float = 5e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = float(x2 - x1), float(y2 - y1)

        self.F = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

        self.x: State = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)

    def predict(self) -> State:
        """Advance one time step. Mutates and returns the prior state estimate."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement_xyxy: tuple[float, float, float, float]) -> State:
        """Correct the prediction with an observed box. Returns posterior state."""
        x1, y1, x2, y2 = measurement_xyxy
        z = np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1)],
            dtype=np.float64,
        )
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        cx, cy, _vx, _vy, w, h = self.x
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

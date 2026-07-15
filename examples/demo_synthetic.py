"""
demo_synthetic.py -- synthetic moving-box demo for MultiObjectTracker.

Two scenarios:
  1. "easy": 3 boxes on parallel, non-crossing paths, with per-frame position
     noise and random detection dropout. Expect ZERO ID switches.
  2. "crossing": 3 boxes whose paths cross near frame 15 (classic ambiguous-
     assignment case). Reported honestly -- IoU-only cost has no way to tell
     the boxes apart at the crossing point, so switches there are expected
     unless appearance embeddings are supplied.

Run: python examples/demo_synthetic.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kh_tracker import MultiObjectTracker  # noqa: E402

BOX_W, BOX_H = 20.0, 20.0
N_FRAMES = 30


def make_scenario(crossing: bool, seed: int = 0):
    """Returns list[frame] of list[(true_id, bbox)]."""
    rng = random.Random(seed)
    if crossing:
        # 3 objects moving toward/through a shared point around frame 15
        starts = [(0.0, 0.0), (60.0, 0.0), (30.0, 60.0)]
        vels = [(3.0, 2.0), (-3.0, 2.0), (0.0, -3.5)]
    else:
        # 3 objects moving in parallel lanes, never close to each other
        starts = [(0.0, 0.0), (0.0, 100.0), (0.0, 200.0)]
        vels = [(4.0, 0.0), (4.0, 0.0), (4.0, 0.0)]

    frames = []
    for f in range(N_FRAMES):
        frame_dets = []
        for tid, ((sx, sy), (vx, vy)) in enumerate(zip(starts, vels)):
            cx = sx + vx * f + rng.uniform(-1.0, 1.0)   # detection noise
            cy = sy + vy * f + rng.uniform(-1.0, 1.0)
            if rng.random() < 0.1:                       # 10% dropout
                continue
            bbox = (cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2)
            frame_dets.append((tid, bbox))
        frames.append(frame_dets)
    return frames


def run_scenario(name: str, crossing: bool) -> int:
    frames = make_scenario(crossing)
    tracker = MultiObjectTracker(max_age=5, min_hits=2, appearance_weight=0.0)

    # track_id -> true_id last seen bound to it
    bound_true_id: dict[int, int] = {}
    id_switches = 0

    for frame_dets in frames:
        true_ids = [tid for tid, _ in frame_dets]
        boxes = [b for _, b in frame_dets]
        tracks = tracker.step(boxes)

        # Match returned tracks back to the detection they consumed by nearest
        # box center, so we can check which true_id each track_id is bound to.
        for t in tracks:
            if t.age != 0:      # not matched this frame
                continue
            tcx = (t.bbox[0] + t.bbox[2]) / 2
            tcy = (t.bbox[1] + t.bbox[3]) / 2
            best_true_id, best_dist = None, float("inf")
            for tid, (x1, y1, x2, y2) in zip(true_ids, boxes):
                dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
                d = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
                if d < best_dist:
                    best_dist, best_true_id = d, tid
            prev = bound_true_id.get(t.track_id)
            if prev is not None and prev != best_true_id:
                id_switches += 1
            bound_true_id[t.track_id] = best_true_id

    print(f"[{name}] frames={N_FRAMES} final_tracks={len(tracker.tracks)} "
          f"id_switches={id_switches}")
    return id_switches


if __name__ == "__main__":
    easy_switches = run_scenario("easy (parallel, non-crossing)", crossing=False)
    assert easy_switches == 0, f"expected zero ID switches on easy case, got {easy_switches}"
    print("  -> PASS: zero ID switches on the easy case, as expected.")

    crossing_switches = run_scenario("crossing (paths intersect)", crossing=True)
    print(f"  -> honest result: {crossing_switches} ID switch(es) at the crossing. "
          f"IoU-only cost cannot disambiguate identical-looking boxes at the exact "
          f"crossing point; add appearance embeddings to reduce this (see README).")

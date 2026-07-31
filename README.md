# kalman-hungarian-tracker

Multi-object tracking-by-detection, built from mathematical primitives instead
of a black-box library: a from-scratch 6D constant-velocity **Kalman filter**
for motion prediction, and the **Hungarian algorithm** for globally-optimal
detection-to-track assignment over a blended IoU + appearance cost.

No `filterpy`. No tracking framework. The two algorithms that make
tracking-by-detection work are ~200 lines of numpy each, and this repo shows
them at that resolution.

Author: [Neel Shah](https://github.com/neeljshah) ([neeljshah22@gmail.com](mailto:neeljshah22@gmail.com))
-- distilled from the tracker in a broadcast-sports computer-vision system
([CourtVision](https://github.com/neeljshah)), rebuilt here standalone on a
synthetic domain so the tracking math stands on its own, with no
sports/video dependency.

## The problem: tracking-by-detection

A detector (YOLO, or in this repo, a synthetic box generator) gives you a set
of boxes *per frame*, with no identity attached -- frame 10 doesn't know that
its 3rd box is "the same object" as frame 9's 2nd box. Tracking-by-detection
is the job of gluing those per-frame detections into consistent identities
over time. Two sub-problems have to be solved:

1. **Where will each existing track be in the next frame?** (motion model)
2. **Which detection in the next frame belongs to which existing track?**
   (assignment)

This repo implements one clean answer to each: a Kalman filter for (1), the
Hungarian algorithm for (2).

## Why Kalman: motion prediction under noise

Detections are noisy -- a box's reported center jitters frame to frame even
when the underlying object moves smoothly, and detectors sometimes miss a
frame entirely (occlusion, low confidence). Naively using "last known
position" as the estimate for "next position" throws away the fact that
objects have momentum, and does nothing when a detection is missing.

The Kalman filter keeps a running estimate of a **hidden state** -- position
*and* velocity -- updated by two alternating steps:

- **Predict**: advance the state estimate one time step using a motion model,
  growing the uncertainty (it's a guess, after all).
- **Update**: when a new measurement arrives, blend it with the predicted
  state, weighted by their respective uncertainties (the Kalman gain).

For this project the state is 6-dimensional, constant-velocity:

```
state       x = [cx, cy, vx, vy, w, h]         (position, velocity, box size)
measurement z = [cx, cy, w, h]                  (a detection only gives position + size)

motion model (dt = 1 frame):
  cx' = cx + vx        F = [[1,0,1,0,0,0],
  cy' = cy + vy              [0,1,0,1,0,0],
  vx' = vx                   [0,0,1,0,0,0],
  vy' = vy                   [0,0,0,1,0,0],
  w'  = w                    [0,0,0,0,1,0],
  h'  = h                    [0,0,0,0,0,1]]

measurement model (velocity is never observed directly, only inferred):
  H = [[1,0,0,0,0,0],
       [0,1,0,0,0,0],
       [0,0,0,0,1,0],
       [0,0,0,0,0,1]]
```

`Q` (process noise) and `R` (measurement noise) are diagonal scalars here --
a deliberate simplification; see [Honest limits](#honest-limits). The filter
predicts a box every frame even when no detection arrives that frame, which
is exactly what lets a track survive a brief occlusion: `predict()` still
returns a sane box, so the assignment step in the next frame has something
sensible to match against.

Implementation: [`src/kh_tracker/kalman.py`](src/kh_tracker/kalman.py) (~100 LOC).

## Why Hungarian: globally optimal, not greedy

Once every track has a predicted box, you need to decide which detection
goes with which track. The obvious approach -- for each track, in order, grab
the closest unclaimed detection ("greedy") -- is fast but **not globally
optimal**: an early, locally-good greedy choice can block a much better
overall assignment, causing an avoidable ID switch.

**Worked example.** Three tracks, three detections, cost = 1 − IoU (lower is
better):

```
              det0   det1   det2
track0        0.10   0.11   0.90
track1        0.10   0.90   0.90
track2        0.90   0.90   0.10
```

Greedy processes tracks in order: track0 takes det0 (cost 0.10, its cheapest
option). Now det0 is gone, so track1 -- whose *only* good option was det0 --
is forced onto det1 at cost 0.90. Track2 takes det2 (0.10).
**Greedy total cost = 0.10 + 0.90 + 0.10 = 1.10.**

Hungarian looks at the whole matrix at once and finds track0→det1 (0.11),
track1→det0 (0.10), track2→det2 (0.10).
**Hungarian total cost = 0.11 + 0.10 + 0.10 = 0.31** -- better, and it didn't
strand track1 on a bad match just because track0 went first. This exact
matrix is reproduced as a test in
[`tests/test_tracker.py::test_hungarian_beats_greedy_on_crafted_ambiguity`](tests/test_tracker.py).

This repo uses `scipy.optimize.linear_sum_assignment` (the Jonker-Volgenant
variant of the Hungarian algorithm, O(n^3)) and falls back to the same greedy
strategy above when scipy isn't installed --so the failure mode above is
visible on purpose if you run without the optional dependency.

## The blended cost

```
cost(track, detection) = (1 - appearance_weight) * (1 - IoU)
                        +      appearance_weight  * appearance_distance
```

- **IoU term**: how much the track's *Kalman-predicted* box overlaps the
  detection's box. This is the primary signal -- objects don't teleport, so
  the predicted box should be close to the true next box.
- **Appearance term** (optional): distance between embedding vectors (e.g.
  a color histogram or a re-ID network's output) attached to each detection.
  Lets the tracker disambiguate two boxes that overlap similarly but look
  different -- the crossing-paths case in the demo is exactly where IoU alone
  runs out of signal.
- **Gating**: any candidate pair with cost above `cost_gate` is rejected
  outright, even if it was the Hungarian solver's "best available" pairing.
  Gating stops a track from being assigned to a wildly wrong detection just
  because everything else was worse -- a bad match should become "no match",
  not "the least-bad wrong match."

## Track lifecycle

```
tentative --(min_hits consecutive matches)--> confirmed
confirmed --(one missed frame)--> lost --(matched again)--> confirmed
lost --(age > max_age consecutive misses)--> evicted
```

- **tentative**: a brand-new track from an unmatched detection. Not yet
  trusted -- could be detector noise (a spurious box).
- **confirmed**: matched `min_hits` frames in a row. Prediction and
  reporting both treat it as real.
- **lost**: currently unmatched, but its Kalman filter keeps predicting
  forward so it can be re-acquired if the detector picks the object back up.
- **evicted**: unmatched for more than `max_age` consecutive frames -- the
  track is deleted; if the object reappears later it becomes a *new* track
  (identity is not recovered).

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest tests/test_tracker.py -q
python examples/demo_synthetic.py
```

```python
from kh_tracker import MultiObjectTracker

tracker = MultiObjectTracker(max_age=5, min_hits=2, appearance_weight=0.3)
tracks = tracker.step(detections=[(10, 10, 30, 30), (100, 40, 120, 60)])
for t in tracker.confirmed_tracks():
    print(t.track_id, t.bbox)
```

## Honest limits

- **No re-ID network.** The `appearances` argument accepts any embedding
  vector, but this repo does not ship a feature extractor -- plug in a color
  histogram, a CNN embedding, or nothing (IoU-only). Without appearance
  features, two objects that cross paths at the same time and place are
  fundamentally indistinguishable to this tracker; the demo's crossing
  scenario exists to show that honestly rather than paper over it.
- **Identity is not recovered after eviction.** Once a track exceeds
  `max_age` consecutive misses it is deleted. If the same object reappears
  later, it gets a brand-new `track_id` -- there is no lost-track gallery or
  long-term re-identification here (a real system would add one; that's a
  separate, harder problem than what this repo demonstrates).
- **Q/R are scalar, not tuned per-axis.** Position and size noise almost
  certainly differ in a real system; this repo uses one process-noise scalar
  and one measurement-noise scalar for simplicity of exposition, not because
  that's the recommended production setting.
- **Linear motion only.** The constant-velocity model has no acceleration
  term; fast direction changes (a sharp turn) will show up as several frames
  of prediction error before the filter catches up.

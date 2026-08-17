#!/usr/bin/env python3
"""
Propagate reviewed keyframe boxes to every frame with KLT optical-flow tracking.

This is the keyframe pipeline's fill-in step. `12_extract_keyframes.py` selects
sparse keyframes, `07_run_qwen.py` seeds boxes on them, and `08_click_review_coco.py`
is used to review the keyframe boxes. This script reads that reviewed keyframe
COCO plus the manifest and produces a COCO file with boxes for **every** frame,
which is then handed back to `08_click_review_coco.py` for a final full review.

Method (anchored optical flow):
    For each adjacent keyframe pair (a, b) the reviewed boxes are matched by
    class + nearest center. For each matched pair, Lucas-Kanade optical flow
    (`goodFeaturesToTrack` inside the box, `calcOpticalFlowPyrLK` frame-by-frame)
    gives a raw trajectory of the box center. A linear correction is applied so
    the trajectory starts exactly at keyframe a's reviewed box and lands exactly
    at keyframe b's reviewed box; the box size is linearly interpolated a->b.
    This keeps the flow's motion shape while guaranteeing the reviewed endpoints
    are respected (no drift accumulation across the whole video).

Method (Kalman, ``--interp-method kalman``):
    Same per-frame optical-flow median displacement is fed as the measurement
    into a constant-velocity Kalman filter (one per axis). The KF denoises
    jittery flow and, when the flow track is temporarily lost, predicts
    forward with the learned velocity instead of bailing out. The smoothed
    trajectory is then endpoint-anchored the same way as the raw method so it
    still starts at keyframe a and lands at keyframe b. Process/measurement
    noise are tunable via ``--kf-q`` / ``--kf-r``.

Method (camera model, ``--camera-model global``):
    Per-frame camera motion is estimated with a RANSAC similarity transform
    (rotation + scale + translation; whole-frame features tracked outside
    the object boxes, DIS-median translation fallback for texture-poor
    pairs) and composed across the span. Object trajectories are then
    tracked/anchored as a *residual* relative to the camera-predicted
    position, so non-linear camera motion (handheld shake, walking) is
    absorbed by the per-frame transform chain while the flow/KF only
    denoises the small object-specific residual. Lost KLT tracks are
    re-seeded at the camera-predicted position. Off by default: an accuracy
    test against re-reviewed MTR 4k frames (644 ten-frame windows, median
    center error vs ground truth) showed no net gain on this fisheye camera
    (18.5px with 'global' vs 16.4px with 'none') — the similarity model is a
    rough ego-motion approximation under fisheye distortion, and a full
    homography fits the background parallax instead of the object's motion
    (much worse). Use 'global' for pinhole cameras with strong non-linear
    shake. Every output annotation also carries a ``source``
    (keyframe/flow/kalman/linear/hold) and a ``confidence`` in [0, 1] so the
    downstream review can prioritize uncertain frames.

USAGE:
    python scripts/13_interpolate_tracks.py \
        --keyframes-coco output/MTR_keyframes/reviewed/coco_reviewed.json \
        --manifest Datasets/MTR/MTR_keyframes/keyframe_manifest.json \
        --image-folder Datasets/MTR/rosbags/MTR_metacam_right \
        --output-coco output/MTR_full/interpolated.coco.json \
        --vis-output output/MTR_full/vis

    # MTR 4k exit-sign dataset, 800 keyframes (stride 5) merged COCO;
    # overwrites previous interpolation results + vis
    python scripts/13_interpolate_tracks.py \
        --keyframes-coco output/MTR_4k/MTR_4k_keyframes/coco_reviewed_800_final.json \
        --manifest Datasets/MTR/MTR_4k_keyframes/keyframe_manifest.json \
        --image-folder Datasets/MTR/MTR_4k_dataset_exit_signs \
        --output-coco output/MTR_4k/interpolated_all_coco.json \
        --vis-output output/MTR_4k/interpolated_vis

    # Same but with the Kalman-filter interpolator (constant-velocity smoothing
    # of the optical-flow measurements; better jitter handling and short-gap
    # extrapolation). Dis flow + tuned process/measurement noise.
    python scripts/13_interpolate_tracks.py \
        --keyframes-coco output/MTR_4k/MTR_4k_keyframes/coco_reviewed_800_final.json \
        --manifest Datasets/MTR/MTR_4k_keyframes/keyframe_manifest.json \
        --image-folder Datasets/MTR/MTR_4k_dataset_exit_signs \
        --output-coco output/MTR_4k/interpolated_kalman_coco.json \
        --vis-output output/MTR_4k/interpolated_kalman_vis \
        --interp-method kalman --flow-method dis \
        --kf-q 1.0 --kf-r 4.0

OUTPUT:
    - <output-coco>        COCO with boxes for every frame (keyframes reviewed,
                            in-between frames interpolated). Annotations carry
                            "source" (keyframe/flow/kalman/linear/hold) and
                            "confidence" (0-1) provenance fields.
    - <vis-output>/*.jpg   annotated frames (optional)
    - <vis-output>/tracking_result.mp4   summary video (optional)
"""

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tracking_utils import create_tracking_video


# ---------------------------------------------------------------------------
# COCO / manifest loading
# ---------------------------------------------------------------------------

def load_keyframe_boxes(coco_path, frame_idx_by_name):
    """Return (categories, {frame_idx: [box, ...]}, (W, H)) from reviewed COCO.

    ``box`` is a dict with ``ann_id``, ``category_id``, ``xywh`` (x,y,w,h),
    ``xyxy`` and ``center``. Only keyframe images present in the manifest are
    kept; a warning is printed for unmatched images.
    """
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    img_id_to_name = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    # Pick the image size from the first image that has width/height, else None.
    size = None
    for img in coco.get("images", []):
        if img.get("width") and img.get("height"):
            size = (int(img["width"]), int(img["height"]))
            break

    boxes_by_frame = {}
    used_frame_idxs = set()
    for ann in coco.get("annotations", []):
        name = img_id_to_name.get(ann["image_id"])
        if name is None:
            continue
        fidx = frame_idx_by_name.get(Path(name).name)
        if fidx is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes_by_frame.setdefault(fidx, []).append({
            "ann_id": ann["id"],
            "category_id": int(ann["category_id"]),
            "xywh": [x, y, w, h],
            "xyxy": [x1, y1, x2, y2],
            "center": np.array([x1 + w / 2.0, y1 + h / 2.0]),
        })
        used_frame_idxs.add(fidx)

    if not used_frame_idxs:
        raise RuntimeError(
            "No reviewed keyframe annotations matched a manifest keyframe. "
            "Check that --keyframes-coco image names match the manifest.")

    return categories, boxes_by_frame, size


# ---------------------------------------------------------------------------
# Matching boxes between adjacent keyframes
# ---------------------------------------------------------------------------

def match_pairs(boxes_a, boxes_b, max_dist):
    """Greedy nearest-center match between two same-keyframe box lists.

    Returns (matches, unmatched_a_idx, unmatched_b_idx) where matches is a list
    of (idx_a, idx_b). Matching is restricted to the same category_id.
    """
    # Build cost = center distance for same-class pairs.
    pairs = []  # (dist, i, j)
    for i, ba in enumerate(boxes_a):
        for j, bb in enumerate(boxes_b):
            if ba["category_id"] != bb["category_id"]:
                continue
            d = float(np.linalg.norm(ba["center"] - bb["center"]))
            if d <= max_dist:
                pairs.append((d, i, j))
    pairs.sort()

    matched_a, matched_b = set(), set()
    matches = []
    for d, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        matches.append((i, j, d))

    unmatched_a = [i for i in range(len(boxes_a)) if i not in matched_a]
    unmatched_b = [j for j in range(len(boxes_b)) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------------
# Union-find for persistent track ids across keyframes
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def label(self, items):
        """Return {item: track_id} for all items, ids 0..N-1 by first-seen order."""
        labels = {}
        next_id = 0
        for it in items:
            r = self.find(it)
            if r not in labels:
                labels[r] = next_id
                next_id += 1
            labels.setdefault(it, labels[r])
        return labels


# ---------------------------------------------------------------------------
# Optical flow: sparse KLT and dense DIS/Farnebäck
# ---------------------------------------------------------------------------

def _seed_points(gray, xyxy, min_points=8):
    """Good features to track inside a box; fall back to a grid if too few."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                 minDistance=3, mask=mask)
    if pts is None or len(pts) < min_points:
        # Grid fallback so KLT still has something to follow.
        gx, gy = max(2, (x2 - x1) // 3), max(2, (y2 - y1) // 3)
        xs = np.linspace(x1 + 2, x2 - 2, num=max(3, (x2 - x1) // max(1, gx)))
        ys = np.linspace(y1 + 2, y2 - 2, num=max(3, (y2 - y1) // max(1, gy)))
        grid = np.array([[xx, yy] for yy in ys for xx in xs], dtype=np.float32)
        pts = grid.reshape(-1, 1, 2) if len(grid) else None
    return pts.astype(np.float32) if pts is not None and len(pts) else None


def _lk_step(prev_gray, curr_gray, points):
    """One LK step with forward-backward check and displacement consensus.

    Returns (next_points, valid_mask, med_disp or None). ``valid_mask`` marks
    the inliers that survived the forward-backward test and the MAD
    consensus, so the caller can refresh the tracked point set.
    """
    next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points,
                                               None, winSize=(21, 21),
                                               maxLevel=3)
    if next_pts is None or st is None:
        return None, None, None
    st = st.reshape(-1).astype(bool)
    finite = np.isfinite(next_pts.reshape(-1, 2)).all(axis=1)
    ok = st & finite
    if ok.sum() < 1:
        return next_pts, ok, None

    idx = np.where(ok)[0]
    p0 = points[idx].reshape(-1, 2)
    p1 = next_pts[idx].reshape(-1, 2)

    # Forward-backward check: the reverse flow from p1 must land near p0.
    back, st2, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray,
                                            p1.reshape(-1, 1, 2), None,
                                            winSize=(21, 21), maxLevel=3)
    if back is not None and st2 is not None:
        st2 = st2.reshape(-1).astype(bool)
        fbe = np.full(len(p0), np.inf)
        fbe[st2] = np.linalg.norm(back.reshape(-1, 2)[st2] - p0[st2], axis=1)
        fbe_ok = fbe < 1.0
        if fbe_ok.sum() < 1:
            return next_pts, ok, None
    else:
        fbe_ok = np.ones(len(p0), dtype=bool)

    disp = p1 - p0
    # MAD consensus: reject displacements that deviate from the median.
    med = np.median(disp, axis=0)
    mad = np.median(np.abs(disp - med), axis=0)
    tol = np.maximum(2.0, 3.0 * mad)
    keep = fbe_ok & (np.abs(disp - med) <= tol).all(axis=1)
    if keep.sum() < 1:
        return next_pts, ok, None

    valid = np.zeros(len(points), dtype=bool)
    valid[idx[keep]] = True
    return next_pts, valid, np.median(disp[keep], axis=0)


# ---------------------------------------------------------------------------
# Dense optical flow (DIS / Farnebäck)
# ---------------------------------------------------------------------------

_dense_flow_cache = {}  # (prev_name, curr_name) -> flow (H, W, 2)


def _compute_dense_flow(prev_gray, curr_gray, prev_name, curr_name, method="dis"):
    """Compute dense optical flow between two frames, with caching.

    Returns (H, W, 2) float32 array. Uses DIS (fast, good for real-time) or
    Farnebäck (more accurate, slower) depending on ``method``.
    """
    cache_key = (prev_name, curr_name, method)
    if cache_key in _dense_flow_cache:
        return _dense_flow_cache[cache_key]

    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=5, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.1,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
    else:
        # DIS: Dense Inverse Search. Fast and handles large displacements
        # much better than sparse KLT, ideal for handheld/unstable camera.
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        flow = dis.calc(prev_gray, curr_gray, None)

    if len(_dense_flow_cache) > 64:
        _dense_flow_cache.clear()
    _dense_flow_cache[cache_key] = flow
    return flow


def _box_median_flow(flow, xyxy):
    """Median (dx, dy) of the dense flow field inside a box region."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = flow.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    region = flow[y1:y2, x1:x2]
    dx = np.median(region[:, :, 0])
    dy = np.median(region[:, :, 1])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    return np.array([dx, dy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Global camera-motion estimation (non-linear camera shake handling)
# ---------------------------------------------------------------------------

_cam_stats = {"attempts": 0, "ok": 0, "qualities": []}


def _estimate_global_flow(prev_gray, curr_gray, exclude_boxes=(),
                          gf_quality=0.02, ransac_px=2.0):
    """Estimate the per-frame camera transform between two consecutive frames.

    Features are tracked across the whole frame *outside* ``exclude_boxes``
    (so object motion does not bias the estimate) and a similarity transform
    (rotation + scale + translation, 4 DoF) is fitted with RANSAC. The
    similarity approximates the camera ego-motion that a world-static object
    rides on; a full homography (8 DoF) fits the *background parallax plane*
    instead (higher inlier ratio ~0.55 vs ~0.32 on the MTR metacam data, but
    4x worse at predicting the object's motion), so it is not used.
    Composing the per-frame transforms across a span captures non-linear
    motion.

    Returns (H 3x3 homogeneous, inlier_ratio) or (None, 0.0) when the frame
    pair has too few trackable features.
    """
    h, w = prev_gray.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for xyxy in exclude_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(h, y2), min(w, x2)
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 0
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400,
                                 qualityLevel=gf_quality, minDistance=8,
                                 mask=mask)
    if p0 is None or len(p0) < 30:
        return None, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None,
                                         winSize=(21, 21), maxLevel=4)
    if p1 is None or st is None:
        return None, 0.0
    st = st.reshape(-1).astype(bool)
    if st.sum() < 30:
        return None, 0.0
    M, inliers = cv2.estimateAffinePartial2D(
        p0[st], p1.reshape(-1, 1, 2)[st], method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px, maxIters=3000, confidence=0.999)
    if M is None:
        return None, 0.0
    ratio = float(inliers.reshape(-1).mean()) if inliers is not None else 0.0
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H, ratio


def _median_flow_outside(flow, exclude_boxes=()):
    """Median (dx, dy) of a dense flow field outside the given box regions."""
    h, w = flow.shape[:2]
    keep = np.ones((h, w), dtype=bool)
    for xyxy in exclude_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(h, y2), min(w, x2)
        if y2 > y1 and x2 > x1:
            keep[y1:y2, x1:x2] = False
    if keep.sum() < 64:
        return None
    dx = np.median(flow[:, :, 0][keep])
    dy = np.median(flow[:, :, 1][keep])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    return np.array([dx, dy], dtype=np.float32)


def _build_camera_model(image_folder, frames, a, b, exclude_boxes=(),
                        gf_quality=0.02, ransac_px=2.0):
    """Per-frame camera transforms for the span (a, b].

    Returns (Hs, quality): ``Hs`` is a list of 3x3 matrices mapping frame
    t-1 coordinates to frame t coordinates (t = a+1..b) and ``quality`` the
    mean RANSAC inlier ratio in [0, 1]. Returns (None, 0.0) if the model
    could not be built (missing frames / no features).
    """
    _cam_stats["attempts"] += 1
    Hs, qs = [], []
    prev_gray = _read_gray(image_folder, frames[a])
    if prev_gray is None:
        return None, 0.0
    for t in range(a + 1, b + 1):
        curr_gray = _read_gray(image_folder, frames[t])
        if curr_gray is None:
            return None, 0.0
        H, ratio = _estimate_global_flow(prev_gray, curr_gray, exclude_boxes,
                                         gf_quality=gf_quality,
                                         ransac_px=ransac_px)
        if H is None or ratio < 0.25:
            # Texture-poor pair: fall back to the global translation given by
            # the median dense flow outside the object boxes.
            flow = _compute_dense_flow(prev_gray, curr_gray, frames[t - 1],
                                       frames[t], method="dis")
            med = _median_flow_outside(flow, exclude_boxes)
            if med is None:
                return None, 0.0
            H = np.eye(3, dtype=np.float64)
            H[0, 2], H[1, 2] = float(med[0]), float(med[1])
            ratio = 0.25
        Hs.append(H)
        qs.append(ratio)
        prev_gray = curr_gray
    _cam_stats["ok"] += 1
    _cam_stats["qualities"].append(float(np.mean(qs)))
    return Hs, float(np.mean(qs))


def _compose(Hs, a):
    """Cumulative transforms: {t: H} mapping frame-a coords to frame-t coords."""
    C = {a: np.eye(3)}
    cur = np.eye(3)
    for t, H in zip(range(a + 1, a + 1 + len(Hs)), Hs):
        cur = H @ cur
        C[t] = cur
    return C


def _apply_H(H, pt):
    v = H @ np.array([float(pt[0]), float(pt[1]), 1.0])
    return v[:2]


def _predict_next(raw_centers, last_disp, cam_predict, p):
    """Predict the center at frame p after a flow dropout.

    Prefers the camera-model prediction (a world-static object rides the
    camera transform), else extrapolates the last measured displacement,
    else holds the last position.
    """
    if cam_predict is not None:
        c = cam_predict(p)
        if c is not None:
            return c
    last_p = max(raw_centers)
    if last_disp is not None:
        return raw_centers[last_p] + last_disp
    return raw_centers[last_p].copy()


def _step_residual(med, cam_pos, p):
    """Per-step displacement relative to the camera motion, or None."""
    if cam_pos is None:
        return None
    c0, c1 = cam_pos(p - 1), cam_pos(p)
    if c0 is None or c1 is None:
        return None
    return med - (c1 - c0)


def _clamp_step(med, cam_pos, p, max_step):
    """Clamp a per-step displacement in the space where junk is bounded.

    With a camera model the frame may legitimately move fast, so the *residual*
    relative to the camera motion is clamped (the camera displacement itself
    passes through untouched). Without one, the absolute displacement is
    clamped, as before.
    """
    res = _step_residual(med, cam_pos, p)
    if res is not None:
        cam_d = cam_pos(p) - cam_pos(p - 1)
        m = float(np.hypot(res[0], res[1]))
        if m > max_step and m > 0:
            res = res * (max_step / m)
        return cam_d + res
    m = float(np.hypot(med[0], med[1]))
    if m > max_step and m > 0:
        med = med * (max_step / m)
    return med


# ---------------------------------------------------------------------------
# Kalman-filter smoothing (constant-velocity model)
# ---------------------------------------------------------------------------

class _ConstantVelocityKF:
    """A minimal 1-D constant-velocity Kalman filter (state [pos, vel]).

    Used to smooth noisy optical-flow center trajectories. Operates on scalar
    position per axis; instantiate two (x and y) for a 2-D center.
    """

    def __init__(self, x0, q=1.0, r=4.0):
        # State: [pos, vel], measurement: [pos]
        self.x = np.array([x0, 0.0], dtype=np.float64)
        self.P = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.diag([q, q * 0.25])
        self.R = np.array([[r]], dtype=np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        if z is None:
            # No measurement: pure prediction step.
            return
        z = np.array([z], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    @property
    def pos(self):
        return float(self.x[0])

    @property
    def vel(self):
        return float(self.x[1])


def interpolate_span_kalman(image_folder, frames, a, b, box_a, box_b,
                            min_valid=2, max_step_frac=0.3, flow_method="dis",
                            kf_q=1.0, kf_r=4.0, camera_model="none",
                            gf_quality=0.02, ransac_px=2.0):
    """Kalman-filtered optical flow for one matched pair between keyframe a and b.

    Feeds the per-frame optical-flow median displacement into a
    constant-velocity Kalman filter (one per axis), then applies the
    reviewed-endpoint anchoring. Compared to the plain accumulator, the KF
    denoises jittery flow (gain on the measurement) and extrapolates
    gracefully through short flow-dropout gaps where the KLT/DIS track is
    temporarily lost, instead of bailing out.

    With ``camera_model="global"`` the KF runs on the *residual* (measured
    center minus camera-predicted position): the per-frame camera transform
    absorbs non-linear camera motion, the CV model fits the small smooth
    residual, and dropouts are extrapolated along the camera path. Lost KLT
    tracks are re-seeded at the camera-predicted position.

    Returns {p: {"xyxy": [x1,y1,x2,y2], "source": str, "conf": float}} for
    interior frames p in (a, b).
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return _linear_span(a, b, box_a, box_b)

    center_a = box_a["center"].copy()
    center_b = box_b["center"].copy()
    wh_a = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    wh_b = np.array([box_b["xywh"][2], box_b["xywh"][3]])
    max_step = max_step_frac * float(min(wh_a[0], wh_a[1]))

    C = None
    if camera_model == "global":
        Hs, _cam_q = _build_camera_model(
            image_folder, frames, a, b, [box_a["xyxy"], box_b["xyxy"]],
            gf_quality=gf_quality, ransac_px=ransac_px)
        if Hs is not None:
            C = _compose(Hs, a)

    def cam_pos(t):
        if C is None or t not in C:
            return None
        return _apply_H(C[t], center_a)

    def to_state(t, center):
        c = cam_pos(t)
        return center - c if c is not None else center

    cur_pts = None
    if flow_method == "klt":
        cur_pts = _seed_points(gray_a, box_a["xyxy"])
        if cur_pts is None:
            return _linear_span(a, b, box_a, box_b)

    # Raw absolute measurements; missing frames stay as gaps the KF predicts
    # through.
    raw_centers = {a: center_a.copy()}
    prev_gray = gray_a
    prev_name = frames[a]
    last_disp = None
    reseed_streak = 0
    for p in range(a + 1, b + 1):
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            # Frame missing: leave a gap; the KF will predict through it.
            continue
        curr_name = frames[p]

        if flow_method == "klt":
            next_pts, ok, med = _lk_step(prev_gray, curr_gray, cur_pts)
            if med is not None and int(ok.sum()) >= min_valid:
                cur_pts = next_pts[ok].reshape(-1, 1, 2)
                reseed_streak = 0
            else:
                pred = _predict_next(raw_centers, last_disp, cam_pos, p)
                new_pts = _seed_points(curr_gray, _center_wh_to_xyxy(pred, wh_a))
                if new_pts is not None and reseed_streak < 2:
                    cur_pts = new_pts
                    prev_center = raw_centers[max(raw_centers)]
                    raw_centers[p] = pred.copy()
                    last_disp = pred - prev_center
                    reseed_streak += 1
                    prev_gray = curr_gray
                    prev_name = curr_name
                    continue
                med = None
        else:
            flow = _compute_dense_flow(prev_gray, curr_gray, prev_name, curr_name,
                                       method=flow_method)
            med = _box_median_flow(flow, box_a["xyxy"] if p == a + 1 else
                                   _center_wh_to_xyxy(
                                        raw_centers[list(raw_centers)[-1]], wh_a))

        if med is not None:
            med = _clamp_step(med, cam_pos, p, max_step)
            prev_center = raw_centers[list(raw_centers)[-1]]
            raw_centers[p] = prev_center + med
            last_disp = med
        prev_gray = curr_gray
        prev_name = curr_name

    # Constant-velocity KF on the state (residual when a camera model is set,
    # absolute center otherwise; both start at the reviewed keyframe-a box).
    init = to_state(a, center_a)
    kfx = _ConstantVelocityKF(float(init[0]), q=kf_q, r=kf_r)
    kfy = _ConstantVelocityKF(float(init[1]), q=kf_q, r=kf_r)
    state = {a: init.copy()}
    for p in range(a + 1, b + 1):
        kfx.predict()
        kfy.predict()
        if p in raw_centers:
            z = to_state(p, raw_centers[p])
            kfx.update(float(z[0]))
            kfy.update(float(z[1]))
        state[p] = np.array([kfx.pos, kfy.pos], dtype=np.float64)

    # Endpoint anchoring in state space (exact at both reviewed keyframes).
    s_b_raw = state[b]
    s_b_tgt = to_state(b, center_b)
    for p in range(a + 1, b):
        t = (p - a) / (b - a) if b > a else 0.0
        state[p] = state[p] + t * (s_b_tgt - s_b_raw)
    state[b] = s_b_tgt.copy()

    n_interior = max(1, b - a - 1)
    n_meas = sum(1 for p in range(a + 1, b) if p in raw_centers)
    q_meas = n_meas / n_interior

    result = {}
    for p in range(a + 1, b):
        c = cam_pos(p)
        center = state[p] + c if c is not None else state[p]
        wh = wh_a + (wh_b - wh_a) * ((p - a) / (b - a) if b > a else 0.0)
        conf = (0.9 if p in raw_centers else 0.6) * q_meas
        result[p] = {"xyxy": _center_wh_to_xyxy(center, wh),
                     "source": "kalman", "conf": round(float(conf), 3)}
    return result


def track_forward_kalman(image_folder, frames, a, b, box_a,
                         min_valid=2, max_step_frac=0.3, flow_method="dis",
                         kf_q=1.0, kf_r=4.0, camera_model="none",
                         gf_quality=0.02, ransac_px=2.0):
    """Forward Kalman-filtered tracking for an unmatched-at-a box.

    The measured centers are smoothed by a constant-velocity KF which keeps
    predicting (extrapolating with the learned velocity) through short
    flow-dropout gaps instead of stopping immediately. With
    ``camera_model="global"`` the KF runs on the residual relative to the
    camera-predicted position, so dropouts ride the camera path. Emits boxes
    for p in (a, b). Size stays at box_a's.

    Returns {p: {"xyxy": [x1,y1,x2,y2], "source": str, "conf": float}}.
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return {}

    center = box_a["center"].copy()
    wh = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    max_step = max_step_frac * float(min(wh[0], wh[1]))

    C = None
    if camera_model == "global":
        Hs, _q = _build_camera_model(image_folder, frames, a, b,
                                     [box_a["xyxy"]],
                                     gf_quality=gf_quality, ransac_px=ransac_px)
        if Hs is not None:
            C = _compose(Hs, a)

    def cam_pos(t):
        if C is None or t not in C:
            return None
        return _apply_H(C[t], center)

    def to_state(t, c):
        cp = cam_pos(t)
        return c - cp if cp is not None else c

    def from_state(t, s):
        cp = cam_pos(t)
        return s + cp if cp is not None else s

    init = to_state(a, center)
    kfx = _ConstantVelocityKF(float(init[0]), q=kf_q, r=kf_r)
    kfy = _ConstantVelocityKF(float(init[1]), q=kf_q, r=kf_r)

    result = {}
    raw = {a: center.copy()}
    prev_gray = gray_a
    prev_name = frames[a]
    cur_pts = None
    if flow_method == "klt":
        cur_pts = _seed_points(gray_a, box_a["xyxy"])
        if cur_pts is None:
            return {}

    last_disp = None
    reseed_streak = 0
    for p in range(a + 1, b):
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            kfx.predict(); kfy.predict()
            st = np.array([kfx.pos, kfy.pos])
            result[p] = {"xyxy": _center_wh_to_xyxy(from_state(p, st), wh),
                         "source": "kalman", "conf": 0.45}
            continue
        curr_name = frames[p]

        med = None
        measured_pred = None
        if flow_method == "klt":
            next_pts, ok, m = _lk_step(prev_gray, curr_gray, cur_pts)
            if m is not None and int(ok.sum()) >= min_valid:
                med = m
                cur_pts = next_pts[ok].reshape(-1, 1, 2)
                reseed_streak = 0
            else:
                pred = _predict_next(raw, last_disp, cam_pos, p)
                new_pts = _seed_points(curr_gray, _center_wh_to_xyxy(pred, wh))
                if new_pts is not None and reseed_streak < 2:
                    cur_pts = new_pts
                    prev_center = raw[max(raw)]
                    raw[p] = pred.copy()
                    last_disp = pred - prev_center
                    reseed_streak += 1
                    measured_pred = pred
        else:
            flow = _compute_dense_flow(prev_gray, curr_gray, prev_name, curr_name,
                                       method=flow_method)
            med = _box_median_flow(flow, _center_wh_to_xyxy(raw[max(raw)], wh))

        kfx.predict(); kfy.predict()
        if med is not None:
            res = _step_residual(med, cam_pos, p)
            junk = res if res is not None else med
            if float(np.hypot(junk[0], junk[1])) > float(max(wh)):
                break  # jumped to junk
            med = _clamp_step(med, cam_pos, p, max_step)
            raw[p] = raw[max(raw)] + med
            last_disp = med
            z = to_state(p, raw[p])
            kfx.update(float(z[0]))
            kfy.update(float(z[1]))
            conf = 0.6
        elif measured_pred is not None:
            z = to_state(p, measured_pred)
            kfx.update(float(z[0]))
            kfy.update(float(z[1]))
            conf = 0.5
        else:
            conf = 0.45
        st = np.array([kfx.pos, kfy.pos])
        result[p] = {"xyxy": _center_wh_to_xyxy(from_state(p, st), wh),
                     "source": "kalman", "conf": conf}
        prev_gray = curr_gray
        prev_name = curr_name
    return result


def interpolate_span(image_folder, frames, a, b, box_a, box_b,
                     min_valid=2, max_step_frac=0.3, flow_method="klt",
                     camera_model="none", gf_quality=0.02, ransac_px=2.0):
    """Anchored optical flow for one matched pair between keyframe a and b.

    Returns {p: {"xyxy": [x1,y1,x2,y2], "source": str, "conf": float}} for
    interior frames p in (a, b).

    Per-step displacement is clamped to ``max_step_frac * min(box_a w,h)`` so
    unreliable features cannot run away; the endpoint anchor then guarantees
    the trajectory starts exactly at box_a and lands exactly at box_b. When the
    flow has nothing real to follow, the clamped raw trajectory stays near
    box_a and the anchor term reduces the result to ~linear interpolation.

    With ``camera_model="global"`` a per-frame camera transform is estimated
    for the span and the anchor correction is applied to the *residual*
    (center minus camera-predicted position) instead of the absolute
    trajectory, so non-linear camera motion is absorbed by the transform chain
    and lost KLT tracks are re-seeded at the camera-predicted position.

    Args:
        flow_method: "klt" (sparse Lucas-Kanade), "dis" (dense inverse
            search — handles large displacements from handheld camera shake)
            or "farneback" (dense, more accurate but slower).
        camera_model: "none" (classic absolute flow) or "global" (per-frame
            RANSAC similarity transform).
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return _linear_span(a, b, box_a, box_b)

    center_a = box_a["center"].copy()
    center_b = box_b["center"].copy()
    wh_a = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    wh_b = np.array([box_b["xywh"][2], box_b["xywh"][3]])
    max_step = max_step_frac * float(min(wh_a[0], wh_a[1]))

    C = None
    if camera_model == "global":
        Hs, _cam_q = _build_camera_model(
            image_folder, frames, a, b, [box_a["xyxy"], box_b["xyxy"]],
            gf_quality=gf_quality, ransac_px=ransac_px)
        if Hs is not None:
            C = _compose(Hs, a)

    def cam_pos(t):
        if C is None or t not in C:
            return None
        return _apply_H(C[t], center_a)

    def to_res(t, center):
        c = cam_pos(t)
        return center - c if c is not None else center

    # KLT needs seed points; dense flow doesn't.
    cur_pts = None
    if flow_method == "klt":
        cur_pts = _seed_points(gray_a, box_a["xyxy"])
        if cur_pts is None:
            return _linear_span(a, b, box_a, box_b)

    raw_centers = {a: center_a.copy()}
    prev_gray = gray_a
    prev_name = frames[a]
    lost = False
    last_disp = None
    reseed_streak = 0
    for p in range(a + 1, b + 1):
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            lost = True
            break
        curr_name = frames[p]

        if flow_method == "klt":
            next_pts, ok, med = _lk_step(prev_gray, curr_gray, cur_pts)
            if med is not None and int(ok.sum()) >= min_valid:
                cur_pts = next_pts[ok].reshape(-1, 1, 2)
                reseed_streak = 0
            else:
                # Track failed: re-seed at the predicted position (camera
                # model preferred) so a single bad frame does not kill the span.
                pred = _predict_next(raw_centers, last_disp, cam_pos, p)
                new_pts = _seed_points(curr_gray, _center_wh_to_xyxy(pred, wh_a))
                if new_pts is not None and reseed_streak < 2:
                    cur_pts = new_pts
                    prev_center = raw_centers[max(raw_centers)]
                    raw_centers[p] = pred.copy()
                    last_disp = pred - prev_center
                    reseed_streak += 1
                    prev_gray = curr_gray
                    prev_name = curr_name
                    continue
                lost = True
                break
        else:
            flow = _compute_dense_flow(prev_gray, curr_gray, prev_name, curr_name,
                                       method=flow_method)
            med = _box_median_flow(flow, box_a["xyxy"] if p == a + 1 else
                                   _center_wh_to_xyxy(
                                        raw_centers[list(raw_centers)[-1]], wh_a))
            if med is None:
                lost = True
                break

        # Clamp the step (residual vs camera when a camera model is set).
        med = _clamp_step(med, cam_pos, p, max_step)
        # Accumulate the per-step median displacement into the raw trajectory.
        prev_center = raw_centers[list(raw_centers)[-1]]
        raw_centers[p] = prev_center + med
        last_disp = med
        prev_gray = curr_gray
        prev_name = curr_name

    n_interior = max(1, b - a - 1)
    n_meas = sum(1 for p in range(a + 1, b) if p in raw_centers)
    q_meas = n_meas / n_interior

    def emit_frame(p, center, source, conf):
        t = (p - a) / (b - a) if b > a else 0.0
        wh = wh_a + (wh_b - wh_a) * t
        return {"xyxy": _center_wh_to_xyxy(center, wh), "source": source,
                "conf": round(float(conf), 3)}

    result = {}
    if lost:
        # Use the raw flow up to where the track was lost, then linearly
        # bridge (in residual space) from the last raw center to center_b.
        last_p = max(raw_centers)
        r_last = to_res(last_p, raw_centers[last_p])
        r_b = to_res(b, center_b)
        for p in range(a + 1, b):
            if p <= last_p:
                r = to_res(p, raw_centers[p])
                conf = 0.9 * q_meas
            else:
                t = (p - last_p) / (b - last_p) if b > last_p else 1.0
                r = r_last + t * (r_b - r_last)
                conf = 0.35 * q_meas
            center = r if C is None else cam_pos(p) + r
            result[p] = emit_frame(p, center, "flow", conf)
        return result

    # Not lost: anchored correction using raw endpoint at b (residual space).
    r_b_raw = to_res(b, raw_centers[b])
    r_b_tgt = to_res(b, center_b)
    for p in range(a + 1, b):
        t = (p - a) / (b - a) if b > a else 0.0
        r = to_res(p, raw_centers[p])
        r = r + t * (r_b_tgt - r_b_raw)
        center = r if C is None else cam_pos(p) + r
        result[p] = emit_frame(p, center, "flow", 0.9 * q_meas)
    return result


def track_forward(image_folder, frames, a, b, box_a,
                    min_valid=2, max_step_frac=0.3, flow_method="klt",
                    camera_model="none", gf_quality=0.02, ransac_px=2.0):
    """Forward optical flow for an unmatched-at-a box.

    Emits boxes until the track is lost or the next keyframe is reached. Size
    stays at box_a's; per-step displacement is clamped to bound drift. With
    ``camera_model="global"`` lost KLT tracks are re-seeded at the
    camera-predicted position instead of stopping.

    Returns {p: {"xyxy": [x1,y1,x2,y2], "source": str, "conf": float}} for
    p in (a, b).
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return {}

    center = box_a["center"].copy()
    wh = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    max_step = max_step_frac * float(min(wh[0], wh[1]))

    C = None
    if camera_model == "global":
        Hs, _q = _build_camera_model(image_folder, frames, a, b,
                                     [box_a["xyxy"]],
                                     gf_quality=gf_quality, ransac_px=ransac_px)
        if Hs is not None:
            C = _compose(Hs, a)

    def cam_pos(t):
        if C is None or t not in C:
            return None
        return _apply_H(C[t], center)

    result = {}
    raw = {a: center.copy()}
    prev_gray = gray_a
    prev_name = frames[a]

    cur_pts = None
    if flow_method == "klt":
        cur_pts = _seed_points(gray_a, box_a["xyxy"])
        if cur_pts is None:
            return {}

    last_disp = None
    reseed_streak = 0
    for p in range(a + 1, b):  # stop before the next keyframe (it's reviewed)
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            break
        curr_name = frames[p]

        if flow_method == "klt":
            next_pts, ok, med = _lk_step(prev_gray, curr_gray, cur_pts)
            if med is not None and int(ok.sum()) >= min_valid:
                cur_pts = next_pts[ok].reshape(-1, 1, 2)
                reseed_streak = 0
            else:
                pred = _predict_next(raw, last_disp, cam_pos, p)
                new_pts = _seed_points(curr_gray, _center_wh_to_xyxy(pred, wh))
                if new_pts is not None and reseed_streak < 2:
                    cur_pts = new_pts
                    prev_center = raw[max(raw)]
                    raw[p] = pred.copy()
                    last_disp = pred - prev_center
                    reseed_streak += 1
                    result[p] = {"xyxy": _center_wh_to_xyxy(pred, wh),
                                 "source": "flow", "conf": 0.5}
                    prev_gray = curr_gray
                    prev_name = curr_name
                    continue
                break
        else:
            flow = _compute_dense_flow(prev_gray, curr_gray, prev_name, curr_name,
                                       method=flow_method)
            med = _box_median_flow(flow, _center_wh_to_xyxy(raw[max(raw)], wh))
            if med is None:
                break

        res = _step_residual(med, cam_pos, p)
        junk = res if res is not None else med
        # A step larger than the whole box means the track jumped to junk: stop.
        if float(np.hypot(junk[0], junk[1])) > float(max(wh)):
            break
        med = _clamp_step(med, cam_pos, p, max_step)
        center_p = raw[max(raw)] + med
        raw[p] = center_p
        last_disp = med
        prev_gray = curr_gray
        prev_name = curr_name
        result[p] = {"xyxy": _center_wh_to_xyxy(center_p, wh),
                     "source": "flow", "conf": 0.7}
    return result


def track_backward(image_folder, frames, a, b, box_b,
                   min_valid=2, max_step_frac=0.3, flow_method="klt",
                   camera_model="none", gf_quality=0.02, ransac_px=2.0):
    """Backward optical flow for a box that first appears at keyframe b.

    Runs :func:`track_forward` on the reversed frame sequence so a new object
    is tracked backwards from keyframe b toward keyframe a (emits boxes for
    p in (a, b)).
    """
    rev = frames[:b + 1][::-1]
    res = track_forward(image_folder, rev, 0, b - a, box_b,
                        min_valid=min_valid, max_step_frac=max_step_frac,
                        flow_method=flow_method, camera_model=camera_model,
                        gf_quality=gf_quality, ransac_px=ransac_px)
    return {b - p: v for p, v in res.items()}


def track_backward_kalman(image_folder, frames, a, b, box_b,
                          min_valid=2, max_step_frac=0.3, flow_method="dis",
                          kf_q=1.0, kf_r=4.0, camera_model="none",
                          gf_quality=0.02, ransac_px=2.0):
    """Kalman-filtered :func:`track_backward` (see its docstring)."""
    rev = frames[:b + 1][::-1]
    res = track_forward_kalman(image_folder, rev, 0, b - a, box_b,
                               min_valid=min_valid, max_step_frac=max_step_frac,
                               flow_method=flow_method, kf_q=kf_q, kf_r=kf_r,
                               camera_model=camera_model,
                               gf_quality=gf_quality, ransac_px=ransac_px)
    return {b - p: v for p, v in res.items()}


def _linear_span(a, b, box_a, box_b):
    """Pure linear interpolation fallback (no flow)."""
    result = {}
    ca, cb = box_a["center"], box_b["center"]
    wa = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    wb = np.array([box_b["xywh"][2], box_b["xywh"][3]])
    for p in range(a + 1, b):
        t = (p - a) / (b - a) if b > a else 0.0
        center = ca + t * (cb - ca)
        wh = wa + t * (wb - wa)
        result[p] = {"xyxy": _center_wh_to_xyxy(center, wh),
                     "source": "linear", "conf": 0.15}
    return result


def _center_wh_to_xyxy(center, wh):
    cx, cy = float(center[0]), float(center[1])
    w, h = float(wh[0]), float(wh[1])
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


_gray_cache = {}


def _read_gray(image_folder, name):
    """Read a frame as grayscale (small LRU cache keyed by name)."""
    if name in _gray_cache:
        return _gray_cache[name]
    path = Path(image_folder) / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(_gray_cache) > 64:
        _gray_cache.clear()
    _gray_cache[name] = gray
    return gray


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Propagate reviewed keyframe boxes to every frame with "
                    "anchored KLT optical-flow tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--keyframes-coco", required=True,
                        help="Reviewed keyframe COCO (from 08_click_review_coco.py).")
    parser.add_argument("--manifest", required=True,
                        help="keyframe_manifest.json (from 12_extract_keyframes.py).")
    parser.add_argument("--image-folder", required=True,
                        help="Full frame folder (all frames, chronological order).")
    parser.add_argument("--output-coco", required=True,
                        help="Output COCO file with boxes for every frame.")
    parser.add_argument("--vis-output",
                        help="Optional dir for annotated frames + tracking_result.mp4.")
    parser.add_argument("--match-max-dist", type=float, default=0.2,
                        help="Max center distance to match objects between "
                             "adjacent keyframes, as a fraction of the smaller "
                             "frame dimension (default: 0.2).")
    parser.add_argument("--tail-fill", choices=["hold", "empty"], default="hold",
                        help="How to handle frames after the last keyframe "
                             "(default: hold last keyframe boxes).")
    parser.add_argument("--min-track-points", type=int, default=8,
                        help="Min goodFeatures to seed a KLT track (default: 8).")
    parser.add_argument("--max-step-frac", type=float, default=0.3,
                        help="Max per-frame box displacement as a fraction of "
                             "the smaller box dimension; clamps flow drift "
                             "(default: 0.3).")
    parser.add_argument("--flow-method", choices=["klt", "dis", "farneback"],
                        default="dis",
                        help="Optical flow method for interpolation: 'klt' "
                             "(sparse Lucas-Kanade, fast but fails on large "
                             "displacements), 'dis' (dense inverse search, "
                             "handles handheld camera shake, default), "
                             "'farneback' (dense, most accurate but slowest).")
    parser.add_argument("--interp-method", choices=["flow", "kalman"],
                        default="flow",
                        help="Interpolation strategy: 'flow' (raw optical-flow "
                             "accumulator with endpoint anchoring, default) or "
                             "'kalman' (constant-velocity Kalman filter fed with "
                             "the same flow measurements; smooths jitter and "
                             "predicts through short flow dropouts).")
    parser.add_argument("--kf-q", type=float, default=1.0,
                        help="Kalman process-noise scale (default: 1.0). Only "
                             "used with --interp-method kalman.")
    parser.add_argument("--kf-r", type=float, default=4.0,
                        help="Kalman measurement-noise scale (default: 4.0). "
                             "Only used with --interp-method kalman.")
    parser.add_argument("--camera-model", choices=["none", "global"],
                        default=None,
                        help="Camera-motion handling: 'none' (classic absolute "
                             "flow + linear anchor, default) or 'global' "
                             "(per-frame RANSAC similarity transform; "
                             "tracking/anchoring runs on the residual relative "
                             "to the camera-predicted position, absorbing "
                             "non-linear camera shake). Accuracy test on the "
                             "re-reviewed MTR 4k frames: no net gain on this "
                             "fisheye camera (18.5px vs 16.4px median center "
                             "error), so it stays off by default; use 'global' "
                             "for pinhole cameras with strong shake.")
    parser.add_argument("--gf-quality", type=float, default=0.02,
                        help="goodFeaturesToTrack quality level for the global "
                             "camera-transform estimate (default: 0.02).")
    parser.add_argument("--ransac-reproj-px", type=float, default=2.0,
                        help="RANSAC reprojection threshold (px) for the global "
                             "camera-transform estimate (default: 2.0).")
    args = parser.parse_args()

    camera_model = args.camera_model if args.camera_model is not None else "none"

    image_folder = Path(args.image_folder)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    total = manifest["total_frames"]
    keyframe_idxs = [kf["frame_idx"] for kf in manifest["keyframes"]]
    frame_idx_by_name = {kf["file_name"]: kf["frame_idx"]
                         for kf in manifest["keyframes"]}

    categories, boxes_by_frame, size = load_keyframe_boxes(
        args.keyframes_coco, frame_idx_by_name)

    # Frame size: prefer COCO image entry, else read the first frame.
    if size is None:
        first = cv2.imread(str(image_folder / frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Could not read frame {frames[0]} for size")
        size = (int(first.shape[1]), int(first.shape[0]))
    W, H = size
    max_dist = args.match_max_dist * min(W, H)

    # Union-find across keyframe annotation ids for persistent track ids.
    uf = UnionFind()
    # Pre-register every reviewed box.
    for fidx, boxes in boxes_by_frame.items():
        for b in boxes:
            uf.find(b["ann_id"])

    # Span list: consecutive keyframe pairs.
    spans = list(zip(keyframe_idxs[:-1], keyframe_idxs[1:]))
    # For each span, compute matches and union the ann_ids.
    span_matches = {}  # (a,b) -> list of (idx_a, idx_b, dist)
    span_interp = {}   # (a,b) -> {(box_a_idx): {p: entry}} for matched pairs
    span_forward = {}  # (a,b) -> {(box_a_idx): {p: entry}} for unmatched-at-a
    span_backward = {} # (a,b) -> {(box_b_idx): {p: entry}} for unmatched-at-b

    cam_kwargs = dict(camera_model=camera_model, gf_quality=args.gf_quality,
                      ransac_px=args.ransac_reproj_px)

    for (a, b) in spans:
        ba = boxes_by_frame.get(a, [])
        bb = boxes_by_frame.get(b, [])
        matches, un_a, un_b = match_pairs(ba, bb, max_dist)
        span_matches[(a, b)] = matches
        for i, j, _ in matches:
            uf.union(ba[i]["ann_id"], bb[j]["ann_id"])
        # Interpolate matched pairs.
        interp = {}
        for i, j, _ in matches:
            if args.interp_method == "kalman":
                interp[i] = interpolate_span_kalman(
                    image_folder, frames, a, b, ba[i], bb[j],
                    max_step_frac=args.max_step_frac,
                    flow_method=args.flow_method,
                    kf_q=args.kf_q, kf_r=args.kf_r, **cam_kwargs)
            else:
                interp[i] = interpolate_span(image_folder, frames, a, b, ba[i], bb[j],
                                             max_step_frac=args.max_step_frac,
                                             flow_method=args.flow_method,
                                             **cam_kwargs)
        span_interp[(a, b)] = interp
        # Forward-track unmatched-at-a boxes.
        fwd = {}
        for i in un_a:
            if args.interp_method == "kalman":
                fwd[i] = track_forward_kalman(
                    image_folder, frames, a, b, ba[i],
                    max_step_frac=args.max_step_frac,
                    flow_method=args.flow_method,
                    kf_q=args.kf_q, kf_r=args.kf_r, **cam_kwargs)
            else:
                fwd[i] = track_forward(image_folder, frames, a, b, ba[i],
                                       max_step_frac=args.max_step_frac,
                                       flow_method=args.flow_method,
                                       **cam_kwargs)
        span_forward[(a, b)] = fwd
        # Back-track unmatched-at-b boxes (objects appearing at keyframe b).
        back = {}
        for j in un_b:
            if args.interp_method == "kalman":
                back[j] = track_backward_kalman(
                    image_folder, frames, a, b, bb[j],
                    max_step_frac=args.max_step_frac,
                    flow_method=args.flow_method,
                    kf_q=args.kf_q, kf_r=args.kf_r, **cam_kwargs)
            else:
                back[j] = track_backward(image_folder, frames, a, b, bb[j],
                                         max_step_frac=args.max_step_frac,
                                         flow_method=args.flow_method,
                                         **cam_kwargs)
        span_backward[(a, b)] = back

    # Assign track ids (0..N-1) in first-seen order across all reviewed boxes.
    all_ann_ids = [b["ann_id"] for fidx in sorted(boxes_by_frame)
                   for b in boxes_by_frame[fidx]]
    track_ids = uf.label(all_ann_ids)

    # Build the output COCO over every frame.
    images_out = []
    annotations_out = []
    ann_id = 1
    for p in range(total):
        images_out.append({
            "id": p + 1,
            "file_name": frames[p],
            "width": W,
            "height": H,
        })

    def emit(p, xyxy, category_id, tid, source, confidence):
        nonlocal ann_id
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
        y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        annotations_out.append({
            "id": ann_id,
            "image_id": p + 1,
            "category_id": int(category_id),
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
            "track_id": int(tid),
            "source": str(source),
            "confidence": round(float(confidence), 3),
        })
        ann_id += 1

    # Emit boxes per frame.
    keyframe_list = sorted(keyframe_idxs)
    keyframe_set = set(keyframe_list)
    for p in range(total):
        if p in keyframe_set:
            # Reviewed boxes verbatim.
            for box in boxes_by_frame.get(p, []):
                tid = track_ids[box["ann_id"]]
                emit(p, box["xyxy"], box["category_id"], tid, "keyframe", 1.0)
            continue
        # Find surrounding keyframes (bisect: keyframe_list is sorted).
        i = bisect.bisect_left(keyframe_list, p)
        next_kf = keyframe_list[i] if i < len(keyframe_list) else None
        prev_kf = keyframe_list[i - 1] if i > 0 else None
        if prev_kf is not None and next_kf is not None:
            ba = boxes_by_frame.get(prev_kf, [])
            bb = boxes_by_frame.get(next_kf, [])
            interp = span_interp.get((prev_kf, next_kf), {})
            fwd = span_forward.get((prev_kf, next_kf), {})
            back = span_backward.get((prev_kf, next_kf), {})
            for i2, boxes_p in interp.items():
                if p in boxes_p:
                    e = boxes_p[p]
                    emit(p, e["xyxy"], ba[i2]["category_id"],
                         track_ids[ba[i2]["ann_id"]], e["source"], e["conf"])
            for i2, boxes_p in fwd.items():
                if p in boxes_p:
                    e = boxes_p[p]
                    emit(p, e["xyxy"], ba[i2]["category_id"],
                         track_ids[ba[i2]["ann_id"]], e["source"], e["conf"])
            for j2, boxes_p in back.items():
                if p in boxes_p:
                    e = boxes_p[p]
                    emit(p, e["xyxy"], bb[j2]["category_id"],
                         track_ids[bb[j2]["ann_id"]], e["source"], e["conf"])
        elif prev_kf is not None and next_kf is None:
            # Tail after the last keyframe.
            if args.tail_fill == "hold":
                for box in boxes_by_frame.get(prev_kf, []):
                    tid = track_ids[box["ann_id"]]
                    emit(p, box["xyxy"], box["category_id"], tid, "hold", 0.4)
        # Frames before the first keyframe (prev_kf is None) stay empty.

    out = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories,
    }
    out_path = Path(args.output_coco)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote COCO for {total} frames, {len(annotations_out)} boxes -> {out_path}")
    print(f"  Keyframes: {len(keyframe_idxs)}, spans: {len(spans)}")
    if camera_model == "global" and _cam_stats["attempts"]:
        q = (float(np.mean(_cam_stats["qualities"]))
             if _cam_stats["qualities"] else 0.0)
        print(f"  Camera model: {_cam_stats['ok']}/{_cam_stats['attempts']} spans "
              f"estimated, mean inlier ratio {q:.2f}")

    if args.vis_output:
        vis_dir = Path(args.vis_output)
        vis_dir.mkdir(parents=True, exist_ok=True)
        cat_colors = {}
        for cat in categories:
            cat_colors[cat["id"]] = tuple(int(c) for c in np.random.randint(0, 255, 3))
        anns_by_img = {}
        for a in annotations_out:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        vis_files = []
        for p in range(total):
            img = cv2.imread(str(image_folder / frames[p]))
            if img is None:
                continue
            for a in anns_by_img.get(p + 1, []):
                x, y, w, h = a["bbox"]
                color = cat_colors.get(a["category_id"], (0, 255, 255))
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                              color, 2)
                cv2.putText(img, f"id{a['track_id']} c{a.get('confidence', 1.0):.2f}",
                            (int(x), max(0, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            vp = vis_dir / frames[p]
            cv2.imwrite(str(vp), img)
            vis_files.append(str(vp))
        if vis_files:
            create_tracking_video(vis_dir, [Path(p) for p in vis_files], fps=10)
            print(f"  Vis -> {vis_dir} ({len(vis_files)} frames)")


if __name__ == "__main__":
    main()
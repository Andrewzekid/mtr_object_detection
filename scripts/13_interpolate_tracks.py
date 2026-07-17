#!/usr/bin/env python3
"""
Propagate reviewed keyframe boxes to every frame with KLT optical-flow tracking.

This is the keyframe pipeline's fill-in step. `12_extract_keyframes.py` selects
sparse keyframes, `07_run_qwen.py` seeds boxes on them, and `08_click_review_coco.py`
is used to review the keyframe boxes. This script reads that reviewed keyframe
COCO plus the manifest and produces a COCO file with boxes for **every** frame,
which is then handed back to `08_click_review_coco.py` for a final full review.

Method (anchored KLT):
    For each adjacent keyframe pair (a, b) the reviewed boxes are matched by
    class + nearest center. For each matched pair, Lucas-Kanade optical flow
    (`goodFeaturesToTrack` inside the box, `calcOpticalFlowPyrLK` frame-by-frame)
    gives a raw trajectory of the box center. A linear correction is applied so
    the trajectory starts exactly at keyframe a's reviewed box and lands exactly
    at keyframe b's reviewed box; the box size is linearly interpolated a->b.
    This keeps the flow's motion shape while guaranteeing the reviewed endpoints
    are respected (no drift accumulation across the whole video).

USAGE:
    python scripts/13_interpolate_tracks.py \
        --keyframes-coco output/MTR_keyframes/reviewed/coco_reviewed.json \
        --manifest Datasets/MTR/MTR_keyframes/keyframe_manifest.json \
        --image-folder Datasets/MTR/rosbags/MTR_metacam_right \
        --output-coco output/MTR_full/interpolated.coco.json \
        --vis-output output/MTR_full/vis

OUTPUT:
    - <output-coco>        COCO with boxes for every frame (keyframes reviewed,
                           in-between frames interpolated)
    - <vis-output>/*.jpg   annotated frames (optional)
    - <vis-output>/tracking_result.mp4   summary video (optional)
"""

import argparse
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
# KLT optical flow
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
    """One LK step; returns (next_points, valid_mask, fwd_disp_xy or None)."""
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
    disp = (next_pts[ok].reshape(-1, 2) - points[ok].reshape(-1, 2))
    med = np.median(disp, axis=0)
    return next_pts, ok, med


def interpolate_span(image_folder, frames, a, b, box_a, box_b,
                     min_valid=2, max_step_frac=0.3):
    """Anchored KLT for one matched pair between keyframe a and keyframe b.

    Returns {p: [x1,y1,x2,y2]} for interior frames p in (a, b).

    Per-step displacement is clamped to ``max_step_frac * min(box_a w,h)`` so
    unreliable features cannot run away; the endpoint anchor then guarantees
    the trajectory starts exactly at box_a and lands exactly at box_b. When the
    flow has nothing real to follow, the clamped raw trajectory stays near
    box_a and the anchor term reduces the result to ~linear interpolation.
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return {}

    pts = _seed_points(gray_a, box_a["xyxy"])
    if pts is None:
        # No features: fall back to pure linear interpolation.
        return _linear_span(a, b, box_a, box_b)

    center_a = box_a["center"].copy()
    center_b = box_b["center"].copy()
    wh_a = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    wh_b = np.array([box_b["xywh"][2], box_b["xywh"][3]])
    max_step = max_step_frac * float(min(wh_a[0], wh_a[1]))

    raw_centers = {a: center_a.copy()}
    prev_gray = gray_a
    cur_pts = pts
    lost = False
    for p in range(a + 1, b + 1):
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            lost = True
            break
        next_pts, ok, med = _lk_step(prev_gray, curr_gray, cur_pts)
        if med is None or int(ok.sum()) < min_valid:
            lost = True
            break
        # Clamp per-step displacement to keep unreliable features bounded.
        mag = float(np.hypot(med[0], med[1]))
        if mag > max_step and mag > 0:
            med = med * (max_step / mag)
        # Accumulate the per-step median displacement into the raw trajectory.
        prev_center = raw_centers[list(raw_centers)[-1]]
        raw_centers[p] = prev_center + med
        cur_pts = next_pts[ok].reshape(-1, 1, 2) if int(ok.sum()) >= 1 else cur_pts
        prev_gray = curr_gray

    result = {}
    if lost:
        # Use the raw flow up to where the track was lost, then linearly
        # bridge from the last raw center to the reviewed center_b.
        last_p = max(raw_centers)
        last_raw = raw_centers[last_p]
        for p in range(a + 1, b):
            if p <= last_p:
                center = raw_centers[p]
            else:
                t = (p - last_p) / (b - last_p) if b > last_p else 1.0
                center = last_raw + t * (center_b - last_raw)
            wh = wh_a + (wh_b - wh_a) * ((p - a) / (b - a) if b > a else 0.0)
            result[p] = _center_wh_to_xyxy(center, wh)
        return result

    # Not lost: anchored correction using raw endpoint at b.
    raw_b = raw_centers.get(b, center_a)
    for p in range(a + 1, b):
        t = (p - a) / (b - a) if b > a else 0.0
        raw = raw_centers.get(p, center_a + t * (raw_b - center_a))
        center = raw + t * (center_b - raw_b)
        wh = wh_a + (wh_b - wh_a) * t
        result[p] = _center_wh_to_xyxy(center, wh)
    return result


def track_forward(image_folder, frames, a, b, box_a,
                  min_valid=2, max_step_frac=0.3):
    """Forward KLT for an unmatched-at-a box. Returns {p: [x1,y1,x2,y2]}.

    Emits boxes until the track is lost or the next keyframe is reached. Size
    stays at box_a's; per-step displacement is clamped to bound drift.
    """
    gray_a = _read_gray(image_folder, frames[a])
    if gray_a is None:
        return {}
    pts = _seed_points(gray_a, box_a["xyxy"])
    if pts is None:
        return {}
    center = box_a["center"].copy()
    wh = np.array([box_a["xywh"][2], box_a["xywh"][3]])
    max_step = max_step_frac * float(min(wh[0], wh[1]))
    result = {}
    prev_gray = gray_a
    cur_pts = pts
    for p in range(a + 1, b):  # stop before the next keyframe (it's reviewed)
        curr_gray = _read_gray(image_folder, frames[p])
        if curr_gray is None:
            break
        next_pts, ok, med = _lk_step(prev_gray, curr_gray, cur_pts)
        if med is None or int(ok.sum()) < min_valid:
            break
        mag = float(np.hypot(med[0], med[1]))
        # A step larger than the whole box means the track jumped to junk: stop.
        if mag > float(max(wh)):
            break
        if mag > max_step and mag > 0:
            med = med * (max_step / mag)
        center = center + med
        cur_pts = next_pts[ok].reshape(-1, 1, 2) if int(ok.sum()) >= 1 else cur_pts
        prev_gray = curr_gray
        result[p] = _center_wh_to_xyxy(center, wh)
    return result


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
        result[p] = _center_wh_to_xyxy(center, wh)
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
                             "the smaller box dimension; clamps KLT drift "
                             "(default: 0.3).")
    parser.add_argument("--device", default="cuda",
                        help="Unused placeholder (KLT runs on CPU). Kept for "
                             "pipeline consistency with other scripts.")
    args = parser.parse_args()

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
    span_interp = {}   # (a,b) -> {(box_a_idx): {p: xyxy}} for matched pairs
    span_forward = {}  # (a,b) -> {(box_a_idx): {p: xyxy}} for unmatched-at-a

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
            interp[i] = interpolate_span(image_folder, frames, a, b, ba[i], bb[j],
                                         max_step_frac=args.max_step_frac)
        span_interp[(a, b)] = interp
        # Forward-track unmatched-at-a boxes.
        fwd = {}
        for i in un_a:
            fwd[i] = track_forward(image_folder, frames, a, b, ba[i],
                                   max_step_frac=args.max_step_frac)
        span_forward[(a, b)] = fwd

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

    def emit(p, xyxy, category_id, tid):
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
        })
        ann_id += 1

    # Emit boxes per frame.
    keyframe_set = set(keyframe_idxs)
    for p in range(total):
        if p in keyframe_set:
            # Reviewed boxes verbatim.
            for b in boxes_by_frame.get(p, []):
                tid = track_ids[b["ann_id"]]
                emit(p, b["xyxy"], b["category_id"], tid)
            continue
        # Find surrounding keyframes.
        prev_kf = max((k for k in keyframe_idxs if k <= p), default=None)
        next_kf = min((k for k in keyframe_idxs if k > p), default=None)
        if prev_kf is not None and next_kf is not None:
            ba = boxes_by_frame.get(prev_kf, [])
            interp = span_interp.get((prev_kf, next_kf), {})
            fwd = span_forward.get((prev_kf, next_kf), {})
            for i, boxes_p in interp.items():
                if p in boxes_p:
                    tid = track_ids[ba[i]["ann_id"]]
                    emit(p, boxes_p[p], ba[i]["category_id"], tid)
            for i, boxes_p in fwd.items():
                if p in boxes_p:
                    tid = track_ids[ba[i]["ann_id"]]
                    emit(p, boxes_p[p], ba[i]["category_id"], tid)
        elif prev_kf is not None and next_kf is None:
            # Tail after the last keyframe.
            if args.tail_fill == "hold":
                for b in boxes_by_frame.get(prev_kf, []):
                    tid = track_ids[b["ann_id"]]
                    emit(p, b["xyxy"], b["category_id"], tid)
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
                cv2.putText(img, f"id{a['track_id']}", (int(x), max(0, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            vp = vis_dir / frames[p]
            cv2.imwrite(str(vp), img)
            vis_files.append(str(vp))
        if vis_files:
            create_tracking_video(vis_dir, [Path(p) for p in vis_files], fps=10)
            print(f"  Vis -> {vis_dir} ({len(vis_files)} frames)")


if __name__ == "__main__":
    main()
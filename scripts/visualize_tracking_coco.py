#!/usr/bin/env python3
"""Render COCO tracking JSON (11_run_tracking.py output) over source images.

Draws per-frame boxes/masks with track-id labels, writes tracked_*.jpg
frames and a summary mp4. Works for both detection and segmentation
results (segmentation polygons drawn when present).

USAGE:
    python scripts/visualize_tracking_coco.py \
        --json output/tracking/HKU_GF/run6/left.json \
        --images Datasets/HKU_GH/rosbags/2026-08-25_22-34-54/camera/undistort/left \
        --output output/tracking/HKU_GF/run6/left_vis
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# BGR palette cycled by track id (stable colors per track across frames)
PALETTE = [
    (56, 56, 255), (151, 157, 255), (56, 200, 255), (240, 147, 96),
    (22, 150, 219), (86, 63, 232), (160, 30, 230), (140, 180, 46),
    (44, 156, 56), (36, 86, 224), (0, 156, 255), (151, 90, 222),
    (128, 128, 0), (60, 228, 255), (30, 32, 240), (240, 60, 60),
]


def track_color(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]


def load_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    per_frame = {}
    for ann in data.get("annotations", []):
        per_frame.setdefault(ann["image_id"], []).append(ann)
    return images, categories, per_frame


def draw_polygon_mask(img, seg, color, alpha=0.35):
    pts = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2).astype(np.int32)
    if len(pts) < 3:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)


def draw_frame(img, anns, categories, show_masks=True, show_labels=True,
               thickness=2):
    for ann in anns:
        tid = ann.get("track_id", "?")
        color = track_color(tid)
        x, y, w, h = ann["bbox"]
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        if show_masks and ann.get("segmentation"):
            draw_polygon_mask(img, ann["segmentation"], color)
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
        if show_labels:
            cat = categories.get(ann.get("category_id"), "")
            label = f"id{tid} {cat}"
            if "confidence" in ann:
                label += f" {ann['confidence']:.2f}"
            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = p1[1] - 6 if p1[1] - 6 > th else p1[1] + th + 6
            cv2.rectangle(img, (p1[0], ty - th - 4),
                          (p1[0] + tw + 4, ty + bl - th), color, -1)
            cv2.putText(img, label, (p1[0] + 2, ty - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(
        description="Render COCO tracking JSON over source images")
    ap.add_argument("--json", required=True, help="Tracking results.json")
    ap.add_argument("--images", required=True, help="Source image folder")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--fps", type=int, default=10, help="Output video fps")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Render only the first N frames")
    ap.add_argument("--no-masks", action="store_true",
                    help="Skip polygon fill, draw boxes/labels only")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip the summary mp4 (frames only)")
    ap.add_argument("--sample", type=int, default=1,
                    help="Render every Nth frame (frames+video)")
    args = ap.parse_args()

    images, categories, per_frame = load_results(args.json)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    img_files = sorted(p for p in Path(args.images).glob("*.jpg")
                       if p.suffix.lower() == ".jpg")
    if args.max_frames:
        img_files = img_files[:args.max_frames]

    name_to_meta = {img["file_name"]: (iid, img["width"], img["height"])
                    for iid, img in images.items()}

    writer = None
    n_drawn = 0
    for idx, path in enumerate(img_files):
        if idx % args.sample:
            continue
        meta = name_to_meta.get(path.name)
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"WARNING: could not read {path}")
            continue
        if meta:
            iid, w, h = meta
            draw_frame(frame, per_frame.get(iid, []), categories,
                       show_masks=not args.no_masks)
        cv2.imwrite(str(out / f"tracked_{path.name}"), frame)
        if not args.no_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out / "tracking_result.mp4"),
                                         fourcc, args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        n_drawn += 1
        if n_drawn % 200 == 0:
            print(f"  rendered {n_drawn}/{len(img_files)}")

    if writer is not None:
        writer.release()
    print(f"Done: {n_drawn} frames -> {out}"
          + ("" if args.no_video else " (+ tracking_result.mp4)"))


if __name__ == "__main__":
    main()
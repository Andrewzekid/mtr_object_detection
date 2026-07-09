#!/usr/bin/env python3
"""
Shared helpers for tracking scripts.

Utilities included:
- Ultralytics tracker YAML generation (BoT-SORT).
- COCO-style polygon/area helpers.
- IoU, mask-to-polygon conversions.
- Summary video creation.

These functions are intentionally stateless so they can be reused by multiple
script entry points without creating a tight coupling to any one pipeline.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def find_ultralytics_trackers_dir() -> Path:
    """Locate ultralytics/cfg/trackers inside the active Python env."""
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers"


def build_runtime_tracker_yaml(
    base_yaml: Path,
    tracker_type: str,
    with_cmc: bool,
    cmc_method: str,
    track_buffer: int,
    track_high_thresh: float,
    with_reid: bool,
    output_dir: Path,
) -> Path:
    """Merge CLI overrides into a copy of a tracker YAML for this run.

    Ultralytics resolves the ``tracker`` argument by name inside its trackers
    dir, so we write a single-file config (with the same fields) next to the
    output and pass its path to ``model.track(...)``.
    """
    import yaml

    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{tracker_type}_runtime.yaml"

    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["tracker_type"] = tracker_type
    cfg["track_high_thresh"] = float(track_high_thresh)
    cfg["track_buffer"] = int(track_buffer)
    cfg["with_reid"] = bool(with_reid)

    if with_cmc:
        cfg["gmc_method"] = cmc_method
    else:
        cfg["gmc_method"] = "none"

    with open(runtime_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return runtime_path


def bbox_iou(a, b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert a binary mask to a flattened COCO-style polygon [x1,y1,x2,y2,...].

    Returns the largest external contour. Empty mask -> empty list.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    poly = approx.reshape(-1, 2).astype(float)
    return [float(coord) for pt in poly for coord in pt]


def polygon_area(poly: list) -> float:
    """Shoelace area for a flat polygon list."""
    n = len(poly) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i = poly[2 * i]
        y_i = poly[2 * i + 1]
        x_j = poly[2 * j]
        y_j = poly[2 * j + 1]
        area += x_i * y_j - x_j * y_i
    return abs(area) / 2.0


def find_image_files(data_path: Path, extensions=None):
    """Return sorted list of image files in a directory."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([f for f in data_path.iterdir() if f.suffix.lower() in extensions])


def create_tracking_video(output_dir: Path, image_files, fps: int = 10):
    """Create a video from a list of image paths.

    image_files may be Path objects to either the original images or already
    annotated images. The caller decides which set to pass.
    """
    if not image_files:
        return

    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        return

    height, width = first_img.shape[:2]
    video_path = output_dir / "tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for image_path in image_files:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Tracking video saved to: {video_path}")


def write_coco_results(
    output_dir: Path,
    images: list,
    annotations: list,
    categories: list,
    description: str = "Tracking results",
):
    """Write a COCO-style JSON file to output_dir/results.json."""
    coco_output = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        import json

        json.dump(coco_output, f, indent=2)
    print(f"Tracking JSON saved to: {json_path}")

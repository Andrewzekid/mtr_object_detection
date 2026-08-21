#!/usr/bin/env python3
"""Combine per-image Qwen result JSONs into a single COCO labels file.

Reads a folder of ``<timestamp>_result.json`` files as produced by
``07_run_qwen.py`` (each holding ``image`` and ``parsed_output`` with
``label`` / ``bbox_2d`` [x1, y1, x2, y2]) and writes a
``labels_coco.json`` in the format ``gui/label_review`` expects:

- ``images``: id, file_name, width, height, timestamp_ns (from the file
  name), frame_idx (sorted order), side
- ``annotations``: id, image_id, category_id, bbox [x1, y1, x2, y2], area,
  iscrowd 0
- ``categories``: one per distinct label, ids starting at 0
- ``annotated_image_ids``: ids of images that have at least one annotation

Usage:
    python 17_qwen_results_to_coco.py \
        --qwen-results-dir output/<run>/qwen/left \
        --output output/<run>/qwen/left/labels_coco.json \
        [--side left]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _image_size(path: Path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.width, im.height
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ERROR: cannot read image {path}: {exc}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qwen-results-dir", required=True, type=Path,
                        help="Folder of per-image <timestamp>_result.json files")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output COCO json path (e.g. labels_coco.json)")
    parser.add_argument("--side", default="left", choices=["left", "right"],
                        help="Value written to each image's 'side' field")
    args = parser.parse_args()

    result_files = sorted(args.qwen_results_dir.glob("*_result.json"))
    if not result_files:
        print(f"No *_result.json files found in {args.qwen_results_dir}",
              file=sys.stderr)
        sys.exit(1)

    categories = []          # [{'id': int, 'name': str}]
    cat_id_by_name = {}
    images = []
    annotations = []
    next_image_id = 1
    next_ann_id = 1

    for frame_idx, rf in enumerate(result_files):
        try:
            data = json.loads(rf.read_text())
        except json.JSONDecodeError as exc:
            print(f"WARNING: skipping unreadable {rf.name}: {exc}",
                  file=sys.stderr)
            continue

        image_path = Path(data.get("image", ""))
        if not image_path.is_file():
            # Fall back to looking for the image next to the result file.
            stem = rf.name[: -len("_result.json")]
            candidate = args.qwen_results_dir / f"{stem}.jpg"
            if candidate.is_file():
                image_path = candidate
            else:
                print(f"WARNING: skipping {rf.name}: image not found "
                      f"({image_path})", file=sys.stderr)
                continue

        file_name = image_path.name
        m = re.match(r"^(\d+)", file_name)
        timestamp_ns = int(m.group(1)) if m else None
        width, height = _image_size(image_path)

        images.append({
            "id": next_image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "timestamp_ns": timestamp_ns,
            "log_time_ns": timestamp_ns,
            "frame_idx": frame_idx,
            "side": args.side,
        })

        for det in data.get("parsed_output") or []:
            label = str(det.get("label", "")).strip()
            bbox_2d = det.get("bbox_2d")
            if not label or not bbox_2d or len(bbox_2d) != 4:
                print(f"WARNING: skipping malformed detection in {rf.name}: "
                      f"{det!r}", file=sys.stderr)
                continue

            if label not in cat_id_by_name:
                cat_id_by_name[label] = len(categories)
                categories.append({"id": len(categories), "name": label})

            x1, y1, x2, y2 = (float(v) for v in bbox_2d)
            x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                print(f"WARNING: skipping empty bbox in {rf.name}: {bbox_2d}",
                      file=sys.stderr)
                continue

            annotations.append({
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": cat_id_by_name[label],
                "bbox": [x1, y1, x2, y2],
                "area": w * h,
                "iscrowd": 0,
            })
            next_ann_id += 1

        next_image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "annotated_image_ids": sorted({a["image_id"] for a in annotations}),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {args.output}: {len(images)} images, "
          f"{len(annotations)} annotations, {len(categories)} categories")


if __name__ == "__main__":
    main()

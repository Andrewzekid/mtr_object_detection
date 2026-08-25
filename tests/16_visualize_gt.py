#!/usr/bin/env python3
"""
Visualize ground truth bboxes from the eval dataset.

Saves annotated images to <eval_dir>/gt_vis/.

USAGE:
    # Per-image GT files
    python scripts/16_visualize_gt.py \
        --eval-dir Datasets/MTR/MTR_keyframes_eval \
        --target-class "Exit Sign"

    # COCO-format GT (only images present in eval_dir/images/)
    python scripts/16_visualize_gt.py \
        --eval-dir Datasets/MTR/MTR_keyframes_eval \
        --gt-coco output/MTR_4k/interpolated_all_coco.json \
        --target-class "Exit Sign"
"""

import argparse
import json
from pathlib import Path

import cv2

CLASS_COLORS = {
    "Exit Sign": (0, 255, 0),
    "Advertisement Board": (255, 0, 0),
    "Ceiling light": (0, 255, 255),
    "Map": (255, 0, 255),
    "Ticket Gate": (0, 165, 255),
    "TV": (255, 255, 0),
    "directional or overhead exit signage containing the standard Exit icon": (0, 200, 200),
}
DEFAULT_COLOR = (200, 200, 200)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--gt-coco", default=None,
                        help="Path to COCO-format GT JSON (overrides eval_dir/ground_truth/)")
    parser.add_argument("--target-class", default=None,
                        help="Only draw this class (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: <eval_dir>/gt_vis/)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    img_dir = eval_dir / "images"
    out_dir = Path(args.output) if args.output else eval_dir / "gt_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        sys.exit(f"Eval dir not valid: {eval_dir}")

    import sys

    eval_stems = {p.stem for p in img_dir.glob("*.jpg")}

    gt_data = {}

    if args.gt_coco:
        gt_coco_path = Path(args.gt_coco)
        if not gt_coco_path.is_file():
            sys.exit(f"GT COCO file not found: {gt_coco_path}")
        coco = json.load(open(gt_coco_path))
        cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        img_id_to_name = {im["id"]: im["file_name"] for im in coco.get("images", [])}
        for ann in coco.get("annotations", []):
            fname = img_id_to_name.get(ann["image_id"])
            if not fname:
                continue
            stem = Path(fname).stem
            if stem not in eval_stems:
                continue
            x, y, w, h = ann["bbox"]
            label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            gt_data.setdefault(stem, []).append({
                "label": label,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            })
        for stem in eval_stems:
            gt_data.setdefault(stem, [])
        print(f"Loaded GT from COCO: {gt_coco_path}")
    else:
        gt_dir = eval_dir / "ground_truth"
        if not gt_dir.is_dir():
            sys.exit(f"GT dir not found: {gt_dir} (use --gt-coco for COCO format)")
        for gt_file in sorted(gt_dir.glob("*.json")):
            data = json.load(open(gt_file))
            stem = gt_file.stem
            gt_data[stem] = data.get("annotations", [])
        print(f"Loaded GT from per-image files in {gt_dir}")

    count = 0
    drawn_boxes = 0
    for stem in sorted(eval_stems):
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        anns = gt_data.get(stem, [])
        if args.target_class:
            anns = [a for a in anns if a["label"] == args.target_class]

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for ann in anns:
            x1, y1, x2, y2 = ann["bbox"]
            label = ann["label"]
            color = CLASS_COLORS.get(label, DEFAULT_COLOR)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            drawn_boxes += 1

        cv2.imwrite(str(out_dir / f"{stem}_gt.jpg"), img)
        count += 1

    print(f"Saved {count} GT-annotated images ({drawn_boxes} boxes) to {out_dir}")


if __name__ == "__main__":
    import sys
    main()
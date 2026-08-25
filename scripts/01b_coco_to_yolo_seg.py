#!/usr/bin/env python3
"""
Convert an MS COCO-style annotation file (the label-review GUI's output)
into a FLAT YOLO segmentation dataset — the input format of
scripts/02_augment_data.py.

    COCO labels_coco.json  -->  output_dir/images/ + output_dir/labels/
                                + output_dir/classes.txt

The full post-annotation chain is:

    GUI annotate (labels_coco.json, discards excluded from the final save)
      -> 01b_coco_to_yolo_seg.py   (this script; flat YOLO-seg dataset)
      -> 02_augment_data.py        (augmented flat YOLO-seg dataset)
      -> 03_split_dataset.py       (train/val/test + dataset.yaml)
      -> 04_train_model.py --task segment

or run all of them at once with scripts/run_seg_dataset_pipeline.py.

USAGE:
    # Mono session: images all in one folder
    python scripts/01b_coco_to_yolo_seg.py \
        --coco-json /data/run/labels_coco.json \
        --images-dir /data/run/camera/left \
        --output-dir output/yolo_flat

    # Stereo session (COCO images carry a "side" field): --images-dir is the
    # parent containing left/ and right/ subfolders; output filenames are
    # prefixed with left_/right_ so the identical timestamp names don't collide.
    python scripts/01b_coco_to_yolo_seg.py \
        --coco-json /data/run/camera/left/labels_coco.json \
        --images-dir /data/run/camera \
        --output-dir output/yolo_flat

ANNOTATIONS WITHOUT MASKS:
    Boxes that never got a SAM3 mask have no polygon. They are skipped by
    default (counted in the summary); --bbox-as-rect emits them as rectangle
    polygons so no labeled object is lost from the training set.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert label-review COCO output to a flat YOLO "
                    "segmentation dataset (input for 02_augment_data.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coco-json", required=True,
                        help="COCO annotation file written by the label-review GUI")
    parser.add_argument("--images-dir", required=True,
                        help="Folder with the source images (mono); for stereo "
                             "COCO files, the parent folder containing left/ and "
                             "right/ subfolders")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory (gets images/, labels/, classes.txt)")
    parser.add_argument("--bbox-as-rect", action="store_true",
                        help="Emit mask-less boxes as rectangle polygons instead "
                             "of skipping them")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink images instead of copying")
    return parser.parse_args()


def polygon_to_yolo(class_id: int, polygon: list, img_w: int, img_h: int):
    """COCO polygon (absolute px) -> normalized YOLO seg line, or None."""
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return None
    parts = [str(class_id)]
    for i in range(0, len(polygon), 2):
        nx = max(0.0, min(1.0, float(polygon[i]) / img_w))
        ny = max(0.0, min(1.0, float(polygon[i + 1]) / img_h))
        parts.append(f"{nx:.6f}")
        parts.append(f"{ny:.6f}")
    return " ".join(parts)


def bbox_to_rect_polygon(bbox: list) -> list:
    """xywh bbox -> 4-point rectangle polygon (absolute px)."""
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def convert(coco_json: Path, images_dir: Path, output_dir: Path,
            bbox_as_rect: bool = False, symlink: bool = False) -> dict:
    """Run the conversion. Returns a summary dict (also written to
    <output_dir>/conversion_summary.json)."""
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    missing = {"images", "annotations", "categories"} - set(coco.keys())
    if missing:
        raise ValueError(f"COCO JSON missing required keys: {missing}")

    # Stable class order: sort by COCO id, map to contiguous YOLO ids.
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    coco_to_yolo = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c.get("name", f"class_{i}") for i, c in enumerate(cats)]

    stereo = any(img.get("side") == "right" for img in coco["images"])
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_images = n_skipped_img = n_polygons = n_maskless = 0
    for img in coco["images"]:
        side = img.get("side", "left")
        file_name = img["file_name"]
        src = (images_dir / side / file_name) if stereo \
            else (images_dir / file_name)
        if not src.exists():
            print(f"  Warning: image not found, skipping: {src}")
            n_skipped_img += 1
            continue
        img_w = int(img.get("width", 0))
        img_h = int(img.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            with Image.open(src) as im:
                img_w, img_h = im.size
        # Stereo sides share timestamp filenames — prefix to avoid collisions.
        out_name = f"{side}_{file_name}" if stereo else file_name
        dst = out_images / out_name
        if symlink:
            if not dst.exists():
                dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

        lines = []
        for ann in anns_by_image.get(img["id"], []):
            class_id = coco_to_yolo.get(ann["category_id"])
            if class_id is None:
                continue
            seg = ann.get("segmentation")
            polygons = [p for p in seg if p] if isinstance(seg, list) else []
            if not polygons:
                # Mask-less box: skip, or emit as a rectangle polygon.
                n_maskless += 1
                if bbox_as_rect and ann.get("bbox"):
                    polygons = [bbox_to_rect_polygon(ann["bbox"])]
            for poly in polygons:
                line = polygon_to_yolo(class_id, poly, img_w, img_h)
                if line:
                    lines.append(line)
        with open(out_labels / f"{Path(out_name).stem}.txt", "w") as f:
            for line in lines:
                f.write(line + "\n")
        n_images += 1
        n_polygons += len(lines)

    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(class_names) + "\n")

    summary = {
        "coco_json": str(coco_json),
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "stereo": stereo,
        "classes": class_names,
        "images_written": n_images,
        "images_skipped_missing": n_skipped_img,
        "polygons_written": n_polygons,
        "maskless_annotations": n_maskless,
        "bbox_as_rect": bbox_as_rect,
    }
    with open(output_dir / "conversion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not coco_json.exists():
        print(f"Error: COCO JSON not found: {coco_json}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}")
        sys.exit(1)

    print("=" * 60)
    print("COCO -> YOLO SEGMENTATION (flat dataset for 02_augment_data.py)")
    print("=" * 60)
    summary = convert(coco_json, images_dir, output_dir,
                      bbox_as_rect=args.bbox_as_rect, symlink=args.symlink)
    print(f"\nClasses: {summary['classes']}")
    print(f"Stereo source: {summary['stereo']}")
    print(f"Images written: {summary['images_written']} "
          f"(skipped missing: {summary['images_skipped_missing']})")
    print(f"Polygons written: {summary['polygons_written']}")
    skipped = summary["maskless_annotations"]
    if skipped and not args.bbox_as_rect:
        print(f"Note: {skipped} annotation(s) had no mask and were skipped "
              f"(use --bbox-as-rect to emit them as rectangles)")
    print(f"\nOutput: {output_dir} (images/ labels/ classes.txt)")
    print(f"Next step: python scripts/02_augment_data.py "
          f"--input-dir {output_dir} --output-dir <augmented_dir>")


if __name__ == "__main__":
    main()

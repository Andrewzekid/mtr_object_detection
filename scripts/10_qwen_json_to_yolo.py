#!/usr/bin/env python3
"""
Convert Qwen auto-labels (from `scripts/07_run_qwen.py --split-by-class`) into
YOLO detection labels (``class_id x_center y_center width height``, normalized)
and lay them out as a YOLO dataset matching the structure of
``Datasets/MTR/detect/train_yolo_detection``.

Input
-----
The per-class annotation JSONs written by `07_run_qwen.py` when run with
``--split-by-class --annotations-output <dir>``. Each image gets a subfolder
``<annotations_dir>/<image_stem>/<class>.json`` of the form::

    {
      "class_name": "Ceiling light",
      "image": "<path to the image as passed to qwen>",
      "bboxes": [[x1, y1, x2, y2], ...]   # pixel coordinates
    }

(Qwen emits normalized 0-1000 coords; `07_run_qwen.py` scales them to pixels
using `--coord-scale 1000.0` and the image's real W/H before writing these
files, so the boxes here are already in pixels.)

Output
------
A YOLO dataset at ``--output-dir``::

    <output-dir>/
        images/<stem>.jpg      (copied from --image-folder)
        labels/<stem>.txt      (class_id x y w h, normalized)
        data.yaml              (nc/names matching the existing dataset)

Class ids follow ``Datasets/MTR/detect/train_yolo_detection/data.yaml``:
    0 Advertisement Board, 1 Exit Sign, 2 Lights, 3 Map, 4 TV, 5 Ticket Gate.

Qwen prompt labels are normalized (case-insensitive, synonym map) onto these
six canonical names. Boxes whose label cannot be mapped are dropped (counted).
Images with zero valid boxes still receive an (empty) label file so images and
labels stay paired.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# Project root regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_FOLDER = PROJECT_ROOT / "Datasets" / "MTR" / "MTR_new_1k"
DEFAULT_ANNOTATIONS_DIR = PROJECT_ROOT / "output" / "mtr_new_1k_annotations"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "MTR" / "detect" / "mtr_new_1k_yolo"
DEFAULT_DATA_YAML = PROJECT_ROOT / "Datasets" / "MTR" / "detect" / "train_yolo_detection" / "data.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Canonical class order (id -> name) matching the existing dataset's data.yaml.
CANON_NAMES = ["Advertisement Board", "Exit Sign", "Lights", "Map", "TV", "Ticket Gate"]

# Map Qwen prompt label strings -> canonical name. Matched case-insensitively
# after stripping; synonyms listed explicitly.
LABEL_ALIASES = {
    "advertisement board": "Advertisement Board",
    "advertisement": "Advertisement Board",
    "ad board": "Advertisement Board",
    "ad": "Advertisement Board",
    "exit sign": "Exit Sign",
    "exit": "Exit Sign",
    "sign": "Exit Sign",
    "ceiling light": "Lights",
    "ceiling lights": "Lights",
    "light": "Lights",
    "lights": "Lights",
    "map": "Map",
    "maps": "Map",
    "tv": "TV",
    "tvs": "TV",
    "television": "TV",
    "ticket gate": "Ticket Gate",
    "ticket gates": "Ticket Gate",
    "gate": "Ticket Gate",
    "turnstile": "Ticket Gate",
    "turnstiles": "Ticket Gate",
}


def normalize_label(raw: str):
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    # Fuzzy fallback: any alias that is a substring of the raw label or vice versa.
    for alias, canon in LABEL_ALIASES.items():
        if alias in key or key in alias:
            return canon
    return None


def load_class_names(data_yaml: Path):
    """Read names list from a data.yaml; fall back to CANON_NAMES."""
    if data_yaml.exists() and yaml is not None:
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, list) and names:
            return names
    return list(CANON_NAMES)


def read_image_size(image_path: Path):
    """Return (W, H) using PIL if available, else cv2."""
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size  # (W, H)
    except Exception:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)


def collect_boxes_for_stem(stem: str, annotations_dir: Path):
    """Read all <stem>/*.json and return list of (class_name, [x1,y1,x2,y2])."""
    stem_dir = annotations_dir / stem
    boxes = []
    if not stem_dir.is_dir():
        return boxes
    for jf in sorted(stem_dir.glob("*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: could not read {jf}: {e}")
            continue
        class_name = data.get("class_name")
        for bbox in data.get("bboxes", []) or []:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                boxes.append((class_name, [float(v) for v in bbox]))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR,
                    help=f"Qwen --split-by-class annotations dir (default: {DEFAULT_ANNOTATIONS_DIR})")
    ap.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER,
                    help=f"Folder of source images to copy into the dataset (default: {DEFAULT_IMAGE_FOLDER})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"YOLO dataset output dir (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML,
                    help=f"data.yaml whose nc/names to mirror (default: {DEFAULT_DATA_YAML})")
    ap.add_argument("--copy-images", action="store_true", default=True,
                    help="Copy images into <output-dir>/images/ (default).")
    ap.add_argument("--symlink-images", action="store_true",
                    help="Symlink images instead of copying.")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="Skip stems with no matching image instead of erroring.")
    args = ap.parse_args()

    names = load_class_names(args.data_yaml)
    name_to_id = {n: i for i, n in enumerate(names)}
    print(f"Class map (id -> name): {dict(enumerate(names))}")

    if not args.annotations_dir.exists():
        sys.exit(f"Error: annotations dir not found: {args.annotations_dir}")
    if not args.image_folder.exists():
        sys.exit(f"Error: image folder not found: {args.image_folder}")

    images_out = args.output_dir / "images"
    labels_out = args.output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Index images by stem so we can find the file regardless of extension.
    image_by_stem = {}
    for f in args.image_folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            image_by_stem[f.stem] = f

    # Stems that have annotations.
    annotated_stems = sorted(
        d.name for d in args.annotations_dir.iterdir() if d.is_dir()
    )
    if not annotated_stems:
        sys.exit(f"Error: no per-image annotation subfolders in {args.annotations_dir}")

    total_boxes = 0
    written_boxes = 0
    dropped_boxes = 0
    dropped_label_counter = Counter()
    images_with_boxes = 0
    images_no_boxes = 0
    images_missing = 0
    per_class_counts = Counter()

    for stem in annotated_stems:
        img_path = image_by_stem.get(stem)
        if img_path is None:
            if args.allow_missing_images:
                images_missing += 1
                continue
            sys.exit(f"Error: no image found in {args.image_folder} for stem '{stem}'. "
                     f"Pass --allow-missing-images to skip.")

        size = read_image_size(img_path)
        if size is None:
            print(f"  Warning: could not read image {img_path}; skipping.")
            continue
        W, H = size

        boxes = collect_boxes_for_stem(stem, args.annotations_dir)
        total_boxes += len(boxes)

        yolo_lines = []
        for class_name, (x1, y1, x2, y2) in boxes:
            canon = normalize_label(class_name)
            if canon is None or canon not in name_to_id:
                dropped_boxes += 1
                dropped_label_counter[class_name] += 1
                continue
            cid = name_to_id[canon]
            # Pixel xyxy -> normalized xywh, clamped to [0,1].
            xc = ((x1 + x2) / 2.0) / W
            yc = ((y1 + y2) / 2.0) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                dropped_boxes += 1
                continue
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            written_boxes += 1
            per_class_counts[canon] += 1

        # Write label file (empty if no valid boxes -> keeps image/label pairing).
        label_path = labels_out / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        # Copy/symlink image into dataset.
        dst_img = images_out / img_path.name
        if not dst_img.exists():
            if args.symlink_images:
                if not dst_img.is_symlink():
                    dst_img.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)

        if yolo_lines:
            images_with_boxes += 1
        else:
            images_no_boxes += 1

    # Write data.yaml mirroring the existing dataset's nc/names.
    data_yaml_out = args.output_dir / "data.yaml"
    yaml_block = (
        f"train: ./images\n"
        f"val: ./images\n"
        f"test: ./images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    with open(data_yaml_out, "w") as f:
        f.write(yaml_block)

    print("\n" + "=" * 60)
    print(f"Output dataset : {args.output_dir}")
    print(f"Images        : {images_with_boxes + images_no_boxes} copied "
          f"({images_with_boxes} with boxes, {images_no_boxes} empty labels)")
    if images_missing:
        print(f"Images skipped (no file): {images_missing}")
    print(f"Total boxes   : {total_boxes}")
    print(f"Written boxes : {written_boxes}")
    print(f"Dropped boxes : {dropped_boxes}")
    if dropped_label_counter:
        print("  Dropped label breakdown:")
        for label, c in dropped_label_counter.most_common():
            print(f"    {c:5d}  {label!r}")
    print("Per-class written counts:")
    for name in names:
        print(f"  {name_to_id[name]}  {name:<22s} {per_class_counts.get(name, 0)}")
    print(f"data.yaml     : {data_yaml_out}")


if __name__ == "__main__":
    main()
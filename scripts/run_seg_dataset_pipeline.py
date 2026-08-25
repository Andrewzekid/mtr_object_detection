#!/usr/bin/env python3
"""
Run the full post-annotation segmentation dataset pipeline in one shot:

    label-review GUI output (COCO labels_coco.json)
      -> 01b_coco_to_yolo_seg.py   COCO -> flat YOLO-seg (images/ + labels/ + classes.txt)
      -> 02_augment_data.py        augmented flat YOLO-seg dataset
      -> 03_split_dataset.py       train/val/test split + dataset.yaml

After each step the intermediate dataset is validated (image/label pairing,
label line syntax, coordinate range, class ids) so a format mismatch fails
loudly at the step that introduced it instead of at training time.

The result is a YOLO segmentation dataset ready for:

    python scripts/04_train_model.py --config <output-dir>/dataset/dataset.yaml --task segment

USAGE:
    python scripts/run_seg_dataset_pipeline.py \
        --coco-json /data/run/labels_coco.json \
        --images-dir /data/run/camera \
        --output-dir output/my_dataset

    # Skip augmentation (split the converted dataset directly):
    python scripts/run_seg_dataset_pipeline.py \
        --coco-json labels_coco.json --images-dir camera --output-dir out \
        --skip-augment
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPTS_DIR = Path(__file__).parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
COORD_TOLERANCE = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO (label-review GUI) -> augment -> split -> YOLO seg dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coco-json", required=True,
                        help="COCO annotation file written by the label-review GUI")
    parser.add_argument("--images-dir", required=True,
                        help="Source image folder (mono) or parent of left/ + right/ (stereo)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Root output dir (gets yolo_flat/, augmented/, dataset/)")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Split the converted dataset directly, no augmentation")
    parser.add_argument("--bbox-as-rect", action="store_true",
                        help="Emit mask-less COCO boxes as rectangle polygons")
    # Split options
    parser.add_argument("--ratios", "-r", type=float, nargs=3,
                        default=[0.7, 0.15, 0.15], metavar=("TRAIN", "TEST", "VAL"),
                        help="Split ratios, must sum to 1.0 (default: 0.7 0.15 0.15)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for the split")
    # Augmentation pass-through options (forwarded to 02_augment_data.py)
    parser.add_argument("--augmentations", "-a", type=str, nargs="+",
                        default=["flip_horizontal", "rotate", "brightness"],
                        help="Augmentations for step 02 (default: flip_horizontal rotate brightness)")
    parser.add_argument("--multiplier", "-m", type=int, default=2,
                        help="Augmented copies per source image (default: 2)")
    parser.add_argument("--rotation-range", type=float, nargs=2, default=[-15, 15],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--brightness-range", type=float, nargs=2, default=[0.8, 1.2],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--hue-range", type=float, nargs=2, default=[-15, 15],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--blur-range", type=int, nargs=2, default=[3, 9],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"),
                        help="Target size when 'resize' is among the augmentations")
    return parser.parse_args()


def run_step(title: str, cmd: list):
    """Run one pipeline step; abort the pipeline on failure."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nERROR: step failed (exit {result.returncode}): {title}")
        sys.exit(result.returncode)


def _iter_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_flat_dataset(dataset_dir: Path, n_classes: int = None) -> dict:
    """Validate a flat YOLO-seg dataset (images/ + labels/ siblings).

    Checks: every image has a label file, every label line is
    "<int class> <float x> <float y> ..." with an even number of coordinates
    (>= 6), all coordinates in [0, 1], and class ids within range.
    Returns stats; raises ValueError on the first problem found.
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"{dataset_dir}: expected images/ and labels/ subdirectories")
    if n_classes is None:
        classes_file = dataset_dir / "classes.txt"
        if classes_file.is_file():
            n_classes = len([l for l in classes_file.read_text().splitlines() if l.strip()])

    images = _iter_images(images_dir)
    if not images:
        raise ValueError(f"{dataset_dir}: no images found in {images_dir}")
    n_lines = 0
    for img in images:
        label_file = labels_dir / f"{img.stem}.txt"
        if not label_file.is_file():
            raise ValueError(f"missing label file for image: {img.name}")
        for lineno, raw in enumerate(label_file.read_text().splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            where = f"{label_file.name}:{lineno}"
            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(f"{where}: class id is not an int: {parts[0]!r}")
            if n_classes and not (0 <= class_id < n_classes):
                raise ValueError(f"{where}: class id {class_id} out of range "
                                 f"(classes.txt has {n_classes})")
            coords = parts[1:]
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise ValueError(f"{where}: expected an even number of >= 6 "
                                 f"coordinates, got {len(coords)}")
            for tok in coords:
                try:
                    v = float(tok)
                except ValueError:
                    raise ValueError(f"{where}: coordinate is not a float: {tok!r}")
                if not (-COORD_TOLERANCE <= v <= 1.0 + COORD_TOLERANCE):
                    raise ValueError(f"{where}: coordinate {v} outside [0, 1]")
            n_lines += 1
    return {"images": len(images), "label_lines": n_lines}


def validate_split_dataset(dataset_dir: Path, n_classes: int = None) -> dict:
    """Validate a split YOLO dataset (train/val/test, each with images/ + labels/)."""
    splits = sorted(p for p in dataset_dir.iterdir()
                    if p.is_dir() and (p / "images").is_dir())
    if not splits:
        raise ValueError(f"{dataset_dir}: no split folders with images/ found")
    if n_classes is None:
        # classes.txt is not copied into the split output; count via dataset.yaml
        yaml_file = dataset_dir / "dataset.yaml"
        if yaml_file.is_file():
            import yaml
            names = yaml.safe_load(yaml_file.read_text()).get("names", {})
            n_classes = len(names)
    stats = {}
    for split in splits:
        if not _iter_images(split / "images"):
            # Empty split is legitimate (e.g. a 0.0 ratio)
            stats[split.name] = {"images": 0, "label_lines": 0}
            continue
        stats[split.name] = validate_flat_dataset(split, n_classes)
    return stats


def main():
    args = parse_args()
    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    flat_dir = output_dir / "yolo_flat"
    aug_dir = output_dir / "augmented"
    dataset_dir = output_dir / "dataset"

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"Error: --ratios must sum to 1.0, got {args.ratios}")
        sys.exit(1)
    if not coco_json.exists():
        print(f"Error: COCO JSON not found: {coco_json}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}")
        sys.exit(1)
    if not args.skip_augment and "resize" in args.augmentations and not args.resize:
        print("Error: 'resize' augmentation requires --resize WIDTH HEIGHT")
        sys.exit(1)

    # Step 1: COCO -> flat YOLO-seg
    cmd = [sys.executable, SCRIPTS_DIR / "01b_coco_to_yolo_seg.py",
           "--coco-json", coco_json, "--images-dir", images_dir,
           "--output-dir", flat_dir]
    if args.bbox_as_rect:
        cmd.append("--bbox-as-rect")
    run_step("STEP 1/3: COCO -> flat YOLO segmentation dataset", cmd)
    stats = validate_flat_dataset(flat_dir)
    print(f"  [validated] {stats['images']} images, {stats['label_lines']} polygons")

    # Step 2: augment (optional)
    split_input = flat_dir
    if not args.skip_augment:
        cmd = [sys.executable, SCRIPTS_DIR / "02_augment_data.py",
               "--input-dir", flat_dir, "--output-dir", aug_dir,
               "--augmentations", *args.augmentations,
               "--multiplier", str(args.multiplier),
               "--rotation-range", *(str(v) for v in args.rotation_range),
               "--brightness-range", *(str(v) for v in args.brightness_range),
               "--hue-range", *(str(v) for v in args.hue_range),
               "--blur-range", *(str(v) for v in args.blur_range)]
        if args.resize:
            cmd += ["--resize", *(str(v) for v in args.resize)]
        run_step("STEP 2/3: image augmentation", cmd)
        stats = validate_flat_dataset(aug_dir)
        print(f"  [validated] {stats['images']} images, {stats['label_lines']} polygons")
        split_input = aug_dir
    else:
        print("\nSTEP 2/3: augmentation skipped (--skip-augment)")

    # Step 3: split + dataset.yaml (class names come from classes.txt fallback)
    cmd = [sys.executable, SCRIPTS_DIR / "03_split_dataset.py",
           "--input-dir", split_input, "--output-dir", dataset_dir,
           "--ratios", *(str(v) for v in args.ratios),
           "--generate-yaml"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    run_step("STEP 3/3: train/test/val split", cmd)
    split_stats = validate_split_dataset(dataset_dir)
    for name, s in split_stats.items():
        print(f"  [validated] {name}: {s['images']} images, {s['label_lines']} polygons")

    yaml_path = dataset_dir / "dataset.yaml"
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Flat YOLO-seg:  {flat_dir}")
    if not args.skip_augment:
        print(f"  Augmented:      {aug_dir}")
    print(f"  Split dataset:  {dataset_dir}")
    if yaml_path.is_file():
        print(f"\nReady for training:")
        print(f"  python scripts/04_train_model.py --config {yaml_path} --task segment")
    else:
        print(f"\nWarning: {yaml_path} was not created; generate it with "
              f"03_split_dataset.py --generate-yaml before training.")


if __name__ == "__main__":
    main()

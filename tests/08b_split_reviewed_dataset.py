#!/usr/bin/env python3
"""
Split a reviewed YOLO detection dataset into train/val/test splits.

This is the intermediate step between ``08_click_review_coco.py`` (which emits a
flat ``yolo_reviewed/`` dataset) and ``09_create_seg_dataset.py`` (which expects
a split layout with ``images/{train,val,test}`` and ``labels/{train,val,test}``
plus a ``data.yaml`` carrying the class names).

It reads the class names from the input ``data.yaml`` so the split output stays
consistent with the reviewed labels, then writes a new ``data.yaml`` at the
output root pointing at the split directories.

USAGE:
    python scripts/08b_split_reviewed_dataset.py \\
        --input-dir output/MTR_new_10_images/reviewed/yolo_reviewed \\
        --output-dir output/MTR_new_10_images/reviewed/yolo_split \\
        --ratios 0.7 0.15 0.15 --seed 42

INPUT DIRECTORY STRUCTURE (flat, from 08_click_review_coco.py):
    input_dir/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    ├── labels/
    │   ├── img1.txt
    │   └── img2.txt
    └── data.yaml          (must contain a ``names`` list)

OUTPUT DIRECTORY STRUCTURE (Layout A, consumed by 09_create_seg_dataset.py):
    output_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

# Supported image extensions (kept in sync with 09_create_seg_dataset.py)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

SPLITS = ["train", "val", "test"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a reviewed YOLO dataset into train/val/test for 09_create_seg_dataset.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Flat reviewed YOLO dataset dir (images/, labels/, data.yaml) from 08_click_review_coco.py",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output dir for the split dataset (Layout A: images/{train,val,test}, labels/...)",
    )
    parser.add_argument(
        "--ratios", "-r",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "TEST", "VAL"),
        help="Split ratios for train/test/val, must sum to 1.0 (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink images into splits instead of copying (saves disk space)",
    )
    return parser.parse_args()


def load_class_names(input_dir: Path) -> list:
    """Read the ``names`` list from the input data.yaml."""
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Error: data.yaml not found in {input_dir}")
        sys.exit(1)
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names") if isinstance(data, dict) else None
    if not names or not isinstance(names, list):
        print(f"Error: data.yaml at {yaml_path} has no 'names' list")
        sys.exit(1)
    return names


def write_data_yaml(output_dir: Path, class_names: list):
    """Write a data.yaml for the split dataset (Layout A paths)."""
    config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path


def link_or_copy(src: Path, dst: Path, symlink: bool):
    """Copy or symlink ``src`` to ``dst`` if it does not already exist."""
    if dst.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.exists():
        print(f"Error: images/ directory not found in {input_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"Error: labels/ directory not found in {input_dir}")
        sys.exit(1)

    ratios = args.ratios
    if abs(sum(ratios) - 1.0) > 0.001:
        print(f"Error: --ratios must sum to 1.0, got {sum(ratios):.3f}")
        sys.exit(1)
    if any(r < 0 or r > 1 for r in ratios):
        print(f"Error: each ratio must be in [0, 1], got {ratios}")
        sys.exit(1)

    class_names = load_class_names(input_dir)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"Error: no images found in {images_dir}")
        sys.exit(1)

    print("=" * 60)
    print("SPLIT REVIEWED DATASET")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_files)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Ratios: train={ratios[0]:.2f}, test={ratios[1]:.2f}, val={ratios[2]:.2f}")
    if args.seed is not None:
        print(f"Seed:   {args.seed}")
    print(f"Images: {'symlink' if args.symlink_images else 'copy'}")

    # Shuffle and partition.
    if args.seed is not None:
        random.seed(args.seed)
    shuffled = image_files[:]
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * ratios[0])
    test_end = train_end + int(total * ratios[1])
    partitions = {
        "train": shuffled[:train_end],
        "test": shuffled[train_end:test_end],
        "val": shuffled[test_end:],
    }

    # Warn on empty splits — 09_create_seg_dataset.py will just skip them, but
    # an empty train split is almost certainly a mistake.
    for split in SPLITS:
        if not partitions[split]:
            print(f"  Warning: {split} split is empty (ratio {ratios[SPLITS.index(split)]:.2f} "
                  f"yielded 0 of {total} images)")

    # Create Layout A directory structure.
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Distribute images + matching labels.
    counts = {}
    for split, files in partitions.items():
        n_img = n_lbl = 0
        for img_file in files:
            link_or_copy(img_file, output_dir / "images" / split / img_file.name,
                         symlink=args.symlink_images)
            n_img += 1
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                link_or_copy(label_file, output_dir / "labels" / split / label_file.name,
                             symlink=args.symlink_images)
                n_lbl += 1
            else:
                # 09_create_seg_dataset.py handles missing labels by writing an
                # empty label file, so just warn here.
                print(f"  Warning: no label for {img_file.name} in {split} split")
        counts[split] = (n_img, n_lbl)

    write_data_yaml(output_dir, class_names)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    for split in SPLITS:
        n_img, n_lbl = counts[split]
        pct = (n_img / total * 100) if total else 0
        print(f"  {split:5s}: {n_img:5d} images ({pct:5.1f}%), {n_lbl} labels")
    print(f"\nOutput: {output_dir}")
    print("\nNext step: run 09_create_seg_dataset.py with this dir as --input-dir.")


if __name__ == "__main__":
    main()
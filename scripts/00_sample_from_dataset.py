#!/usr/bin/env python3
"""
Select 1000 images from `MTR_metacam_right` that are NOT already present in the
existing YOLO dataset `train_yolo_detection`, and copy them into a new folder
ready for labeling with `scripts/07_run_qwen.py`.

Matching rule
-------------
Source images   : Datasets/MTR/MTR_metacam_right/<19-digit-id>.jpg
Already-labeled : Datasets/MTR/detect/train_yolo_detection/images/{train,valid,test}/<id>_jpg.rf.<hash>.jpg

The Roboflow-style filenames embed the original id as their leading digits
(possibly followed by an augmentation suffix like `_1` before `_jpg.rf.`).
We extract the source id as the leading digit run via ``re.match(r"(\\d+)", ...)``,
which correctly handles the ``_1`` augmented variants (they map back to the
original numeric id).

The selection is a reproducible random sample (fixed seed) of the complement.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Project root (repo root) regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "Datasets" / "MTR" / "MTR_metacam_right"
DEFAULT_LABELED_DIRS = [
    PROJECT_ROOT / "Datasets" / "MTR" / "detect" / "train_yolo_detection" / "images" / "train",
    PROJECT_ROOT / "Datasets" / "MTR" / "detect" / "train_yolo_detection" / "images" / "valid",
    PROJECT_ROOT / "Datasets" / "MTR" / "detect" / "train_yolo_detection" / "images" / "test",
]
DEFAULT_OUT_DIR = PROJECT_ROOT / "Datasets" / "MTR" / "MTR_new_1k"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_ID_RE = re.compile(r"(\d+)")


def list_source_ids(source_dir: Path):
    """Return {id: Path} for every image in the source pool, keyed by stem id."""
    ids = {}
    if not source_dir.exists():
        sys.exit(f"Error: source dir not found: {source_dir}")
    for f in source_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            ids[f.stem] = f
    return ids


def list_labeled_ids(labeled_dirs):
    """Return the set of source ids already covered by the YOLO dataset.

    Extracts the leading digit run of each Roboflow filename so augmented
    variants (e.g. ``<id>_1_jpg.rf.<hash>.jpg``) map back to the original id.
    """
    labeled = set()
    for d in labeled_dirs:
        if not d.exists():
            print(f"Warning: labeled dir not found, skipping: {d}")
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                m = _ID_RE.match(f.stem)
                if m:
                    labeled.add(m.group(1))
    return labeled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                    help=f"Source image pool (default: {DEFAULT_SOURCE_DIR})")
    ap.add_argument("--labeled-dir", action="append", type=Path, default=None,
                    help="YOLO images dir already labeled (repeatable). "
                         "Defaults to train_yolo_detection images/{train,valid,test}.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Destination folder for the selected images (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("-n", "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                    help=f"Number of images to select (default: {DEFAULT_SAMPLE_SIZE})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED})")
    ap.add_argument("--copy", action="store_true", default=True,
                    help="Copy selected images into --out-dir (default).")
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy (useful to save disk).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print the selection without writing files.")
    args = ap.parse_args()

    labeled_dirs = args.labeled_dir if args.labeled_dir else DEFAULT_LABELED_DIRS

    source = list_source_ids(args.source_dir)
    labeled = list_labeled_ids(labeled_dirs)

    source_ids = set(source.keys())
    complement = sorted(source_ids - labeled)

    print(f"Source images      : {len(source_ids)}  ({args.source_dir})")
    print(f"Already-labeled ids: {len(labeled)}")
    print(f"Complement size    : {len(complement)}")

    if len(complement) < args.sample_size:
        sys.exit(f"Error: complement ({len(complement)}) smaller than requested sample "
                 f"size ({args.sample_size}). Reduce --sample-size.")

    # Reproducible random sample.
    import random
    rng = random.Random(args.seed)
    selected = rng.sample(complement, args.sample_size)
    selected_set = set(selected)

    # Sanity: no overlap with already-labeled ids.
    overlap = selected_set & labeled
    assert not overlap, f"BUG: selection overlaps labeled set: {sorted(overlap)[:5]}"

    print(f"Selected           : {len(selected)} (seed={args.seed})")
    print(f"First 5 ids        : {selected[:5]}")
    print(f"Last 5 ids         : {selected[-5:]}")

    manifest = {
        "source_dir": str(args.source_dir),
        "labeled_dirs": [str(d) for d in labeled_dirs],
        "out_dir": str(args.out_dir),
        "source_count": len(source_ids),
        "labeled_count": len(labeled),
        "complement_count": len(complement),
        "sample_size": len(selected),
        "seed": args.seed,
        "selected_ids": selected,
    }

    if args.dry_run:
        print("\n--dry-run: not writing files.")
        print(json.dumps({k: v for k, v in manifest.items() if k != "selected_ids"}, indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_id in selected:
        src = source[img_id]
        dst = args.out_dir / src.name
        if dst.exists():
            continue
        if args.symlink:
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        written += 1

    manifest_path = args.out_dir / "selected_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {written} new images to: {args.out_dir}")
    print(f"Total images now in out_dir: {sum(1 for _ in args.out_dir.glob('*.jpg'))}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
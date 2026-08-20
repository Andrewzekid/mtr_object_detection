#!/usr/bin/env python3
"""
Split a folder of images into subfolders, or combine COCO-style annotation
JSONs into one file.

Split mode
----------
Split the images of a folder (sorted by name, timestamp-aware like
gui.label_review) into subfolders, either by explicit index ranges
or by an even N-way divide:

    # one subfolder per range (indices inclusive, into the sorted file list):
    python scripts/09b_split_and_combine.py split /path/to/folder \
        --ranges 0-100 200-500

    # even 3-way divide into split_00/ split_01/ split_02/:
    python scripts/09b_split_and_combine.py split /path/to/folder \
        --divide-evenly 3

    # copy instead of move, and/or put subfolders somewhere else:
    python scripts/09b_split_and_combine.py split /path/to/folder \
        --ranges 0-100 --copy --output-dir /path/to/splits

Subfolders are created inside the source folder by default, named
``range_000000-000100`` / ``split_00`` etc. Images are MOVED by default
(use ``--copy`` to keep the originals in place). Images not covered by any
range stay where they are; ranges may not overlap.

Combine mode
------------
Merge several COCO-style annotation files (the shape produced by this
toolchain — see scripts/results.json or the label reviewer's output:
top-level ``images`` / ``annotations`` / ``categories``) into one file:

    python scripts/09b_split_and_combine.py combine part1.json part2.json \
        --output combined.json

Assumes the images (by file_name) and annotations in each input are unique
across all inputs. Category lists are merged by NAME (ids are remapped);
image and annotation ids are re-assigned sequentially and all references
(``image_id`` / ``category_id``) are fixed up, so inputs with clashing or
string ids (like results.json) combine cleanly. Extra fields
(``segmentation``, ``track_id``, ``confidence``, ...) are preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# split mode
# ---------------------------------------------------------------------------

def list_images(folder: Path) -> List[Path]:
    """Images in `folder` (non-recursive), sorted like the label reviewer:
    numerically when every stem is a bare integer (timestamp-named series),
    otherwise by name."""
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if files and all(p.stem.isdigit() for p in files):
        files.sort(key=lambda p: (int(p.stem), p.name))
    if not files:
        sys.exit(f"Error: no image files found in {folder}")
    return files


def parse_ranges(specs: List[str], n_files: int) -> List[Tuple[int, int]]:
    """Parse ['0-100', '200-500'] into inclusive (start, end) index pairs."""
    ranges: List[Tuple[int, int]] = []
    for spec in specs:
        try:
            lo_s, hi_s = spec.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            sys.exit(f"Error: bad range {spec!r} — expected e.g. 0-100")
        if lo < 0 or hi < lo or hi >= n_files:
            sys.exit(f"Error: range {spec!r} out of bounds for "
                     f"{n_files} images (valid: 0-{n_files - 1})")
        ranges.append((lo, hi))
    # Reject overlaps — moving the same file twice would corrupt the split.
    covered: set = set()
    for lo, hi in ranges:
        span = set(range(lo, hi + 1))
        if covered & span:
            sys.exit(f"Error: range {lo}-{hi} overlaps an earlier range")
        covered |= span
    return ranges


def even_splits(n_files: int, n_ways: int) -> List[Tuple[int, int]]:
    """Inclusive (start, end) pairs dividing n_files into n_ways ~equal parts
    (earlier parts get the remainder)."""
    if n_ways < 1 or n_ways > n_files:
        sys.exit(f"Error: --divide-evenly must be 1..{n_files} "
                 f"(got {n_ways})")
    base, rem = divmod(n_files, n_ways)
    out, start = [], 0
    for i in range(n_ways):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size - 1))
        start += size
    return out


def run_split(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Error: folder not found: {folder}")
    files = list_images(folder)
    print(f"📁 {len(files)} images in {folder}")

    if args.ranges:
        spans = parse_ranges(args.ranges, len(files))
        names = [f"range_{lo:06d}-{hi:06d}" for lo, hi in spans]
    else:
        spans = even_splits(len(files), args.divide_evenly)
        width = max(2, len(str(args.divide_evenly - 1)))
        names = [f"split_{i:0{width}d}" for i in range(len(spans))]

    out_base = Path(args.output_dir) if args.output_dir else folder
    verb = "Copied" if args.copy else "Moved"
    op = shutil.copy2 if args.copy else shutil.move
    for name, (lo, hi) in zip(names, spans):
        dest_dir = out_base / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i in range(lo, hi + 1):
            op(str(files[i]), str(dest_dir / files[i].name))
        print(f"  ✅ {verb} {hi - lo + 1} images [{lo}–{hi}] → {dest_dir}")
    print("Done.")


# ---------------------------------------------------------------------------
# combine mode
# ---------------------------------------------------------------------------

def _to_int(value: Any) -> int:
    """Coerce COCO ids that may be strings (results.json) to int."""
    return int(value)


def run_combine(args: argparse.Namespace) -> None:
    cat_id_by_name: Dict[str, int] = {}
    categories: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    seen_files: set = set()
    next_img_id = 1
    next_ann_id = 1

    for path_str in args.jsons:
        path = Path(path_str)
        if not path.is_file():
            sys.exit(f"Error: json not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not all(k in data for k in ("images", "annotations", "categories")):
            sys.exit(f"Error: {path} is missing images/annotations/categories")

        # Map this file's category ids → merged ids (merge by name).
        local_cat_map: Dict[int, int] = {}
        for cat in data["categories"]:
            name = cat["name"]
            if name not in cat_id_by_name:
                new_id = len(categories)
                cat_id_by_name[name] = new_id
                merged = dict(cat)
                merged["id"] = new_id
                categories.append(merged)
            local_cat_map[_to_int(cat["id"])] = cat_id_by_name[name]

        # Map this file's image ids → fresh sequential ids.
        local_img_map: Dict[int, int] = {}
        for img in data["images"]:
            fname = img["file_name"]
            if fname in seen_files:
                sys.exit(f"Error: duplicate image {fname!r} in {path} — "
                         "inputs are assumed to be unique")
            seen_files.add(fname)
            merged = dict(img)
            merged["id"] = next_img_id
            local_img_map[_to_int(img["id"])] = next_img_id
            next_img_id += 1
            images.append(merged)

        n_skipped = 0
        for ann in data["annotations"]:
            img_key = _to_int(ann["image_id"])
            cat_key = _to_int(ann["category_id"])
            if img_key not in local_img_map or cat_key not in local_cat_map:
                n_skipped += 1
                continue
            merged = dict(ann)
            merged["id"] = next_ann_id
            merged["image_id"] = local_img_map[img_key]
            merged["category_id"] = local_cat_map[cat_key]
            next_ann_id += 1
            annotations.append(merged)
        warn = (f" (⚠️ {n_skipped} annotation(s) skipped: unknown "
                f"image/category id)") if n_skipped else ""
        print(f"  ✅ {path.name}: {len(local_img_map)} images, "
              f"{len(data['annotations']) - n_skipped} annotations{warn}")

    out = {
        "info": {"description": "Combined annotations "
                                "(09b_split_and_combine.py)"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"✅ Combined {len(args.jsons)} files → {out_path}: "
          f"{len(images)} images, {len(annotations)} annotations, "
          f"{len(categories)} categories")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_split = sub.add_parser(
        "split", help="Split a folder of images into subfolders.")
    p_split.add_argument("folder", help="Folder containing the images.")
    group = p_split.add_mutually_exclusive_group(required=True)
    group.add_argument("--ranges", nargs="+", metavar="LO-HI",
                       help="Inclusive index ranges into the sorted image "
                            "list, e.g. --ranges 0-100 200-500 (one "
                            "subfolder per range).")
    group.add_argument("--divide-evenly", type=int, metavar="N",
                       help="Divide all images evenly into N subfolders.")
    p_split.add_argument("--copy", action="store_true",
                         help="Copy instead of move (keep originals).")
    p_split.add_argument("--output-dir",
                         help="Where to create the subfolders "
                              "(default: inside the source folder).")
    p_split.set_defaults(func=run_split)

    p_comb = sub.add_parser(
        "combine", help="Combine COCO-style annotation JSONs into one file.")
    p_comb.add_argument("jsons", nargs="+",
                        help="Input JSON files (images/annotations assumed "
                             "unique across inputs).")
    p_comb.add_argument("--output", required=True,
                        help="Output JSON path.")
    p_comb.set_defaults(func=run_combine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

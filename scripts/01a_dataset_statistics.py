#!/usr/bin/env python3
"""
Compute dataset statistics for a YOLO-format dataset (detection or
segmentation labels — only the class id, the first token of each label
line, is used).

Accepts any of the common YOLO layouts — a single dataset directory:

    input_dir/
    ├── images/
    └── labels/

a split dataset produced by 03_split_dataset.py:

    input_dir/
    ├── train/{images,labels}
    ├── val/{images,labels}
    └── test/{images,labels}

or the YOLO training layout (images/<split> + labels/<split>):

    input_dir/
    ├── images/{train,val,test}
    └── labels/{train,val,test}

Class names are read from classes.txt or dataset.yaml inside the input
directory when present, otherwise classes are named class_<id>.

REPORTED STATISTICS (per split and per class):
    - number of images / label files / background (unlabeled) images
    - total instances
    - per class: instance count, % of all instances, number of images
      containing the class, % of images, avg instances per image

USAGE:
    # Single dataset
    python scripts/01a_dataset_statistics.py --input-dir output/augmented

    # Split dataset, also write CSV
    python scripts/01a_dataset_statistics.py --input-dir output/split \
        --csv output/split/dataset_statistics.csv

CSV COLUMNS:
    split, class_id, class_name, instances, pct_instances,
    images_with_class, pct_images, avg_instances_per_image,
    total_images, labeled_images, background_images.
    An "all" row per split carries the totals (the last three columns are
    only filled on "all" rows). Note: per-class images_with_class counts
    images containing that class, so class rows do not sum to the "all"
    row — an image containing several classes is counted once per class.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")

CSV_FIELDS = [
    "split", "class_id", "class_name", "instances", "pct_instances",
    "images_with_class", "pct_images", "avg_instances_per_image",
    # Split-level totals, filled only on the "all" summary row:
    "total_images", "labeled_images", "background_images",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", "-i", required=True,
                   help="Dataset dir (images/ + labels/) or split root "
                        "(train/val/test subdirs)")
    p.add_argument("--csv", "-c", default=None,
                   help="Optional CSV output path for the per-class statistics")
    return p.parse_args()


def find_splits(input_dir: Path) -> dict:
    """Return {split_name: labels_dir}. Supports:
    - <split>/labels      (03_split_dataset.py layout)
    - labels/<split>      (YOLO training layout: images/<split> + labels/<split>)
    - labels/             (single dataset, reported as split 'all')
    "val" also matches a "valid" directory (Roboflow exports).
    """
    splits = {}
    for name in SPLIT_NAMES:
        aliases = (name, "valid") if name == "val" else (name,)
        for alias in aliases:
            if (input_dir / alias / "labels").is_dir():
                splits[name] = input_dir / alias / "labels"
                break
            if (input_dir / "labels" / alias).is_dir():
                splits[name] = input_dir / "labels" / alias
                break
    if splits:
        return splits
    if (input_dir / "labels").is_dir():
        return {"all": input_dir / "labels"}
    raise RuntimeError(
        f"No labels found under {input_dir} — expected labels/, "
        f"labels/<split>/ or <split>/labels subdirectories")


def load_class_names(input_dir: Path) -> dict:
    """{class_id: name} from classes.txt or dataset.yaml, else {}."""
    classes_txt = input_dir / "classes.txt"
    if classes_txt.is_file():
        names = [l.strip() for l in classes_txt.read_text().splitlines()
                 if l.strip()]
        return {i: n for i, n in enumerate(names)}
    for yaml_name in ("dataset.yaml", "dataset.yml", "data.yaml"):
        yaml_path = input_dir / yaml_name
        if yaml_path.is_file():
            try:
                import yaml
                data = yaml.safe_load(yaml_path.read_text())
                names = data.get("names", [])
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                return {i: n for i, n in enumerate(names)}
            except Exception:
                return {}
    return {}


def compute_split_stats(labels_dir: Path) -> dict:
    """Statistics for one labels dir.

    Returns dict with image_count, labeled_count, total_instances and
    per_class {class_id: {"instances": int, "images": int}}.
    """
    image_count = 0
    labeled_count = 0
    total_instances = 0
    per_class = {}

    for txt in sorted(labels_dir.glob("*.txt")):
        image_count += 1
        classes_in_image = set()
        with open(txt, "r") as f:
            for line in f:
                toks = line.split()
                if len(toks) < 5:
                    continue
                try:
                    cid = int(float(toks[0]))
                except ValueError:
                    continue
                classes_in_image.add(cid)
                total_instances += 1
                entry = per_class.setdefault(cid, {"instances": 0, "images": 0})
                entry["instances"] += 1
        if classes_in_image:
            labeled_count += 1
            for cid in classes_in_image:
                per_class[cid]["images"] += 1

    return {
        "image_count": image_count,
        "labeled_count": labeled_count,
        "background_count": image_count - labeled_count,
        "total_instances": total_instances,
        "per_class": per_class,
    }


def build_rows(split: str, stats: dict, class_names: dict) -> list:
    """CSV rows for one split: an "all" summary row plus one row per class.

    Note: per-class images_with_class counts images containing THAT class, so
    the class rows do NOT sum to labeled_images — an image containing several
    classes is counted once per class. The "all" row's images_with_class is
    the number of labeled images overall (same as labeled_images).
    """
    n_images = stats["image_count"]
    total = stats["total_instances"]
    rows = [{
        "split": split, "class_id": "all", "class_name": "all",
        "instances": total, "pct_instances": 100.0 if total else 0.0,
        "images_with_class": stats["labeled_count"],
        "pct_images": (100.0 * stats["labeled_count"] / n_images
                       if n_images else 0.0),
        "avg_instances_per_image": (total / n_images if n_images else 0.0),
        "total_images": n_images,
        "labeled_images": stats["labeled_count"],
        "background_images": stats["background_count"],
    }]
    for cid in sorted(stats["per_class"]):
        c = stats["per_class"][cid]
        rows.append({
            "split": split,
            "class_id": cid,
            "class_name": class_names.get(cid, f"class_{cid}"),
            "instances": c["instances"],
            "pct_instances": 100.0 * c["instances"] / total if total else 0.0,
            "images_with_class": c["images"],
            "pct_images": 100.0 * c["images"] / n_images if n_images else 0.0,
            "avg_instances_per_image": (c["instances"] / n_images
                                        if n_images else 0.0),
        })
    return rows


def print_split(split: str, stats: dict, class_names: dict):
    n_images = stats["image_count"]
    total = stats["total_instances"]
    print(f"\n{split.upper()}:")
    print(f"  Images:            {n_images}")
    print(f"  Labeled images:    {stats['labeled_count']}")
    print(f"  Background images: {stats['background_count']}")
    print(f"  Total instances:   {total}")
    if stats["per_class"]:
        print(f"\n  {'class':<30} {'inst':>7} {'%inst':>7} {'imgs':>7} "
              f"{'%imgs':>7} {'inst/img':>8}")
        for cid in sorted(stats["per_class"]):
            c = stats["per_class"][cid]
            name = class_names.get(cid, f"class_{cid}")
            print(f"  {name:<30} {c['instances']:>7} "
                  f"{100.0 * c['instances'] / total if total else 0:>6.1f}% "
                  f"{c['images']:>7} "
                  f"{100.0 * c['images'] / n_images if n_images else 0:>6.1f}% "
                  f"{c['instances'] / n_images if n_images else 0:>8.2f}")


def write_csv(rows: list, csv_path: str):
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("pct_instances", "pct_images",
                        "avg_instances_per_image"):
                formatted[key] = f"{row[key]:.2f}"
            writer.writerow(formatted)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    try:
        splits = find_splits(input_dir)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    class_names = load_class_names(input_dir)

    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Input: {input_dir}")
    if class_names:
        print(f"Classes: {len(class_names)} "
              f"({', '.join(class_names[i] for i in sorted(class_names))})")

    all_rows = []
    for split, labels_dir in splits.items():
        stats = compute_split_stats(labels_dir)
        print_split(split, stats, class_names)
        all_rows.extend(build_rows(split, stats, class_names))

    if args.csv:
        write_csv(all_rows, args.csv)
        print(f"\n✓ CSV written to: {args.csv}")


if __name__ == "__main__":
    main()

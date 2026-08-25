#!/usr/bin/env python3
"""
Create a YOLO segmentation dataset from an MS COCO-style segmentation annotation file.

This script reads a COCO JSON with polygon segmentations, converts the polygons to
YOLO segmentation format (class_id x1 y1 x2 y2 ... xn yn, normalized), splits the
images into train/test/val sets, and writes the same directory layout as
scripts/09_create_seg_dataset.py.

USAGE:
    python scripts/10_create_seg_dataset_from_coco.py \
        --coco-json "Datasets/HKU GH/HKU_GH_Segmentation_1k.json" \
        --images-dir "Datasets/HKU GH/HKU_GH_left" \
        --output-dir "Datasets/HKU GH/hku_gh_yolo_seg"

    # Quick dry run (first 5 images per split):
    python scripts/10_create_seg_dataset_from_coco.py \
        --coco-json "Datasets/HKU GH/HKU_GH_Segmentation_1k.json" \
        --images-dir "Datasets/HKU GH/HKU_GH_left" \
        --output-dir "output/hku_gh_yolo_seg_test" \
        --max-images 5

OUTPUT:
    A YOLO segmentation dataset with:
    - images/{train,val,test}/  (copied or symlinked from source)
    - labels/{train,val,test}/  (YOLO seg format polygons)
    - data.yaml                 (dataset config for YOLO training)
    - creation_summary.json     (generation statistics)
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation dataset from MS COCO annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        default="Datasets/HKU GH/HKU_GH_Segmentation_1k.json",
        help="Path to MS COCO JSON annotation file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="Datasets/HKU GH/HKU_GH_left",
        help="Directory containing the source images referenced by the COCO file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="Datasets/HKU GH/hku_gh_yolo_seg",
        help="Output YOLO segmentation dataset directory",
    )
    parser.add_argument(
        "--ratios",
        "-r",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "TEST", "VAL"),
        help="Split ratios for train/test/val (must sum to 1.0, default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process per split (for testing)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (saves disk space)",
    )

    return parser.parse_args()


def load_coco_data(coco_path: Path) -> dict:
    """Load COCO JSON data."""
    if not coco_path.exists():
        print(f"Error: COCO JSON not found: {coco_path}")
        sys.exit(1)

    with open(coco_path, "r") as f:
        data = json.load(f)

    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"Error: COCO JSON missing required keys: {missing}")
        sys.exit(1)

    return data


def build_category_mapping(categories: list) -> tuple[dict, list]:
    """Build mapping from COCO category_id to contiguous YOLO class_id.

    Returns:
        (coco_id_to_yolo_id, class_names)
    """
    # Sort categories by id to keep a stable order
    sorted_categories = sorted(categories, key=lambda c: c["id"])
    coco_id_to_yolo_id = {}
    class_names = []

    for yolo_id, cat in enumerate(sorted_categories):
        coco_id = cat["id"]
        coco_id_to_yolo_id[coco_id] = yolo_id
        class_names.append(cat.get("name", f"class_{yolo_id}"))

    return coco_id_to_yolo_id, class_names


def build_image_lookup(images: list) -> dict:
    """Build lookup from image_id to image metadata."""
    lookup = {}
    for img in images:
        img_id = img["id"]
        lookup[img_id] = {
            "file_name": img["file_name"],
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
        }
    return lookup


def group_annotations_by_image(annotations: list) -> dict:
    """Group COCO annotations by image_id."""
    grouped = {}
    for ann in annotations:
        img_id = ann["image_id"]
        grouped.setdefault(img_id, []).append(ann)
    return grouped


def split_image_ids(image_ids: list, ratios: list, seed: int) -> dict:
    """Randomly split image IDs into train/test/val sets.

    Args:
        image_ids: List of image IDs to split
        ratios: [train, test, val] fractions that sum to 1.0
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'train', 'test', 'val' mapping to lists of image IDs
    """
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.3f}")

    rng = random.Random(seed)
    shuffled = image_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_test = int(n * ratios[1])
    # Assign remaining images to val to avoid rounding issues
    n_val = n - n_train - n_test

    splits = {
        "train": shuffled[:n_train],
        "test": shuffled[n_train : n_train + n_test],
        "val": shuffled[n_train + n_test : n_train + n_test + n_val],
    }
    return splits


def coco_polygon_to_yolo_seg_format(
    class_id: int, coco_polygon: list, img_w: int, img_h: int
) -> str | None:
    """Convert a COCO polygon to YOLO segmentation label format.

    COCO format: [x1, y1, x2, y2, ...] in absolute pixel coordinates.
    YOLO seg format: class_id x1 y1 x2 y2 ... xn yn (normalized).

    Returns None if the polygon is invalid or has fewer than 3 points.
    """
    if len(coco_polygon) < 6 or len(coco_polygon) % 2 != 0:
        return None

    parts = [str(class_id)]
    for i in range(0, len(coco_polygon), 2):
        x = float(coco_polygon[i])
        y = float(coco_polygon[i + 1])
        nx = max(0.0, min(1.0, x / img_w)) if img_w > 0 else 0.0
        ny = max(0.0, min(1.0, y / img_h)) if img_h > 0 else 0.0
        parts.append(f"{nx:.6f}")
        parts.append(f"{ny:.6f}")

    return " ".join(parts)


def process_image_annotations(
    img_id: int,
    image_lookup: dict,
    annotations_by_image: dict,
    coco_id_to_yolo_id: dict,
    images_dir: Path,
) -> dict:
    """Process annotations for a single image and prepare YOLO seg labels.

    Returns:
        Dict with:
        - success: bool
        - src_image_path: Path | None
        - dst_image_name: str | None
        - label_lines: list[str]
        - error: str | None
    """
    img_meta = image_lookup.get(img_id)
    if img_meta is None:
        return {"success": False, "error": f"Image id {img_id} not found in image lookup"}

    file_name = img_meta["file_name"]
    img_w = img_meta["width"]
    img_h = img_meta["height"]

    src_image_path = images_dir / file_name
    if not src_image_path.exists():
        return {
            "success": False,
            "error": f"Image file not found: {src_image_path}",
        }

    # If dimensions are missing, try to infer from the image file
    if img_w <= 0 or img_h <= 0:
        try:
            from PIL import Image

            with Image.open(src_image_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            print(f"    Warning: Could not read dimensions for {file_name}, skipping")
            return {
                "success": False,
                "error": f"Missing width/height and could not read image: {file_name}",
            }

    anns = annotations_by_image.get(img_id, [])
    label_lines = []

    for ann in anns:
        coco_cat_id = ann.get("category_id")
        if coco_cat_id not in coco_id_to_yolo_id:
            continue
        class_id = coco_id_to_yolo_id[coco_cat_id]

        segmentation = ann.get("segmentation")
        if not segmentation:
            continue

        # COCO segmentation can be a list of polygons (list of lists) or RLE dict.
        # We only support polygon lists here.
        if isinstance(segmentation, list):
            polygons = segmentation
        elif isinstance(segmentation, dict):
            print(f"    Warning: RLE segmentation not supported for annotation {ann.get('id')}")
            continue
        else:
            continue

        for polygon in polygons:
            label_line = coco_polygon_to_yolo_seg_format(class_id, polygon, img_w, img_h)
            if label_line:
                label_lines.append(label_line)

    return {
        "success": True,
        "src_image_path": src_image_path,
        "dst_image_name": file_name,
        "label_lines": label_lines,
    }


def setup_output_directories(output_dir: Path):
    """Create output directory structure for a YOLO segmentation dataset."""
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def create_data_yaml(output_dir: Path, class_names: list):
    """Create data.yaml for the segmentation dataset."""
    config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "test": "images/test",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path


def process_split(
    split: str,
    image_ids: list,
    image_lookup: dict,
    annotations_by_image: dict,
    coco_id_to_yolo_id: dict,
    images_dir: Path,
    output_dir: Path,
    symlink: bool = False,
    max_images: int = None,
):
    """Process all images in a split (train/val/test)."""
    if max_images:
        image_ids = image_ids[:max_images]

    print(f"\n{'=' * 60}")
    print(f"Processing {split} split: {len(image_ids)} images")
    print(f"{'=' * 60}")

    dst_img_dir = output_dir / "images" / split
    dst_label_dir = output_dir / "labels" / split

    successful = 0
    failed = 0
    total_annotations = 0
    total_polygons = 0

    for i, img_id in enumerate(image_ids, 1):
        result = process_image_annotations(
            img_id=img_id,
            image_lookup=image_lookup,
            annotations_by_image=annotations_by_image,
            coco_id_to_yolo_id=coco_id_to_yolo_id,
            images_dir=images_dir,
        )

        if not result["success"]:
            failed += 1
            print(f"\n[{i}/{len(image_ids)}] Error: {result.get('error', 'Unknown error')}")
            continue

        src_image_path = result["src_image_path"]
        dst_image_name = result["dst_image_name"]
        label_lines = result["label_lines"]

        print(f"\n[{i}/{len(image_ids)}] {dst_image_name}")
        print(f"  Annotations: {len(annotations_by_image.get(img_id, []))}, Polygons: {len(label_lines)}")

        # Copy or symlink image
        dst_image_path = dst_img_dir / dst_image_name
        if symlink:
            if not dst_image_path.exists():
                dst_image_path.symlink_to(src_image_path.resolve())
        else:
            if not dst_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)

        # Write YOLO segmentation labels
        label_path = dst_label_dir / f"{Path(dst_image_name).stem}.txt"
        with open(label_path, "w") as f:
            for line in label_lines:
                f.write(line + "\n")

        successful += 1
        total_annotations += len(annotations_by_image.get(img_id, []))
        total_polygons += len(label_lines)

    print(f"\n{'=' * 60}")
    print(f"{split} split complete:")
    print(f"  Processed: {len(image_ids)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Total polygons: {total_polygons}")
    print(f"{'=' * 60}")

    return {
        "processed": len(image_ids),
        "successful": successful,
        "failed": failed,
        "total_annotations": total_annotations,
        "total_polygons": total_polygons,
    }


def main():
    args = parse_args()

    coco_path = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load COCO data
    print(f"Loading COCO annotations from: {coco_path}")
    coco_data = load_coco_data(coco_path)
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")

    # Build mappings
    coco_id_to_yolo_id, class_names = build_category_mapping(coco_data["categories"])
    image_lookup = build_image_lookup(coco_data["images"])
    annotations_by_image = group_annotations_by_image(coco_data["annotations"])

    print(f"\nClass mapping:")
    for coco_id, yolo_id in sorted(coco_id_to_yolo_id.items(), key=lambda x: x[1]):
        print(f"  COCO id {coco_id} -> YOLO id {yolo_id}: {class_names[yolo_id]}")

    # Split image IDs
    image_ids = sorted(image_lookup.keys())
    print(f"\nSplitting {len(image_ids)} images with ratios {args.ratios} and seed {args.seed}")
    splits = split_image_ids(image_ids, args.ratios, args.seed)
    for split_name, split_ids in splits.items():
        print(f"  {split_name}: {len(split_ids)} images")

    # Setup output directories
    print(f"\nSetting up output directories at: {output_dir}")
    setup_output_directories(output_dir)

    # Create data.yaml
    create_data_yaml(output_dir, class_names)

    # Process each split
    summary = {}
    for split in ["train", "val", "test"]:
        result = process_split(
            split=split,
            image_ids=splits[split],
            image_lookup=image_lookup,
            annotations_by_image=annotations_by_image,
            coco_id_to_yolo_id=coco_id_to_yolo_id,
            images_dir=images_dir,
            output_dir=output_dir,
            symlink=args.symlink,
            max_images=args.max_images,
        )
        summary[split] = result

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    total_processed = sum(s["processed"] for s in summary.values())
    total_successful = sum(s["successful"] for s in summary.values())
    total_failed = sum(s["failed"] for s in summary.values())
    total_annotations = sum(s.get("total_annotations", 0) for s in summary.values())
    total_polygons = sum(s.get("total_polygons", 0) for s in summary.values())

    print(f"Total images processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Total input annotations: {total_annotations}")
    print(f"Total output polygons: {total_polygons}")
    print(f"\nOutput dataset saved to: {output_dir}")
    print(f"\nTo train a YOLO segmentation model:")
    print(f"  python scripts/04_train_model.py --data {output_dir}/data.yaml --task segment")

    # Save summary JSON
    summary_path = output_dir / "creation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "coco_json": str(coco_path),
                "images_dir": str(images_dir),
                "output_dir": str(output_dir),
                "classes": class_names,
                "class_mapping": {
                    str(coco_id): yolo_id
                    for coco_id, yolo_id in coco_id_to_yolo_id.items()
                },
                "ratios": args.ratios,
                "seed": args.seed,
                "symlink": args.symlink,
                "summary": summary,
                "totals": {
                    "processed": total_processed,
                    "successful": total_successful,
                    "failed": total_failed,
                    "annotations": total_annotations,
                    "polygons": total_polygons,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

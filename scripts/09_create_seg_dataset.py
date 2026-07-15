#!/usr/bin/env python3
"""
Create a YOLO segmentation dataset from an existing detection dataset using SAM3.

This script reads YOLO detection labels (class_id x_center y_center width height),
runs SAM3 segmentation using the bboxes as exemplars per class, extracts polygon
contours from the resulting masks, and writes YOLO segmentation labels
(class_id x1 y1 x2 y2 ... xn yn) with normalized polygon coordinates.

USAGE:
    python scripts/09_create_seg_dataset.py \
        --input-dir ./train_yolo_augmented \
        --output-dir ./train_yolo_seg \
        --conf 0.25 \
        --device cuda

    # Dry run (process first 5 images only):
    python scripts/09_create_seg_dataset.py \
        --input-dir ./train_yolo_augmented \
        --output-dir ./train_yolo_seg \
        --max-images 5

OUTPUT:
    A new YOLO segmentation dataset with:
    - images/{train,val,test}/  (copied from input)
    - labels/{train,val,test}/  (YOLO seg format polygons)
    - data.yaml                 (updated for segmentation task)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models_inference import run_sam3


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation dataset from detection dataset using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default="./train_yolo_augmented",
        help="Input YOLO detection dataset directory (default: ./train_yolo_augmented)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./train_yolo_seg",
        help="Output YOLO segmentation dataset directory (default: ./train_yolo_seg)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to SAM3 model weights",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for SAM3 segmentation (default: 0.25)",
    )
    parser.add_argument(
        "--poly-epsilon",
        type=float,
        default=0.02,
        help="Polygon simplification epsilon as fraction of perimeter (default: 0.02)",
    )
    parser.add_argument(
        "--min-polygon-area",
        type=int,
        default=50,
        help="Minimum polygon area in pixels to keep (default: 50)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process per split (for testing)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy images to output directory (default: True)",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        default=False,
        help="Symlink images instead of copying (saves disk space)",
    )

    return parser.parse_args()


def load_data_yaml(input_dir: Path) -> dict:
    """Load the data.yaml from the input dataset."""
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Error: data.yaml not found in {input_dir}")
        sys.exit(1)
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    return data


def read_yolo_detection_labels(label_path: Path, img_w: int, img_h: int) -> list:
    """Read YOLO labels and convert to pixel-coordinate bboxes.
    
    Supports both formats:
    - Detection: class_id x_center y_center width height (5 values, normalized)
    - Segmentation: class_id x1 y1 x2 y2 ... xn yn (polygon, normalized)
    
    For segmentation format, the bounding box is computed from polygon extents.
    
    Returns:
        List of dicts: [{"class_id": int, "bbox": [x1, y1, x2, y2]}]
    """
    if not label_path.exists():
        return []
    
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
            
            if len(values) == 4:
                # Detection format: x_center y_center width height
                x_center = values[0] * img_w
                y_center = values[1] * img_h
                width = values[2] * img_w
                height = values[3] * img_h
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
            elif len(values) >= 6 and len(values) % 2 == 0:
                # Segmentation polygon format: x1 y1 x2 y2 ... xn yn
                xs = [values[i] * img_w for i in range(0, len(values), 2)]
                ys = [values[i] * img_h for i in range(1, len(values), 2)]
                x1 = min(xs)
                y1 = min(ys)
                x2 = max(xs)
                y2 = max(ys)
            else:
                continue
            
            # Clip to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            if x2 > x1 and y2 > y1:
                detections.append({
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                })
    
    return detections


def group_bboxes_by_class(detections: list) -> dict:
    """Group bboxes by class_id.
    
    Returns:
        Dict: {class_id: [[x1, y1, x2, y2], ...]}
    """
    grouped = {}
    for det in detections:
        class_id = det["class_id"]
        if class_id not in grouped:
            grouped[class_id] = []
        grouped[class_id].append(det["bbox"])
    return grouped


def mask_to_polygons(mask: np.ndarray, epsilon_factor: float = 0.02, 
                     min_area: int = 50) -> list:
    """Convert a binary mask to a list of polygon contours.
    
    Args:
        mask: Binary mask (H, W) bool or uint8
        epsilon_factor: Polygon simplification epsilon as fraction of perimeter
        min_area: Minimum polygon area to keep
    
    Returns:
        List of polygons, each polygon is a list of [x, y] points (pixel coordinates)
    """
    # Ensure mask is uint8
    if mask.dtype == bool:
        mask_uint8 = (mask.astype(np.uint8)) * 255
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify polygon
        perimeter = cv2.contourLength(contour) if hasattr(cv2, 'contourLength') else cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of [x, y] points
        polygon = approx.reshape(-1, 2).tolist()
        
        # Ensure polygon has at least 3 points
        if len(polygon) >= 3:
            polygons.append(polygon)
    
    return polygons


def polygon_to_yolo_seg_format(class_id: int, polygon: list, img_w: int, img_h: int) -> str:
    """Convert a polygon to YOLO segmentation label format.
    
    YOLO seg format: class_id x1 y1 x2 y2 ... xn yn (normalized)
    """
    parts = [str(class_id)]
    for x, y in polygon:
        # Normalize to [0, 1]
        nx = max(0.0, min(1.0, x / img_w))
        ny = max(0.0, min(1.0, y / img_h))
        parts.append(f"{nx:.6f}")
        parts.append(f"{ny:.6f}")
    return " ".join(parts)


def bbox_to_polygon(bbox: list) -> list:
    """Convert a bbox [x1, y1, x2, y2] to a 4-point polygon as fallback."""
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def process_image_for_segmentation(
    image_path: Path,
    label_path: Path,
    class_names: list,
    model_path: str = None,
    device: str = "cuda",
    conf: float = 0.25,
    poly_epsilon: float = 0.02,
    min_polygon_area: int = 50,
) -> dict:
    """Process a single image: read detection labels, run SAM3, extract polygons.
    
    Returns:
        Dict with:
        - success: bool
        - seg_labels: list of YOLO seg format strings
        - num_detections: int
        - num_polygons: int
        - error: str (if failed)
    """
    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        return {"success": False, "error": f"Could not read image: {image_path}"}
    
    img_h, img_w = img.shape[:2]
    
    # Read detection labels
    detections = read_yolo_detection_labels(label_path, img_w, img_h)
    
    if not detections:
        # No detections - return empty labels
        return {
            "success": True,
            "seg_labels": [],
            "num_detections": 0,
            "num_polygons": 0,
        }
    
    # Group bboxes by class
    bboxes_by_class = group_bboxes_by_class(detections)
    
    # Run SAM3 per class and collect all polygons
    all_seg_labels = []
    total_polygons = 0
    
    for class_id, bboxes in bboxes_by_class.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Run SAM3 with bboxes as exemplars for this class
        result = run_sam3(
            image_path=str(image_path),
            bboxes=bboxes,
            concepts=[class_name],
            model_path=model_path,
            device=device,
            conf=conf,
            log_callback=lambda msg: print(f"      {msg}"),
        )
        
        if not result.get("success"):
            print(f"    Warning: SAM3 failed for class '{class_name}': {result.get('error', 'Unknown error')}")
            # Fallback: use bboxes as rectangles
            for bbox in bboxes:
                polygon = bbox_to_polygon(bbox)
                label_line = polygon_to_yolo_seg_format(class_id, polygon, img_w, img_h)
                all_seg_labels.append(label_line)
                total_polygons += 1
            continue
        
        masks = result.get("masks", [])
        
        if masks:
            # Extract polygons from each mask
            for mask in masks:
                polygons = mask_to_polygons(
                    mask, 
                    epsilon_factor=poly_epsilon,
                    min_area=min_polygon_area
                )
                for polygon in polygons:
                    label_line = polygon_to_yolo_seg_format(class_id, polygon, img_w, img_h)
                    all_seg_labels.append(label_line)
                    total_polygons += 1
        else:
            # No masks returned - fallback to bbox rectangles
            print(f"    Warning: No masks for class '{class_name}', using bbox fallback")
            for bbox in bboxes:
                polygon = bbox_to_polygon(bbox)
                label_line = polygon_to_yolo_seg_format(class_id, polygon, img_w, img_h)
                all_seg_labels.append(label_line)
                total_polygons += 1
    
    return {
        "success": True,
        "seg_labels": all_seg_labels,
        "num_detections": len(detections),
        "num_polygons": total_polygons,
    }


def _find_split_img_dir(input_dir: Path, split: str) -> Path:
    """Find the image directory for a split, supporting both layouts:
    - Layout A: input_dir/images/{split}/
    - Layout B: input_dir/{split}/images/
    Also handles 'valid' as alias for 'val'.
    """
    candidates = [
        input_dir / "images" / split,
        input_dir / split / "images",
    ]
    # Also try 'valid' as alias for 'val'
    if split == "val":
        candidates.extend([
            input_dir / "images" / "valid",
            input_dir / "valid" / "images",
        ])
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # fallback


def _find_split_label_dir(input_dir: Path, split: str) -> Path:
    """Find the label directory for a split, supporting both layouts."""
    candidates = [
        input_dir / "labels" / split,
        input_dir / split / "labels",
    ]
    if split == "val":
        candidates.extend([
            input_dir / "labels" / "valid",
            input_dir / "valid" / "labels",
        ])
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # fallback


def setup_output_directories(input_dir: Path, output_dir: Path, symlink_images: bool = False):
    """Create output directory structure and copy/symlink images.
    
    Creates:
        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    """
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Copy or symlink images
    for split in ["train", "val", "test"]:
        src_img_dir = _find_split_img_dir(input_dir, split)
        dst_img_dir = output_dir / "images" / split
        
        if not src_img_dir.exists():
            print(f"  Warning: Source image directory not found: {src_img_dir}")
            continue
        
        image_files = [f for f in src_img_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        
        for img_file in image_files:
            dst_path = dst_img_dir / img_file.name
            if symlink_images:
                if not dst_path.exists():
                    dst_path.symlink_to(img_file.resolve())
            else:
                if not dst_path.exists():
                    shutil.copy2(img_file, dst_path)
        
        print(f"  Copied {len(image_files)} images to {dst_img_dir}")


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
    input_dir: Path,
    output_dir: Path,
    class_names: list,
    model_path: str = None,
    device: str = "cuda",
    conf: float = 0.25,
    poly_epsilon: float = 0.02,
    min_polygon_area: int = 50,
    max_images: int = None,
):
    """Process all images in a split (train/val/test)."""
    src_img_dir = _find_split_img_dir(input_dir, split)
    src_label_dir = _find_split_label_dir(input_dir, split)
    dst_label_dir = output_dir / "labels" / split
    
    if not src_img_dir.exists():
        print(f"  Skipping {split}: source directory not found")
        return {"processed": 0, "successful": 0, "failed": 0}
    
    # Find all image files
    image_files = sorted([
        f for f in src_img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split: {len(image_files)} images")
    print(f"{'=' * 60}")
    
    successful = 0
    failed = 0
    total_detections = 0
    total_polygons = 0
    
    for i, image_path in enumerate(image_files, 1):
        # Find corresponding label file
        label_path = src_label_dir / f"{image_path.stem}.txt"
        
        print(f"\n[{i}/{len(image_files)}] {image_path.name}")
        
        if not label_path.exists():
            print(f"  Warning: No label file found, creating empty label")
            # Create empty label file
            dst_label_path = dst_label_dir / f"{image_path.stem}.txt"
            dst_label_path.touch()
            successful += 1
            continue
        
        result = process_image_for_segmentation(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            model_path=model_path,
            device=device,
            conf=conf,
            poly_epsilon=poly_epsilon,
            min_polygon_area=min_polygon_area,
        )
        
        if result["success"]:
            successful += 1
            total_detections += result["num_detections"]
            total_polygons += result["num_polygons"]
            
            # Write segmentation labels
            dst_label_path = dst_label_dir / f"{image_path.stem}.txt"
            with open(dst_label_path, "w") as f:
                for label_line in result["seg_labels"]:
                    f.write(label_line + "\n")
            
            print(f"  Detections: {result['num_detections']}, Polygons: {result['num_polygons']}")
        else:
            failed += 1
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 60}")
    print(f"{split} split complete:")
    print(f"  Processed: {len(image_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total input detections: {total_detections}")
    print(f"  Total output polygons: {total_polygons}")
    print(f"{'=' * 60}")
    
    return {
        "processed": len(image_files),
        "successful": successful,
        "failed": failed,
        "total_detections": total_detections,
        "total_polygons": total_polygons,
    }


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Load data.yaml to get class names
    data_config = load_data_yaml(input_dir)
    class_names = data_config.get("names", [])
    
    if not class_names:
        print("Error: No class names found in data.yaml")
        sys.exit(1)
    
    print(f"Input dataset: {input_dir}")
    print(f"Output dataset: {output_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    print(f"Polygon epsilon: {args.poly_epsilon}")
    print(f"Min polygon area: {args.min_polygon_area}")
    
    # Setup output directories and copy/symlink images
    print(f"\nSetting up output directories...")
    setup_output_directories(input_dir, output_dir, symlink_images=args.symlink_images)
    
    # Create data.yaml
    create_data_yaml(output_dir, class_names)
    
    # Process each split
    summary = {}
    for split in ["train", "val", "test"]:
        result = process_split(
            split=split,
            input_dir=input_dir,
            output_dir=output_dir,
            class_names=class_names,
            model_path=args.model,
            device=args.device,
            conf=args.conf,
            poly_epsilon=args.poly_epsilon,
            min_polygon_area=args.min_polygon_area,
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
    total_detections = sum(s.get("total_detections", 0) for s in summary.values())
    total_polygons = sum(s.get("total_polygons", 0) for s in summary.values())
    
    print(f"Total images processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Total input detections: {total_detections}")
    print(f"Total output polygons: {total_polygons}")
    print(f"\nOutput dataset saved to: {output_dir}")
    print(f"\nTo train a YOLO segmentation model with focal loss:")
    print(f"  python scripts/04_train_model.py --data {output_dir}/data.yaml --task segment --loss focal")
    
    # Save summary JSON
    summary_path = output_dir / "creation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "classes": class_names,
            "device": args.device,
            "conf": args.conf,
            "poly_epsilon": args.poly_epsilon,
            "min_polygon_area": args.min_polygon_area,
            "summary": summary,
            "totals": {
                "processed": total_processed,
                "successful": total_successful,
                "failed": total_failed,
                "detections": total_detections,
                "polygons": total_polygons,
            }
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
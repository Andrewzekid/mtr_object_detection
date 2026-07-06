#!/usr/bin/env python3
"""
Visualize YOLO segmentation dataset annotations.

This script reads YOLO segmentation format labels and draws segmentation masks
and polygon outlines on images for visualization.

USAGE:
    python scripts/10_visualize_yolo_seg.py \
        --dataset-path ./train_yolo_seg \
        --split train \
        --output ./output/vis_yolo_seg \
        --max-images 20

YOLO SEGMENTATION FORMAT:
    Each label file contains lines in format:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn
    
    Where:
    - class_id: integer class index (0 to nc-1)
    - xi, yi: normalized polygon coordinates (0-1 range)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# Color palette for different classes (BGR format for OpenCV)
CLASS_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Light Green
    (255, 0, 128),    # Pink
    (0, 255, 128),    # Spring Green
    (128, 128, 255),  # Salmon
    (255, 128, 128),  # Light Cyan
    (128, 255, 128),  # Light Yellow
    (255, 255, 128),  # Light Magenta
]


def get_color_for_class(class_id):
    """Get a color for a given class ID, cycling through the palette."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def load_data_yaml(dataset_path):
    """Load data.yaml to get class names.
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        dict with class names and other config
    """
    yaml_path = Path(dataset_path) / "data.yaml"
    if not yaml_path.exists():
        print(f"Warning: data.yaml not found at {yaml_path}")
        return None
    
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    return data


def parse_yolo_seg_label(label_path, img_width, img_height):
    """Parse a YOLO segmentation label file.
    
    Args:
        label_path: Path to the label .txt file
        img_width: Width of the corresponding image
        img_height: Height of the corresponding image
        
    Returns:
        List of dicts with 'class_id' and 'polygon' (list of (x, y) pixel coordinates)
    """
    annotations = []
    
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * img_width
                    y = coords[i + 1] * img_height
                    polygon.append((int(x), int(y)))
            
            if len(polygon) >= 3:  # Valid polygon needs at least 3 points
                annotations.append({
                    "class_id": class_id,
                    "polygon": polygon
                })
    
    return annotations


def draw_legend(image, class_names, class_ids_present):
    """Draw a legend in the top right corner of the image.
    
    Args:
        image: Image to draw on
        class_names: List of class names
        class_ids_present: Set of class IDs present in the image
        
    Returns:
        Image with legend drawn
    """
    if not class_ids_present:
        return image
    
    img_h, img_w = image.shape[:2]
    
    # Filter to only classes present
    labels = [class_names[i] for i in sorted(class_ids_present) if i < len(class_names)]
    if not labels:
        return image
    
    # Legend parameters
    padding = 10
    box_size = 20
    text_padding = 5
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Calculate legend dimensions
    max_text_width = 0
    for label in labels:
        (text_width, _), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
    
    legend_width = box_size + text_padding * 2 + max_text_width + padding * 2
    legend_height = len(labels) * line_height + padding * 2
    
    # Position legend in top right corner
    legend_x = img_w - legend_width - 10
    legend_y = 10
    
    # Draw legend background (semi-transparent white)
    legend_bg = image.copy()
    cv2.rectangle(
        legend_bg, 
        (legend_x, legend_y), 
        (legend_x + legend_width, legend_y + legend_height),
        (255, 255, 255), 
        -1
    )
    cv2.addWeighted(legend_bg, 0.8, image, 0.2, 0, image)
    
    # Draw legend border
    cv2.rectangle(
        image,
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height),
        (0, 0, 0),
        2
    )
    
    # Draw each legend item
    for i, label in enumerate(labels):
        class_id = sorted(class_ids_present)[i]
        color = get_color_for_class(class_id)
        
        # Y position for this item
        item_y = legend_y + padding + i * line_height
        
        # Draw color box
        cv2.rectangle(
            image,
            (legend_x + padding, item_y),
            (legend_x + padding + box_size, item_y + box_size),
            color,
            -1
        )
        cv2.rectangle(
            image,
            (legend_x + padding, item_y),
            (legend_x + padding + box_size, item_y + box_size),
            (0, 0, 0),
            1
        )
        
        # Draw label text
        text_x = legend_x + padding + box_size + text_padding
        text_y = item_y + box_size - 3
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness
        )
    
    return image


def visualize_yolo_seg(dataset_path, split, output_dir, max_images=None, mask_alpha=0.4):
    """Visualize YOLO segmentation annotations.
    
    Args:
        dataset_path: Path to dataset root
        split: Dataset split (train/val/test)
        output_dir: Output directory for visualizations
        max_images: Maximum number of images to process (None for all)
        mask_alpha: Alpha value for mask overlay (0-1)
        
    Returns:
        dict with processing summary
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    # Load class names from data.yaml
    data_config = load_data_yaml(dataset_path)
    if data_config and "names" in data_config:
        class_names = data_config["names"]
    else:
        class_names = [f"class_{i}" for i in range(10)]
    
    print(f"Class names: {class_names}")
    
    # Find images and labels directories
    images_dir = dataset_path / "images" / split
    labels_dir = dataset_path / "labels" / split
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\nFound {len(image_files)} image(s) in {split} split")
    print(f"Output directory: {output_dir}")
    print(f"Mask alpha: {mask_alpha}")
    
    # Statistics
    successful = 0
    failed = 0
    total_annotations = 0
    class_counts = {i: 0 for i in range(len(class_names))}
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Error: Could not read image")
            failed += 1
            continue
        
        img_h, img_w = img.shape[:2]
        print(f"  Image size: {img_w}x{img_h}")
        
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"  Warning: No label file found: {label_path.name}")
            # Still save the image without annotations
            vis_path = output_dir / f"{image_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), img)
            successful += 1
            continue
        
        # Parse annotations
        annotations = parse_yolo_seg_label(label_path, img_w, img_h)
        print(f"  Found {len(annotations)} annotation(s)")
        
        # Create mask overlay
        mask_overlay = img.copy()
        class_ids_present = set()
        
        for ann in annotations:
            class_id = ann["class_id"]
            polygon = np.array(ann["polygon"], dtype=np.int32)
            color = get_color_for_class(class_id)
            class_ids_present.add(class_id)
            
            if class_id < len(class_names):
                class_counts[class_id] += 1
            
            # Draw filled polygon on mask overlay
            cv2.fillPoly(mask_overlay, [polygon], color)
        
        # Blend mask overlay with original image
        cv2.addWeighted(mask_overlay, mask_alpha, img, 1 - mask_alpha, 0, img)
        
        # Draw polygon outlines and labels
        for ann in annotations:
            class_id = ann["class_id"]
            polygon = np.array(ann["polygon"], dtype=np.int32)
            color = get_color_for_class(class_id)
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Draw polygon outline
            cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=2)
            
            # Calculate centroid for label
            M = cv2.moments(polygon)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw label background
                label_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, 
                              (cx - label_size[0] // 2 - 2, cy - label_size[1] // 2 - 2),
                              (cx + label_size[0] // 2 + 2, cy + label_size[1] // 2 + 2),
                              color, -1)
                cv2.putText(img, class_name, 
                            (cx - label_size[0] // 2, cy + label_size[1] // 2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw legend
        img = draw_legend(img, class_names, class_ids_present)
        
        # Save visualization
        vis_path = output_dir / f"{image_path.stem}_vis.jpg"
        cv2.imwrite(str(vis_path), img)
        print(f"  Saved: {vis_path}")
        
        total_annotations += len(annotations)
        successful += 1
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Visualization complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total annotations: {total_annotations}")
    print(f"\n  Class distribution:")
    for class_id, count in class_counts.items():
        if count > 0:
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"    {class_name}: {count}")
    print(f"\n  Output saved to: {output_dir}")
    
    return {
        "successful": successful,
        "failed": failed,
        "total_annotations": total_annotations,
        "class_counts": class_counts
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation dataset annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize training set
    python scripts/10_visualize_yolo_seg.py \\
        --dataset-path ./train_yolo_seg \\
        --split train \\
        --output ./output/vis_yolo_seg

    # Visualize validation set with limit
    python scripts/10_visualize_yolo_seg.py \\
        --dataset-path ./train_yolo_seg \\
        --split val \\
        --output ./output/vis_yolo_seg_val \\
        --max-images 10

    # Adjust mask transparency
    python scripts/10_visualize_yolo_seg.py \\
        --dataset-path ./train_yolo_seg \\
        --split train \\
        --output ./output/vis_yolo_seg \\
        --mask-alpha 0.6
        """,
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./train_yolo_seg",
        help="Path to YOLO dataset root directory (default: ./train_yolo_seg)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/vis_yolo_seg",
        help="Output directory for visualization images (default: ./output/vis_yolo_seg)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.4,
        help="Alpha value for mask overlay, 0-1 (default: 0.4)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("YOLO Segmentation Dataset Visualization")
    print("=" * 60)
    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output}")
    
    visualize_yolo_seg(
        dataset_path=args.dataset_path,
        split=args.split,
        output_dir=args.output,
        max_images=args.max_images,
        mask_alpha=args.mask_alpha,
    )
    
    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
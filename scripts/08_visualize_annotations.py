#!/usr/bin/env python3
"""
Visualize Qwen annotations by drawing bounding boxes on images.

This script reads per-class annotation JSON files and generates visualization
images with bounding boxes drawn for each class, using different colors per class.

USAGE:
    python scripts/08_visualize_annotations.py \
        --annotations-folder ./output/annotations/ \
        --image-folder ./qwen_test/ \
        --vis-output ./output/vis_annotations/

ANNOTATIONS FOLDER FORMAT:
    The folder should contain subfolders per image, each with JSON files per class:
    annotations/
    ├── image0/
    │   ├── Ceiling_light.json
    │   └── Advertisement_Board.json
    ├── image1/
    │   └── ...
    
    Each class JSON file should contain:
    {
        "class_name": "Ceiling_light",
        "image": "path/to/image0.jpg",
        "bboxes": [[x1, y1, x2, y2], ...]
    }
    
    NOTE: The bboxes should be in PIXEL COORDINATES (not normalized).

OUTPUT:
    - Visualization images with bounding boxes drawn
    - Different colors per class
    - Legend showing class-color mapping
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


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


def load_class_annotation(json_path):
    """Load a single class annotation file.
    
    Expected format:
    {
        "class_name": "Ceiling_light",
        "image": "path/to/image.jpg",
        "bboxes": [[x1, y1, x2, y2], ...]
    }
    
    Returns:
        dict with class_name, image_path, and bboxes
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: Annotation file not found: {json_file}")
        return None
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        print(f"Error: Annotation file must be a dict: {json_file}")
        return None
    
    class_name = data.get("class_name", "unknown")
    image_path = data.get("image", "")
    bboxes = data.get("bboxes", [])
    
    return {
        "class_name": class_name,
        "image": image_path,
        "bboxes": bboxes,
    }


def draw_legend(image, labels, label_to_color):
    """Draw a legend in the top right corner of the image.
    
    Args:
        image: Image to draw on
        labels: List of label names
        label_to_color: Mapping from label to BGR color
        
    Returns:
        Image with legend drawn
    """
    if not labels:
        return image
    
    img_h, img_w = image.shape[:2]
    
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
        color = label_to_color[label]
        
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
        text_y = item_y + box_size - 3  # Adjust for text baseline
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


def visualize_annotations(args):
    """Visualize annotations by drawing bounding boxes on images.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict with processing summary
    """
    annotations_folder = Path(args.annotations_folder)
    image_folder = Path(args.image_folder)
    vis_output = Path(args.vis_output)
    
    if not annotations_folder.exists():
        print(f"Error: Annotations folder not found: {annotations_folder}")
        sys.exit(1)
    
    if not image_folder.exists():
        print(f"Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    # Find all image subfolders
    image_subfolders = sorted([
        f for f in annotations_folder.iterdir()
        if f.is_dir()
    ])
    
    if not image_subfolders:
        print(f"Error: No image subfolders found in {annotations_folder}")
        sys.exit(1)
    
    print(f"Found {len(image_subfolders)} image annotation folder(s) in {annotations_folder}")
    
    # Setup output directory
    vis_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Image folder: {image_folder}")
    print(f"Visualization output: {vis_output}")
    
    # Process each image folder
    successful = 0
    failed = 0
    
    for i, image_subfolder in enumerate(image_subfolders, 1):
        print(f"\n[{i}/{len(image_subfolders)}] Processing: {image_subfolder.name}")
        
        # Find the corresponding image file
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = image_folder / f"{image_subfolder.name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        # If not found in image folder, check annotation files for image path
        if image_path is None:
            annotation_files = list(image_subfolder.glob("*.json"))
            if annotation_files:
                ann_data = load_class_annotation(annotation_files[0])
                if ann_data and ann_data.get("image"):
                    image_path = Path(ann_data["image"])
                    if not image_path.exists():
                        image_path = image_folder / image_path.name
                    if not image_path.exists():
                        image_path = None
        
        if image_path is None or not image_path.exists():
            print(f"  Error: Could not find image for folder: {image_subfolder.name}")
            failed += 1
            continue
        
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Error: Could not read image: {image_path}")
            failed += 1
            continue
        
        img_h, img_w = img.shape[:2]
        print(f"  Image: {image_path.name} ({img_w}x{img_h})")
        
        # Find all class annotation files
        annotation_files = sorted(image_subfolder.glob("*.json"))
        if not annotation_files:
            print(f"  Warning: No annotation files found in {image_subfolder}")
            continue
        
        print(f"  Found {len(annotation_files)} class annotation file(s)")
        
        # Build class list and color mapping
        class_data = []  # List of (class_name, bboxes, color)
        label_to_color = {}
        unique_labels = []
        
        for ann_file in annotation_files:
            ann_data = load_class_annotation(ann_file)
            if not ann_data:
                continue
            
            class_name = ann_data["class_name"]
            bboxes = ann_data["bboxes"]
            
            if not bboxes:
                print(f"    Skipping class '{class_name}': no bboxes")
                continue
            
            # Assign color
            if class_name not in label_to_color:
                class_id = len(unique_labels)
                label_to_color[class_name] = get_color_for_class(class_id)
                unique_labels.append(class_name)
            
            color = label_to_color[class_name]
            class_data.append((class_name, bboxes, color))
            print(f"    Class '{class_name}': {len(bboxes)} bbox(es)")
        
        # Draw bounding boxes
        for class_name, bboxes, color in class_data:
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    print(f"    Warning: Skipping invalid bbox: {bbox}")
                    continue
                
                # Clip to image bounds
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1 - label_size[1] - 8), 
                              (x1 + label_size[0] + 4, y1), color, -1)
                cv2.putText(img, class_name, (x1 + 2, y1 - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw legend
        img = draw_legend(img, unique_labels, label_to_color)
        
        # Save visualization
        vis_path = vis_output / f"{image_subfolder.name}_vis.jpg"
        cv2.imwrite(str(vis_path), img)
        print(f"  Saved visualization to: {vis_path}")
        successful += 1
    
    print(f"\n{'=' * 60}")
    print(f"Visualization complete!")
    print(f"  Total image folders: {len(image_subfolders)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output saved to: {vis_output}")
    
    return {"successful": successful, "failed": failed}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Qwen annotations by drawing bounding boxes on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize all annotations
    python scripts/08_visualize_annotations.py \\
        --annotations-folder ./output/annotations/ \\
        --image-folder ./qwen_test/ \\
        --vis-output ./output/vis_annotations/
        """,
    )
    
    parser.add_argument(
        "--annotations-folder",
        type=str,
        required=True,
        help="Path to folder with per-class annotations",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to folder containing source images",
    )
    parser.add_argument(
        "--vis-output",
        type=str,
        default="./output/vis_annotations",
        help="Output directory for visualization images (default: ./output/vis_annotations)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Visualizing Qwen annotations...")
    print(f"  Annotations folder: {args.annotations_folder}")
    print(f"  Image folder: {args.image_folder}")
    print(f"  Visualization output: {args.vis_output}")
    
    visualize_annotations(args)
    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
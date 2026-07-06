#!/usr/bin/env python3
"""
Visualize bounding boxes from YOLO format dataset.
Reads images and labels, draws bounding boxes with class names, and saves annotated images.
"""

import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path


def load_data_yaml(yaml_path):
    """Load class names from data.yaml file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


def get_color_for_class(class_id, num_classes=6):
    """Generate a distinct color for each class."""
    # Use HSV color space for better color distribution
    hue = int((class_id / max(num_classes, 1)) * 180)
    hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def draw_bounding_boxes(image, label_path, class_names, conf_threshold=0.0):
    """
    Draw bounding boxes on image from YOLO format label file.
    
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    """
    h, w = image.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return image
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert from normalized to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Get color and class name
            color = get_color_for_class(class_id, len(class_names))
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = class_name
            (label_w, label_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def visualize_dataset(dataset_path, split='train', output_dir=None, max_images=None):
    """
    Visualize bounding boxes for all images in a dataset split.
    
    Args:
        dataset_path: Path to the YOLO dataset root
        split: Dataset split ('train', 'val', or 'test')
        output_dir: Output directory for visualized images
        max_images: Maximum number of images to process (None for all)
    """
    dataset_path = Path(dataset_path)
    
    # Load class names
    yaml_path = dataset_path / 'data.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    
    class_names = load_data_yaml(yaml_path)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Set up paths
    images_dir = dataset_path / 'images' / split
    labels_dir = dataset_path / 'labels' / split
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path('output') / f'vis_yolo_{split}'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    image_files = sorted(list(set(image_files)))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images in {split} split")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            continue
        
        # Get corresponding label file
        label_path = labels_dir / (image_path.stem + '.txt')
        
        # Draw bounding boxes
        vis_image = draw_bounding_boxes(image.copy(), label_path, class_names)
        
        # Save output
        output_path = output_dir / f"{image_path.stem}_vis.jpg"
        cv2.imwrite(str(output_path), vis_image)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print(f"\nVisualization complete! Output saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset bounding boxes')
    parser.add_argument('--dataset', type=str, default='train_yolo_augmented',
                        help='Path to YOLO dataset root')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: output/vis_yolo_<split>)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    visualize_dataset(
        dataset_path=args.dataset,
        split=args.split,
        output_dir=args.output,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
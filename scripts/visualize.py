#!/usr/bin/env python3
"""
Unified visualization script for YOLO datasets and annotations.

Supports multiple visualization modes:
    --mode annotations     Visualize Qwen per-class annotations (bounding boxes)
    --mode yolo-detect     Visualize YOLO detection dataset (bounding boxes)
    --mode yolo-seg        Visualize YOLO segmentation dataset (polygons + masks)
    --mode predictions     Visualize YOLO-seg model predictions on images

USAGE:
    # Visualize Qwen annotations
    python scripts/visualize.py --mode annotations \\
        --annotations-folder ./output/annotations/ \\
        --image-folder ./qwen_test/ \\
        --output ./output/vis_annotations/

    # Visualize YOLO detection dataset
    python scripts/visualize.py --mode yolo-detect \\
        --dataset ./train_yolo_augmented \\
        --split train \\
        --output ./output/vis_yolo_train_detection

    # Visualize YOLO segmentation dataset
    python scripts/visualize.py --mode yolo-seg \\
        --dataset ./train_yolo_seg \\
        --split train \\
        --output ./output/vis_yolo_seg

    # Visualize model predictions
    python scripts/visualize.py --mode predictions \\
        --model runs/segment/.../weights/best.pt \\
        --images-dir test/images \\
        --output ./output/vis_predictions
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Shared Constants and Utilities
# =============================================================================

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


def get_hsv_color(class_id, num_classes):
    """Return an HSV-derived color for a class id."""
    hue = int((class_id / max(num_classes, 1)) * 180)
    hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


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


def load_data_yaml(dataset_path):
    """Load data.yaml to get class names."""
    yaml_path = Path(dataset_path) / "data.yaml"
    if not yaml_path.exists():
        return None
    
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Mode: annotations - Visualize Qwen per-class annotations
# =============================================================================

def load_class_annotation(json_path):
    """Load a single class annotation file."""
    json_file = Path(json_path)
    if not json_file.exists():
        return None
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        return None
    
    return {
        "class_name": data.get("class_name", "unknown"),
        "image": data.get("image", ""),
        "bboxes": data.get("bboxes", []),
    }


def visualize_annotations(args):
    """Visualize Qwen annotations by drawing bounding boxes on images."""
    annotations_folder = Path(args.annotations_folder)
    image_folder = Path(args.image_folder)
    vis_output = Path(args.output)
    
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
    
    print(f"Found {len(image_subfolders)} image annotation folder(s)")
    vis_output.mkdir(parents=True, exist_ok=True)
    
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
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Error: Could not read image: {image_path}")
            failed += 1
            continue
        
        img_h, img_w = img.shape[:2]
        
        # Find all class annotation files
        annotation_files = sorted(image_subfolder.glob("*.json"))
        if not annotation_files:
            continue
        
        # Build class list and color mapping
        class_data = []
        label_to_color = {}
        unique_labels = []
        
        for ann_file in annotation_files:
            ann_data = load_class_annotation(ann_file)
            if not ann_data:
                continue
            
            class_name = ann_data["class_name"]
            bboxes = ann_data["bboxes"]
            
            if not bboxes:
                continue
            
            if class_name not in label_to_color:
                class_id = len(unique_labels)
                label_to_color[class_name] = get_color_for_class(class_id)
                unique_labels.append(class_name)
            
            color = label_to_color[class_name]
            class_data.append((class_name, bboxes, color))
        
        # Draw bounding boxes
        for class_name, bboxes, color in class_data:
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
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
        successful += 1
    
    print(f"\nVisualization complete! Successful: {successful}, Failed: {failed}")
    return {"successful": successful, "failed": failed}


# =============================================================================
# Mode: yolo-detect - Visualize YOLO detection dataset
# =============================================================================

def visualize_yolo_detect(args):
    """Visualize bounding boxes from YOLO format dataset."""
    dataset_path = Path(args.dataset)
    
    # Load class names
    yaml_path = dataset_path / 'data.yaml'
    if not yaml_path.exists():
        print(f"Error: data.yaml not found at {yaml_path}")
        sys.exit(1)
    
    class_names = load_data_yaml(dataset_path).get('names', [])
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Set up paths
    images_dir = dataset_path / 'images' / args.split
    labels_dir = dataset_path / 'labels' / args.split
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f'*{ext}'))
    image_files = sorted(list(set(image_files)))
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images in {args.split} split")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Get corresponding label file
        label_path = labels_dir / (image_path.stem + '.txt')
        
        if label_path.exists():
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
                    color = get_hsv_color(class_id, len(class_names))
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    (label_w, label_h), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save output
        output_path = output_dir / f"{image_path.stem}_vis.jpg"
        cv2.imwrite(str(output_path), image)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print(f"\nVisualization complete! Output saved to: {output_dir}")


# =============================================================================
# Mode: yolo-seg - Visualize YOLO segmentation dataset
# =============================================================================

def parse_yolo_seg_label(label_path, img_width, img_height):
    """Parse a YOLO segmentation label file."""
    annotations = []
    
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:
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
            
            if len(polygon) >= 3:
                annotations.append({
                    "class_id": class_id,
                    "polygon": polygon
                })
    
    return annotations


def visualize_yolo_seg(args):
    """Visualize YOLO segmentation annotations."""
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    
    # Load class names from data.yaml
    data_config = load_data_yaml(dataset_path)
    if data_config and "names" in data_config:
        class_names = data_config["names"]
    else:
        class_names = [f"class_{i}" for i in range(10)]
    
    print(f"Class names: {class_names}")
    
    # Find images and labels directories
    images_dir = dataset_path / "images" / args.split
    labels_dir = dataset_path / "labels" / args.split
    
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
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nFound {len(image_files)} image(s) in {args.split} split")
    
    # Statistics
    successful = 0
    failed = 0
    total_annotations = 0
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            failed += 1
            continue
        
        img_h, img_w = img.shape[:2]
        
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            vis_path = output_dir / f"{image_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), img)
            successful += 1
            continue
        
        # Parse annotations
        annotations = parse_yolo_seg_label(label_path, img_w, img_h)
        
        # Create mask overlay
        mask_overlay = img.copy()
        class_ids_present = set()
        
        for ann in annotations:
            class_id = ann["class_id"]
            polygon = np.array(ann["polygon"], dtype=np.int32)
            color = get_color_for_class(class_id)
            class_ids_present.add(class_id)
            
            # Draw filled polygon on mask overlay
            cv2.fillPoly(mask_overlay, [polygon], color)
        
        # Blend mask overlay with original image
        mask_alpha = args.mask_alpha
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
        labels = [class_names[i] for i in sorted(class_ids_present) if i < len(class_names)]
        label_to_color = {name: get_color_for_class(i) for i, name in enumerate(class_names) if i in class_ids_present}
        img = draw_legend(img, labels, label_to_color)
        
        # Save visualization
        vis_path = output_dir / f"{image_path.stem}_vis.jpg"
        cv2.imwrite(str(vis_path), img)
        
        total_annotations += len(annotations)
        successful += 1
    
    print(f"\nVisualization complete! Successful: {successful}, Failed: {failed}")
    print(f"Total annotations: {total_annotations}")


# =============================================================================
# Mode: predictions - Visualize YOLO-seg model predictions
# =============================================================================

def visualize_predictions(args):
    """Visualize YOLO segmentation model predictions on test images."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        sys.exit(1)
    
    model_path = Path(args.model)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: Images dir not found: {images_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Force model onto the requested device (e.g. GPU 0) at load time so
    # nvidia-smi shows utilization and per-image predict() stays on GPU.
    import torch
    device = args.device
    if device is None or device == "":
        device = "0"
    try:
        if device.isdigit() or ":" in device:
            model.to(f"cuda:{device}" if device.isdigit() else f"cuda:{device.split(':')[1]}")
        elif device.startswith("cuda"):
            model.to(device)
        print(f"Model moved to device: {device}")
    except Exception as e:
        print(f"Warning: could not move model to device '{device}': {e}")
    
    # Get class names from model
    class_names = list(model.names.values()) if hasattr(model, 'names') and model.names else [f"class_{i}" for i in range(10)]
    print(f"Classes: {class_names}")
    
    # Parse excluded classes
    excluded_cls_ids = set()
    if getattr(args, "exclude_classes", None):
        tokens = [t.strip() for t in args.exclude_classes.split(",") if t.strip()]
        for tok in tokens:
            if tok.isdigit():
                excluded_cls_ids.add(int(tok))
            elif tok in class_names:
                excluded_cls_ids.add(class_names.index(tok))
            else:
                print(f"Warning: exclude-classes token '{tok}' did not match any class name or ID; ignoring")
        print(f"Excluded class IDs: {excluded_cls_ids} ({[class_names[i] for i in sorted(excluded_cls_ids) if i < len(class_names)]})")
    
    # Find images
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images")
    print(f"Confidence: {args.conf}")
    
    num_classes = len(class_names)
    successful = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        
        if not results:
            continue
        
        result = results[0]
        img = result.orig_img.copy()
        
        # Draw masks
        if result.masks is not None and len(result.masks) > 0:
            mask_overlay = img.copy()
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(masks)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                if cls_id in excluded_cls_ids:
                    continue
                
                if args.color_scheme == "hsv":
                    color = get_hsv_color(cls_id, num_classes)
                else:
                    color = get_color_for_class(cls_id)
                
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                
                # Get mask as binary array
                mask = masks.data[i].cpu().numpy().astype(np.uint8)
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Draw filled mask
                mask_bool = mask > 0
                mask_overlay[mask_bool] = color
                
                # Draw contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color, 2)
                
                # Draw label at centroid
                M = cv2.moments(contours[0]) if contours else None
                if M and M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (cx - tw//2 - 2, cy - th//2 - 2), (cx + tw//2 + 2, cy + th//2 + 2), color, -1)
                    cv2.putText(img, label, (cx - tw//2, cy + th//2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Blend mask overlay
            cv2.addWeighted(mask_overlay, args.mask_alpha, img, 1 - args.mask_alpha, 0, img)
        
        # Save
        out_path = output_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), img)
        successful += 1
    
    print(f"\nVisualization complete! {successful}/{len(image_files)} images processed.")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified visualization script for YOLO datasets and annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize Qwen annotations
    python scripts/visualize.py --mode annotations \\
        --annotations-folder ./output/annotations/ \\
        --image-folder ./qwen_test/ \\
        --output ./output/vis_annotations/

    # Visualize YOLO detection dataset
    python scripts/visualize.py --mode yolo-detect \\
        --dataset ./train_yolo_augmented \\
        --split train \\
        --output ./output/vis_yolo_detection

    # Visualize YOLO segmentation dataset
    python scripts/visualize.py --mode yolo-seg \\
        --dataset ./train_yolo_seg \\
        --split train \\
        --output ./output/vis_yolo_seg

    # Visualize model predictions
    python scripts/visualize.py --mode predictions \\
        --model runs/segment/.../weights/best.pt \\
        --images-dir test/images \\
        --output ./output/vis_predictions
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["annotations", "yolo-detect", "yolo-seg", "predictions"],
        help="Visualization mode",
    )
    
    # Common arguments
    parser.add_argument("--output", "-o", type=str, default="./output/visualization", 
                        help="Output directory")
    parser.add_argument("--max-images", type=int, default=None, 
                        help="Maximum number of images to process")
    
    # Annotations mode arguments
    parser.add_argument("--annotations-folder", type=str, 
                        help="Path to folder with per-class annotations (for annotations mode)")
    parser.add_argument("--image-folder", type=str, 
                        help="Path to folder containing source images (for annotations mode)")
    
    # YOLO dataset mode arguments
    parser.add_argument("--dataset", type=str, 
                        help="Path to YOLO dataset root (for yolo-detect/yolo-seg modes)")
    parser.add_argument("--split", type=str, default="train", 
                        choices=["train", "val", "test"],
                        help="Dataset split to visualize")
    parser.add_argument("--mask-alpha", type=float, default=0.4, 
                        help="Alpha value for mask overlay (for yolo-seg/predictions)")
    
    # Predictions mode arguments
    parser.add_argument("--model", "-m", type=str, 
                        help="Path to trained .pt model (for predictions mode)")
    parser.add_argument("--images-dir", "-i", type=str, 
                        help="Directory of test images (for predictions mode)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold (for predictions mode)")
    parser.add_argument("--iou", type=float, default=0.45, 
                        help="IoU threshold for NMS (for predictions mode)")
    parser.add_argument("--imgsz", type=int, default=640, 
                        help="Image size for inference (for predictions mode)")
    parser.add_argument("--device", type=str, default="0", 
                        help="Device (for predictions mode)")
    parser.add_argument("--color-scheme", type=str, choices=("distinct", "hsv"), 
                        default="distinct",
                        help="Color palette (for predictions mode)")
    parser.add_argument("--exclude-classes", type=str, default=None,
                        help="Comma-separated class names or IDs to exclude from visualization (for predictions mode). Example: 'Ticket Gate' or '5' or 'Ticket Gate,TV'")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"Visualization Mode: {args.mode}")
    print("=" * 60)
    
    if args.mode == "annotations":
        if not args.annotations_folder or not args.image_folder:
            print("Error: --annotations-folder and --image-folder required for annotations mode")
            sys.exit(1)
        visualize_annotations(args)
    
    elif args.mode == "yolo-detect":
        if not args.dataset:
            print("Error: --dataset required for yolo-detect mode")
            sys.exit(1)
        visualize_yolo_detect(args)
    
    elif args.mode == "yolo-seg":
        if not args.dataset:
            print("Error: --dataset required for yolo-seg mode")
            sys.exit(1)
        visualize_yolo_seg(args)
    
    elif args.mode == "predictions":
        if not args.model or not args.images_dir:
            print("Error: --model and --images-dir required for predictions mode")
            sys.exit(1)
        visualize_predictions(args)
    
    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
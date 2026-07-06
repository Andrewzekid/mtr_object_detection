#!/usr/bin/env python3
"""
Run SAM3 segmentation on an image or folder of images.

This script runs Ultralytics SAM3SemanticPredictor on an input image,
optionally using bounding boxes as exemplars for segmentation.

USAGE:
    # Single image mode
    python scripts/06_run_sam3.py --image ./sample.jpg --bbox 100 100 300 300
    python scripts/06_run_sam3.py --image ./sample.jpg --bbox-json bboxes.json
    python scripts/06_run_sam3.py --image ./sample.jpg --concept "the red car"

    # Batch mode (folder of images)
    python scripts/06_run_sam3.py --image-folder ./images/ --segmented-output ./output/segmented/

    # Individual class mode (per-class annotations from Qwen --split-by-class)
    # Note: --image-folder is required because annotations only contain bounding boxes, not images.
    #       The script matches images by filename (e.g., annotations/image0/ -> images/image0.jpg)
    python scripts/06_run_sam3.py --annotations-folder ./annotations/ --image-folder ./images/ --segmented-output ./output/segmented/

OUTPUT:
    - Single image mode: Segmented regions with bounding boxes, areas, and centers
    - Batch mode: Folder of segmented images
    - Individual class mode: Folder of segmented images (combined from all classes)

ANNOTATIONS FOLDER FORMAT (for --annotations-folder):
    The folder should contain subfolders per image, each with JSON files per class:
    annotations/
    ├── image1/
    │   ├── Ceiling_light.json
    │   ├── Sign.json
    │   └── Advertisement_Board.json
    ├── image2/
    │   └── ...
    
    Each class JSON file should contain:
    {
        "class_name": "Ceiling_light",
        "image": "path/to/image1.jpg",
        "bboxes": [[x1, y1, x2, y2], ...]
    }
    
    NOTE: The annotation files contain bounding boxes in PIXEL COORDINATES (not normalized).
    The bboxes are in [x1, y1, x2, y2] format.
    The annotation files contain bounding boxes only, NOT the actual images.
    You must provide --image-folder to specify where the source images are located.
    The script matches images by filename: annotations/image1/ -> image_folder/image1.jpg

BBOX JSON FORMAT (for --bbox-json):
    The JSON file should contain a list of bounding boxes:
    [
        [480.0, 290.0, 590.0, 650.0],
        [539.0, 599.0, 589.0, 639.0]
    ]
    
    Or a wrapped format:
    {
        "image_path": "data/sample.jpg",
        "bboxes": [
            [480.0, 290.0, 590.0, 650.0],
            [539.0, 599.0, 589.0, 639.0]
        ]
    }
    
    Or Qwen output format:
    {
        "model": "qwen3.6:27b",
        "format": "json",
        "raw_response": "...",
        "parsed_output": [
            {"bbox_2d": [275, 295, 493, 592], "label": "Advertisement Board"},
            {"bbox_2d": [483, 359, 583, 545], "label": "Advertisement Board"}
        ]
    }
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models_inference import run_sam3


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


def create_colored_overlay(image, masks, detections, concepts):
    """Create a colored overlay with different colors per class and a legend.
    
    Args:
        image: Original image (BGR format)
        masks: List of binary masks from SAM3
        detections: List of detection dicts with 'label' keys
        concepts: List of concept names used
        
    Returns:
        Overlay image with colored masks and legend
    """
    overlay = image.copy()
    
    # Create a mapping from concept/label to class ID
    unique_labels = []
    for det in detections:
        label = det.get("label", "unknown")
        if label not in unique_labels:
            unique_labels.append(label)
    
    # If no labels from detections, use concepts
    if not unique_labels and concepts:
        unique_labels = list(concepts)
    
    # If still no labels, use a single "object" label
    if not unique_labels:
        unique_labels = ["object"]
    
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    # Apply colored masks
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            if mask is not None:
                # Determine the label for this mask
                if i < len(detections):
                    label = detections[i].get("label", "unknown")
                elif i < len(concepts):
                    label = concepts[i]
                else:
                    label = "object"
                
                class_id = label_to_id.get(label, 0)
                color = get_color_for_class(class_id)
                
                # Ensure mask is binary and same size as image
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                
                mask_bool = mask > 0
                
                # Apply semi-transparent colored overlay
                alpha = 0.5
                overlay[mask_bool] = (
                    overlay[mask_bool] * (1 - alpha) + 
                    np.array(color) * alpha
                ).astype(np.uint8)
    
    # Draw legend in top right corner
    overlay = draw_legend(overlay, unique_labels, label_to_id)
    
    return overlay


def draw_legend(image, labels, label_to_id):
    """Draw a legend in the top right corner of the image.
    
    Args:
        image: Image to draw on
        labels: List of label names
        label_to_id: Mapping from label to class ID
        
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
        class_id = label_to_id.get(label, 0)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 segmentation on an image or folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image mode - Segment with a single bounding box
    python scripts/06_run_sam3.py --image ./sample.jpg --bbox 100 100 300 300

    # Single image mode - Segment with multiple bbox exemplars (same concept)
    python scripts/06_run_sam3.py --image ./sample.jpg --bbox 100 100 300 300 200 150 400 350

    # Single image mode - Segment with text concept
    python scripts/06_run_sam3.py --image ./sample.jpg --concept "the red car"

    # Single image mode - Segment with both bbox and concept
    python scripts/06_run_sam3.py --image ./sample.jpg --bbox 100 100 300 300 --concept "car"

    # Use CPU instead of GPU
    python scripts/06_run_sam3.py --image ./sample.jpg --device cpu

    # Multiple concepts (space-separated)
    python scripts/06_run_sam3.py --image ./test.jpg --concept "car" "person" "bicycle"

    # Multiple concepts with bbox json file
    python scripts/06_run_sam3.py --image ./test.jpg --bbox-json qwenoutput.json --concept "Advertisement Board" "Sign"

    # Batch mode - Process all images in a folder
    python scripts/06_run_sam3.py --image-folder ./images/ --segmented-output ./output/segmented/

    # Individual class mode - Process per-class annotations from Qwen --split-by-class
    python scripts/06_run_sam3.py --annotations-folder ./annotations/ --image-folder ./images/ --segmented-output ./output/segmented/

    #Multiclass
    python scripts/06_run_sam3.py --image ./test_multiclass.jpg --bbox-json qwenoutputmulticlass.json --concept "Advertisement Display" "Sign" "Monitor" "Ceiling light" "Ticket Gate" --conf 0.5
        """,
    )

    # Input mode group - mutually exclusive (single image vs batch mode)
    # Note: --annotations-folder is NOT in this group because it requires --image-folder
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to input image (single image mode)",
    )
    input_group.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Path to folder containing images (batch mode). "
             "Also required with --annotations-folder to provide source images.",
    )
    parser.add_argument(
        "--annotations-folder",
        type=str,
        default=None,
        help="Path to folder with per-class annotations (individual class mode). "
             "Requires --image-folder because annotations only contain bounding boxes, not images. "
             "Images are matched by filename: annotations/image1/ -> image_folder/image1.jpg",
    )
    parser.add_argument(
        "--bbox", "-b",
        type=float,
        nargs="+",
        default=None,
        help="Bounding box(es) as [x1 y1 x2 y2] or multiple: [x1 y1 x2 y2 x1 y1 x2 y2 ...]",
    )
    parser.add_argument(
        "--bbox-json",
        type=str,
        default=None,
        help="Path to JSON file containing bounding boxes. Format: [[x1,y1,x2,y2], ...] or {\"bboxes\": [[x1,y1,x2,y2], ...]}",
    )
    parser.add_argument(
        "--concept", "-c",
        type=str,
        nargs="+",
        default=None,
        help="Concept label(s) for segmentation (e.g., 'car', 'person')",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to SAM3 model weights (default: ./core/sam3/models/sam3-model/sam3.pt)",
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
        help="Confidence threshold for segmentation (default: 0.25)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output/sam3_results",
        help="Output directory for results (default: ./output/sam3_results)",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=True,
        help="Save mask overlay image (default: True)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_false",
        dest="save_overlay",
        help="Don't save mask overlay image",
    )
    parser.add_argument(
        "--segmented-output",
        type=str,
        default="./output/segmented",
        help="Output directory for segmented images (batch/annotations mode). Default: ./output/segmented",
    )

    return parser.parse_args()


def parse_bboxes(bbox_list):
    """Parse flat bbox list into list of [x1, y1, x2, y2] lists."""
    if bbox_list is None:
        return None

    if len(bbox_list) % 4 != 0:
        print("Error: Bounding boxes must be specified as groups of 4 values: x1 y1 x2 y2")
        sys.exit(1)
    
    bboxes = []
    for i in range(0, len(bbox_list), 4):
        bboxes.append(bbox_list[i:i+4])
    
    return bboxes


def load_bboxes_from_json(json_path):
    """Load bounding boxes from a JSON file.
    
    Supports multiple formats:
    1. Bare list: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    2. Wrapped: {"image_path": "...", "bboxes": [[x1, y1, x2, y2], ...]}
    3. Qwen output: {"parsed_output": [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}, ...]}
    
    Returns:
        list: List of bounding boxes in [x1, y1, x2, y2] format (pixel coordinates)
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Handle bare list format
    if isinstance(data, list):
        return data
    
    # Handle wrapped format
    if isinstance(data, dict):
        # Qwen output format with parsed_output
        if "parsed_output" in data:
            parsed = data["parsed_output"]
            if isinstance(parsed, list):
                bboxes = []
                for item in parsed:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        bboxes.append(item["bbox_2d"])
                    elif isinstance(item, list) and len(item) == 4:
                        bboxes.append(item)
                return bboxes
        
        # Standard wrapped format
        if "bboxes" in data:
            return data["bboxes"]
        elif "bbox" in data:
            bbox = [data["bbox"]] if isinstance(data["bbox"], list) and len(data["bbox"]) == 4 else data["bbox"]
            return bbox
    
    print("Error: JSON file must contain a list of bboxes or a dict with 'bboxes' or 'parsed_output' key")
    sys.exit(1)


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


def process_single_image_sam3(args, image_path, bboxes=None, concepts=None):
    """Process a single image with SAM3.
    
    Args:
        args: Parsed command line arguments
        image_path: Path to input image
        bboxes: Optional list of bounding boxes (pixel coordinates)
        concepts: Optional list of concept labels
    
    Returns:
        dict with processing results including masks and detections
    """
    # Get image dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Error: Could not read image: {image_path}")
        return {"success": False, "error": f"Could not read image: {image_path}"}
    
    # Run SAM3 (bboxes are already in pixel coordinates)
    result = run_sam3(
        image_path=str(image_path),
        bboxes=bboxes,
        concepts=concepts,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        log_callback=lambda msg: print(f"    {msg}"),
    )
    
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Unknown error")}
    
    return {
        "success": True,
        "image": img,
        "masks": result.get("masks", []),
        "detections": result.get("detections", []),
        "combined_mask": result.get("combined_mask"),
    }


def process_image_folder_batch(args):
    """Process all images in a folder with SAM3 (batch mode).
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict with batch processing summary
    """
    folder_path = Path(args.image_folder)
    
    # Find all image files
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if not image_files:
        print(f"Error: No image files found in {folder_path}")
        print(f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s) in {folder_path}")
    
    # Setup output directory
    output_dir = Path(args.segmented_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        result = process_single_image_sam3(args, image_path)
        
        if result["success"]:
            successful += 1
            # Save segmented image
            if result.get("combined_mask") is not None:
                mask = result["combined_mask"]
                if isinstance(mask, np.ndarray) and mask.size > 0:
                    # Create colored overlay
                    overlay = create_colored_overlay(
                        image=result["image"],
                        masks=result["masks"],
                        detections=result["detections"],
                        concepts=[d.get("label", "object") for d in result["detections"]],
                    )
                    output_path = output_dir / f"{image_path.stem}_segmented.png"
                    cv2.imwrite(str(output_path), overlay)
                    print(f"  Saved segmented image to: {output_path}")
        else:
            failed += 1
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 60}")
    print(f"Batch processing complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output saved to: {output_dir}")
    
    return {"successful": successful, "failed": failed}


def validate_bbox(bbox, img_w, img_h):
    """Validate and clip a bounding box to image bounds.

    Args:
        bbox: [x1, y1, x2, y2] bounding box
        img_w: Image width
        img_h: Image height

    Returns:
        Validated and clipped bbox, or None if invalid
    """
    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox

    # Check for valid dimensions
    if x2 <= x1 or y2 <= y1:
        return None

    # Clip to image bounds
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    # Check if bbox has any area after clipping
    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def process_annotations_folder(args):
    """Process per-class annotations folder with SAM3 (individual class mode).
    
    For each image folder in the annotations folder:
    - For each class JSON file:
        - Load class name and bboxes
        - Run SAM3 with those bboxes and class name as concept
        - Store segmentation mask
    - Combine all masks for that image into a single segmented output
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict with processing summary
    """
    annotations_folder = Path(args.annotations_folder)
    image_folder = Path(args.image_folder) if args.image_folder else None
    
    if not annotations_folder.exists():
        print(f"Error: Annotations folder not found: {annotations_folder}")
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
    output_dir = Path(args.segmented_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    
    # Process each image folder
    successful = 0
    failed = 0
    total_classes_processed = 0
    
    for i, image_subfolder in enumerate(image_subfolders, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(image_subfolders)}] Processing image folder: {image_subfolder.name}")
        
        # Find the corresponding image file
        image_path = None
        if image_folder:
            # Look for image in the image folder
            for ext in IMAGE_EXTENSIONS:
                candidate = image_folder / f"{image_subfolder.name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        
        # If not found in image folder, check annotation files for image path
        if image_path is None:
            # Try to get image path from first annotation file
            annotation_files = list(image_subfolder.glob("*.json"))
            if annotation_files:
                ann_data = load_class_annotation(annotation_files[0])
                if ann_data and ann_data.get("image"):
                    image_path = Path(ann_data["image"])
                    if not image_path.exists():
                        # Try relative to image folder
                        if image_folder:
                            image_path = image_folder / image_path.name
                        if not image_path.exists():
                            image_path = None
        
        if image_path is None or not image_path.exists():
            print(f"  Error: Could not find image for folder: {image_subfolder.name}")
            failed += 1
            continue
        
        print(f"  Image: {image_path}")
        
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  Error: Could not read image: {image_path}")
            failed += 1
            continue
        
        img_h, img_w = img.shape[:2]
        
        # Find all class annotation files
        annotation_files = sorted(image_subfolder.glob("*.json"))
        if not annotation_files:
            print(f"  Warning: No annotation files found in {image_subfolder}")
            continue
        
        print(f"  Found {len(annotation_files)} class annotation file(s)")
        
        # Collect all masks and detections from all classes
        all_masks = []
        all_detections = []
        all_concepts = []
        
        # Process each class annotation file
        for ann_file in annotation_files:
            ann_data = load_class_annotation(ann_file)
            if not ann_data:
                print(f"  Warning: Could not load annotation file: {ann_file}")
                continue
            
            class_name = ann_data["class_name"]
            bboxes = ann_data["bboxes"]
            
            if not bboxes:
                print(f"  Skipping class '{class_name}': no bboxes")
                continue
            
            # Validate and clip bboxes to image bounds
            valid_bboxes = []
            for bbox in bboxes:
                validated = validate_bbox(bbox, img_w, img_h)
                if validated:
                    valid_bboxes.append(validated)
                else:
                    print(f"    Warning: Skipping invalid bbox: {bbox}")
            
            if not valid_bboxes:
                print(f"  Skipping class '{class_name}': no valid bboxes after validation")
                continue
            
            print(f"  Processing class: {class_name} ({len(valid_bboxes)} bbox(es))")
            
            # Run SAM3 for this class
            result = run_sam3(
                image_path=str(image_path),
                bboxes=valid_bboxes,
                concepts=[class_name],
                model_path=args.model,
                device=args.device,
                conf=args.conf,
                log_callback=lambda msg: print(f"    {msg}"),
            )
            
            if result.get("success"):
                masks = result.get("masks", [])
                detections = result.get("detections", [])
                
                all_masks.extend(masks)
                all_detections.extend(detections)
                all_concepts.append(class_name)
                total_classes_processed += 1
                
                print(f"    Found {len(detections)} detection(s) for class '{class_name}'")
            else:
                print(f"    Error running SAM3 for class '{class_name}': {result.get('error', 'Unknown error')}")
        
        # Combine all masks and create segmented output
        if all_masks:
            # Create combined overlay with all classes
            overlay = create_colored_overlay(
                image=img,
                masks=all_masks,
                detections=all_detections,
                concepts=all_concepts,
            )
            output_path = output_dir / f"{image_subfolder.name}_segmented.png"
            cv2.imwrite(str(output_path), overlay)
            print(f"  Saved combined segmented image to: {output_path}")
            successful += 1
        else:
            print(f"  Warning: No masks found for image: {image_subfolder.name}")
            # Save original image with no segmentation
            output_path = output_dir / f"{image_subfolder.name}_segmented.png"
            cv2.imwrite(str(output_path), img)
            successful += 1
        
        # Save JSON results for this image
        json_output = {
            "image": str(image_path),
            "image_name": image_subfolder.name,
            "concepts": all_concepts,
            "num_detections": len(all_detections),
            "num_masks": len(all_masks),
            "detections": [
                {
                    "label": d.get("label"),
                    "bbox": d.get("bbox"),
                    "confidence": d.get("confidence"),
                    "area": d.get("area"),
                    "center": d.get("center"),
                }
                for d in all_detections
            ],
            "input_annotations": [
                {
                    "class_name": load_class_annotation(ann_file)["class_name"],
                    "bboxes": load_class_annotation(ann_file)["bboxes"],
                }
                for ann_file in annotation_files
                if load_class_annotation(ann_file)
            ],
        }
        json_path = output_dir / f"{image_subfolder.name}_results.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"  Saved JSON results to: {json_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Annotations processing complete!")
    print(f"  Total image folders: {len(image_subfolders)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total classes processed: {total_classes_processed}")
    print(f"  Output saved to: {output_dir}")
    
    return {"successful": successful, "failed": failed, "total_classes": total_classes_processed}


def main():
    args = parse_args()
    
    # Determine mode
    if args.annotations_folder:
        # Individual class mode (annotations folder)
        if not args.image_folder:
            print("Error: --image-folder is required when using --annotations-folder")
            sys.exit(1)
        
        print("Running SAM3 in individual class mode (annotations folder)...")
        print(f"  Annotations folder: {args.annotations_folder}")
        print(f"  Image folder: {args.image_folder}")
        
        process_annotations_folder(args)
        print("\nSAM3 annotations processing completed successfully!")
        return
    
    elif args.image_folder:
        # Batch mode (image folder)
        print("Running SAM3 in batch mode (image folder)...")
        print(f"  Image folder: {args.image_folder}")
        
        process_image_folder_batch(args)
        print("\nSAM3 batch processing completed successfully!")
        return
    
    # Single image mode
    if not args.image:
        print("Error: --image, --image-folder, or --annotations-folder is required")
        sys.exit(1)
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Get image dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)
    
    # Parse bounding boxes from --bbox or --bbox-json (already in pixel coordinates)
    bboxes = None
    if args.bbox:
        bboxes = parse_bboxes(args.bbox)
    elif args.bbox_json:
        bboxes = load_bboxes_from_json(args.bbox_json)
    
    # Parse concepts
    concepts = args.concept
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running SAM3 segmentation on: {image_path}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    
    if bboxes:
        print(f"Bounding boxes: {bboxes}")
    if concepts:
        print(f"Concepts: {concepts}")
    
    # Run SAM3
    result = run_sam3(
        image_path=str(image_path),
        bboxes=bboxes,
        concepts=concepts,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        log_callback=lambda msg: print(f"  {msg}"),
    )
    
    if not result.get("success"):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Print results
    detections = result.get("detections", [])
    masks = result.get("masks", [])
    concepts_used = result.get("concepts", [])
    
    print(f"\nResults:")
    print(f"  Concepts queried: {', '.join(concepts_used) if concepts_used else 'N/A'}")
    print(f"  Detections found: {len(detections)}")
    print(f"  Masks found: {len(masks)}")
    
    if detections:
        # Group detections by label
        detections_by_label = {}
        for det in detections:
            label = det.get("label", "unknown")
            if label not in detections_by_label:
                detections_by_label[label] = []
            detections_by_label[label].append(det)
        
        print(f"\nDetections by label:")
        for label, dets in detections_by_label.items():
            print(f"  {label}: {len(dets)} instance(s)")
        
        print(f"\nDetection details:")
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            conf = det.get("confidence", 0)
            area = det.get("area", 0)
            print(f"  Detection {i+1} [{det.get('label', 'unknown')}]:")
            print(f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            print(f"    Confidence: {conf:.2f}")
            print(f"    Area: {area:.1f} pixels")
    else:
        print("\nNo detections found. Try different concepts or lower confidence.")
    
    # Save overlay with per-class colors and legend
    if args.save_overlay:
        # Create colored overlay with legend using the masks and detections
        overlay = create_colored_overlay(
            image=img,
            masks=masks,
            detections=detections,
            concepts=concepts_used,
        )
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"\nOverlay saved to: {overlay_path}")
    
    # Save JSON results
    json_output = {
        "image": str(image_path),
        "concepts": concepts_used,
        "num_detections": len(detections),
        "num_masks": len(masks),
        "detections": [
            {
                "label": d.get("label"),
                "bbox": d.get("bbox"),
                "confidence": d.get("confidence"),
                "area": d.get("area"),
                "center": d.get("center"),
            }
            for d in detections
        ],
    }
    
    json_path = output_dir / f"{image_path.stem}_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    print("\nSAM3 segmentation completed successfully!")


if __name__ == "__main__":
    main()
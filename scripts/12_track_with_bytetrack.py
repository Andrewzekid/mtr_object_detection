#!/usr/bin/env python3
"""
Tracking script using Ultralytics ByteTrack with YOLO segmentation model.
Performs object tracking on training data with visualization.
Outputs results in COCO-style format.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import cv2
import numpy as np


def parse_args():
    """Parse command line arguments."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    parser = argparse.ArgumentParser(
        description="Track objects in images using ByteTrack with YOLO segmentation model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(project_dir / "runs" / "segment" / "output" / "training" / "yolo_training" / "weights" / "best.pt"),
        help="Path to the trained YOLO model weights (default: runs/segment/output/training/yolo_training/weights/best.pt)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_dir / "MTR_dataset"),
        help="Path to the directory containing images to track (default: MTR_dataset)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_dir / "output" / "tracking" / "bytetrack"),
        help="Directory to save tracking results (default: output/tracking/bytetrack)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference (default: 640)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output video (default: 10)"
    )
    
    return parser.parse_args()


def run_tracking(args):
    """Run ByteTrack tracking on training data using the trained YOLO segmentation model."""
    
    # Define paths from arguments
    model_path = Path(args.model)
    train_data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Verify training data exists
    if not train_data_path.exists():
        print(f"Error: Training data not found at {train_data_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Image size: {args.imgsz}")
    
    # Load the trained YOLO segmentation model
    model = YOLO(str(model_path))
    
    # Get list of images in training data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in train_data_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    print(f"Found {len(image_files)} images to process")
    
    # Get class names from model
    class_names = model.names if hasattr(model, 'names') else {}
    
    # Build COCO-style categories
    categories = [
        {"id": int(cat_id), "name": cat_name}
        for cat_id, cat_name in class_names.items()
    ]
    
    # COCO-style output structures
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    
    # Process each image with tracking
    for idx, image_path in enumerate(image_files):
        print(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")
        
        # Read the image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Warning: Could not read {image_path}")
            continue
        
        img_h, img_w = frame.shape[:2]
        
        # Add image entry to COCO format
        image_id = idx + 1
        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_w,
            "height": img_h
        })
        
        # Run tracking with ByteTrack
        # track=True enables tracking, persist=True maintains track IDs across frames
        results = model.track(
            source=frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False
        )
        
        # Process results and visualize
        if results and len(results) > 0:
            result = results[0]
            
            # Get the annotated frame with tracking
            annotated_frame = result.plot()
            
            # Extract tracking data
            if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Extract masks/polygons if available
                mask_polygons = None
                if result.masks is not None and hasattr(result.masks, 'xy'):
                    mask_polygons = result.masks.xy  # List of polygon coordinates
                
                for i in range(len(track_ids)):
                    # Get bbox in COCO format [x, y, width, height]
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    bbox_width = float(x2 - x1)
                    bbox_height = float(y2 - y1)
                    bbox_coco = [float(x1), float(y1), bbox_width, bbox_height]
                    
                    # Calculate area
                    area = bbox_width * bbox_height
                    
                    # Get segmentation polygon if available
                    segmentation = []
                    if mask_polygons is not None and i < len(mask_polygons):
                        polygon = mask_polygons[i]
                        if polygon is not None and len(polygon) > 0:
                            # Convert polygon to flat list [x1, y1, x2, y2, ...]
                            if hasattr(polygon, 'tolist'):
                                polygon_list = polygon.tolist()
                            else:
                                polygon_list = list(polygon)
                            # Flatten the polygon coordinates
                            segmentation = [coord for point in polygon_list for coord in point]
                            # Recalculate area from polygon if available
                            if len(segmentation) >= 6:
                                area = calculate_polygon_area(segmentation)
                    
                    # Create COCO-style annotation
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_ids[i]),
                        "bbox": bbox_coco,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [segmentation] if segmentation else [],
                        "track_id": int(track_ids[i]),
                        "confidence": float(confidences[i])
                    }
                    
                    coco_annotations.append(annotation)
                    annotation_id += 1
                
                # Display tracking statistics
                unique_ids = np.unique(track_ids)
                info_text = f"Objects: {len(unique_ids)} | Tracks: {len(track_ids)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save the annotated frame
            output_path = output_dir / f"tracked_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"  Saved: {output_path}")
        else:
            # If no results, copy original image
            output_path = output_dir / f"tracked_{image_path.name}"
            cv2.imwrite(str(output_path), frame)
            print(f"  No detections, saved original: {output_path}")
    
    # Save COCO-style JSON results
    coco_output = {
        "info": {
            "description": "ByteTrack tracking results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }
    
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Tracking JSON saved to: {json_path}")
    
    print(f"\nTracking complete! Results saved to: {output_dir}")
    print(f"Total images: {len(coco_images)}")
    print(f"Total annotations: {len(coco_annotations)}")
    
    # Create a summary video from tracked images
    create_tracking_video(output_dir, image_files, args.fps)


def calculate_polygon_area(polygon_flat):
    """Calculate the area of a polygon using the shoelace formula.
    
    Args:
        polygon_flat: Flat list of coordinates [x1, y1, x2, y2, ...]
    
    Returns:
        Area of the polygon
    """
    n = len(polygon_flat) // 2
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i = polygon_flat[2 * i]
        y_i = polygon_flat[2 * i + 1]
        x_j = polygon_flat[2 * j]
        y_j = polygon_flat[2 * j + 1]
        area += x_i * y_j
        area -= x_j * y_i
    
    return abs(area) / 2.0


def create_tracking_video(output_dir, image_files, fps=10):
    """Create a video from the tracked images."""
    if not image_files:
        return
    
    video_path = output_dir / "tracking_result.mp4"
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(output_dir / f"tracked_{image_files[0].name}"))
    if first_img is None:
        return
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Write frames
    for image_path in image_files:
        tracked_path = output_dir / f"tracked_{image_path.name}"
        if tracked_path.exists():
            frame = cv2.imread(str(tracked_path))
            if frame is not None:
                out.write(frame)
    
    out.release()
    print(f"Tracking video saved to: {video_path}")


if __name__ == "__main__":
    args = parse_args()
    run_tracking(args)
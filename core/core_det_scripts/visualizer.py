#    core/visualizer.py - Prediction overlays & timeline visualizations.
#
#    USAGE (OOP & module-level helpers):
#        from core.visualizer import generate_prediction_visualizations
#        r = generate_prediction_visualizations(
#            model_path="./best.pt",
#            images_dir="./test_images",
#            output_dir="./output/visualizations",
#            conf_threshold=0.5,
#        )
#        print(r["output_dir"], r["summary_file"])
#
#        # Create a chronological pipeline-timeline PNG:
#        from core.visualizer import create_timeline_visualization
#        runs = [
#            {"timestamp": "2024-01-01 12:00:00", "step": "Labeling",
#             "status": "success", "duration": 120.5},
#        ]
#        r = create_timeline_visualization(pipeline_runs=runs,
#                                           output_dir="./output/visualizations")
#        print(r["timeline_image"])
#
#        # Browse existing visualizations on disk:
#        from core.visualizer import browse_timeline_outputs
#        files = browse_timeline_outputs("./output/visualizations")
#        print(files["timeline_images"], files["prediction_summaries"])
#
#    RUN AS A ONE-LINER:
#        python -c "from core.visualizer import generate_prediction_visualizations; \
#            print(generate_prediction_visualizations( \
#            model_path='./best.pt', images_dir='./test', output_dir='./viz'))"
#
#    ARGUMENTS:
#        generate_prediction_visualizations(model_path,
#                                            images_dir,
#                                            output_dir,
#                                            conf_threshold)  - overlays boxes
#        create_timeline_visualization(pipeline_runs, output_dir) - PNG timeline
#        browse_timeline_outputs(output_dir)                     - list files
#
#    REQUIREMENTS:
#        pip install ultralytics opencv-python-headless numpy

"""
Visualization utilities: generate prediction folders and timeline visualizations.
All functions are headless and return structured results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
import json


def generate_prediction_visualizations(
    model_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    conf_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Generate visualization images with predicted bounding boxes.
    
    Args:
        model_path: Path to trained model (.pt file)
        images_dir: Directory containing images to run inference on
        output_dir: Directory to save visualization results
        conf_threshold: Confidence threshold for predictions
        progress_callback: Callback for progress updates
        status_callback: Callback for status messages
        log_callback: Callback for log messages
        is_cancelled: Callback to check if operation should be cancelled
    
    Returns:
        Dictionary with visualization results
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "success": False,
            "error": "Ultralytics YOLO not installed."
        }
    
    model_file = Path(model_path)
    if not model_file.exists():
        return {"success": False, "error": f"Model file not found: {model_file}"}
    
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    total = len(image_files)
    if total == 0:
        return {"success": False, "error": "No images found"}
    
    if status_callback:
        status_callback("Loading model...")
    
    try:
        model = YOLO(str(model_file))
        
        predictions_summary = []
        
        for i, img_file in enumerate(image_files):
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True, "processed": i}
            
            try:
                # Run inference
                results = model(str(img_file), conf=conf_threshold, verbose=False)
                
                # Plot results on image
                annotated = results[0].plot()
                
                # Save annotated image
                output_file = output_path / f"pred_{img_file.name}"
                cv2.imwrite(str(output_file), annotated)
                
                # Extract prediction info
                preds = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        preds.append({
                            "class_id": int(box.cls[0]),
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist(),
                        })
                
                predictions_summary.append({
                    "image": img_file.name,
                    "predictions": preds,
                    "output_file": f"pred_{img_file.name}",
                })
                
                if log_callback:
                    log_callback(f"Processed: {img_file.name} ({len(preds)} detections)")
                
            except Exception as e:
                if log_callback:
                    log_callback(f"Error processing {img_file.name}: {str(e)}")
            
            if progress_callback:
                progress = int(((i + 1) / total) * 100)
                progress_callback(progress)
            
            if status_callback:
                status_callback(f"Processing {i + 1}/{total}")
        
        # Save predictions summary as JSON
        summary_file = output_path / "predictions_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(predictions_summary, f, indent=2)
        
        return {
            "success": True,
            "processed": total,
            "output_dir": str(output_path),
            "summary_file": str(summary_file),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Visualization generation failed: {str(e)}",
        }


def create_timeline_visualization(
    pipeline_runs: List[Dict],
    output_dir: str | Path,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Create timeline visualization of pipeline runs.
    
    Args:
        pipeline_runs: List of pipeline run dictionaries with keys:
            - timestamp: Run timestamp
            - step: Pipeline step name
            - status: 'success' or 'failed'
            - duration: Duration in seconds
            - metrics: Optional metrics dictionary
        output_dir: Directory to save timeline visualization
        progress_callback: Callback for progress updates
        status_callback: Callback for status messages
        log_callback: Callback for log messages
        is_cancelled: Callback to check if operation should be cancelled
    
    Returns:
        Dictionary with timeline results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not pipeline_runs:
        return {"success": False, "error": "No pipeline runs provided"}
    
    # Sort runs by timestamp
    sorted_runs = sorted(pipeline_runs, key=lambda x: x.get("timestamp", ""))
    
    # Create timeline image
    width = 1200
    row_height = 60
    padding = 20
    header_height = 80
    
    # Calculate image height
    num_runs = len(sorted_runs)
    height = header_height + (num_runs * row_height) + (padding * 2)
    
    # Create white background
    timeline_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw header
    cv2.putText(timeline_img, "Pipeline Timeline", (padding, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.line(timeline_img, (padding, 60), (width - padding, 60), (0, 0, 0), 2)
    
    # Draw each run
    for i, run in enumerate(sorted_runs):
        if is_cancelled and is_cancelled():
            return {"success": False, "cancelled": True}
        
        y = header_height + padding + (i * row_height)
        
        # Determine color based on status
        status = run.get("status", "unknown")
        if status == "success":
            color = (0, 200, 0)  # Green
        elif status == "failed":
            color = (0, 0, 200)  # Red
        else:
            color = (200, 200, 0)  # Yellow
        
        # Draw timeline connector
        if i > 0:
            prev_y = header_height + padding + ((i - 1) * row_height) + row_height // 2
            cv2.line(timeline_img, (padding + 20, prev_y), 
                    (padding + 20, y + row_height // 2), (150, 150, 150), 2)
        
        # Draw circle marker
        cv2.circle(timeline_img, (padding + 20, y + row_height // 2), 10, color, -1)
        
        # Draw step name
        step = run.get("step", "Unknown")
        cv2.putText(timeline_img, step, (padding + 50, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw timestamp
        timestamp = run.get("timestamp", "")
        if timestamp:
            cv2.putText(timeline_img, timestamp, (padding + 50, y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Draw duration
        duration = run.get("duration", 0)
        cv2.putText(timeline_img, f"{duration:.1f}s", (width - 150, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw status
        cv2.putText(timeline_img, status.upper(), (width - 80, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if progress_callback:
            progress = int(((i + 1) / num_runs) * 100)
            progress_callback(progress)
    
    # Save timeline image
    timeline_file = output_path / "timeline.png"
    cv2.imwrite(str(timeline_file), timeline_img)
    
    # Save timeline data as JSON
    timeline_data_file = output_path / "timeline_data.json"
    with open(timeline_data_file, 'w') as f:
        json.dump(sorted_runs, f, indent=2)
    
    if log_callback:
        log_callback(f"Timeline saved to: {timeline_file}")
    
    return {
        "success": True,
        "timeline_image": str(timeline_file),
        "timeline_data": str(timeline_data_file),
        "total_runs": num_runs,
    }


def browse_timeline_outputs(
    output_dir: str | Path,
) -> Dict:
    """
    Browse timeline outputs in a directory.
    
    Args:
        output_dir: Directory containing timeline outputs
    
    Returns:
        Dictionary with available timeline files
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return {"success": False, "error": "Directory not found"}
    
    # Find timeline files
    timeline_files = list(output_path.glob("timeline*.json"))
    timeline_images = list(output_path.glob("timeline*.png"))
    
    # Find prediction summaries
    prediction_files = list(output_path.glob("predictions_summary.json"))
    
    return {
        "success": True,
        "timeline_data_files": [str(f) for f in timeline_files],
        "timeline_images": [str(f) for f in timeline_images],
        "prediction_summaries": [str(f) for f in prediction_files],
    }
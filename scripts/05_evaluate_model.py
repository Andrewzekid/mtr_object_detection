#!/usr/bin/env python3
"""
Evaluate a trained YOLO model and generate visualizations.

This script evaluates a trained YOLO model on test data, calculates metrics
(mAP, precision, recall), and generates prediction visualizations.

USAGE:
    python scripts/05_evaluate_model.py --model ./output/training/yolo_training/weights/best.pt \\
        --test-data ./output/split/dataset.yaml
    python scripts/05_evaluate_model.py --model ./best.pt --test-data ./dataset.yaml \\
        --visualize --images-dir ./test_images
    python scripts/05_evaluate_model.py --model ./best.pt --test-data ./dataset.yaml \\
        --conf 0.5 --iou 0.5

OUTPUT:
    - Evaluation metrics (mAP, precision, recall)
    - Per-class performance breakdown
    - Prediction visualization images with bounding boxes
    - JSON summary of results
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_evaluator import ModelEvaluator
from core.visualizer import generate_prediction_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO model and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation on test set
    python scripts/05_evaluate_model.py \\
        --model ./output/training/yolo_training/weights/best.pt \\
        --test-data ./output/split/dataset.yaml

    # Evaluation with custom thresholds
    python scripts/05_evaluate_model.py \\
        --model ./output/training/yolo_training/weights/best.pt \\
        --test-data ./output/split/dataset.yaml \\
        --conf 0.5 --iou 0.5

    # Evaluation with prediction visualizations
    python scripts/05_evaluate_model.py \\
        --model ./output/training/yolo_training/weights/best.pt \\
        --test-data ./output/split/dataset.yaml \\
        --visualize --images-dir ./test_images \\
        --output-dir ./output/evaluation

    # Save results to JSON
    python scripts/05_evaluate_model.py \\
        --model ./output/training/yolo_training/weights/best.pt \\
        --test-data ./output/split/dataset.yaml \\
        --output ./output/evaluation/results.json

Metrics Explained:
    mAP50      - Mean Average Precision at IoU=0.50
    mAP50-95   - Mean Average Precision averaged over IoU thresholds 0.50-0.95
    Precision  - Ratio of true positives to all positive predictions
    Recall     - Ratio of true positives to all actual positives

Output Structure:
    output_dir/
    ├── evaluation_results.json
    ├── prediction_images/
    │   ├── img1_pred.jpg
    │   └── img2_pred.jpg
    └── summary.json
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)",
    )
    parser.add_argument(
        "--test-data", "-t",
        type=str,
        required=True,
        help="Path to dataset YAML file or test directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0.0-1.0, default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truth (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate prediction visualization images",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory of images for visualization (required with --visualize if not in test-data)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output/evaluation",
        help="Output directory for evaluation results (default: ./output/evaluation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for detailed results",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="0",
        help="Device to run evaluation on: '0' for GPU 0, 'cpu' for CPU (default: 0)",
    )

    return parser.parse_args()


def print_metrics(metrics, per_class=None):
    """Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dict with overall metrics
        per_class: Optional dict with per-class metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    # Overall metrics
    print("\nOverall Performance:")
    
    # Handle different metric key names
    map50 = metrics.get('mAP50', metrics.get('map50', metrics.get('metrics/mAP50(B)', 'N/A')))
    map50_95 = metrics.get('mAP50_95', metrics.get('map50_95', metrics.get('metrics/mAP50-95(B)', 'N/A')))
    precision = metrics.get('precision', metrics.get('metrics/precision(B)', 'N/A'))
    recall = metrics.get('recall', metrics.get('metrics/recall(B)', 'N/A'))
    
    if isinstance(map50, (int, float)):
        print(f"  mAP50:      {map50:.4f} ({map50*100:.1f}%)")
    else:
        print(f"  mAP50:      {map50}")
    
    if isinstance(map50_95, (int, float)):
        print(f"  mAP50-95:   {map50_95:.4f} ({map50_95*100:.1f}%)")
    else:
        print(f"  mAP50-95:   {map50_95}")
    
    if isinstance(precision, (int, float)):
        print(f"  Precision:  {precision:.4f} ({precision*100:.1f}%)")
    else:
        print(f"  Precision:  {precision}")
    
    if isinstance(recall, (int, float)):
        print(f"  Recall:     {recall:.4f} ({recall*100:.1f}%)")
    else:
        print(f"  Recall:     {recall}")
    
    # Per-class metrics
    if per_class:
        print("\nPer-Class Performance:")
        print(f"  {'Class':<20} {'AP50':>10} {'Precision':>10} {'Recall':>10}")
        print("  " + "-" * 52)
        
        for class_name, class_metrics in per_class.items():
            # Try multiple key names for AP50
            c_ap50 = class_metrics.get('AP50', class_metrics.get('mAP50', class_metrics.get('ap50', 'N/A')))
            c_prec = class_metrics.get('precision', None)
            c_rec = class_metrics.get('recall', None)
            
            c_ap50_str = f"{c_ap50:.4f}" if isinstance(c_ap50, (int, float)) else str(c_ap50)
            c_prec_str = f"{c_prec:.4f}" if isinstance(c_prec, (int, float)) else "-"
            c_rec_str = f"{c_rec:.4f}" if isinstance(c_rec, (int, float)) else "-"
            
            print(f"  {class_name:<20} {c_ap50_str:>10} {c_prec_str:>10} {c_rec_str:>10}")


def main():
    args = parse_args()
    
    # Validate model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Validate test data
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data not found: {test_data_path}")
        sys.exit(1)
    
    # Validate thresholds
    if args.conf < 0 or args.conf > 1:
        print(f"Error: Confidence threshold must be between 0 and 1, got {args.conf}")
        sys.exit(1)
    
    if args.iou < 0 or args.iou > 1:
        print(f"Error: IoU threshold must be between 0 and 1, got {args.iou}")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO MODEL EVALUATION")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    print("\nRunning evaluation...")
    print("-" * 60)
    
    # Define callbacks for debug output
    def log_callback(msg):
        print(f"  [DEBUG] {msg}")
    
    def status_callback(msg):
        print(f"  [STATUS] {msg}")
    
    evaluator = ModelEvaluator(
        model_path=str(model_path),
        test_data_path=str(test_data_path),
    )
    
    result = evaluator.evaluate_unseen(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        log_callback=log_callback,
        status_callback=status_callback,
    )
    
    if not result.get("success", False):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Get metrics
    metrics = result.get("metrics", {})
    per_class = result.get("per_class", None)
    
    # Print metrics
    print_metrics(metrics, per_class)
    
    # Generate visualizations if requested
    vis_results = None
    if args.visualize:
        images_dir = args.images_dir
        
        # Try to get images directory from test data path
        if not images_dir:
            # Check if test_data is a directory with test/images
            if test_data_path.is_dir():
                test_images = test_data_path / "test" / "images"
                if test_images.exists():
                    images_dir = str(test_images)
            elif test_data_path.suffix in ['.yaml', '.yml']:
                # Try to parse YAML to find images path
                try:
                    import yaml
                    with open(test_data_path) as f:
                        yaml_data = yaml.safe_load(f)
                    
                    # Get the base path from YAML
                    base_path = yaml_data.get('path', '')
                    test_path = yaml_data.get('test', yaml_data.get('val'))
                    
                    if test_path:
                        # Construct full path
                        full_test_path = Path(base_path) / test_path if base_path else Path(test_path)
                        
                        # Check if it's an images directory or a text file list
                        if full_test_path.exists():
                            if full_test_path.is_dir():
                                images_dir = str(full_test_path)
                            elif full_test_path.suffix in ['.txt', '.json']:
                                # It's a file list, read the first image path to get the directory
                                with open(full_test_path) as f:
                                    first_line = f.readline().strip()
                                    if first_line:
                                        first_img = Path(first_line)
                                        if first_img.exists():
                                            images_dir = str(first_img.parent)
                        else:
                            # Try relative to the YAML file location
                            yaml_dir = test_data_path.parent
                            relative_test = Path(test_path)
                            if relative_test.exists():
                                images_dir = str(relative_test)
                            elif (yaml_dir / relative_test).exists():
                                images_dir = str(yaml_dir / relative_test)
                except Exception as e:
                    print(f"  Warning: Could not parse YAML for images: {e}")
        
        if images_dir and Path(images_dir).exists():
            print(f"\nGenerating prediction visualizations...")
            vis_dir = output_dir / "prediction_images"
            
            vis_results = generate_prediction_visualizations(
                model_path=str(model_path),
                images_dir=images_dir,
                output_dir=str(vis_dir),
                conf_threshold=args.conf,
            )
            
            if vis_results.get("success", True):
                print(f"  Visualizations saved to: {vis_dir}")
                print(f"  Images processed: {vis_results.get('images_processed', 'N/A')}")
            else:
                print(f"  Visualization failed: {vis_results.get('error', 'Unknown error')}")
        else:
            print("\nWarning: Could not find images directory for visualization")
            print("  Use --images-dir to specify the images directory")
    
    # Prepare output data
    output_data = {
        "model": str(model_path),
        "test_data": str(test_data_path),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "metrics": metrics,
        "per_class": per_class,
    }
    
    if vis_results:
        output_data["visualizations"] = {
            "output_dir": str(output_dir / "prediction_images"),
            "images_processed": vis_results.get("images_processed"),
        }
    
    # Save results to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "evaluation_results.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    
    print("\nYOLO model evaluation completed successfully!")


if __name__ == "__main__":
    main()
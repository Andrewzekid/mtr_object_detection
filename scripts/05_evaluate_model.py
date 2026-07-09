#!/usr/bin/env python3
"""
Script to evaluate detection/segmentation models using ModelEvaluator class.

USAGE:
    python scripts/05_evaluate_model.py --model yolo26l.pt --data ../Datasets/yolo26l/dataset.yaml
    python scripts/05_evaluate_model.py --model yolo26l-seg.pt --data ../Datasets/yolo26l_seg/dataset.yaml

    # Compare predictions with ground truth
    python scripts/05_evaluate_model.py --pred-json predictions.json --gt-json ground_truth.json

    # Load GT from YOLO format labels
    python scripts/05_evaluate_model.py --labels-dir ../Datasets/yolo26l/labels --images-dir ../Datasets/yolo26l/images
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.model_evaluator import ModelEvaluator


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_bboxes_from_json(json_path):
    """Load bounding boxes from a JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)

    with open(json_file, "r") as f:
        data = json.load(f)

    # Handle bare list format
    if isinstance(data, list):
        return data

    # Handle wrapped format or Qwen output format
    if isinstance(data, dict):
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

        if "bboxes" in data:
            return data["bboxes"]
        elif "bbox" in data:
            bbox = data["bbox"]
            return [bbox] if isinstance(bbox, list) and len(bbox) == 4 else bbox

    print("Error: JSON file must contain a list of bboxes or a dict with 'bboxes' or 'parsed_output' key")
    sys.exit(1)


def load_predictions_from_json(json_path: str) -> list:
    """Load predictions from a JSON file.

    Expected format:
    [
        {"image_id": "img1.jpg", "class_id": 0, "bbox": [x1, y1, x2, y2], ...},
        ...
    ]

    Returns:
        List of prediction dictionaries
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: Predictions JSON file not found: {json_file}")
        sys.exit(1)

    with open(json_file, "r") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Try common keys
        for key in ["predictions", "results", "data", "bboxes"]:
            if key in data:
                return data[key]

        # Try direct use of dict if it looks like predictions
        return data

    print("Error: Predictions JSON file must contain a list or dict with 'predictions'/'results' key")
    sys.exit(1)


def load_gt_from_yolo(labels_dir: str, images_dir: str) -> list:
    """Load ground truth from YOLO format label files.

    Args:
        labels_dir: Path to directory containing .txt label files
        images_dir: Path to directory containing images

    Returns:
        List of ground truth dictionaries: {"image_id": ..., "class_id": ..., "bbox": [...]}
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)

    ground_truth = []

    for label_file in labels_path.glob("*.txt"):
        image_id = label_file.stem
        image_file = images_path / f"{image_id}.jpg"

        # Try different extensions
        if not image_file.exists():
            image_file = images_path / f"{image_id}.png"

        if not image_file.exists():
            continue

        # Validate image is readable
        import cv2
        img = cv2.imread(str(image_file))
        if img is None:
            continue

        # Read label file
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to bounding box
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                ground_truth.append({
                    "image_id": image_id,
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                })

    return ground_truth


def evaluate_model(model_path: str, data_path: str, split: str = "val", conf_threshold: float = 0.5, iou_threshold: float = 0.5, log_callback=None):
    """Evaluate model on unseen data.

    Args:
        model_path: Path to trained model (.pt file)
        data_path: Path to dataset YAML file (or image folder)
        split: Dataset split to use (val/test/unsupervised/default)
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching
        log_callback: Optional callback for logging messages

    Returns:
        Evaluation results dictionary
    """
    try:
        evaluator = ModelEvaluator(model_path=model_path)
    except Exception as e:
        if log_callback:
            log_callback(f"Error initializing model: {e}")
        return {"success": False, "error": f"Failed to load model: {e}"}

    if not evaluator.model_path or not evaluator.model_path.exists():
        if log_callback:
            log_callback(f"Error: Model file not found: {evaluator.model_path}")
        return {"success": False, "error": f"Model file not found: {evaluator.model_path}"}

    # Set test data path (with split support)
    if log_callback:
        log_callback(f"Evaluating model: {evaluator.model_path}")
        log_callback(f"Test data: {data_path}")
        log_callback(f"Split: {split}")

    try:
        result = evaluator.evaluate_unseen(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_path=model_path,
            test_data_path=data_path,
            split=split,
            log_callback=log_callback,
        )

        if result.get("success"):
            if log_callback:
                log_callback(f"✓ Evaluation successful")
                log_callback(f"  mAP50: {result['metrics'].get('mAP50', 'N/A'):.4f}")
                log_callback(f"  mAP50-95: {result['metrics'].get('mAP50_95', 'N/A'):.4f}")

            return result
        else:
            if log_callback:
                log_callback(f"✗ Evaluation failed: {result.get('error', 'Unknown error')}")
            return result

    except Exception as e:
        if log_callback:
            log_callback(f"✗ Evaluation failed with exception: {e}")
        return {"success": False, "error": f"Evaluation failed: {e}"}


def compare_with_gt(predictions: list, ground_truth: list, iou_threshold: float = 0.5):
    """Compare predictions with ground truth.

    Args:
        predictions: List of prediction dicts with "image_id", "class_id", "bbox"
        ground_truth: List of ground truth dicts with "image_id", "class_id", "bbox"
        iou_threshold: IoU threshold for matching

    Returns:
        Comparison results dictionary
    """
    try:
        evaluator = ModelEvaluator()
        result = evaluator.compare_with_gt(
            predictions=predictions,
            ground_truth=ground_truth,
            iou_threshold=iou_threshold,
        )
        return result
    except Exception as e:
        return {"success": False, "error": f"Comparison failed: {e}"}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Evaluate detection/segmentation models using ModelEvaluator class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Evaluate model on unseen data
    python 05_evaluate_model.py --model yolo26l.pt --data ../Datasets/yolo26l/dataset.yaml

    # Evaluate segmentation model
    python 05_evaluate_model.py --model yolo26l-seg.pt --data ../Datasets/yolo26l_seg/dataset.yaml

    # Compare predictions with ground truth
    python 05_evaluate_model.py --pred-json predictions.json --gt-json ground_truth.json

    # Load GT from YOLO format
    python 05_evaluate_model.py --labels-dir ../Datasets/yolo26l/labels --images-dir ../Datasets/yolo26l/images

    # Use custom thresholds
    python 05_evaluate_model.py --model yolo26l.pt --data dataset.yaml --conf 0.4
        """
    )

    # Model and data arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained model (.pt file). Default: ./best.pt",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to dataset YAML file (for evaluation on unseen data)",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="val",
        help="Dataset split to use (val/test/unsupervised/default)",
    )

    # Prediction/ground truth arguments
    parser.add_argument(
        "--pred-json",
        type=str,
        default=None,
        help="Path to predictions JSON file (for comparison with GT)",
    )
    parser.add_argument(
        "--gt-json",
        type=str,
        default=None,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Path to YOLO format label files directory",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to images directory (for YOLO GT loading)",
    )

    # Thresholds
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", "-i",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )

    # Logging
    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
        help="Suppress output",
    )

    args = parser.parse_args()

    # Determine operation mode
    if args.model and args.data:
        # Evaluation mode
        model_path = args.model
        data_path = args.data
        split = args.split  # Default to "val" for evaluation
        conf_threshold = args.conf
        iou_threshold = args.iou

        print("=" * 70)
        print("Model Evaluation")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Data: {data_path}")
        print(f"Split: {split}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        print()

        log_callback = lambda msg: print(msg) if not args.silent else None
        result = evaluate_model(
            model_path=model_path,
            data_path=data_path,
            split=split,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            log_callback=log_callback,
        )

        if result.get("success"):
            print(f"\n{'=' * 70}")
            print(f"✓ Evaluation completed successfully!")
            print(f"{'=' * 70}")

            # Print metrics
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                if value is not None:
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

            # Print per-class metrics if available
            per_class = result.get("per_class", {})
            if per_class:
                print(f"\nPer-class mAP50:")
                for cls_name, cls_metrics in per_class.items():
                    print(f"  {cls_name}: {cls_metrics.get('AP50', 'N/A'):.4f}")

            print()
        else:
            print(f"\n{'=' * 70}")
            print(f"✗ Evaluation failed!")
            print(f"{'=' * 70}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print()

    elif args.pred_json or args.gt_json:
        # Comparison mode
        if not args.pred_json:
            print("Error: --pred-json is required for comparison mode")
            sys.exit(1)

        if not args.gt_json and not (args.labels_dir and args.images_dir):
            print("Error: --gt-json or --labels-dir/--images-dir is required")
            sys.exit(1)

        predictions = load_predictions_from_json(args.pred_json)

        if args.gt_json:
            # Load from JSON
            with open(args.gt_json, "r") as f:
                ground_truth = json.load(f)
            # Ensure it's a list
            if isinstance(ground_truth, dict):
                if "bboxes" in ground_truth:
                    ground_truth = [{"image_id": k, "class_id": v["class_id"], "bbox": v["bbox"]} for k, v in ground_truth.items()]
                elif "data" in ground_truth:
                    ground_truth = [{"image_id": v["image_id"], "class_id": v["class_id"], "bbox": v["bbox"]} for v in ground_truth["data"]]
        else:
            # Load from YOLO format
            ground_truth = load_gt_from_yolo(args.labels_dir, args.images_dir)

        if args.silent:
            result = compare_with_gt(predictions, ground_truth, iou_threshold=args.iou)
        else:
            print("=" * 70)
            print("Ground Truth Comparison")
            print("=" * 70)
            print(f"Predictions: {len(predictions)}")
            print(f"Ground truth: {len(ground_truth)}")
            print(f"IoU threshold: {args.iou}")
            print()
            result = compare_with_gt(predictions, ground_truth, iou_threshold=args.iou)

        if result.get("success"):
            print(f"\n{'=' * 70}")
            print(f"✓ Comparison completed successfully!")
            print(f"{'=' * 70}")

            # Print overall metrics
            overall = result.get("overall", {})
            print(f"\nOverall metrics:")
            for key, value in overall.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

            # Print per-class metrics
            per_class = result.get("per_class", {})
            if per_class:
                print(f"\nPer-class metrics:")
                for cls_name, cls_metrics in per_class.items():
                    print(f"  {cls_name}:")
                    for metric, value in cls_metrics.items():
                        print(f"    {metric}: {value:.4f}" if isinstance(value, float) else f"    {metric}: {value}")

            # Print confusion matrix
            confusion = result.get("confusion_matrix", {})
            if confusion:
                print(f"\nConfusion matrix:")
                for cls_name, cls_preds in confusion.items():
                    print(f"  {cls_name}:")
                    for target, count in cls_preds.items():
                        print(f"    {target}: {count}")

            print()
        else:
            print(f"\n{'=' * 70}")
            print(f"✗ Comparison failed!")
            print(f"{'=' * 70}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print()

    else:
        # No valid arguments
        parser.print_help()
        print("\nError: Please provide either model+data for evaluation, or pred-json+gt-json/labels for comparison")
        sys.exit(1)


if __name__ == "__main__":
    main()
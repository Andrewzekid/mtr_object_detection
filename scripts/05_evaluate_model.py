#!/usr/bin/env python3
"""
Evaluate detection/segmentation models using the ModelEvaluator class.

Modes:
1. Model evaluation (--model + --data): runs ultralytics val on one or more
   dataset splits (default: test AND train) and reports overall metrics plus
   PER-CLASS precision / recall / F1 / AP50 / AP50-95. For segmentation
   models, mask (seg) metrics are reported alongside box metrics.
   With --csv, the per-class metrics for every evaluated split are written to
   a CSV file (one row per split x class, plus an "all" summary row).

2. Comparison mode (--pred-json + --gt-json or --labels-dir/--images-dir):
   matches prediction boxes against ground truth and reports overall and
   per-class precision / recall / F1 / TP / FP / FN.

USAGE:
    # Evaluate on test + train splits, write per-class CSV
    python scripts/05_evaluate_model.py --model best.pt \
        --data output/split/dataset.yaml --csv output/eval/metrics.csv

    # Evaluate a single split
    python scripts/05_evaluate_model.py --model yolo26l.pt \
        --data ../Datasets/yolo26l/dataset.yaml --split val

    # Segmentation model (box + mask metrics)
    python scripts/05_evaluate_model.py --model yolo26l-seg.pt \
        --data ../Datasets/yolo26l_seg/dataset.yaml --csv seg_metrics.csv

    # Compare predictions with ground truth
    python scripts/05_evaluate_model.py --pred-json predictions.json --gt-json ground_truth.json

    # Load GT from YOLO format labels
    python scripts/05_evaluate_model.py --labels-dir ../Datasets/yolo26l/labels --images-dir ../Datasets/yolo26l/images

CSV COLUMNS:
    split, class_id, class_name, instances, precision, recall, f1, ap50,
    ap50_95 and, for segmentation models, mask_precision, mask_recall,
    mask_f1, mask_ap50, mask_ap50_95. An "all" row per split carries the
    overall means. "instances" is the number of GT instances for that class
    in the split (0 when unavailable).
"""

import argparse
import csv
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


CSV_FIELDS = [
    "split", "class_id", "class_name", "instances",
    "precision", "recall", "f1", "ap50", "ap50_95",
    "mask_precision", "mask_recall", "mask_f1", "mask_ap50", "mask_ap50_95",
]


def count_gt_instances(data_path: str, split: str) -> dict:
    """Count GT instances per class id for a dataset-YAML split.

    Resolves the split's images dir from the YAML and reads the sibling
    labels/ dir, falling back to common layouts (<yaml_dir>/<split>/labels,
    labels/<split>, "valid" alias for "val"). Returns {class_id: count},
    or {} when unavailable.
    """
    counts = {}
    try:
        import yaml
        data_path = Path(data_path)
        with open(data_path, "r") as f:
            data = yaml.safe_load(f)
        yaml_dir = data_path.parent.resolve()
        root = Path(data.get("path", yaml_dir))
        if not root.is_absolute():
            root = (yaml_dir / root).resolve()
        entry = data.get(split)
        if not entry:
            return {}
        # entry may be a path or list of paths
        img_rel = entry[0] if isinstance(entry, list) else entry
        img_dir = (root / img_rel).resolve()

        candidates = []
        parts = list(img_dir.parts)
        if "images" in parts:
            lbl_parts = list(parts)
            lbl_parts[lbl_parts.index("images")] = "labels"
            candidates.append(Path(*lbl_parts))
        # Roboflow-style: <yaml_dir>/<split>/labels (val may be "valid")
        for s in ({split, "valid"} if split == "val" else {split}):
            candidates.append(yaml_dir / s / "labels")
            candidates.append(yaml_dir / "labels" / s)

        labels_dir = next((c for c in candidates if c.is_dir()), None)
        if labels_dir is None:
            return {}
        for txt in labels_dir.glob("*.txt"):
            if txt.suffix != ".txt":
                continue
            with open(txt, "r") as f:
                for line in f:
                    toks = line.split()
                    if len(toks) >= 5:
                        cid = int(float(toks[0]))
                        counts[cid] = counts.get(cid, 0) + 1
    except Exception:
        return {}
    return counts


def build_csv_rows(split: str, result: dict, gt_counts: dict = None) -> list:
    """Build CSV rows (list of dicts) for one split's evaluation result.

    One row per class plus an "all" summary row. Mask columns are filled for
    segmentation results, left blank otherwise.
    """
    rows = []
    metrics = result.get("metrics", {})
    per_class = result.get("per_class", {})
    class_names = result.get("class_names", {})
    name_to_id = {name: idx for idx, name in class_names.items()}
    gt_counts = gt_counts or {}

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    overall_p = metrics.get("precision", 0.0)
    overall_r = metrics.get("recall", 0.0)
    rows.append({
        "split": split, "class_id": "all", "class_name": "all",
        "instances": sum(gt_counts.values()) if gt_counts else "",
        "precision": overall_p, "recall": overall_r,
        "f1": _f1(overall_p, overall_r),
        "ap50": metrics.get("mAP50", 0.0),
        "ap50_95": metrics.get("mAP50_95", 0.0),
        "mask_precision": metrics.get("mask_precision", ""),
        "mask_recall": metrics.get("mask_recall", ""),
        "mask_f1": (_f1(metrics["mask_precision"], metrics["mask_recall"])
                    if "mask_precision" in metrics else ""),
        "mask_ap50": metrics.get("mask_mAP50", ""),
        "mask_ap50_95": metrics.get("mask_mAP50_95", ""),
    })

    for cls_name, m in sorted(per_class.items(),
                              key=lambda kv: name_to_id.get(kv[0], 1 << 30)):
        cid = name_to_id.get(cls_name, "")
        rows.append({
            "split": split, "class_id": cid, "class_name": cls_name,
            "instances": gt_counts.get(cid, "") if cid != "" else "",
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "f1": m.get("f1", 0.0),
            "ap50": m.get("AP50", 0.0),
            "ap50_95": m.get("AP50_95", 0.0),
            "mask_precision": m.get("mask_precision", ""),
            "mask_recall": m.get("mask_recall", ""),
            "mask_f1": m.get("mask_f1", ""),
            "mask_ap50": m.get("mask_AP50", ""),
            "mask_ap50_95": m.get("mask_AP50_95", ""),
        })
    return rows


def write_metrics_csv(rows: list, csv_path: str):
    """Write per-class metric rows to a CSV file."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = {
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            }
            writer.writerow(formatted)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Evaluate detection/segmentation models using ModelEvaluator class.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Evaluate on test + train splits, write per-class metrics CSV
    python 05_evaluate_model.py --model best.pt --data dataset.yaml --csv metrics.csv

    # Evaluate a single split
    python 05_evaluate_model.py --model yolo26l.pt --data ../Datasets/yolo26l/dataset.yaml --split val

    # Evaluate segmentation model (box + mask metrics)
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
        nargs="+",
        default=["test", "train"],
        help="Dataset split(s) to evaluate (default: test train). "
             "Any of train/val/test supported by the dataset YAML.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write per-class metrics for all evaluated splits to this CSV file",
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
        # Evaluation mode (one or more splits)
        model_path = args.model
        data_path = args.data
        conf_threshold = args.conf
        iou_threshold = args.iou

        print("=" * 70)
        print("Model Evaluation")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Data: {data_path}")
        print(f"Splits: {', '.join(args.split)}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        print()

        log_callback = lambda msg: print(msg) if not args.silent else None
        csv_rows = []
        any_failed = False
        for split in args.split:
            print(f"--- Split: {split} " + "-" * 50)
            result = evaluate_model(
                model_path=model_path,
                data_path=data_path,
                split=split,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                log_callback=log_callback,
            )

            if result.get("success"):
                print(f"\n✓ Evaluation of split '{split}' completed successfully!")

                # Print overall metrics
                metrics = result.get("metrics", {})
                for key, value in metrics.items():
                    if value is not None:
                        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

                # Print per-class metrics if available
                per_class = result.get("per_class", {})
                if per_class:
                    print(f"\nPer-class metrics ({split}):")
                    for cls_name, cls_metrics in per_class.items():
                        parts = [f"{k}={v:.4f}" for k, v in cls_metrics.items()
                                 if isinstance(v, float)]
                        print(f"  {cls_name}: {', '.join(parts)}")

                gt_counts = count_gt_instances(data_path, split)
                csv_rows.extend(build_csv_rows(split, result, gt_counts))
                print()
            else:
                any_failed = True
                print(f"\n✗ Evaluation of split '{split}' failed: "
                      f"{result.get('error', 'Unknown error')}\n")

        if args.csv and csv_rows:
            write_metrics_csv(csv_rows, args.csv)
            print(f"✓ Per-class metrics CSV written to: {args.csv}\n")

        if any_failed and not csv_rows:
            sys.exit(1)

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
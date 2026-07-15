#    core/model_evaluator.py - Model evaluation & ground-truth comparison.
#
#    USAGE (OOP):
#        from core.model_evaluator import ModelEvaluator
#        e = ModelEvaluator(
#            model_path="./best.pt",
#            test_data_path="./dataset.yaml",
#        )
#        r = e.evaluate_unseen(conf_threshold=0.5, iou_threshold=0.5)
#        print(r["metrics"])      # {"mAP50": ..., "mAP50_95": ..., "precision": ..., "recall": ...}
#
#        # Compare prediction dicts against YOLO ground truth:
#        preds = [{"image_id": "img1", "class_id": 0, "bbox": [10, 10, 100, 100]}]
#        gt   = [{"image_id": "img1", "class_id": 0, "bbox": [12, 12, 98, 98]}]
#        r = e.compare_with_gt(preds, gt, iou_threshold=0.5)
#        print(r["overall"], r["confusion_matrix"])
#
#        # Static IoU between two boxes:
#        iou = ModelEvaluator.calculate_iou([0,0,100,100], [10,10,110,110])
#
#    USAGE (BACKWARD-COMPATIBLE FUNCTION):
#        from core.model_evaluator import evaluate_unseen, compare_with_gt
#        evaluate_unseen("./best.pt", "./test", conf_threshold=0.5)
#
#    RUN AS A ONE-LINER:
#        python -c "from core.model_evaluator import ModelEvaluator; \
#            print(ModelEvaluator('./best.pt','./test.yaml').evaluate_unseen(0.5,0.5))"
#
#    ARGUMENTS:
#        evaluate_unseen(model_path, test_data_path,
#                        conf_threshold, iou_threshold)   - mAP / precision / recall
#        compare_with_gt(predictions, ground_truth,
#                         iou_threshold)                   - precision / recall / f1
#        calculate_iou(box1, box2)                         - static IoU
#        load_ground_truth_from_yolo(labels_dir, images_dir)  - GT dicts
#
#    REQUIREMENTS:
#        pip install ultralytics opencv-python-headless numpy

"""
Model evaluation utilities: evaluation on unseen data and ground truth comparison.
OOP paradigm with ModelEvaluator class.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from collections import defaultdict
import json


class ModelEvaluator:
    """Class for evaluating trained models (detection and segmentation)."""

    def __init__(self, model_path: Optional[str] = None, test_data_path: Optional[str] = None):
        """Initialize with model path and test data path.

        Args:
            model_path: Path to trained model file (.pt, .onnx, etc.)
            test_data_path: Path to test dataset YAML file or directory
        """
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.test_data_path: Optional[Path] = Path(test_data_path) if test_data_path else None
        self.model: Any = None
        self.task_type: str = "detect"

    def set_model_path(self, model_path: str):
        """Set the model path."""
        self.model_path = Path(model_path)

    def set_test_data_path(self, test_data_path: str):
        """Set the test data path."""
        self.test_data_path = Path(test_data_path)

    def _detect_task_type(self, model) -> str:
        """Detect whether the model is a detection or segmentation model.

        Args:
            model: YOLO model object

        Returns:
            "detect" or "segment"
        """
        if hasattr(model, 'task'):
            return str(model.task)
        if hasattr(model, 'model') and hasattr(model.model, 'task'):
            return str(model.model.task)
        if self.model_path and '-seg' in str(self.model_path).lower():
            return 'segment'
        return 'detect'

    def _extract_class_names(self, results, model) -> Dict[int, str]:
        """Extract class names from results or model.

        Args:
            results: YOLO results object
            model: YOLO model object

        Returns:
            Dict mapping class index to class name
        """
        class_names = {}
        # Try multiple sources for class names
        if hasattr(results, 'names') and results.names:
            class_names = results.names
        elif hasattr(model, 'names') and model.names:
            class_names = model.names
        # Convert to dict if it's a list
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        return class_names

    def evaluate_unseen(
        self,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        model_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        split: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance on unseen test data.

        Supports both detection and segmentation models. For segmentation models,
        both box metrics and mask (segment) metrics are reported.

        Args:
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for non-maximum suppression
            model_path: Path to model file (optional, uses instance if not provided)
            test_data_path: Path to test dataset (optional, uses instance if not provided)
            split: Dataset split to use ("val" or "test")
            progress_callback: Callback for progress percentage
            status_callback: Callback for status messages
            log_callback: Callback for log messages
            is_cancelled: Callback to check if evaluation should stop

        Returns:
            Dict with evaluation metrics
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"success": False, "error": "Ultralytics YOLO not installed."}

        m_path = Path(model_path) if model_path else self.model_path
        t_path = Path(test_data_path) if test_data_path else self.test_data_path

        # If test_data_path is not provided, try to get it from the model path
        if not t_path:
            model_dir = m_path.parent if m_path else None
            if model_dir:
                yaml_file = model_dir / f"{m_path.stem}.yaml"
                if yaml_file.exists():
                    t_path = yaml_file

        # Validate paths
        if not m_path or not m_path.exists():
            return {"success": False, "error": "Model file not found"}

        if t_path and not t_path.exists():
            return {"success": False, "error": "Test data path not found"}

        # Load model and detect task type
        self.model = YOLO(str(m_path))
        self.task_type = self._detect_task_type(self.model)

        if status_callback:
            status_callback("Loading model...")

        if log_callback:
            log_callback(f"Evaluating model: {m_path}, Task: {self.task_type}")

        # Determine the split to use for evaluation
        val_args = {
            "data": str(t_path) if t_path.suffix in ['.yaml', '.yml'] else None,
            "source": str(t_path) if t_path.suffix not in ['.yaml', '.yml'] else None,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "verbose": True,
        }
        if split:
            val_args["split"] = split

        results = self.model.val(**val_args)

        # Extract box (detection) metrics
        metrics = {}
        if hasattr(results, 'box') and results.box is not None:
            metrics["mAP50"] = float(results.box.map50) if hasattr(results.box, 'map50') else 0
            metrics["mAP50_95"] = float(results.box.map) if hasattr(results.box, 'map') else 0
            metrics["precision"] = float(results.box.mp) if hasattr(results.box, 'mp') else 0
            metrics["recall"] = float(results.box.mr) if hasattr(results.box, 'mr') else 0

        # Extract mask (segmentation) metrics
        if self.task_type == 'segment' and hasattr(results, 'seg') and results.seg is not None:
            seg = results.seg
            metrics["mask_mAP50"] = float(seg.map50) if hasattr(seg, 'map50') else 0
            metrics["mask_mAP50_95"] = float(seg.map) if hasattr(seg, 'map') else 0
            metrics["mask_precision"] = float(seg.mp) if hasattr(seg, 'mp') else 0
            metrics["mask_recall"] = float(seg.mr) if hasattr(seg, 'mr') else 0

        # Get class names
        class_names = self._extract_class_names(results, self.model)

        # Extract per-class metrics
        per_class = {}
        if hasattr(results.box, 'ap50_values') and results.box.ap50_values:
            for i, ap50 in enumerate(results.box.ap50_values):
                class_name = class_names.get(i, f"class_{i}")
                per_class[class_name] = {
                    "AP50": float(ap50) if ap50 is not None else 0,
                    "mAP50": float(ap50) if ap50 is not None else 0,
                }
        elif hasattr(results.box, 'ap50') and results.box.ap50:
            for cls_idx, ap_val in results.box.ap50.items():
                class_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
                per_class[class_name] = {
                    "AP50": float(ap_val) if ap_val is not None else 0,
                    "mAP50": float(ap_val) if ap_val is not None else 0,
                }
        else:
            for cls_idx, class_name in class_names.items():
                per_class[class_name] = {"AP50": 0.0, "mAP50": 0.0}

        # For segmentation, also extract per-class mask AP50
        if self.task_type == 'segment':
            seg = results.seg
            if hasattr(seg, 'ap50_values') and seg.ap50_values:
                for i, ap50 in enumerate(seg.ap50_values):
                    class_name = class_names.get(i, f"class_{i}")
                    if class_name in per_class:
                        per_class[class_name]["mask_AP50"] = float(ap50) if ap50 is not None else 0
                    else:
                        per_class[class_name] = {
                            "AP50": 0,
                            "mAP50": 0,
                            "mask_AP50": float(ap50) if ap50 is not None else 0,
                        }

        return {
            "success": True,
            "metrics": metrics,
            "per_class": per_class,
            "task_type": self.task_type,
            "results": results,
        }

    def compare_with_gt(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Compare predictions with ground truth annotations.

        Args:
            predictions: List of prediction dicts with image_id, class_id, bbox
            ground_truth: List of ground truth dicts with image_id, class_id, bbox
            iou_threshold: IoU threshold for matching
            progress_callback: Progress callback
            status_callback: Status callback
            log_callback: Log callback
            is_cancelled: Cancel check callback

        Returns:
            Dict with overall and per-class metrics
        """
        if not predictions:
            return {"success": False, "error": "No predictions provided"}
        if not ground_truth:
            return {"success": False, "error": "No ground truth provided"}

        # Group by image
        pred_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)

        for pred in predictions:
            pred_by_image[pred["image_id"]].append(pred)
        for gt in ground_truth:
            gt_by_image[gt["image_id"]].append(gt)

        all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
        total_images = len(all_image_ids)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        processed = 0

        for image_id in all_image_ids:
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True}

            preds = pred_by_image.get(image_id, [])
            gts = gt_by_image.get(image_id, [])
            gt_matched = [False] * len(gts)

            for pred in preds:
                pred_bbox = pred["bbox"]
                pred_class = pred["class_id"]

                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    gt_bbox = gt["bbox"]
                    iou = self.calculate_iou(pred_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_class = gts[best_gt_idx]["class_id"]
                    gt_matched[best_gt_idx] = True

                    if pred_class == gt_class:
                        true_positives += 1
                        class_tp[pred_class] += 1
                        confusion_matrix[gt_class][pred_class] += 1
                    else:
                        false_positives += 1
                        class_fp[pred_class] += 1
                        class_fn[gt_class] += 1
                        confusion_matrix[gt_class][pred_class] += 1
                else:
                    false_positives += 1
                    class_fp[pred_class] += 1

            for gt_idx, gt in enumerate(gts):
                if not gt_matched[gt_idx]:
                    false_negatives += 1
                    class_fn[gt["class_id"]] += 1

            processed += 1
            if progress_callback:
                progress_callback(int((processed / total_images) * 100))
            if status_callback:
                status_callback(f"Comparing {processed}/{total_images}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics = {}
        all_classes = set(class_tp.keys()) | set(class_fp.keys()) | set(class_fn.keys())

        for cls in all_classes:
            tp = class_tp[cls]
            fp = class_fp[cls]
            fn = class_fn[cls]
            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            per_class_metrics[cls] = {
                "precision": cls_precision,
                "recall": cls_recall,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }

        confusion_dict = {k: dict(v) for k, v in confusion_matrix.items()}

        return {
            "success": True,
            "overall": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "per_class": per_class_metrics,
            "confusion_matrix": confusion_dict,
            "total_images": total_images,
        }

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: First box in [x1, y1, x2, y2] format
            box2: Second box in [x1, y1, x2, y2] format

        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def load_predictions_from_json(self, json_path: str) -> List[Dict]:
        """Load predictions from JSON file.

        Args:
            json_path: Path to JSON file containing predictions

        Returns:
            List of prediction dicts
        """
        with open(json_path, 'r') as f:
            return json.load(f)

    def load_ground_truth_from_yolo(
        self, labels_dir: str, images_dir: str
    ) -> List[Dict]:
        """Load ground truth from YOLO format label files.

        Args:
            labels_dir: Directory containing .txt label files
            images_dir: Directory containing corresponding images

        Returns:
            List of ground truth dicts
        """
        labels_path = Path(labels_dir)
        images_path = Path(images_dir)

        ground_truth = []

        for label_file in labels_path.glob("*.txt"):
            image_id = label_file.stem

            image_file = images_path / f"{image_id}.jpg"
            if not image_file.exists():
                image_file = images_path / f"{image_id}.png"

            if not image_file.exists():
                continue

            img = cv2.imread(str(image_file))
            if img is None:
                continue

            h, w = img.shape[:2]

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h

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


# Backward compatibility functions
def evaluate_unseen(
    model_path: Optional[str],
    test_data_path: Optional[str],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Module-level evaluation function."""
    evaluator = ModelEvaluator(model_path, test_data_path)
    return evaluator.evaluate_unseen(conf_threshold, iou_threshold,
                                     progress_callback, status_callback, log_callback, is_cancelled)


def compare_with_gt(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Module-level comparison function."""
    evaluator = ModelEvaluator()
    return evaluator.compare_with_gt(predictions, ground_truth, iou_threshold,
                                    progress_callback, status_callback, log_callback, is_cancelled)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Module-level IoU calculation."""
    return ModelEvaluator.calculate_iou(box1, box2)


def load_predictions_from_json(json_path: str) -> List[Dict]:
    """Module-level predictions loader."""
    evaluator = ModelEvaluator()
    return evaluator.load_predictions_from_json(json_path)


def load_ground_truth_from_yolo(labels_dir: str, images_dir: str) -> List[Dict]:
    """Module-level ground truth loader."""
    evaluator = ModelEvaluator()
    return evaluator.load_ground_truth_from_yolo(labels_dir, images_dir)
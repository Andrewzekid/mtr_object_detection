#    core/model_visualizer.py - YOLO model visualization and inference with annotations.
#
#    USAGE (OOP):
#        from core.model_visualizer import ModelVisualizer
#        v = ModelVisualizer(model_path="./best.pt", device="0")
#        r = v.predict_image(image_path_or_bytes, conf_threshold=0.5, iou_threshold=0.5)
#        print(r["predictions"], r["image_with_boxes"])
#
#        # Load from file or bytes:
#        with open("test.jpg", "rb") as f:
#            image_bytes = f.read()
#        r = v.predict_image(image_bytes, conf_threshold=0.25)
#
#    USAGE (BACKWARD-COMPATIBLE FUNCTION):
#        from core.model_visualizer import predict_image
#        predict_image("./best.pt", image_path, conf_threshold=0.5)
#
#    RUN AS A ONE-LINER:
#        python -c "from core.model_visualizer import ModelVisualizer; \
#            print(ModelVisualizer('./best.pt','0').predict_image('test.jpg',0.5))"
#
#    ARGUMENTS:
#        predict_image(image_path_or_bytes,
#                     conf_threshold, iou_threshold,
#                     device, max_det, imgsz)
#        draw_boxes_on_image(image_path,
#                           detections, conf_threshold)
#        compare_predictions(preds_model, preds_gt)  # confusion matrix
#
#    REQUIREMENTS:
#        pip install ultralytics opencv-python-headless numpy

"""
YOLO model visualization and inference.
OOP paradigm with ModelVisualizer class.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
import cv2
import numpy as np


class ModelVisualizer:
    """Class for visualizing YOLO model predictions and performing inference.

    Supports image files, bytes, PIL images, and NumPy arrays.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "0",
    ):
        """Initialize with model path and device.

        Args:
            model_path: Path to trained model file (.pt, .onnx, etc.)
            device: Device to use for inference ("0" for first GPU, "cpu" for CPU)
        """
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.device: str = device
        self.model: Any = None

    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        return YOLO(str(self.model_path)).to(self.device)

    def predict_image(
        self,
        image_input: Union[str, bytes, np.ndarray, Dict],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        max_det: int = 100,
        imgsz: int = 640,
        half: bool = False,
        dnn: bool = False,
        augment: bool = False,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run inference on an image and return predictions + annotated image.

        Supports image files, bytes, PIL images, and NumPy arrays.

        Args:
            image_input: Image path (str), bytes, PIL, or NumPy array
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            device: Device to use (optional, uses instance if not provided)
            max_det: Maximum detections per image
            imgsz: Image size
            half: Use FP16
            dnn: Use DNN backend
            augment: Apply data augmentation
            verbose: Print progress
            progress_callback: Progress callback
            status_callback: Status callback
            log_callback: Log callback
            is_cancelled: Cancel check callback

        Returns:
            Dict with predictions and annotated image
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {
                "success": False,
                "error": "Ultralytics YOLO not installed. Install with: pip install ultralytics",
            }

        # Use provided device or instance device
        dev = device if device is not None else self.device

        # Load model if not loaded
        if self.model is None:
            self.model = YOLO(str(self.model_path)).to(dev) if self.model_path else None

        if self.model is None:
            return {
                "success": False,
                "error": "Model not provided and no model path found.",
            }

        # Check for cancellation
        if is_cancelled and is_cancelled():
            return {"success": False, "error": "Inference cancelled by user."}

        # Convert image_input to appropriate format
        if isinstance(image_input, bytes):
            image = cv2.imdecode(np.frombuffer(image_input, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, Path) and image_input.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
            image = cv2.imread(str(image_input))
        elif isinstance(image_input, Dict) and 'image' in image_input and 'bbox' in image_input:
            # Already in format for compare_predictions
            return {
                "success": True,
                "predictions": image_input,
                "image_with_boxes": None,
                "bbox": image_input.get("bbox"),
                "class_id": image_input.get("class_id"),
                "confidence": image_input.get("confidence"),
            }
        else:
            # Assume it's already a PIL, numpy array, or similar
            image = image_input

        if image is None:
            return {
                "success": False,
                "error": "Failed to load image.",
            }

        # Convert to RGB if needed (YOLO expects RGB)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            imgsz=imgsz,
            half=half,
            dnn=dnn,
            augment=augment,
            verbose=verbose,
        )

        if log_callback:
            log_callback(f"Predictions: {len(results[0].boxes)} detections found.")

        # Extract detections
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, class_ids):
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": int(cls),
                    "class_name": self.model.names.get(int(cls), f"class_{int(cls)}"),
                })

        # Convert to BGR for visualization
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw boxes
        annotated_image = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            conf = det["confidence"]
            class_name = det.get("class_name", f"class_{class_id}")

            # Draw rectangle
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(annotated_image, label, (int(x1), int(y1) - text_height - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return {
            "success": True,
            "predictions": detections,
            "image_with_boxes": annotated_image,
            "image": image,
            "image_bgr": annotated_image,
            "image_path": image_input if isinstance(image_input, str) else None,
        }

    def draw_boxes_on_image(
        self,
        image_input: Union[str, bytes, np.ndarray, Dict],
        detections: List[Dict],
        conf_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Manually draw detections on an image.

        Args:
            image_input: Image path or image object
            detections: List of detection dicts with "bbox" and "class_name"/"class_id" keys
            conf_threshold: Optional confidence threshold for filtering

        Returns:
            Annotated image
        """
        if image_input is None or detections is None or len(detections) == 0:
            return None

        # Convert image to BGR if needed
        if isinstance(image_input, Dict) and 'image' in image_input:
            image_bgr = image_input['image']
        elif isinstance(image_input, bytes):
            image_bgr = cv2.imdecode(np.frombuffer(image_input, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, Path) and image_input.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
            image_bgr = cv2.imread(str(image_input))
        elif isinstance(image_input, np.ndarray):
            # Convert to BGR if in RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_input
        else:
            return None

        # Filter by confidence if threshold provided
        if conf_threshold is not None:
            detections = [d for d in detections if d.get("confidence", 0) >= conf_threshold]

        annotated_image = image_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det.get("class_id", 0)
            conf = det.get("confidence", 1.0)
            class_name = det.get("class_name", f"class_{class_id}")

            # Draw rectangle
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(annotated_image, label, (int(x1), int(y1) - text_height - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_image

    def compare_predictions(
        self,
        preds_model: List[Dict],
        preds_gt: List[Dict],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Compare model predictions against ground truth.

        Args:
            preds_model: List of model predictions
            preds_gt: List of ground truth annotations
            conf_threshold: Confidence threshold for matching
            iou_threshold: IoU threshold for matching

        Returns:
            Dict with precision, recall, F1, per-class metrics, confusion matrix
        """
        from collections import defaultdict

        if not preds_model or not preds_gt:
            return {
                "success": False,
                "error": "No predictions or ground truth provided.",
            }

        # Group by image
        pred_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)

        for pred in preds_model:
            pred_by_image[pred.get("image_id", pred.get("filename", ""))].append(pred)
        for gt in preds_gt:
            gt_by_image[gt.get("image_id", gt.get("filename", ""))].append(gt)

        all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        class_tp = defaultdict(int)
        class_fp = defaultdict(int)
        class_fn = defaultdict(int)
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for image_id in all_image_ids:
            preds = pred_by_image.get(image_id, [])
            gts = gt_by_image.get(image_id, [])
            gt_matched = [False] * len(gts)

            for pred in preds:
                pred_bbox = pred["bbox"]
                pred_class = pred.get("class_id", 0)

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
        }

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


# Backward compatibility functions
def predict_image(
    model_path: Optional[Union[str, Path]] = None,
    image_input: Union[str, bytes, np.ndarray, Dict] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: Optional[str] = None,
    max_det: int = 100,
    imgsz: int = 640,
    half: bool = False,
    dnn: bool = False,
    augment: bool = False,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Module-level prediction function."""
    visualizer = ModelVisualizer(model_path, device)
    return visualizer.predict_image(
        image_input, conf_threshold, iou_threshold, device, max_det, imgsz,
        half, dnn, augment, verbose, progress_callback, status_callback, log_callback, is_cancelled
    )


def draw_boxes_on_image(
    image_input: Union[str, bytes, np.ndarray, Dict],
    detections: List[Dict],
    conf_threshold: Optional[float] = None,
) -> np.ndarray:
    """Module-level box drawing function."""
    return ModelVisualizer().draw_boxes_on_image(image_input, detections, conf_threshold)


def compare_predictions(
    preds_model: List[Dict],
    preds_gt: List[Dict],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Module-level comparison function."""
    visualizer = ModelVisualizer()
    return visualizer.compare_predictions(preds_model, preds_gt, conf_threshold, iou_threshold)
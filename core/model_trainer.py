#    core/model_trainer.py - YOLO model training pipeline (Ultralytics).
#
#    USAGE (OOP):
#        from core.model_trainer import ModelTrainer
#        t = ModelTrainer(
#            config_path="./dataset.yaml",
#            output_dir="./output/training",
#        )
#        r = t.train_yolo(
#            epochs=100,
#            batch_size=16,
#            model_type="yolov8n",   # yolov8n | yolov8s | yolov8m | yolov8l | yolov8x
#            device="0",             # "0" first GPU, "cpu" CPU
#        )
#        print(r["model_path"], r["metrics"])
#
#        # Create dataset YAML for YOLO training:
#        t.create_dataset_yaml(
#            class_names=["car", "person", "dog"],
#            dataset_path="./split_dataset",
#        )
#
#        # Export trained model to ONNX / TorchScript / TFLite:
#        t.export_model(
#            export_format="onnx",
#            model_path="./output/training/yolo_training/weights/best.pt",
#        )
#
#    USAGE (BACKWARD-COMPATIBLE FUNCTION):
#        from core.model_trainer import train_yolo
#        train_yolo("./dataset.yaml", epochs=50, batch_size=16, model_type="yolov8s")
#
#    RUN AS A ONE-LINER:
#        python -c "from core.model_trainer import ModelTrainer; \
#            ModelTrainer('./dataset.yaml','./output/training').train_yolo( \
#            epochs=10, batch_size=8, model_type='yolo26l-seg', device='0')"
#
#    ARGUMENTS:
#        config_path              - path to dataset YAML
#        output_dir               - training output directory
#        train_yolo(epochs, batch_size,
#                   model_type,
#                   device)      - training hyperparameters
#        create_dataset_yaml(class_names,
#                             dataset_path)         - writes dataset.yaml
#        export_model(export_format,
#                     model_path)                    - converts model
#
#    REQUIREMENTS:
#        pip install ultralytics torch torchvision

"""
YOLO model training pipeline.
OOP paradigm with ModelTrainer class.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
import yaml

from utils.constants import (
    MODEL_SIZE_NAMES,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
)


class ModelTrainer:
    """Class for training and managing YOLO models.

    Supports YOLOv8/v11/v12/v26 series with detection and segmentation tasks.
    Provides full control over hyperparameters, callbacks, and export options.

    Example:
        >>> trainer = ModelTrainer(
        ...     config_path="./dataset.yaml",
        ...     output_dir="./output/training",
        ... )
        >>> result = trainer.train_yolo(epochs=100, batch_size=16, model_type="yolov8n")
        >>> print(result["model_path"], result["metrics"])
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize with config path and output directory.

        Args:
            config_path: Path to dataset YAML configuration file
            output_dir: Output directory for training results
        """
        self.config_path = Path(config_path) if config_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.model: Any = None
        self.current_epoch: int = 0
        self.training_history: List[Dict] = []

    def train_yolo(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        model_type: str = "yolov8n",
        device: str = "0",
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume: bool = False,
        # Learning rate & scheduler
        lr0: float = 0.01,
        lrf: float = 0.01,
        cos_lr: bool = True,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        # Image size
        imgsz: int = 640,
        # Optimizer settings
        optimizer: str = "SGD",
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        # Augmentation parameters
        mosaic: float = 1.0,
        mixup: float = 0.0,
        copy_paste: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        # Loss function weights
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        # Loss function type
        loss_type: str = "auto",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        # Early stopping & checkpointing
        patience: int = 50,
        save_period: int = -1,
        # Task type
        task: str = "detect",
        # Paths
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        # Callbacks
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Train YOLO model using Ultralytics with full hyperparameter control.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_type: YOLO model variant (e.g., "yolov8n", "yolo26l-seg")
            device: Device to use ("0" for first GPU, "cpu" for CPU)
            checkpoint_path: Optional path to resume training from
            resume: Whether to resume training from checkpoint
            lr0: Initial learning rate
            lrf: Learning rate factor (lr0 * lrf = actual lr)
            cos_lr: Use cosine annealing schedule
            warmup_epochs: Number of warmup epochs
            warmup_momentum: Momentum for linear warmup
            imgsz: Image size for training
            optimizer: Optimizer type ("SGD", "Adam", etc.)
            momentum: Momentum for SGD
            weight_decay: Weight decay for SGD
            mosaic: Mosaic augmentation probability
            mixup: Mixup augmentation probability
            copy_paste: Copy-paste augmentation probability
            hsv_h: HSV hue random factor
            hsv_s: HSV saturation random factor
            hsv_v: HSV value random factor
            fliplr: Horizontal flip probability
            flipud: Vertical flip probability
            degrees: Random rotation degrees
            translate: Translation probability
            scale: Scale augmentation range
            shear: Shear augmentation
            perspective: Perspective transformation
            box: Box loss weight
            cls: Classification loss weight
            dfl: DFL loss weight
            patience: Patience for early stopping
            save_period: Save checkpoint every N epochs (-1 = never)
            task: Task type ("detect" or "segment")
            config_path: Override config path (if different from init)
            output_dir: Override output directory (if different from init)
            progress_callback: Callback for progress percentage (epoch/total)
            status_callback: Callback for status messages
            log_callback: Callback for log messages
            is_cancelled: Callback to check if training should stop

        Returns:
            Dict with training results including model_path, metrics, and hyperparameters.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {
                "success": False,
                "error": "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            }

        cfg_path = Path(config_path) if config_path else self.config_path
        out_dir = Path(output_dir) if output_dir else self.output_dir

        if not cfg_path or not cfg_path.exists():
            return {
                "success": False,
                "error": f"Config file not found: {cfg_path}"
            }

        if not out_dir:
            out_dir = Path("output/training")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Status callback
        if status_callback:
            status_callback(f"Loading {model_type} model...")
        if log_callback:
            log_callback(f"Starting YOLO training with {model_type}")
            log_callback(f"Config: {cfg_path}, Epochs: {epochs}, Batch size: {batch_size}")
            log_callback(f"LR: {lr0} (scheduler: {'cosine' if cos_lr else 'linear'}), "
                        f"Image size: {imgsz}, Optimizer: {optimizer}")

        try:
            # Load model - support checkpoint loading for fine-tuning
            if checkpoint_path:
                ckpt = Path(checkpoint_path)
                if not ckpt.exists():
                    return {
                        "success": False,
                        "error": f"Checkpoint file not found: {ckpt}"
                    }
                if log_callback:
                    log_callback(f"Loading custom checkpoint: {ckpt}")
                self.model = YOLO(str(ckpt))
            else:
                model_suffix = "-seg" if task == "segment" else ""
                model_name = f"{model_type}{model_suffix}.pt"
                if log_callback:
                    log_callback(f"Loading pretrained model: {model_name}")
                self.model = YOLO(model_name)

            self.current_epoch = [0]
            self._loss_type = loss_type

            if status_callback:
                status_callback("Training started...")

            # Progress callback closure
            def on_train_epoch_end(trainer: Any):
                self.current_epoch[0] += 1
                if progress_callback:
                    progress_callback(int((self.current_epoch[0] / epochs) * 100))
                if status_callback:
                    status_callback(f"Epoch {self.current_epoch[0]}/{epochs}")
                if is_cancelled and is_cancelled():
                    trainer.stop = True

            self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Build training kwargs
            train_kwargs: Dict[str, Any] = {
                "task": task,
                "data": str(cfg_path),
                "epochs": epochs,
                "batch": batch_size,
                "device": device,
                "imgsz": imgsz,
                "project": str(out_dir),
                "name": "yolo_training",
                "exist_ok": True,
                "verbose": True,
                # LR / scheduler
                "lr0": lr0,
                "lrf": lrf,
                "cos_lr": cos_lr,
                "warmup_epochs": warmup_epochs,
                "warmup_momentum": warmup_momentum,
                # Optimizer / regularization
                "optimizer": optimizer,
                "momentum": momentum,
                "weight_decay": weight_decay,
                # Augmentations
                "mosaic": mosaic,
                "mixup": mixup,
                "copy_paste": copy_paste,
                "hsv_h": hsv_h,
                "hsv_s": hsv_s,
                "hsv_v": hsv_v,
                "fliplr": fliplr,
                "flipud": flipud,
                "degrees": degrees,
                "translate": translate,
                "scale": scale,
                "shear": shear,
                "perspective": perspective,
                # Loss function weights
                "box": box,
                "cls": cls,
                "dfl": dfl,
                # Early stopping & checkpointing
                "patience": 0 if patience == 0 else patience,
                "save_period": save_period if save_period > 0 else -1,
                # Resume training
                "resume": resume,
            }

            # Focal Loss override via callback
            if loss_type := getattr(self, "_loss_type", "auto"):
                if loss_type.lower() == "focal":
                    if log_callback:
                        log_callback(
                            f"Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}"
                        )

                    def on_train_start(trainer: Any):
                        """Replace the loss function with Focal Loss."""
                        try:
                            from ultralytics.utils.loss import FocalLoss
                            existing_loss = trainer.model.criterion
                            trainer.model.criterion = FocalLoss(
                                existing_loss,
                                gamma=focal_gamma,
                                alpha=focal_alpha
                            )
                            if log_callback:
                                log_callback("Focal Loss applied successfully")
                        except Exception as e:
                            if log_callback:
                                log_callback(f"Warning: Could not apply Focal Loss: {e}")

                    self.model.add_callback("on_train_start", on_train_start)
                elif loss_type.lower() != "auto":
                    if log_callback:
                        log_callback(
                            f"Requested loss type: {loss_type} (gamma={focal_gamma}, alpha={focal_alpha}). "
                            f"Ultralytics applies its built-in VFL/DFL; focal_alpha/gamma are noted for reference."
                        )

            results = self.model.train(**train_kwargs)

            best_model_path = out_dir / "yolo_training" / "weights" / "best.pt"

            if log_callback:
                log_callback(f"Training completed. Best model: {best_model_path}")

            return {
                "success": True,
                "model_path": str(best_model_path),
                "training_dir": str(out_dir / "yolo_training"),
                "epochs_completed": self.current_epoch[0],
                "metrics": self._extract_metrics(results, task),
                "hyperparameters": {
                    "lr0": lr0,
                    "lrf": lrf,
                    "cos_lr": cos_lr,
                    "scheduler": "cosine" if cos_lr else "linear",
                    "imgsz": imgsz,
                    "optimizer": optimizer,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "mosaic": mosaic,
                    "mixup": mixup,
                    "copy_paste": copy_paste,
                    "loss_type": loss_type,
                    "focal_gamma": focal_gamma,
                    "focal_alpha": focal_alpha,
                    "box": box,
                    "cls": cls,
                    "dfl": dfl,
                    "patience": patience,
                    "save_period": save_period,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "epochs_completed": self.current_epoch[0] if hasattr(self, 'current_epoch') else 0,
            }

    def _extract_metrics(self, results: Any, task: str = "detect") -> Dict[str, float]:
        """Extract metrics from training results.

        Args:
            results: Results object from model.train()
            task: Task type ("detect" or "segment")

        Returns:
            Dict with mAP, precision, recall metrics
        """
        try:
            metrics = {
                "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
            # Also extract mask metrics for segmentation
            if task == "segment":
                metrics["mask_map50"] = float(results.results_dict.get("metrics/mAP50(M)", 0))
                metrics["mask_map50_95"] = float(results.results_dict.get("metrics/mAP50-95(M)", 0))
                metrics["mask_precision"] = float(results.results_dict.get("metrics/precision(M)", 0))
                metrics["mask_recall"] = float(results.results_dict.get("metrics/recall(M)", 0))
            return metrics
        except Exception:
            return {"map50": 0, "map50_95": 0, "precision": 0, "recall": 0}

    def create_dataset_yaml(
        self,
        class_names: List[str],
        dataset_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create YAML configuration file for YOLO training.

        Args:
            class_names: List of class names for the dataset
            dataset_path: Path to the dataset directory (optional)
            output_path: Output path for dataset.yaml (optional)

        Returns:
            Path to the created dataset.yaml file
        """
        dataset = Path(dataset_path) if dataset_path else self.output_dir

        if output_path is None:
            output_path = dataset / "dataset.yaml"
        else:
            output_path = Path(output_path)

        config = {
            "path": str(dataset.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return output_path

    def export_model(
        self,
        export_format: str = "onnx",
        model_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Export trained YOLO model to different formats.

        Args:
            export_format: Export format ("onnx", "torchscript", "tflite", "pb", etc.)
            model_path: Path to model file (optional)
            output_dir: Output directory (optional)
            progress_callback: Progress callback
            status_callback: Status callback
            log_callback: Log callback
            is_cancelled: Cancel check callback

        Returns:
            Dict with export results
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {
                "success": False,
                "error": "Ultralytics YOLO not installed."
            }

        model_file = Path(model_path) if model_path else (
            self.output_dir / "yolo_training" / "weights" / "best.pt" if self.output_dir else None
        )

        if not model_file or not model_file.exists():
            return {
                "success": False,
                "error": f"Model file not found: {model_file}"
            }

        out_dir = Path(output_dir) if output_dir else model_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if status_callback:
            status_callback(f"Exporting model to {export_format}...")

        try:
            model = YOLO(str(model_file))
            results = model.export(format=export_format)

            if log_callback:
                log_callback(f"Model exported to: {results}")

            return {
                "success": True,
                "exported_path": str(results),
                "format": export_format,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}",
            }


# Backward compatibility - module-level functions
def train_yolo(
    config_path,
    epochs: int = 100,
    batch_size: int = 16,
    model_type: str = "yolov8n",
    device: str = "0",
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
):
    """Module-level training function for backward compatibility."""
    trainer = ModelTrainer(config_path, output_dir)
    return trainer.train_yolo(
        epochs, batch_size, model_type, device,
        progress_callback, status_callback, log_callback, is_cancelled
    )


def create_dataset_yaml(
    dataset_path: Optional[Union[str, Path]] = None,
    class_names: List[str]=None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Module-level dataset YAML creation function."""
    trainer = ModelTrainer()
    return trainer.create_dataset_yaml(class_names, dataset_path, output_path)


def export_model(
    model_path: Optional[Union[str, Path]],
    export_format: str = "onnx",
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Module-level model export function."""
    trainer = ModelTrainer()
    return trainer.export_model(export_format, model_path, output_dir,
                               progress_callback, status_callback, log_callback, is_cancelled)
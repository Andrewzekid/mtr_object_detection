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
#            epochs=10, batch_size=8, model_type='yolov8n', device='0')"
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
from typing import Optional, Callable, Dict, Any, List
import yaml
import json


class ModelTrainer:
    """Class for training and managing YOLO models."""
    
    AVAILABLE_MODELS = [
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
        "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    ]
    
    def __init__(self, config_path: Optional[str | Path] = None, output_dir: Optional[str | Path] = None):
        """Initialize with config path and output directory."""
        self.config_path = Path(config_path) if config_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.model = None
        self.current_epoch = 0
        self.training_history = []
    
    def set_config_path(self, config_path: str | Path):
        """Set the dataset config path."""
        self.config_path = Path(config_path)
    
    def set_output_dir(self, output_dir: str | Path):
        """Set the output directory for training results."""
        self.output_dir = Path(output_dir)
    
    def train_yolo(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        model_type: str = "yolov8n",
        device: str = "0",
        checkpoint_path: Optional[str | Path] = None,
        resume: bool = False,
        # Learning rate & scheduler
        lr0: float = 0.01,
        lrf: float = 0.01,
        cos_lr: bool = True,
        warmup_epochs: float = 3.0,
        warmup_momentum: float = 0.8,
        # Image size
        imgsz: int = 640,
        # Optimizer
        optimizer: str = "SGD",
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        # Augmentations
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
        # Loss function & class balance
        loss_type: str = "auto",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        # Early stopping & checkpointing
        patience: int = 50,
        save_period: int = -1,
        # Task type
        task: str = "detect",
        # Paths
        config_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        # Callbacks
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Train YOLO model using Ultralytics with full hyperparameter control."""
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"success": False, "error": "Ultralytics YOLO not installed. Install with: pip install ultralytics"}

        cfg_path = Path(config_path) if config_path else self.config_path
        out_dir = Path(output_dir) if output_dir else self.output_dir

        if not cfg_path or not cfg_path.exists():
            return {"success": False, "error": f"Config file not found: {cfg_path}"}

        if not out_dir:
            out_dir = Path("output/training")
        out_dir.mkdir(parents=True, exist_ok=True)

        if status_callback:
            status_callback(f"Loading {model_type} model...")
        if log_callback:
            log_callback(f"Starting YOLO training with {model_type}")
            log_callback(f"Config: {cfg_path}, Epochs: {epochs}, Batch size: {batch_size}")
            log_callback(f"LR: {lr0} (scheduler: {'cosine' if cos_lr else 'linear'}), Image size: {imgsz}, Optimizer: {optimizer}")

        try:
            # If a checkpoint path is supplied, load it directly (supports fine-tuning).
            # Otherwise, fall back to a pretrained model name like "yolov8n.pt".
            # For segmentation tasks, use "-seg" suffix (e.g. "yolo26l-seg.pt").
            if checkpoint_path:
                ckpt = Path(checkpoint_path)
                if not ckpt.exists():
                    return {"success": False, "error": f"Checkpoint file not found: {ckpt}"}
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

            if status_callback:
                status_callback("Training started...")

            def on_train_epoch_end(trainer):
                self.current_epoch[0] += 1
                if progress_callback:
                    progress_callback(int((self.current_epoch[0] / epochs) * 100))
                if status_callback:
                    status_callback(f"Epoch {self.current_epoch[0]}/{epochs}")
                if is_cancelled and is_cancelled():
                    trainer.stop = True

            self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Build the kwargs dict for model.train() with all hyperparameters
            train_kwargs = dict(
                task=task,
                data=str(cfg_path),
                epochs=epochs,
                batch=batch_size,
                device=device,
                imgsz=imgsz,
                project=str(out_dir),
                name="yolo_training",
                exist_ok=True,
                verbose=True,
                # LR / scheduler
                lr0=lr0,
                lrf=lrf,
                cos_lr=cos_lr,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                # Optimizer / regularization
                optimizer=optimizer,
                momentum=momentum,
                weight_decay=weight_decay,
                # Augmentations
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                fliplr=fliplr,
                flipud=flipud,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                # Loss function weights
                box=box,
                cls=cls,
                dfl=dfl,
                # Early stopping & checkpointing
                patience=0 if patience == 0 else patience,  # 0 disables
                save_period=save_period if save_period > 0 else -1,
                # Resume training
                resume=resume,
            )

            # Focal Loss override via callback: Ultralytics doesn't accept loss/fl_gamma/fl_alpha
            # as direct training arguments, so we use a callback to replace the loss function
            if loss_type.lower() == "focal":
                if log_callback:
                    log_callback(
                        f"Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}"
                    )
                
                def on_train_start(trainer):
                    """Replace the loss function with Focal Loss."""
                    try:
                        from ultralytics.utils.loss import FocalLoss
                        # Get the existing loss function and wrap it with Focal Loss
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
                    "lr0": lr0, "lrf": lrf, "cos_lr": cos_lr, "scheduler": ("cosine" if cos_lr else "linear"),
                    "imgsz": imgsz, "optimizer": optimizer, "momentum": momentum, "weight_decay": weight_decay,
                    "mosaic": mosaic, "mixup": mixup, "copy_paste": copy_paste,
                    "loss_type": loss_type, "focal_gamma": focal_gamma, "focal_alpha": focal_alpha,
                    "box": box, "cls": cls, "dfl": dfl,
                    "patience": patience, "save_period": save_period,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}",
                "epochs_completed": self.current_epoch[0] if hasattr(self, 'current_epoch') else 0,
            }
    
    def _extract_metrics(self, results, task: str = "detect") -> Dict:
        """Extract metrics from training results.
        
        For detection: extracts box mAP50, mAP50-95, precision, recall.
        For segmentation: also extracts mask mAP50, mAP50-95, precision, recall.
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
        dataset_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Create YAML configuration file for YOLO training."""
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
        model_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Export trained YOLO model to different formats."""
        try:
            from ultralytics import YOLO
        except ImportError:
            return {"success": False, "error": "Ultralytics YOLO not installed."}
        
        model_file = Path(model_path) if model_path else (self.output_dir / "yolo_training" / "weights" / "best.pt" if self.output_dir else None)
        
        if not model_file or not model_file.exists():
            return {"success": False, "error": f"Model file not found: {model_file}"}
        
        out_dir = Path(output_dir) if output_dir else model_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if status_callback:
            status_callback(f"Exporting model to {export_format}...")
        
        try:
            model = YOLO(str(model_file))
            results = model.export(format=export_format)
            
            if log_callback:
                log_callback(f"Model exported to: {results}")
            
            return {"success": True, "exported_path": str(results), "format": export_format}
        except Exception as e:
            return {"success": False, "error": f"Export failed: {str(e)}"}


# Backward compatibility
def train_yolo(config_path, epochs=100, batch_size=16, model_type="yolov8n",
               device="0", output_dir=None,
               progress_callback=None, status_callback=None,
               log_callback=None, is_cancelled=None):
    trainer = ModelTrainer(config_path, output_dir)
    return trainer.train_yolo(epochs, batch_size, model_type, device,
                              progress_callback, status_callback, log_callback, is_cancelled)


def create_dataset_yaml(dataset_path, class_names, output_path=None):
    trainer = ModelTrainer()
    return trainer.create_dataset_yaml(class_names, dataset_path, output_path)


def export_model(model_path, export_format="onnx", output_dir=None,
                 progress_callback=None, status_callback=None,
                 log_callback=None, is_cancelled=None):
    trainer = ModelTrainer()
    return trainer.export_model(export_format, model_path, output_dir,
                                progress_callback, status_callback, log_callback, is_cancelled)
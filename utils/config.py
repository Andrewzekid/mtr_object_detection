#    utils/config.py - Centralized application state (singleton).
#
#    USAGE (OOP):
#        from utils.config import Config
#        config = Config()
#        config.set_dataset_path("./my_dataset")
#        config.set_trained_model_path("./model.pt")
#        config.update_pipeline_state("labeling_complete", True)
#
#    USAGE (SINGLETON - PREFERRED IN GUI / SCRIPTS):
#        from utils.config import config           # auto-instantiated
#        print(config.project_root)                # absolute Path of repo
#        print(config.current_dataset_path)        # Path or None
#        print(config.trained_model_path)
#        print(config.pipeline_state)
#
#    USAGE (PERSISTENCE):
#        config.save_config()                      # writes project_root/config.json
#        config.save_config("./my_config.json")    # custom path
#        config.load_config()                      # reads project_root/config.json
#        config.load_config("./my_config.json")
#
#    RUN AS A ONE-LINER:
#        python -c "from utils.config import config; print(config.project_root)"
#
#    ARGUMENTS (Config class):
#        No constructor arguments. All state is set via setters:
#            set_dataset_path(path)        - sets current_dataset_path
#                                           (auto-derives current_images_path
#                                            and current_labels_path)
#            set_trained_model_path(path)   - sets trained_model_path
#            update_pipeline_state(step, completed)  - toggles pipeline flags
#
#    ATTRIBUTES YOU CAN READ:
#        project_root, core_dir, gui_dir, utils_dir   - Paths
#        sam3_models_dir                               - core/sam3/models
#        current_dataset_path, current_images_path,
#          current_labels_path, trained_model_path,
#          yolo_config_path
#        output_dir, augmented_dir, split_dir,
#          training_dir, evaluation_dir, visualizations_dir
#        pipeline_state                                - dict of 6 booleans
#        dataset_stats                                 - dict
#        training_params                               - dict
#        ollama_config, sam3_config                    - dicts
#
#    REQUIREMENTS:
#        None (stdlib only - json, pathlib, typing)

"""
Global settings and path management for the Object Detection Application.
Acts as a centralized state manager (singleton pattern).
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Centralized configuration and state manager."""

    _instance: Optional['Config'] = None

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Base paths
        self.project_root = Path(__file__).parent.parent
        self.core_dir = self.project_root / "core"
        self.gui_dir = self.project_root / "gui"
        self.utils_dir = self.project_root / "utils"

        # SAM3 model path (local weights)
        self.sam3_models_dir = self.core_dir / "sam3" / "models"

        # Data paths (to be set by user)
        self.current_dataset_path: Optional[Path] = None
        self.current_images_path: Optional[Path] = None
        self.current_labels_path: Optional[Path] = None

        # Model paths
        self.trained_model_path: Optional[Path] = None
        self.yolo_config_path: Optional[Path] = None

        # Output paths
        self.output_dir = self.project_root / "output"
        self.augmented_dir = self.output_dir / "augmented"
        self.split_dir = self.output_dir / "split"
        self.training_dir = self.output_dir / "training"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.visualizations_dir = self.output_dir / "visualizations"

        # Pipeline progress states
        self.pipeline_state: Dict[str, Any] = {
            "labeling_complete": False,
            "augmentation_complete": False,
            "split_complete": False,
            "training_complete": False,
            "evaluation_complete": False,
            "visualization_complete": False,
        }

        # Dataset statistics
        self.dataset_stats: Dict[str, Any] = {
            "total_images": 0,
            "class_distribution": {},
            "image_dimensions": [],
        }

        # Training parameters
        self.training_params: Dict[str, Any] = {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.01,
            "model_type": "yolov8n",  # nano, small, medium, large, xlarge
        }

        # Ollama configuration for Qwen3.6
        self.ollama_config: Dict[str, Any] = {
            "base_url": "http://localhost:11434",
            "model_name": "qwen3.6:27b",
            "timeout": 120,
        }

        # SAM3 configuration (local)
        self.sam3_config: Dict[str, Any] = {
            "model_type": "sam3",  # Model variant
            "device": "cuda",  # cuda or cpu
            "confidence_threshold": 0.5,
        }

        # Ensure output directories exist
        self._create_directories()

        self._initialized = True

    def _create_directories(self):
        """Create necessary output directories if they don't exist."""
        dirs_to_create = [
            self.output_dir,
            self.augmented_dir,
            self.split_dir,
            self.training_dir,
            self.evaluation_dir,
            self.visualizations_dir,
            self.sam3_models_dir,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions (used by auto-detection)
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def set_dataset_path(self, path: str | Path):
        """Set the current dataset path.

        Auto-detects the dataset layout:
          - 'unlabelled'    : a flat folder of images (no images/ subdir)
          - 'yolo_flat'     : has images/ and labels/ subdirs (flat)
          - 'yolo_nested'   : has images/{train,val,test}/ and labels/{train,val,test}/
          - 'yolo_split'    : has train/, test/, val/ subdirs (already split)
        """
        root = Path(path)
        self.current_dataset_path = root

        # --- auto-detect layout ---
        has_images_subdir = (root / "images").is_dir()
        has_labels_subdir = (root / "labels").is_dir()
        has_train = (root / "train").is_dir()
        has_test  = (root / "test").is_dir()
        has_val   = (root / "val").is_dir()
        
        # Check for nested YOLO structure: images/{train,val,test}/
        has_nested_train = (root / "images" / "train").is_dir()
        has_nested_val = (root / "images" / "val").is_dir()
        has_nested_test = (root / "images" / "test").is_dir()

        if has_train or has_test or has_val:
            # Already-split YOLO dataset (train/images/, etc.)
            self.dataset_type = "yolo_split"
            self.current_images_path = root          # images live under train/images etc.
            self.current_labels_path = root
        elif has_images_subdir and has_labels_subdir and (has_nested_train or has_nested_val or has_nested_test):
            # Nested YOLO dataset (images/train/, images/val/, images/test/)
            self.dataset_type = "yolo_nested"
            self.current_images_path = root / "images"
            self.current_labels_path = root / "labels"
        elif has_images_subdir and has_labels_subdir:
            # Flat YOLO dataset (images/ + labels/)
            self.dataset_type = "yolo_flat"
            self.current_images_path = root / "images"
            self.current_labels_path = root / "labels"
        else:
            # Unlabelled folder of images
            self.dataset_type = "unlabelled"
            self.current_images_path = root
            self.current_labels_path = root / "labels"

    def get_image_files(self) -> list:
        """Return a list of image files for the current dataset, regardless of type.
        
        For nested YOLO datasets (yolo_nested), recursively gathers images from
        train/, val/, test/ subdirectories inside images/.
        """
        if not self.current_images_path or not self.current_images_path.exists():
            return []
        
        # For nested YOLO structure, recursively gather from split subdirectories
        if getattr(self, 'dataset_type', None) == 'yolo_nested':
            image_files = []
            for split_dir in ['train', 'val', 'test']:
                split_path = self.current_images_path / split_dir
                if split_path.exists():
                    image_files.extend(
                        f for f in split_path.iterdir()
                        if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
                    )
            return sorted(image_files)
        
        # For other types, look at direct children only
        return sorted(
            f for f in self.current_images_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def set_trained_model_path(self, path: str | Path):
        """Set the trained model path."""
        self.trained_model_path = Path(path)

    def update_pipeline_state(self, step: str, completed: bool):
        """Update the pipeline progress state."""
        if step in self.pipeline_state:
            self.pipeline_state[step] = completed

    def get_pipeline_progress(self) -> float:
        """Calculate overall pipeline progress percentage."""
        total_steps = len(self.pipeline_state)
        completed_steps = sum(1 for v in self.pipeline_state.values() if v)
        return (completed_steps / total_steps) * 100 if total_steps > 0 else 0

    def save_config(self, filepath: Optional[str | Path] = None):
        """Save configuration to JSON file."""
        if filepath is None:
            filepath = self.project_root / "config.json"

        config_data = {
            "current_dataset_path": str(self.current_dataset_path) if self.current_dataset_path else None,
            "trained_model_path": str(self.trained_model_path) if self.trained_model_path else None,
            "pipeline_state": self.pipeline_state,
            "training_params": self.training_params,
            "ollama_config": self.ollama_config,
            "sam3_config": self.sam3_config,
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

    def load_config(self, filepath: Optional[str | Path] = None):
        """Load configuration from JSON file."""
        if filepath is None:
            filepath = self.project_root / "config.json"

        if not Path(filepath).exists():
            return

        with open(filepath, 'r') as f:
            config_data = json.load(f)

        if config_data.get("current_dataset_path"):
            self.set_dataset_path(config_data["current_dataset_path"])

        if config_data.get("trained_model_path"):
            self.set_trained_model_path(config_data["trained_model_path"])

        if "pipeline_state" in config_data:
            self.pipeline_state.update(config_data["pipeline_state"])

        if "training_params" in config_data:
            self.training_params.update(config_data["training_params"])

        if "ollama_config" in config_data:
            self.ollama_config.update(config_data["ollama_config"])

        if "sam3_config" in config_data:
            self.sam3_config.update(config_data["sam3_config"])


# Global config instance
config = Config()

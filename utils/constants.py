"""
Shared constants and configuration values across the application.

This module provides centralized configuration that can be imported
by any other module in the project.
"""

from pathlib import Path
from typing import Final

# ============ Image Extensions ============
IMAGE_EXTENSIONS: Final[set[str]] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
}
DEFAULT_IMAGE_EXTENSIONS: Final[set[str]] = IMAGE_EXTENSIONS

LABEL_EXTENSIONS: Final[set[str]] = {
    ".txt", ".xml"
}
DEFAULT_LABEL_EXTENSIONS: Final[set[str]] = LABEL_EXTENSIONS

MODEL_EXTENSIONS: Final[set[str]] = {
    ".pt", ".pth", ".onnx", ".torchscript", ".tflite", ".pb"
}
DEFAULT_MODEL_EXTENSIONS: Final[set[str]] = MODEL_EXTENSIONS

# ============ Default Dataset Splits ============
DEFAULT_TRAIN_RATIO: Final[float] = 0.7
DEFAULT_VAL_RATIO: Final[float] = 0.15
DEFAULT_TEST_RATIO: Final[float] = 0.15

DEFAULT_SPLIT_RATIOS: Final[list[float]] = [DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO]
DEFAULT_SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")

# ============ YOLO Model Families ============
YOLO_MODEL_FAMILIES: Final[list[str]] = ["yolov8", "yolo11", "yolo12", "yolo26"]

# Model size prefixes ordered by parameter count
MODEL_SIZES: Final[list[str]] = ["n", "s", "m", "l", "x"]
MODEL_SIZE_NAMES: Final[dict[str, str]] = {
    "n": "nano",
    "s": "small",
    "m": "medium",
    "l": "large",
    "x": "extra large",
}

# ============ Default Paths ============
DEFAULT_PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
DEFAULT_DATASET_DIR: Final[Path] = DEFAULT_PROJECT_ROOT / "datasets"
DEFAULT_MODELS_DIR: Final[Path] = DEFAULT_PROJECT_ROOT / "models"
DEFAULT_LOG_DIR: Final[Path] = DEFAULT_PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_DIR: Final[Path] = DEFAULT_PROJECT_ROOT / "output"

# ============ Output Directory Structure ============
TRAINING_OUTPUT_SUBDIR: Final[str] = "yolo_training"
CHECKPOINT_SUBDIR: Final[str] = "weights"
BEST_MODEL_FILE: Final[str] = "best.pt"
CURRENT_MODEL_FILE: Final[str] = "last.pt"

# ============ Directory Structure for Dataset Splits ============
SPLIT_IMAGE_DIR_NAME: Final[str] = "images"
SPLIT_LABELS_DIR_NAME: Final[str] = "labels"

# ============ Hyperparameter Defaults ============
DEFAULT_LR0: Final[float] = 0.01
DEFAULT_LRF: Final[float] = 0.01
DEFAULT_WARMUP_EPOCHS: Final[float] = 3.0
DEFAULT_WARMUP_MOMENTUM: Final[float] = 0.8

DEFAULT_MOMENTUM: Final[float] = 0.937
DEFAULT_WEIGHT_DECAY: Final[float] = 0.0005

DEFAULT_BOX_LOSS: Final[float] = 7.5
DEFAULT_CLASS_LOSS: Final[float] = 0.5
DEFAULT_DFL_LOSS: Final[float] = 1.5

DEFAULT_PATIENCE: Final[int] = 50
DEFAULT_SAVE_PERIOD: Final[int] = -1

DEFAULT_IMG_SIZE: Final[int] = 640

# ============ SAM Defaults ============
DEFAULT_SOURCE_TYPE: Final[str] = "image"
DEFAULT_Prompt_TYPE: Final[str] = "point"
DEFAULT_HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.65
DEFAULT_POINT_PROMPT: Final[str] = "point, + 1"

# ============ Qwen Model Config ============
DEFAULT_QWEN_MODEL: Final[str] = "Qwen/Qwen3.5-0.6B-Instruct"
DEFAULT_QWEN_TEMPLATE_ID: Final[str] = "chat"
DEFAULT_QWEN_TOKEN_LIMIT: Final[int] = 8192

#!/usr/bin/env python3
"""
Train a YOLO object detection model.

This script trains a YOLO model using Ultralytics, with support for various
model sizes and training configurations.

USAGE:
    python scripts/04_train_model.py --config ./output/split/dataset.yaml
    python scripts/04_train_model.py --config ./output/split/dataset.yaml --epochs 100 --batch-size 16
    python scripts/04_train_model.py --config ./output/split/dataset.yaml --model-type yolov8s --device 0

PREREQUISITES:
    - Ultralytics installed: pip install ultralytics
    - Dataset YAML file created (use 03_split_dataset.py --generate-yaml)
    - GPU recommended for faster training

OUTPUT:
    - Trained model weights (best.pt, last.pt)
    - Training logs and metrics
    - Confusion matrix and PR curves
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO object detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/04_train_model.py --config ./Datasets/HKU_GH/HKU_GH_1k/data.yaml --epochs 1000 --batch-size 32 --task segment -m yolo26n --imgsz 1024 --output-dir ./output/training/yolo_training/HKU_GH/yolo26n
    # Basic training with defaults
    python scripts/04_train_model.py --config ./output/split/dataset.yaml

    # Custom epochs and batch size
    python scripts/04_train_model.py --config ./output/split/dataset.yaml \\
        --epochs 100 --batch-size 16

    # Use larger model for better accuracy
    python scripts/04_train_model.py --config ./output/split/dataset.yaml \\
        --model-type yolov8l --epochs 200

    # Train on CPU (slower)
    python scripts/04_train_model.py --config ./output/split/dataset.yaml --device cpu

    # Custom output directory
    python scripts/04_train_model.py --config ./output/split/dataset.yaml \\
        --output-dir ./output/training/my_experiment

    # Export trained model to ONNX
    python scripts/04_train_model.py --config ./output/split/dataset.yaml \\
        --export onnx

Model Types:
    yolov8n   - Nano (fastest, smallest, ~3.2M params)
    yolov8s   - Small (fast, ~11.2M params)
    yolov8m   - Medium (balanced, ~25.9M params)
    yolov8l   - Large (accurate, ~43.7M params)
    yolov8x   - Extra large (most accurate, ~68.2M params)

Device Options:
    0, 1, 2...  - GPU device index
    cpu          - CPU training (slower)

Output Structure:
    output_dir/
    └── yolo_training/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        └── args.yaml
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output/training",
        help="Output directory for training results (default: ./output/training)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (1-1000, default: 100)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for training (1-128, default: 16)",
    )
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        default="yolov8n",
        choices=[
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
            "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
        ],
        help="YOLO model type (default: yolov8n)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "segment"],
        help="Task type: 'detect' for object detection, 'segment' for instance segmentation (default: detect)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="0",
        help="Device to train on: '0' for GPU 0, 'cpu' for CPU (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "torchscript", "tflite", "engine"],
        help="Export trained model to specified format after training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained weights to fine-tune",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="auto",
        choices=["auto", "focal"],
        help="Loss function type: 'auto' (default VFL/DFL) or 'focal' for Focal Loss (default: auto)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter - focusing parameter (default: 2.0)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal Loss alpha parameter - balancing parameter (default: 0.25)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor: lr0 * lrf = final LR (default: 0.01)",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Use cosine annealing LR schedule (default: True)",
    )
    parser.add_argument(
        "--no-cos-lr",
        dest="cos_lr",
        action="store_false",
        help="Use linear LR schedule instead of cosine",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs (default: 3.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Validate epochs
    if args.epochs < 1 or args.epochs > 1000:
        print(f"Error: Epochs must be between 1 and 1000, got {args.epochs}")
        sys.exit(1)
    
    # Validate batch size
    if args.batch_size < 1 or args.batch_size > 128:
        print(f"Error: Batch size must be between 1 and 128, got {args.batch_size}")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO MODEL TRAINING")
    print("=" * 60)
    print(f"\nConfig file: {config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.imgsz}")
    print(f"Task: {args.task}")
    if args.resume:
        print("Resume: enabled")
    if args.pretrained:
        print(f"Pretrained weights: {args.pretrained}")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == "focal":
        print(f"Focal gamma: {args.focal_gamma}")
        print(f"Focal alpha: {args.focal_alpha}")
    print(f"Learning rate: {args.lr0} (decay factor: {args.lrf}, final LR: {args.lr0 * args.lrf:.6f})")
    print(f"LR scheduler: {'cosine' if args.cos_lr else 'linear'}, warmup: {args.warmup_epochs} epochs")
    
    # Create trainer
    trainer = ModelTrainer(
        config_path=str(config_path),
        output_dir=args.output_dir,
    )
    
    # Run training
    print("\nStarting training...")
    print("-" * 60)
    
    result = trainer.train_yolo(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        device=args.device,
        imgsz=args.imgsz,
        resume=args.resume,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        task=args.task,
        checkpoint_path=args.pretrained,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    if result.get("success", True):
        model_path = result.get("model_path")
        metrics = result.get("metrics", {})
        
        print(f"\nModel saved to: {model_path}")
        
        if metrics:
            print(f"\nTraining Metrics:")
            print(f"  mAP50:    {metrics.get('map50', metrics.get('mAP50', 'N/A')):.4f}" if isinstance(metrics.get('map50', metrics.get('mAP50')), (int, float)) else f"  mAP50:    N/A")
            print(f"  mAP50-95: {metrics.get('map50_95', metrics.get('mAP50_95', 'N/A')):.4f}" if isinstance(metrics.get('map50_95', metrics.get('mAP50_95')), (int, float)) else f"  mAP50-95: N/A")
            print(f"  Precision:{metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"  Precision: N/A")
            print(f"  Recall:   {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"  Recall:    N/A")
        
        # Export model if requested
        if args.export and model_path:
            print(f"\nExporting model to {args.export}...")
            export_result = trainer.export_model(
                export_format=args.export,
                model_path=model_path,
            )
            if export_result.get("success"):
                print(f"  Exported to: {export_result.get('export_path')}")
            else:
                print(f"  Export failed: {export_result.get('error', 'Unknown error')}")
        
        print(f"\nOutput directory: {args.output_dir}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("\nYOLO training completed successfully!")


if __name__ == "__main__":
    main()
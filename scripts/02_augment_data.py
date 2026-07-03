#!/usr/bin/env python3
"""
Augment a labeled dataset with various transformations.

This script applies augmentations (flip, rotate, brightness, contrast, mosaic)
to a YOLO-format labeled dataset, producing augmented images and corresponding labels.

USAGE:
    python scripts/02_augment_data.py --input-dir ./data/labeled --output-dir ./output/augmented
    python scripts/02_augment_data.py --input-dir ./data/labeled --output-dir ./output/augmented --multiplier 3
    python scripts/02_augment_data.py --input-dir ./data/labeled --output-dir ./output/augmented \\
        --augmentations flip_horizontal rotate brightness mosaic

INPUT DIRECTORY STRUCTURE:
    input_dir/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt
        └── img2.txt

OUTPUT:
    Augmented dataset with same structure:
    output_dir/
    ├── images/
    │   ├── img1.jpg
    │   ├── img1_aug_001_flip_horizontal.jpg
    │   └── ...
    └── labels/
        ├── img1.txt
        ├── img1_aug_001_flip_horizontal.txt
        └── ...

AVAILABLE AUGMENTATIONS:
    - flip_horizontal: Mirror image horizontally
    - flip_vertical: Mirror image vertically
    - rotate: Rotate image by random angle within range
    - brightness: Adjust image brightness
    - contrast: Adjust image contrast
    - mosaic: Combine 4 images into a 2x2 grid
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_processor import DataProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment a YOLO-format labeled dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic augmentation with defaults (flip_horizontal, rotate, brightness)
    python scripts/02_augment_data.py --input-dir ./data/labeled \\
        --output-dir ./output/augmented

    # Custom multiplier (3 augmented copies per image)
    python scripts/02_augment_data.py --input-dir ./data/labeled \\
        --output-dir ./output/augmented --multiplier 3

    # Specific augmentation types
    python scripts/02_augment_data.py --input-dir ./data/labeled \\
        --output-dir ./output/augmented \\
        --augmentations flip_horizontal flip_vertical rotate mosaic

    # Custom rotation and brightness ranges
    python scripts/02_augment_data.py --input-dir ./data/labeled \\
        --output-dir ./output/augmented \\
        --rotation-range -30 30 \\
        --brightness-range 0.7 1.3

    # All augmentation types
    python scripts/02_augment_data.py --input-dir ./data/labeled \\
        --output-dir ./output/augmented \\
        --augmentations flip_horizontal flip_vertical rotate brightness contrast mosaic

Available Augmentations:
    flip_horizontal   - Mirror image horizontally (adjusts bbox accordingly)
    flip_vertical     - Mirror image vertically
    rotate            - Random rotation within specified range
    brightness        - Random brightness adjustment
    contrast          - Random contrast adjustment
    mosaic            - Combine 4 random images into 2x2 grid

Input Directory Structure:
    input_dir/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt
        └── img2.txt
        """,
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing images/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Path to output directory for augmented dataset",
    )
    parser.add_argument(
        "--augmentations", "-a",
        type=str,
        nargs="+",
        default=["flip_horizontal", "rotate", "brightness"],
        choices=["flip_horizontal", "flip_vertical", "rotate", "brightness", "contrast", "mosaic"],
        help="Augmentation types to apply (default: flip_horizontal rotate brightness)",
    )
    parser.add_argument(
        "--multiplier", "-m",
        type=int,
        default=2,
        help="Number of augmented copies per source image (1-10, default: 2)",
    )
    parser.add_argument(
        "--rotation-range",
        type=float,
        nargs=2,
        default=[-15, 15],
        metavar=("MIN", "MAX"),
        help="Rotation angle range in degrees (default: -15 15)",
    )
    parser.add_argument(
        "--brightness-range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        metavar=("MIN", "MAX"),
        help="Brightness/contrast factor range (default: 0.8 1.2)",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Name of images subdirectory (default: images)",
    )
    parser.add_argument(
        "--labels-subdir",
        type=str,
        default="labels",
        help="Name of labels subdirectory (default: labels)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    images_path = input_path / args.images_subdir
    labels_path = input_path / args.labels_subdir
    
    if not images_path.exists():
        print(f"Error: Images directory not found: {images_path}")
        sys.exit(1)
    
    if not labels_path.exists():
        print(f"Error: Labels directory not found: {labels_path}")
        sys.exit(1)
    
    # Validate multiplier
    if args.multiplier < 1 or args.multiplier > 10:
        print(f"Error: Multiplier must be between 1 and 10, got {args.multiplier}")
        sys.exit(1)
    
    print("=" * 60)
    print("DATASET AUGMENTATION")
    print("=" * 60)
    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentations: {', '.join(args.augmentations)}")
    print(f"Multiplier: {args.multiplier}")
    print(f"Rotation range: {args.rotation_range}")
    print(f"Brightness range: {args.brightness_range}")
    
    # Create DataProcessor configuration
    config = {
        "input_dir": str(input_path),
        "output_dir": args.output_dir,
        "augmentation_types": args.augmentations,
        "multiplier": args.multiplier,
        "rotation_range": tuple(args.rotation_range),
        "brightness_range": tuple(args.brightness_range),
    }
    
    # Create processor and run augmentation
    print("\nStarting augmentation...")
    processor = DataProcessor(config)
    
    # Define callbacks for progress reporting
    def progress_callback(value):
        pass  # Silent progress
    
    def status_callback(msg):
        print(f"  {msg}")
    
    def log_callback(msg):
        print(f"  {msg}")
    
    result = processor.augment_dataset(
        progress_callback=progress_callback,
        status_callback=status_callback,
        log_callback=log_callback,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    
    if result.get("success", True):
        print(f"\nTotal images processed: {result.get('total_images', 'N/A')}")
        print(f"Total augmented images: {result.get('augmented_images', 'N/A')}")
        print(f"Output directory: {args.output_dir}")
        
        # Count output files
        output_path = Path(args.output_dir)
        if output_path.exists():
            out_images = output_path / args.images_subdir
            out_labels = output_path / args.labels_subdir
            
            if out_images.exists():
                num_out_images = len(list(out_images.glob("*")))
                print(f"Output images: {num_out_images}")
            
            if out_labels.exists():
                num_out_labels = len(list(out_labels.glob("*.txt")))
                print(f"Output labels: {num_out_labels}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("\nDataset augmentation completed successfully!")


if __name__ == "__main__":
    main()
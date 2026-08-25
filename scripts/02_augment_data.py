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
    Augmented dataset with the same structure:
    output_dir/
    ├── images/
    │   ├── img1.jpg
    │   ├── img1_aug0_rotate.jpg
    │   └── ...
    └── labels/
        ├── img1.txt
        ├── img1_aug0_rotate.txt
        └── ...

AVAILABLE AUGMENTATIONS:
    - flip_horizontal: Mirror image horizontally (mirrors bbox/polygon coords)
    - flip_vertical: Mirror image vertically (mirrors bbox/polygon coords)
    - rotate: Rotate image and bounding boxes/polygons by a random angle
    - brightness: Adjust image brightness
    - contrast: Adjust image contrast
    - hue: Shift image hue (HSV) by a random amount
    - blur: Gaussian blur with a random odd kernel size
    - resize: Resize image to a fixed target size (normalized labels carry over)
    - mosaic: Combine 4 images into a 2x2 grid (bbox/polygon aware)

NOTE:
    Geometric augmentations (flip/rotate/mosaic) transform BOTH bounding-box
    and polygon (segmentation) labels; photometric ones (brightness, contrast,
    hue, blur) and resize leave the normalized labels unchanged.
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
    flip_horizontal   - Mirror image horizontally (adjusts bbox/polygon coords)
    flip_vertical     - Mirror image vertically
    rotate            - Random rotation within specified range (rotates bboxes/polygons)
    brightness        - Random brightness adjustment
    contrast          - Random contrast adjustment
    hue               - Random hue shift (--hue-range, degrees)
    blur              - Random Gaussian blur (--blur-range, kernel size)
    resize            - Resize to fixed target (--resize WIDTH HEIGHT)
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
        choices=["flip_horizontal", "flip_vertical", "rotate", "brightness",
                 "contrast", "hue", "blur", "resize", "mosaic"],
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
        "--hue-range",
        type=float,
        nargs=2,
        default=[-15, 15],
        metavar=("MIN", "MAX"),
        help="Hue shift range in degrees (default: -15 15)",
    )
    parser.add_argument(
        "--blur-range",
        type=int,
        nargs=2,
        default=[3, 9],
        metavar=("MIN", "MAX"),
        help="Gaussian blur kernel range (rounded to odd, default: 3 9)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for the 'resize' augmentation (e.g. --resize 640 640)",
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

    # Validate rotation range when rotation is requested
    if "rotate" in args.augmentations:
        rot_min, rot_max = args.rotation_range
        if rot_min > rot_max:
            print(f"Error: --rotation-range must be min max, got {args.rotation_range}")
            sys.exit(1)

    # Validate resize when requested
    if "resize" in args.augmentations and not args.resize:
        print("Error: --resize WIDTH HEIGHT is required when using the 'resize' augmentation")
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
        "hue_range": tuple(args.hue_range),
        "blur_range": tuple(args.blur_range),
        "resize": tuple(args.resize) if args.resize else None,
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
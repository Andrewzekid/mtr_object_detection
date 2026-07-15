#!/usr/bin/env python3
"""
Split a labeled dataset into train/test/val sets for YOLO training.

This script splits a YOLO-format dataset into training, testing, and validation
sets, and optionally generates a dataset.yaml file for YOLO training.

USAGE:
    python scripts/03_split_dataset.py --input-dir ./data/augmented --output-dir ./output/split
    python scripts/03_split_dataset.py --input-dir ./data/augmented --output-dir ./output/split --ratios 0.7 0.15 0.15
    python scripts/03_split_dataset.py --input-dir ./data/augmented --output-dir ./output/split \\
        --class-names car person dog --generate-yaml

INPUT DIRECTORY STRUCTURE:
    input_dir/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt
        └── img2.txt

OUTPUT:
    Split dataset with YOLO training structure:
    output_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml (if --generate-yaml)
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset_creator import DatasetCreator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a YOLO-format dataset into train/test/val sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic split with default ratios (70/15/15)
    python scripts/03_split_dataset.py --input-dir ./data/augmented \\
        --output-dir ./output/split

    # Custom split ratios (80/10/10)
    python scripts/03_split_dataset.py --input-dir ./data/augmented \\
        --output-dir ./output/split --ratios 0.8 0.1 0.1

    # With class names and YAML generation
    python scripts/03_split_dataset.py --input-dir ./data/augmented \\
        --output-dir ./output/split \\
        --class-names car person dog bicycle --generate-yaml

    # Reproducible split with seed
    python scripts/03_split_dataset.py --input-dir ./data/augmented \\
        --output-dir ./output/split --seed 42

    # Get statistics of an existing split
    python scripts/03_split_dataset.py --stats ./output/split

Split Ratios:
    The --ratios argument specifies [train, test, val] fractions.
    They must sum to 1.0. Default is [0.7, 0.15, 0.15].

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
        default=None,
        help="Path to directory containing images/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Path to output directory for split dataset",
    )
    parser.add_argument(
        "--ratios", "-r",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "TEST", "VAL"),
        help="Split ratios for train/test/val (must sum to 1.0, default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="List of class names for YAML generation (e.g., car person dog)",
    )
    parser.add_argument(
        "--generate-yaml",
        action="store_true",
        help="Generate dataset.yaml file for YOLO training",
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
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Print statistics for an existing split directory and exit",
    )

    return parser.parse_args()


def print_split_statistics(stats_path):
    """Print statistics for an existing split dataset.
    
    Args:
        stats_path: Path to split dataset directory
    """
    creator = DatasetCreator()
    stats = creator.get_split_statistics(stats_path)
    
    print("=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)
    print(f"\nDirectory: {stats_path}")
    
    for split_name in ["train", "val", "test"]:
        if split_name in stats:
            split_data = stats[split_name]
            print(f"\n{split_name.upper()}:")
            print(f"  Images: {split_data.get('image_count', 0)}")
            print(f"  Labels: {split_data.get('label_count', 0)}")
    
    total = sum(stats.get(s, {}).get("image_count", 0) for s in ["train", "val", "test"])
    print(f"\nTOTAL IMAGES: {total}")

    if total > 0:
        print(f"\nSplit Distribution:")
        for split_name in ["train", "val", "test"]:
            count = stats.get(split_name, {}).get("image_count", 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {split_name:5s}: {count:5d} ({pct:5.1f}%) {bar}")


def copy_labels_to_splits(output_dir: str, labels_path: Path):
    """Copy the correct .txt labels into each split after DatasetCreator ran.

    DatasetCreator.split_dataset expects labels under ``src/labels``; when the
    caller keeps labels next to the ``images/`` subdirectory, the labels are
    not found there. This helper fixes the resulting split by copying each
    label from the real ``labels_path`` into ``<output_dir>/<split>/labels/``.
    """
    out_path = Path(output_dir)
    for split in ["train", "test", "val"]:
        split_img_dir = out_path / split / "images"
        split_lbl_dir = out_path / split / "labels"
        if not split_img_dir.exists():
            continue
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img_file in split_img_dir.iterdir():
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, split_lbl_dir / f"{img_file.stem}.txt")
                copied += 1
        if copied:
            print(f"  Copied {copied} label(s) into {split}/labels")


def main():
    args = parse_args()
    
    # Stats mode - just print statistics and exit
    if args.stats:
        stats_path = Path(args.stats)
        if not stats_path.exists():
            print(f"Error: Directory not found: {stats_path}")
            sys.exit(1)
        print_split_statistics(str(stats_path))
        return
    
    # Validate required arguments for split mode
    if not args.input_dir:
        print("Error: --input-dir is required (unless using --stats)")
        sys.exit(1)
    
    if not args.output_dir:
        print("Error: --output-dir is required (unless using --stats)")
        sys.exit(1)
    
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
    
    # Validate ratios
    ratio_sum = sum(args.ratios)
    if abs(ratio_sum - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1.0, got {ratio_sum:.3f}")
        sys.exit(1)
    
    for ratio in args.ratios:
        if ratio < 0 or ratio > 1:
            print(f"Error: Each ratio must be between 0 and 1, got {ratio}")
            sys.exit(1)
    
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)
    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: train={args.ratios[0]:.2f}, test={args.ratios[1]:.2f}, val={args.ratios[2]:.2f}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    
    # Count input files
    num_images = len(list(images_path.glob("*")))
    num_labels = len(list(labels_path.glob("*.txt")))
    print(f"\nInput images: {num_images}")
    print(f"Input labels: {num_labels}")
    
    # Create DatasetCreator and run split
    print("\nSplitting dataset...")
    creator = DatasetCreator()
    
    result = creator.split_dataset(
        ratios=args.ratios,
        src=str(images_path),
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # DatasetCreator looks for labels under ``src/labels``; when the dataset
    # uses ``images/`` + ``labels/`` siblings we need to copy them ourselves.
    copy_labels_to_splits(args.output_dir, labels_path)

    # Print results
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    if result.get("success", True):
        splits = result.get("splits", {})
        print(f"\nSplit results:")
        for split_name, split_data in splits.items():
            count = split_data.get("image_count", split_data.get("count", 0))
            print(f"  {split_name}: {count} images")
        
        # Generate dataset.yaml if requested
        if args.generate_yaml and args.class_names:
            print(f"\nGenerating dataset.yaml...")
            from core.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            yaml_path = trainer.create_dataset_yaml(
                class_names=args.class_names,
                dataset_path=args.output_dir,
            )
            print(f"  Created: {yaml_path}")
        elif args.generate_yaml:
            print("\nWarning: --generate-yaml requires --class-names to be specified")
        
        print(f"\nOutput directory: {args.output_dir}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Print final statistics
    print("\n" + "-" * 60)
    print_split_statistics(args.output_dir)
    
    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    main()
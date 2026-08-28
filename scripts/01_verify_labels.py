#!/usr/bin/env python3
"""
Verify and validate YOLO-format labels.

This script checks YOLO label files for correctness, identifies missing labels,
and provides statistics about the labelled dataset.

USAGE:
    python scripts/01_verify_labels.py --input-dir ./data/labeled
    python scripts/01_verify_labels.py --input-dir ./data/labeled --fix
    python scripts/01_verify_labels.py --input-dir ./data/labeled --class-names car person dog

INPUT DIRECTORY STRUCTURE:
    input_dir/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels/
        ├── img1.txt
        ├── img2.txt
        └── ...

YOLO LABEL FORMAT:
    Each line in a label file: <class_id> <x_center> <y_center> <width> <height>
    - class_id: integer (0-indexed)
    - x_center, y_center, width, height: normalized (0-1) coordinates

OUTPUT:
    - Validation report with errors and warnings
    - Class distribution statistics
    - Summary of images with/without labels
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and validate YOLO-format labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic verification
    python scripts/01_verify_labels.py --input-dir ./data/labeled

    # With class names for better statistics
    python scripts/01_verify_labels.py --input-dir ./data/labeled \\
        --class-names car person dog bicycle

    # Auto-fix common issues (normalize coordinates > 1)
    python scripts/01_verify_labels.py --input-dir ./data/labeled --fix

    # Specify custom subdirectory names
    python scripts/01_verify_labels.py --input-dir ./data \\
        --images-subdir photos --labels-subdir annotations

Input Directory Structure:
    input_dir/
    ├── images/       (or custom name via --images-subdir)
    │   ├── img1.jpg
    │   └── img2.png
    └── labels/       (or custom name via --labels-subdir)
        ├── img1.txt
        └── img2.txt

YOLO Label Format:
    Each line: <class_id> <x_center> <y_center> <width> <height>
    - class_id: integer starting from 0
    - All coordinates normalized to 0-1 range
        """,
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing images/ and labels/ subdirectories",
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
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="List of class names for statistics (e.g., car person dog)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix common issues (normalize coordinates > 1, remove invalid lines)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for validation report (default: print to stdout)",
    )

    return parser.parse_args()


def validate_label_file(label_path, num_classes=None, fix=False):
    """Validate a single YOLO label file.
    
    Args:
        label_path: Path to label file
        num_classes: Expected number of classes (optional)
        fix: Whether to auto-fix issues
        
    Returns:
        dict with validation results
    """
    errors = []
    warnings = []
    valid_lines = []
    fixed_lines = []
    class_ids = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Cannot read file: {e}"],
            "warnings": [],
            "objects": 0,
            "class_ids": [],
        }
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # Check format
        if len(parts) != 5:
            errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
            continue
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            errors.append(f"Line {line_num}: Invalid number format - {e}")
            continue
        
        # Validate class_id
        if class_id < 0:
            errors.append(f"Line {line_num}: Negative class_id ({class_id})")
            continue
        
        if num_classes is not None and class_id >= num_classes:
            warnings.append(f"Line {line_num}: class_id {class_id} >= num_classes {num_classes}")
        
        # Validate coordinates
        needs_fix = False
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            if fix and 0 < x_center <= 1000 and 0 < y_center <= 1000:
                # Assume pixel coordinates, normalize
                x_center = x_center / 1000
                y_center = y_center / 1000
                needs_fix = True
                warnings.append(f"Line {line_num}: Normalized x_center/y_center from pixel range")
            else:
                errors.append(f"Line {line_num}: x_center/y_center out of range [0,1]: ({x_center}, {y_center})")
                continue
        
        if width < 0 or width > 1 or height < 0 or height > 1:
            if fix and 0 < width <= 1000 and 0 < height <= 1000:
                width = width / 1000
                height = height / 1000
                needs_fix = True
                warnings.append(f"Line {line_num}: Normalized width/height from pixel range")
            else:
                errors.append(f"Line {line_num}: width/height out of range [0,1]: ({width}, {height})")
                continue
        
        # Check for very small boxes
        if width < 0.001 or height < 0.001:
            warnings.append(f"Line {line_num}: Very small box (width={width:.4f}, height={height:.4f})")
        
        valid_lines.append(line)
        class_ids.append(class_id)
        
        if needs_fix:
            fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            fixed_lines.append(line)
    
    # Write fixed file if needed
    if fix and fixed_lines != [l.strip() for l in lines if l.strip()]:
        with open(label_path, 'w') as f:
            f.write('\n'.join(fixed_lines) + '\n')
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "objects": len(valid_lines),
        "class_ids": class_ids,
    }


def verify_dataset(input_dir, images_subdir="images", labels_subdir="labels", 
                   class_names=None, fix=False):
    """Verify a YOLO-format dataset.
    
    Args:
        input_dir: Path to dataset directory
        images_subdir: Name of images subdirectory
        labels_subdir: Name of labels subdirectory
        class_names: Optional list of class names
        fix: Whether to auto-fix issues
        
    Returns:
        dict with verification results
    """
    input_path = Path(input_dir)
    images_path = input_path / images_subdir
    labels_path = input_path / labels_subdir
    
    results = {
        "input_dir": str(input_dir),
        "images_dir_exists": images_path.exists(),
        "labels_dir_exists": labels_path.exists(),
        "total_images": 0,
        "total_labels": 0,
        "images_without_labels": [],
        "labels_without_images": [],
        "valid_labels": 0,
        "invalid_labels": 0,
        "total_objects": 0,
        "class_distribution": defaultdict(int),
        "errors": [],
        "warnings": [],
        "label_details": [],
    }
    
    # Check directories exist
    if not images_path.exists():
        results["errors"].append(f"Images directory not found: {images_path}")
        return results
    
    if not labels_path.exists():
        results["errors"].append(f"Labels directory not found: {labels_path}")
        return results
    
    # Find all images
    image_files = {
        f.stem: f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }
    results["total_images"] = len(image_files)
    
    # Find all labels
    label_files = {
        f.stem: f for f in labels_path.iterdir()
        if f.is_file() and f.suffix == '.txt'
    }
    results["total_labels"] = len(label_files)
    
    # Check for mismatches
    image_stems = set(image_files.keys())
    label_stems = set(label_files.keys())
    
    results["images_without_labels"] = sorted(image_stems - label_stems)
    results["labels_without_images"] = sorted(label_stems - image_stems)
    
    # Validate each label file
    num_classes = len(class_names) if class_names else None
    
    for stem in sorted(label_stems):
        label_path = label_files[stem]
        validation = validate_label_file(label_path, num_classes, fix)
        
        detail = {
            "file": stem,
            "valid": validation["valid"],
            "objects": validation["objects"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }
        results["label_details"].append(detail)
        
        if validation["valid"]:
            results["valid_labels"] += 1
        else:
            results["invalid_labels"] += 1
            results["errors"].extend([f"{stem}: {e}" for e in validation["errors"]])
        
        results["warnings"].extend([f"{stem}: {w}" for w in validation["warnings"]])
        results["total_objects"] += validation["objects"]
        
        for class_id in validation["class_ids"]:
            results["class_distribution"][class_id] += 1
    
    return results


def print_report(results, class_names=None):
    """Print verification report.
    
    Args:
        results: Verification results dict
        class_names: Optional list of class names
    """
    print("=" * 60)
    print("YOLO LABEL VERIFICATION REPORT")
    print("=" * 60)
    print(f"\nInput directory: {results['input_dir']}")
    
    # Directory status
    print(f"\nDirectory Status:")
    print(f"  Images directory: {'✓ exists' if results['images_dir_exists'] else '✗ missing'}")
    print(f"  Labels directory: {'✓ exists' if results['labels_dir_exists'] else '✗ missing'}")
    
    if not results['images_dir_exists'] or not results['labels_dir_exists']:
        print("\nCannot proceed without both directories.")
        return
    
    # File counts
    print(f"\nFile Counts:")
    print(f"  Total images: {results['total_images']}")
    print(f"  Total labels: {results['total_labels']}")
    print(f"  Valid labels: {results['valid_labels']}")
    print(f"  Invalid labels: {results['invalid_labels']}")
    
    # Mismatches
    if results['images_without_labels']:
        print(f"\n⚠ Images without labels ({len(results['images_without_labels'])}):")
        for stem in results['images_without_labels'][:10]:
            print(f"    - {stem}")
        if len(results['images_without_labels']) > 10:
            print(f"    ... and {len(results['images_without_labels']) - 10} more")
    
    if results['labels_without_images']:
        print(f"\n⚠ Labels without images ({len(results['labels_without_images'])}):")
        for stem in results['labels_without_images'][:10]:
            print(f"    - {stem}")
        if len(results['labels_without_images']) > 10:
            print(f"    ... and {len(results['labels_without_images']) - 10} more")
    
    # Object statistics
    print(f"\nObject Statistics:")
    print(f"  Total objects: {results['total_objects']}")
    print(f"  Unique classes: {len(results['class_distribution'])}")
    
    # Class distribution
    if results['class_distribution']:
        print(f"\nClass Distribution:")
        max_class_id = max(results['class_distribution'].keys())
        for class_id in range(max_class_id + 1):
            count = results['class_distribution'].get(class_id, 0)
            name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            bar = "█" * min(50, count)
            print(f"  {class_id:3d} ({name:15s}): {count:5d} {bar}")
    
    # Errors
    if results['errors']:
        print(f"\n✗ Errors ({len(results['errors'])}):")
        for error in results['errors'][:20]:
            print(f"    - {error}")
        if len(results['errors']) > 20:
            print(f"    ... and {len(results['errors']) - 20} more errors")
    
    # Warnings
    if results['warnings']:
        print(f"\n⚠ Warnings ({len(results['warnings'])}):")
        for warning in results['warnings'][:10]:
            print(f"    - {warning}")
        if len(results['warnings']) > 10:
            print(f"    ... and {len(results['warnings']) - 10} more warnings")
    
    # Summary
    print(f"\n{'=' * 60}")
    if results['errors']:
        print("RESULT: ✗ VALIDATION FAILED - Fix errors before training")
    elif results['warnings'] or results['images_without_labels']:
        print("RESULT: ⚠ VALIDATION PASSED WITH WARNINGS")
    else:
        print("RESULT: ✓ VALIDATION PASSED")
    print("=" * 60)


def main():
    args = parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)
    
    print(f"Verifying YOLO labels in: {input_path}")
    print(f"  Images subdirectory: {args.images_subdir}")
    print(f"  Labels subdirectory: {args.labels_subdir}")
    if args.fix:
        print("  Auto-fix: enabled")
    
    # Run verification
    results = verify_dataset(
        input_dir=args.input_dir,
        images_subdir=args.images_subdir,
        labels_subdir=args.labels_subdir,
        class_names=args.class_names,
        fix=args.fix,
    )
    
    # Capture output
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    print_report(results, args.class_names)
    
    sys.stdout = old_stdout
    report = buffer.getvalue()
    
    # Print to console
    print(report)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    # Exit with error code if validation failed
    if results['errors']:
        sys.exit(1)


if __name__ == "__main__":
    main()
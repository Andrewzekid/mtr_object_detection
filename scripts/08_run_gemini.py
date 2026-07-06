#!/usr/bin/env python3
"""
Run Gemini inference via HKU API proxy.

This script sends a prompt (with optional image) to Gemini via the HKU API proxy
and returns structured output in the requested format.

USAGE:
    # Single image mode
    python scripts/08_run_gemini.py --prompt "Describe this image" --image ./sample.jpg
    python scripts/08_run_gemini.py --prompt "Detect all cars" --template object_detection --format json --vis-output ./result.jpg
    python scripts/08_run_gemini.py --prompt "Count objects" --template counting --format text

    # Batch mode (folder of images)
    python scripts/08_run_gemini.py --prompt "Detect all cars" --image-folder ./images/ --output ./output/gemini_results/
    python scripts/08_run_gemini.py --prompt "Describe scene" --image-folder ./photos/ --output ./results/ --vis-output ./vis/

    # With generation config
    python scripts/08_run_gemini.py --prompt "Detect objects" --image ./scene.jpg --temperature 0.2 --thinking-budget 1024

PREREQUISITES:
    - Access to HKU Gemini API proxy (no authentication required)

OUTPUT:
    - Raw model response
    - Parsed output in requested format (json/yaml/bbox/text)
    - Optional visualization image with bounding boxes drawn
    - For batch mode: individual JSON files per image + summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models_inference import run_gemini


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Gemini inference via HKU API proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prompt with image
    python scripts/08_run_gemini.py --prompt "Describe this image" --image ./sample.jpg

    # Object detection with JSON output
    python scripts/08_run_gemini.py --prompt "Detect all cars and people" \\
        --template object_detection --format json --image ./street.jpg

    # Image captioning
    python scripts/08_run_gemini.py --prompt "What is happening?" \\
        --template image_captioning --format text --image ./scene.jpg

    # Counting objects
    python scripts/08_run_gemini.py --prompt "Count all objects" \\
        --template counting --format json --image ./parking.jpg

    # Spatial reasoning
    python scripts/08_run_gemini.py --prompt "Describe object positions" \\
        --template spatial_reasoning --format text --image ./room.jpg

    # Text-only prompt (no image)
    python scripts/08_run_gemini.py --prompt "What is object detection?"

    # Use custom API endpoint
    python scripts/08_run_gemini.py --prompt "test" --api-url https://custom.api.com/gemini

    # With generation config
    python scripts/08_run_gemini.py --prompt "Detect objects" --image ./scene.jpg \\
        --temperature 0.2 --top-p 0.95 --top-k 40 --thinking-budget 1024

Batch Mode (folder of images):
    # Process all images in a folder
    python scripts/08_run_gemini.py --prompt "Detect all objects" \\
        --template object_detection --image-folder ./images/ \\
        --output ./output/results/ --vis-output ./output/vis/ --split-by-class

Templates:
    object_detection    - Detect objects with bounding boxes
    image_captioning    - Describe the image in detail
    scene_understanding - Analyze the scene and context
    counting            - Count objects by category
    spatial_reasoning   - Analyze spatial relationships

Output Formats:
    json    - Valid JSON format
    yaml    - Valid YAML format
    bbox    - Bounding box format: [class, x1, y1, x2, y2]
    text    - Plain text format
        """,
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt for the model",
    )
    # Image input group - mutually exclusive
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to input image for multimodal inference",
    )
    image_group.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Path to folder containing images for batch inference",
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default=None,
        choices=["object_detection", "image_captioning", "scene_understanding", 
                 "counting", "spatial_reasoning"],
        help="Predefined prompt template",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="json",
        choices=["json", "yaml", "bbox", "text"],
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemini-3.5-flash",
        help="Model deployment ID (default: gemini-3.5-flash)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.hku.hk/gemini/student",
        help="Gemini API base URL (default: https://api.hku.hk/gemini/student)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default=None,
        help="System instruction to set model role/style",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Controls randomness (0.0-1.0, higher = more creative)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Controls nucleus sampling (0.0-1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limits sampling pool to top K tokens",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Max tokens for model's internal reasoning",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--vis-output",
        type=str,
        default=None,
        help="Path to save visualization image with bounding boxes drawn",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1000.0,
        help="Coordinate scale factor: Gemini outputs normalized 0-1000 coordinates in x1,y1,x2,y2 format, which are scaled to pixel coordinates (default: 1000). Set to 1 if Gemini already outputs pixel coordinates.",
    )
    parser.add_argument(
        "--split-by-class",
        action="store_true",
        help="Split annotations by class into separate files (for use with SAM3 individual mode). Creates a folder per image with a JSON file per class.",
    )
    parser.add_argument(
        "--annotations-output",
        type=str,
        default=None,
        help="Output folder for split-by-class annotations (default: <output>/annotations/). Only used with --split-by-class.",
    )

    return parser.parse_args()


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def extract_bboxes_from_output(parsed_output):
    """Extract bounding boxes from parsed output.
    
    Handles various formats:
    - List of dicts with bbox_2d or bbox key
    - Dict with bbox_2d or bbox key
    - List of lists [x1, y1, x2, y2]
    """
    bboxes = []
    
    if isinstance(parsed_output, list):
        for item in parsed_output:
            if isinstance(item, dict):
                bbox = item.get("bbox_2d") or item.get("bbox")
                if bbox:
                    bboxes.append({"bbox": bbox, "label": item.get("label", "object")})
            elif isinstance(item, list) and len(item) >= 4:
                bboxes.append({"bbox": item[:4], "label": "object"})
    elif isinstance(parsed_output, dict):
        bbox = parsed_output.get("bbox_2d") or parsed_output.get("bbox")
        if bbox:
            bboxes.append({"bbox": bbox, "label": parsed_output.get("label", "object")})
    
    return bboxes


def generate_visualization(image_path, vis_path, parsed_output, coord_scale):
    """Generate visualization image with bounding boxes drawn.
    
    Args:
        image_path: Path to input image
        vis_path: Path to save visualization image
        parsed_output: Parsed output from Gemini containing bboxes
        coord_scale: Coordinate scale factor (1000 for normalized, 1 for pixel)
    
    Returns:
        True if visualization was generated successfully, False otherwise
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read image for visualization: {image_path}")
        return False
    
    img_h, img_w = img.shape[:2]
    
    # Extract bboxes from parsed output
    bboxes = extract_bboxes_from_output(parsed_output)
    
    # Draw bboxes on image
    for bbox_info in bboxes:
        bbox = bbox_info.get("bbox_2d") or bbox_info.get("bbox")
        label = bbox_info.get("label", "object")
        
        if bbox and len(bbox) == 4:
            # Scale coordinates from normalized range to pixel coordinates
            # Gemini outputs x1, y1, x2, y2 format
            x1 = int(bbox[0] / coord_scale * img_w)
            y1 = int(bbox[1] / coord_scale * img_h)
            x2 = int(bbox[2] / coord_scale * img_w)
            y2 = int(bbox[3] / coord_scale * img_h)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save visualization
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), img)
    return True


def process_single_image(args, image_path, output_dir=None, vis_dir=None):
    """Process a single image with Gemini.
    
    Args:
        args: Parsed command line arguments
        image_path: Path to input image
        output_dir: Directory to save JSON output (for batch mode)
        vis_dir: Directory to save visualization (for batch mode)
    
    Returns:
        dict with processing results
    """
    print(f"\nProcessing: {image_path.name}")
    print(f"  Image: {image_path}")
    
    # Run Gemini
    result = run_gemini(
        prompt=args.prompt,
        template_id=args.template,
        output_format=args.format,
        image_path=str(image_path),
        api_base_url=args.api_url,
        deployment_id=args.model,
        timeout=args.timeout,
        system_instruction=args.system_instruction,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        thinking_budget=args.thinking_budget,
        log_callback=lambda msg: print(f"  {msg}"),
    )
    
    if not result.get("success"):
        print(f"  Error: {result.get('error', 'Unknown error')}")
        return {
            "image": str(image_path),
            "success": False,
            "error": result.get("error", "Unknown error"),
        }
    
    # Get outputs
    raw_response = result.get("response", "")
    parsed_output = result.get("parsed_output")
    
    # Prepare output
    output_data = {
        "image": str(image_path),
        "model": result.get("model"),
        "format": result.get("format"),
        "raw_response": raw_response,
        "parsed_output": parsed_output,
    }
    
    # Save JSON output
    if output_dir:
        json_path = output_dir / f"{image_path.stem}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to: {json_path}")
    
    # Generate visualization if requested
    if vis_dir and parsed_output:
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"
        if generate_visualization(image_path, vis_path, parsed_output, args.coord_scale):
            print(f"  Visualization saved to: {vis_path}")
    
    return {
        "image": str(image_path),
        "success": True,
        "parsed_output": parsed_output,
    }


def split_annotations_by_class(parsed_output, image_path, annotations_dir, coord_scale):
    """Split annotations by class into separate files.
    
    Creates a folder per image (named after image stem) with a JSON file per class.
    Each class JSON file contains the class name and all bounding boxes for that class.
    
    Args:
        parsed_output: Parsed output from Gemini containing bboxes (list of dicts with bbox_2d and label)
        image_path: Path to the input image
        annotations_dir: Base directory for annotations output
        coord_scale: Coordinate scale factor (1000 for normalized, 1 for pixel)
    
    Returns:
        dict mapping class names to their annotation file paths
    """
    # Create image-specific folder
    image_folder = annotations_dir / image_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)
    
    # Get image dimensions for coordinate scaling
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read image for scaling: {image_path}")
        return {}
    img_h, img_w = img.shape[:2]
    
    # Group bboxes by class
    class_bboxes = {}
    
    if isinstance(parsed_output, list):
        for item in parsed_output:
            if isinstance(item, dict):
                bbox = item.get("bbox_2d") or item.get("bbox")
                label = item.get("label", "object")
                if bbox and len(bbox) == 4:
                    # Gemini outputs x1, y1, x2, y2 format (normalized 0-1000)
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox

                    # Scale to pixel coordinates
                    x1 = int(x1_norm / coord_scale * img_w)
                    y1 = int(y1_norm / coord_scale * img_h)
                    x2 = int(x2_norm / coord_scale * img_w)
                    y2 = int(y2_norm / coord_scale * img_h)

                    # Store in x1, y1, x2, y2 format in pixel coordinates
                    if label not in class_bboxes:
                        class_bboxes[label] = []
                    class_bboxes[label].append([x1, y1, x2, y2])
    
    # Save each class's annotations to a separate file
    saved_files = {}
    for class_name, bboxes in class_bboxes.items():
        # Sanitize class name for filename (replace spaces/special chars with underscores)
        safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
        annotation_file = image_folder / f"{safe_class_name}.json"
        
        annotation_data = {
            "class_name": class_name,
            "image": str(image_path),
            "bboxes": bboxes,
        }
        
        with open(annotation_file, "w") as f:
            json.dump(annotation_data, f, indent=2)
        
        saved_files[class_name] = str(annotation_file)
        print(f"    Saved {len(bboxes)} bbox(es) for class '{class_name}' to: {annotation_file.name}")
    
    return saved_files


def process_image_folder(args):
    """Process all images in a folder with Gemini.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict with batch processing summary
    """
    folder_path = Path(args.image_folder)
    
    # Find all image files
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    
    if not image_files:
        print(f"Error: No image files found in {folder_path}")
        print(f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s) in {folder_path}")
    
    # Setup output directories
    output_dir = Path(args.output) if args.output else folder_path / "gemini_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Always create visualization directory in batch mode
    # If --vis-output is not provided, default to <output_dir>/visualizations/
    if args.vis_output:
        vis_dir = Path(args.vis_output)
    else:
        vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup annotations output directory for split-by-class mode
    annotations_dir = None
    if args.split_by_class:
        if args.annotations_output:
            annotations_dir = Path(args.annotations_output)
        else:
            annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotations output directory (split-by-class): {annotations_dir}")
    
    print(f"Output directory: {output_dir}")
    if vis_dir:
        print(f"Visualization directory: {vis_dir}")
    
    # Process each image
    results = []
    successful = 0
    failed = 0
    total_classes_found = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        result = process_single_image(args, image_path, output_dir, vis_dir)
        
        # If split-by-class mode, process the annotations
        if args.split_by_class and result["success"] and result.get("parsed_output"):
            print(f"  Splitting annotations by class...")
            saved_files = split_annotations_by_class(
                result["parsed_output"], 
                image_path, 
                annotations_dir,
                args.coord_scale
            )
            result["annotation_files"] = saved_files
            total_classes_found += len(saved_files)
        
        results.append(result)
        
        if result["success"]:
            successful += 1
        else:
            failed += 1
    
    # Generate summary
    summary = {
        "folder": str(folder_path),
        "total_images": len(image_files),
        "successful": successful,
        "failed": failed,
        "model": args.model,
        "template": args.template,
        "format": args.format,
        "prompt": args.prompt,
        "split_by_class": args.split_by_class,
        "total_classes_found": total_classes_found,
        "results": results,
    }
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Batch processing complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if args.split_by_class:
        print(f"  Total classes found: {total_classes_found}")
        print(f"  Annotations saved to: {annotations_dir}")
    print(f"  Summary saved to: {summary_path}")
    
    return summary


def main():
    args = parse_args()
    
    # Validate prompt
    if not args.prompt:
        print("Error: --prompt is required")
        sys.exit(1)
    
    # Batch mode: process folder of images
    if args.image_folder:
        folder_path = Path(args.image_folder)
        if not folder_path.exists():
            print(f"Error: Image folder not found: {folder_path}")
            sys.exit(1)
        if not folder_path.is_dir():
            print(f"Error: Path is not a directory: {folder_path}")
            sys.exit(1)
        
        print(f"Running Gemini batch inference...")
        print(f"  Model: {args.model}")
        print(f"  Template: {args.template or 'none'}")
        print(f"  Output format: {args.format}")
        print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
        print(f"  Image folder: {args.image_folder}")
        
        process_image_folder(args)
        print("\nGemini batch inference completed successfully!")
        return
    
    # Single image mode
    # Validate image if provided
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
    
    print(f"Running Gemini inference...")
    print(f"  Model: {args.model}")
    print(f"  Template: {args.template or 'none'}")
    print(f"  Output format: {args.format}")
    print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    if args.image:
        print(f"  Image: {args.image}")
    
    # Run Gemini
    result = run_gemini(
        prompt=args.prompt,
        template_id=args.template,
        output_format=args.format,
        image_path=args.image,
        api_base_url=args.api_url,
        deployment_id=args.model,
        timeout=args.timeout,
        system_instruction=args.system_instruction,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        thinking_budget=args.thinking_budget,
        log_callback=lambda msg: print(f"  {msg}"),
    )
    
    if not result.get("success"):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Get outputs
    raw_response = result.get("response", "")
    parsed_output = result.get("parsed_output")
    
    # Prepare output
    output_data = {
        "model": result.get("model"),
        "format": result.get("format"),
        "raw_response": raw_response,
        "parsed_output": parsed_output,
    }
    
    # Output results
    output_str = json.dumps(output_data, indent=2)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_str)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n" + "=" * 60)
        print("RAW RESPONSE:")
        print("=" * 60)
        print(raw_response)
        print("\n" + "=" * 60)
        print("PARSED OUTPUT:")
        print("=" * 60)
        if isinstance(parsed_output, (dict, list)):
            print(json.dumps(parsed_output, indent=2))
        else:
            print(parsed_output)
    
    # Generate visualization if requested
    if args.vis_output and args.image and parsed_output:
        vis_path = Path(args.vis_output)
        # If vis_path is a directory or has no valid image extension, generate a filename
        if vis_path.is_dir() or vis_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
            vis_path = vis_path / f"{Path(args.image).stem}_vis.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        
        if generate_visualization(Path(args.image), vis_path, parsed_output, args.coord_scale):
            print(f"\nVisualization saved to: {vis_path}")
    
    print("\nGemini inference completed successfully!")


if __name__ == "__main__":
    main()
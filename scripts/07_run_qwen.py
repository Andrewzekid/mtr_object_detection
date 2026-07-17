#!/usr/bin/env python3
"""
Run Qwen3.6 inference via Ollama API.

This script sends a prompt (with optional image) to Qwen3.6 via Ollama
and returns structured output in the requested format.

USAGE:
    # Single image mode
    python scripts/07_run_qwen.py --prompt "Describe this image" --image ./sample.jpg
    python scripts/07_run_qwen.py --prompt "Detect all cars" --template object_detection --format json --vis-output ./result.jpg
    python scripts/07_run_qwen.py --prompt "Count objects" --template counting --format text

    # Batch mode (folder of images)
    python scripts/07_run_qwen.py --prompt "Detect all cars" --image-folder ./images/ --output ./output/qwen_results/
    python scripts/07_run_qwen.py --prompt "Describe scene" --image-folder ./photos/ --output ./results/ --vis-output ./vis/

    EXAMPLE:
     python scripts/07_run_qwen.py --prompt "Detect all objects: Ceiling light, Sign, Advertisement Board, Ticket Gate, Map. 
 Signs are hanging lcd screens from the ceiling which show directions. They can contain arrows, characters like A B C, and X's. Do not categorize the X's on ticket gates as a sign. Do not classify posters as signs. Only hanging monitors can be classified as signs.
 Ceiling light are a flat and horizontal rectangular strip, do not detect reflections of lights in the glass or wall. If there are lights in the green wall, they are likely to be reflections. Consider carefully whether or not they are actually reflections. Detect individual ceiling lights and do not cluster them together. 
 Advertisements are flat lcd screens on the green wall that only display commercial content, not directions. Also do not mistake them for posters.  
 Maps are posters which show the directions in the MTR and which way to go.
 Ticket gates are turnstiles. Do not classify ticket vending machines as ticket gates.
 " --template object_detection --image-folder ./qwen_test --annotations-output ./output/annotations/ --vis-output ./output/vis_qwen_batch/ --split-by-class

PREREQUISITES:
    - Ollama installed and running: ollama serve
    - Qwen3.6 model pulled: ollama pull qwen3.6

OUTPUT:
    - Raw model response
    - Parsed output in requested format (json/yaml/bbox/text)
      NOTE: bounding boxes are scaled to PIXEL coordinates (Qwen's raw 0-1000
      normalized coords are converted using --coord-scale and the image's real
      W/H) before being saved, so the JSON can be fed directly to SAM3.
    - Optional visualization image with bounding boxes drawn
    - For batch mode: individual JSON files per image + summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / "core" / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, will use os.environ directly

from core.models_inference import run_qwen, run_qwen_api, list_ollama_models


def get_api_key(args):
    """Resolve the API key from args or environment variables.
    
    Priority:
    1. --api-key-env: Use the specified environment variable (e.g., PREMIUM_KEY)
    2. --api-key: Use the directly provided API key
    3. API_KEY: Use the default API_KEY environment variable
    """
    if args.api_key_env:
        key = os.getenv(args.api_key_env)
        if key:
            return key
        else:
            print(f"Warning: Environment variable '{args.api_key_env}' not found or empty")
    
    if args.api_key:
        return args.api_key
    
    return os.getenv("API_KEY")


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# Default class descriptions for MTR station object detection.
# Keys match the conditioning image filenames (underscores replaced with spaces).
MTR_CLASS_DESCRIPTIONS = {
    "ceiling light": "a flat and horizontal rectangular strip light mounted on the ceiling. Do NOT detect reflections of lights in glass or walls — if there are lights visible in the green wall, they are likely reflections. Consider carefully whether each detection is an actual light or a reflection. Detect individual ceiling lights separately; do not cluster them together.",
    "light": "a flat and horizontal rectangular strip light mounted on the ceiling. Do NOT detect reflections of lights in glass or walls — if there are lights visible in the green wall, they are likely reflections. Consider carefully whether each detection is an actual light or a reflection. Detect individual ceiling lights separately; do not cluster them together.",
    "exit sign": "a hanging monitor/display showing the lime-green character 出 along with letters and arrows indicating directions. It is a hanging LCD screen, not a wall poster.",
    "advertisement board": "a flat LCD screen mounted on the green wall that displays commercial/advertising content only, NOT directions. These are NOT TVs and NOT maps.",
    "ticket gate": "turnstiles/gates that passengers pass through after tapping their ticket/card. Do NOT classify ticket vending machines as ticket gates.",
    "map": "a poster/sign that shows MTR station directions, routes, and which way to go. It is a static poster, not an electronic display.",
    "tv": "an LCD screen hanging from the ceiling that does NOT contain directions or symbols. It displays general content (news, ads, etc.), not wayfinding information.",
}


def load_conditioning_images(folder_path):
    """Load conditioning images from a folder.
    
    Each image file in the folder becomes a conditioning image.
    The class name is derived from the filename (without extension),
    with underscores replaced by spaces.
    
    Args:
        folder_path: Path to folder containing reference images
    
    Returns:
        List of dicts with 'label' (class name) and 'path' (image path)
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Conditioning images folder not found: {folder}")
        return []
    
    conditioning = []
    for img_file in sorted(folder.iterdir()):
        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
            # Derive class name from filename (without extension)
            class_name = img_file.stem.replace("_", " ")
            conditioning.append({
                "label": class_name,
                "path": str(img_file),
            })
    
    return conditioning


def build_conditioning_prompt(conditioning_images, original_prompt, class_descriptions=None):
    """Build an enhanced prompt that references conditioning images.
    
    Args:
        conditioning_images: List of dicts with 'label' and 'path'
        original_prompt: The user's original prompt
        class_descriptions: Optional dict mapping class names to detailed descriptions.
                           If None, uses MTR_CLASS_DESCRIPTIONS as default.
    
    Returns:
        Enhanced prompt string
    """
    if not conditioning_images:
        return original_prompt
    
    if class_descriptions is None:
        class_descriptions = MTR_CLASS_DESCRIPTIONS
    
    class_names = [img["label"] for img in conditioning_images]
    num_refs = len(conditioning_images)
    main_image_num = num_refs + 1
    
    # Build reference description with per-class details
    ref_lines = []
    for i, img in enumerate(conditioning_images, 1):
        label = img["label"]
        desc = class_descriptions.get(label, "")
        if desc:
            ref_lines.append(f"- Image {i}: Reference for \"{label}\" — {desc}")
        else:
            ref_lines.append(f"- Image {i}: Reference for \"{label}\"")
    
    ref_text = "\n".join(ref_lines)
    
    # Build the enhanced prompt
    enhanced = (
        f"You will be given {main_image_num} images. "
        f"The first {num_refs} are reference examples (one per class), "
        f"and image {main_image_num} is the main scene to analyze.\n\n"
        f"REFERENCE IMAGES:\n"
        f"{ref_text}\n\n"
        f"MAIN IMAGE:\n"
        f"- Image {main_image_num}: The scene to analyze.\n\n"
        f"Detect all instances of: {', '.join(class_names)}.\n"
        f"Use the reference images to understand what each class looks like "
        f"and compare objects in the main image against these references.\n\n"
        f"{original_prompt}"
    )
    
    return enhanced


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3.6 inference via Ollama API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prompt with image
    python scripts/07_run_qwen.py --prompt "Describe this image" --image ./sample.jpg

    # Object detection with JSON output
    python scripts/07_run_qwen.py --prompt "Detect all cars and people" \\
        --template object_detection --format json --image ./street.jpg

    # Image captioning
    python scripts/07_run_qwen.py --prompt "What is happening?" \\
        --template image_captioning --format text --image ./scene.jpg

    # Counting objects
    python scripts/07_run_qwen.py --prompt "Count all objects" \\
        --template counting --format json --image ./parking.jpg

    # Spatial reasoning
    python scripts/07_run_qwen.py --prompt "Describe object positions" \\
        --template spatial_reasoning --format text --image ./room.jpg

    # Text-only prompt (no image)
    python scripts/07_run_qwen.py --prompt "What is object detection?"

    # List available Ollama models
    python scripts/07_run_qwen.py --list-models

    # Use custom Ollama server
    python scripts/07_run_qwen.py --prompt "test" --ollama-url http://192.168.1.100:11434

    #Multiclass
    python scripts/07_run_qwen.py     --prompt "Detect all: Ceiling light, Sign, Advertisement Board, Ticket Gate. Monitors are hanging display boards which only show a blank screen. Signs show directions and are also hanging lcd s
creens. Ceiling light are a flat and horizontal rectangular strip, do not detect reflections in the glass or wall. Detect individual ceiling light
s and do not cluster them together. Advertisement boards are flat lcd screens on the green wall which display colorful advertisements. They are not posters or wall drawings. "     --template object_detection     --format json  --output ./qwenoutputmulticlass.json  --image test_multiclass.jpg
   --vis-output ./qwen_vis_multiclass.jpg 

Batch Mode (folder of images):
    #Ceiling light
    python scripts/07_run_qwen.py --prompt "Detect all objects: Ceiling light. 
    Ceiling light are a flat and horizontal rectangular strip, do not detect reflections of lights in the glass or wall. If there are lights in the green wall, they are likely to be reflections. Consider carefully whether or not they are actually reflections. Detect individual ceiling lights and do not cluster them together. " --template object_detection --image ./qwen_test/image4.jpg --output lights.json --vis-output ./output/vis_lights.jpg
    
    # Process all images in a folder
     python scripts/07_run_qwen.py --prompt "Detect all objects: Ceiling light, Sign, Advertisement Board, Ticket Gate, Map. 
 Signs are hanging lcd screens from the ceiling which show directions. They can contain arrows, characters like A B C, and X's. Do not categorize the X's on ticket gates as a sign.
 Ceiling light are a flat and horizontal rectangular strip, do not detect reflections of lights in the glass or wall. If there are lights in the green wall, they are likely to be reflections. Consider carefully whether or not they are actually reflections. Detect individual ceiling lights and do not cluster them together. 
 Advertisement boards are flat lcd screens on the green wall. They are not posters or wall drawings. They only display commercial concent and do not display directions.  
 Maps are posters which show the directions in the MTR and which way to go." --template object_detection --image-folder ./qwen_test --annotations-output ./output/annotations/ --vis-output ./output/vis_qwen_batch/ --split-by-class
    # Batch with visualizations
    python scripts/07_run_qwen.py --prompt "Detect " \\
        --template object_detection --image-folder ./photos/ \\
        --output ./output/results/ --vis-output ./output/vis/ --split-by-class --annotations-output ./output/annotations/

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
        default="qwen3.6:27b",
        help="Model name in Ollama (default: qwen3.6:27b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
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
        help="Coordinate scale factor: Qwen outputs normalized 0-1000 coordinates in x1,y1,x2,y2 format, which are scaled to pixel coordinates and saved as PIXEL coordinates in the output JSON (default: 1000). Set to 1 if Qwen already outputs pixel coordinates.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
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
    
    # API mode arguments
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use Aliyun DashScope API (qwen-vl-plus) instead of local Ollama",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for Aliyun DashScope (if not provided, loads from API_KEY env variable)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable name to use for API key (e.g., PREMIUM_KEY). "
             "Overrides --api-key and API_KEY env variable.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Resume from this image number (1-indexed) in batch mode. "
             "Images before this number will be skipped.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        help="Base URL for Aliyun DashScope API (default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1)",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="qwen3.7-plus",
        help="Model name for API mode (default: qwen3.7-plus)",
    )

    # Conditioning images argument
    parser.add_argument(
        "--conditioning-images", "-c",
        type=str,
        default=None,
        help="Path to folder containing reference/conditioning images (one per class). "
             "Filename (without extension) becomes the class name. "
             "Underscores in filenames are converted to spaces. "
             "Example: --conditioning-images ./ref_images/ where ref_images/ contains "
             "ceiling_light.jpg, sign.jpg, etc.",
    )

    return parser.parse_args()


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


def read_image_size(image_path):
    """Return (width, height) of an image, or None if it cannot be read."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)


def scale_parsed_output_to_pixels(parsed_output, img_w, img_h, coord_scale):
    """Scale bounding boxes in parsed_output to integer pixel coordinates.

    Qwen emits ``bbox_2d``/``bbox`` values in the normalized 0-coord_scale range
    (0-1000 by default). This converts them in place (on a shallow copy) so the
    saved JSON contains pixel coordinates that can be fed directly to downstream
    consumers such as SAM3 — which expects pixel-space boxes.

    Args:
        parsed_output: Parsed Qwen output (list of dicts, a single dict, or other)
        img_w, img_h: Image dimensions in pixels
        coord_scale: Coordinate scale factor (1000 for Qwen's normalized range,
                     1 if Qwen already emitted pixel coordinates)

    Returns:
        parsed_output with bbox values replaced by [x1, y1, x2, y2] pixel ints.
        If coord_scale == 1 the input is returned unchanged.
    """
    if coord_scale == 1 or parsed_output is None:
        return parsed_output

    def scale_bbox(bbox):
        x1 = int(round(bbox[0] / coord_scale * img_w))
        y1 = int(round(bbox[1] / coord_scale * img_h))
        x2 = int(round(bbox[2] / coord_scale * img_w))
        y2 = int(round(bbox[3] / coord_scale * img_h))
        return [x1, y1, x2, y2]

    def fix_item(item):
        if not isinstance(item, dict):
            return item
        item = dict(item)
        key = "bbox_2d" if "bbox_2d" in item else ("bbox" if "bbox" in item else None)
        if key is not None:
            bbox = item[key]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                item[key] = scale_bbox(bbox)
        return item

    if isinstance(parsed_output, list):
        return [fix_item(item) for item in parsed_output]
    if isinstance(parsed_output, dict):
        return fix_item(parsed_output)
    return parsed_output


def generate_visualization(image_path, vis_path, parsed_output):
    """Generate visualization image with bounding boxes drawn.

    Args:
        image_path: Path to input image
        vis_path: Path to save visualization image
        parsed_output: Parsed output from Qwen containing bboxes in PIXEL
                       coordinates (already scaled upstream before saving JSON)

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
            # Bboxes are already in pixel coordinates (scaled upstream before
            # the JSON was saved). Qwen outputs x1, y1, x2, y2 format.
            x1 = max(0, min(int(bbox[0]), img_w))
            y1 = max(0, min(int(bbox[1]), img_h))
            x2 = max(0, min(int(bbox[2]), img_w))
            y2 = max(0, min(int(bbox[3]), img_h))

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


def process_single_image(args, image_path, output_dir=None, vis_dir=None, conditioning_images=None):
    """Process a single image with Qwen3.6.
    
    Args:
        args: Parsed command line arguments
        image_path: Path to input image
        output_dir: Directory to save JSON output (for batch mode)
        vis_dir: Directory to save visualization (for batch mode)
        conditioning_images: Optional list of conditioning images (dicts with 'label' and 'path')
    
    Returns:
        dict with processing results
    """
    print(f"\nProcessing: {image_path.name}")
    print(f"  Image: {image_path}")
    
    # Build the prompt (with conditioning info if provided)
    prompt = args.prompt
    if conditioning_images:
        prompt = build_conditioning_prompt(conditioning_images, args.prompt)
        print(f"  Conditioning images: {len(conditioning_images)} reference(s)")
        for ci in conditioning_images:
            print(f"    - {ci['label']}: {ci['path']}")
    
    # Choose between API mode and Ollama mode
    if args.use_api:
        # Resolve API key
        api_key = get_api_key(args)
        
        # Run Qwen via Aliyun DashScope API
        result = run_qwen_api(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=str(image_path),
            conditioning_images=conditioning_images,
            api_key=api_key,
            base_url=args.base_url,
            model_name=args.api_model,
            timeout=args.timeout,
            log_callback=lambda msg: print(f"  {msg}"),
        )
    else:
        # Run Qwen3.6 via Ollama
        result = run_qwen(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=str(image_path),
            conditioning_images=conditioning_images,
            ollama_base_url=args.ollama_url,
            model_name=args.model,
            timeout=args.timeout,
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

    # Scale Qwen's normalized 0-1000 bounding boxes to pixel coordinates before
    # saving the JSON and before any downstream use (visualization, split-by-class,
    # SAM3). The saved JSON therefore contains pixel-space boxes.
    size = read_image_size(image_path)
    if size is not None:
        img_w, img_h = size
        parsed_output = scale_parsed_output_to_pixels(
            parsed_output, img_w, img_h, args.coord_scale
        )
    else:
        print(f"  Warning: could not read image size for coordinate scaling: {image_path}")

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
        if generate_visualization(image_path, vis_path, parsed_output):
            print(f"  Visualization saved to: {vis_path}")
    
    return {
        "image": str(image_path),
        "success": True,
        "parsed_output": parsed_output,
    }


def split_annotations_by_class(parsed_output, image_path, annotations_dir, img_w, img_h):
    """Split annotations by class into separate files.

    Creates a folder per image (named after image stem) with a JSON file per class.
    Each class JSON file contains the class name and all bounding boxes for that class.

    Args:
        parsed_output: Parsed output from Qwen containing bboxes in PIXEL
                       coordinates (already scaled upstream before saving JSON)
        image_path: Path to the input image
        annotations_dir: Base directory for annotations output
        img_w, img_h: Image dimensions in pixels (used to clamp boxes to bounds)

    Returns:
        dict mapping class names to their annotation file paths
    """
    # Create image-specific folder
    image_folder = annotations_dir / image_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)

    # Group bboxes by class
    class_bboxes = {}

    if isinstance(parsed_output, list):
        for item in parsed_output:
            if isinstance(item, dict):
                bbox = item.get("bbox_2d") or item.get("bbox")
                label = item.get("label", "object")
                if bbox and len(bbox) == 4:
                    # Bboxes are already in pixel coordinates (scaled upstream).
                    # Clamp to image bounds and store as x1, y1, x2, y2.
                    x1 = max(0, min(int(bbox[0]), img_w))
                    y1 = max(0, min(int(bbox[1]), img_h))
                    x2 = max(0, min(int(bbox[2]), img_w))
                    y2 = max(0, min(int(bbox[3]), img_h))

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
    """Process all images in a folder with Qwen3.6.
    
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
    
    total_images = len(image_files)
    print(f"Found {total_images} image(s) in {folder_path}")
    
    # Handle resume-from: filter images to process
    start_index = 0
    if args.resume_from is not None:
        # resume_from is 1-indexed, convert to 0-indexed
        start_index = max(0, args.resume_from - 1)
        if start_index > 0:
            print(f"Resuming from image {args.resume_from} (skipping first {start_index} images)")
            image_files = image_files[start_index:]
            print(f"Images to process: {len(image_files)}")
    
    # Setup output directories
    output_dir = Path(args.output) if args.output else folder_path / "qwen_results"
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
    
    # Load conditioning images if provided
    conditioning_images = None
    if args.conditioning_images:
        conditioning_images = load_conditioning_images(args.conditioning_images)
        if conditioning_images:
            print(f"Conditioning images: {len(conditioning_images)} reference(s)")
            for ci in conditioning_images:
                print(f"  - {ci['label']}: {ci['path']}")
        else:
            print(f"Warning: No valid conditioning images found in {args.conditioning_images}")
    
    # Process each image
    results = []
    successful = 0
    failed = 0
    total_classes_found = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        result = process_single_image(args, image_path, output_dir, vis_dir, conditioning_images)
        
        # If split-by-class mode, process the annotations
        if args.split_by_class and result["success"] and result.get("parsed_output"):
            print(f"  Splitting annotations by class...")
            # parsed_output is already in pixel coordinates (scaled in
            # process_single_image); pass image dims for clamping.
            size = read_image_size(image_path)
            if size is None:
                print(f"  Warning: could not read image size, skipping split: {image_path}")
                result["annotation_files"] = {}
            else:
                img_w, img_h = size
                saved_files = split_annotations_by_class(
                    result["parsed_output"],
                    image_path,
                    annotations_dir,
                    img_w,
                    img_h,
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
    
    # List models mode
    if args.list_models:
        print("Fetching available Ollama models...")
        result = list_ollama_models(args.ollama_url)
        if result.get("success"):
            models = result.get("models", [])
            if models:
                print(f"\nAvailable models ({len(models)}):")
                for model in models:
                    print(f"  - {model}")
            else:
                print("\nNo models found in Ollama.")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
            print(f"Is Ollama running at {args.ollama_url}?")
        return
    
    # Validate prompt
    if not args.prompt:
        print("Error: --prompt is required (unless using --list-models)")
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
        
        if args.use_api:
            print(f"Running Qwen API batch inference...")
            print(f"  Model: {args.api_model}")
            print(f"  Mode: Aliyun DashScope API")
        else:
            print(f"Running Qwen3.6 batch inference...")
            print(f"  Model: {args.model}")
            print(f"  Mode: Ollama (local)")
        print(f"  Template: {args.template or 'none'}")
        print(f"  Output format: {args.format}")
        print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
        print(f"  Image folder: {args.image_folder}")
        
        process_image_folder(args)
        print("\nQwen3.6 batch inference completed successfully!")
        return
    
    # Single image mode
    # Validate image if provided
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
    
    # Determine mode and model info
    if args.use_api:
        print(f"Running Qwen API inference...")
        print(f"  Model: {args.api_model}")
        print(f"  Mode: Aliyun DashScope API")
    else:
        print(f"Running Qwen3.6 inference...")
        print(f"  Model: {args.model}")
        print(f"  Mode: Ollama (local)")
    
    print(f"  Template: {args.template or 'none'}")
    print(f"  Output format: {args.format}")
    print(f"  Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    if args.image:
        print(f"  Image: {args.image}")
    
    # Load conditioning images if provided
    conditioning_images = None
    if args.conditioning_images:
        conditioning_images = load_conditioning_images(args.conditioning_images)
        if conditioning_images:
            print(f"  Conditioning images: {len(conditioning_images)} reference(s)")
            for ci in conditioning_images:
                print(f"    - {ci['label']}: {ci['path']}")
        else:
            print(f"  Warning: No valid conditioning images found in {args.conditioning_images}")
    
    # Build the prompt (with conditioning info if provided)
    prompt = args.prompt
    if conditioning_images:
        prompt = build_conditioning_prompt(conditioning_images, args.prompt)
    
    # Run inference based on mode
    if args.use_api:
        # Run Qwen via Aliyun DashScope API
        result = run_qwen_api(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=args.image,
            conditioning_images=conditioning_images,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.api_model,
            timeout=args.timeout,
            log_callback=lambda msg: print(f"  {msg}"),
        )
    else:
        # Run Qwen3.6 via Ollama
        result = run_qwen(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=args.image,
            conditioning_images=conditioning_images,
            ollama_base_url=args.ollama_url,
            model_name=args.model,
            timeout=args.timeout,
            log_callback=lambda msg: print(f"  {msg}"),
        )
    
    if not result.get("success"):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Get outputs
    raw_response = result.get("response", "")
    parsed_output = result.get("parsed_output")

    # Scale Qwen's normalized 0-1000 bounding boxes to pixel coordinates before
    # saving the JSON and before any downstream use (visualization, SAM3). The
    # saved JSON therefore contains pixel-space boxes.
    if args.image:
        size = read_image_size(Path(args.image))
        if size is not None:
            img_w, img_h = size
            parsed_output = scale_parsed_output_to_pixels(
                parsed_output, img_w, img_h, args.coord_scale
            )
        else:
            print(f"  Warning: could not read image size for coordinate scaling: {args.image}")

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
        
        if generate_visualization(Path(args.image), vis_path, parsed_output):
            print(f"\nVisualization saved to: {vis_path}")
    
    print("\nQwen3.6 inference completed successfully!")


if __name__ == "__main__":
    main()
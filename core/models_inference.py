#    core/models_inference.py - SAM3 (local) + Qwen3.6 (via Ollama) + Gemini wrappers.
#
#    USAGE (SAM3 - local):
#        from core.models_inference import run_sam3
#        r = run_sam3(
#            image_path="./image.jpg",
#            bbox=[100, 100, 300, 300],      # x1, y1, x2, y2 (optional)
#            text_prompt="the red car",        # optional
#            model_path="./core/sam3/models/sam_vit_h_4b8939.pth",
#            device="cuda",                    # or "cpu"
#            conf_threshold=0.5,
#        )
#        # r["mask"]      - boolean numpy array of the segmentation
#        # r["mask_overlay"] - BGR image with green mask overlay
#        # r["segmented_regions"] - list of {contour, bbox, area, center}
#
#    USAGE (Qwen3.6 - via Ollama):
#        from core.models_inference import run_qwen
#        r = run_qwen(
#            prompt="Detect the cars in this scene",
#            template_id="object_detection",   # or None for custom
#            output_format="json",            # json | yaml | bbox | text
#            image_path="./image.jpg",        # optional multimodal image
#            ollama_base_url="http://localhost:11434",
#            model_name="qwen3.6",
#            timeout=120,
#        )
#        # r["response"]       - raw model text
#        # r["parsed_output"]  - structured JSON/YAML/bbox list
#
#    USAGE (Gemini - via HKU API proxy):
#        from core.models_inference import run_gemini
#        r = run_gemini(
#            prompt="Detect the cars in this scene",
#            template_id="object_detection",   # or None for custom
#            output_format="json",            # json | yaml | bbox | text
#            image_path="./image.jpg",        # required for object detection
#            api_base_url="https://api.hku.hk/gemini/student",
#            deployment_id="gemini-3.5-flash",
#            timeout=120,
#        )
#        # r["response"]       - raw model text
#        # r["parsed_output"]  - structured JSON/YAML/bbox list
#
#    USAGE (list installed Ollama models):
#        from core.models_inference import list_ollama_models
#        print(list_ollama_models())
#
#    PREREQUISITES:
#        SAM3 weights go in ./core/sam3/models/ (sam_vit_h_4b8939.pth or similar)
#        Qwen3.6 needs: ollama serve    ollama pull qwen3.6
#        Gemini needs: access to HKU Gemini API proxy (no authentication required)
#
#    REQUIREMENTS:
#        pip install opencv-python-headless numpy requests torch torchvision
#        pip install segment-anything   # optional, only needed for SAM3

"""
Model inference wrappers for SAM3 (local), Qwen3.6 (via Ollama), and Gemini (via HKU API proxy).
All functions are headless and return structured results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import json
import requests
import base64
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from core/.env
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, will use os.environ directly


def run_sam3(
    image_path: str | Path,
    bboxes: Optional[List[List[float]]] = None,
    concepts: Optional[List[str]] = None,
    model_path: Optional[str | Path] = None,
    device: str = "cuda",
    conf: float = 0.25,
    quantize: Optional[int] = None,
    save: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Run SAM3 semantic segmentation on an image using the Ultralytics SAM API.

    Multiple bounding boxes can be supplied as exemplars of the same visual concept
    (e.g. multiple cars to define the car class). The predictor segments all
    similar objects in the image.

    Args:
        image_path   : Path to input image.
        bboxes       : List of bounding boxes in [x1, y1, x2, y2] format (pixel
                       coordinates). Pass a single bbox or multiple exemplars as
                       a list of lists.
        concepts     : Optional list of concept labels (e.g. ["car", "person"]).
                       When provided alongside bboxes, concepts are logged for
                       reference. SAM3 uses bboxes as exemplars; concepts can
                       be used for downstream filtering or annotation.
        model_path   : Path to SAM3 weights file (e.g. ./core/sam3/models/sam3-model/sam3.pt).
                       If None, defaults to './core/sam3/models/sam3-model/sam3.pt'.
        device       : 'cuda' or 'cpu' (string).
        conf         : Confidence threshold for segmentation mask.
        quantize     : INT8 / INT16 quantization (None to disable).
        save         : Whether Ultralytics should save annotated images.
        progress_callback / status_callback / log_callback / is_cancelled
                     : Optional worker-thread callbacks.

    Returns:
        Dictionary with:
          - success               : bool
          - masks                 : list of numpy bool/HxW arrays (one per region)
          - combined_mask         : single HxW bool array (union of all masks)
          - mask_overlay          : BGR image with green mask overlay
          - segmented_regions     : list of dicts {bbox, area, center, contour}
          - bboxes_used           : the bboxes actually passed to the predictor
    """
    # Backward-compat: allow callers passing a single flat bbox list [x1,y1,x2,y2]
    if bboxes is not None and isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]

    image_file = Path(image_path)
    if not image_file.exists():
        return {"success": False, "error": f"Image not found: {image_file}"}

    # Determine default model path
    if model_path is None:
        model_path = Path(__file__).parent / "sam3" / "models" / "sam3-model" / "sam3.pt"
    model_path = Path(model_path)

    if status_callback:
        status_callback("Loading Ultralytics SAM3...")
    if log_callback:
        log_callback(f"Running SAM3 on: {image_file}")
        log_callback(f"Model path: {model_path}")
        if bboxes:
            log_callback(f"Using {len(bboxes)} bbox exemplar(s): {bboxes}")
        if concepts:
            log_callback(f"Concepts: {concepts}")
        if not bboxes and not concepts:
            log_callback("No bboxes or concepts supplied — predictor will fall back to its defaults.")

    try:
        # Use the high-level Ultralytics SAM API (recommended for SAM3)
        from ultralytics import SAM

        # Read the source image so we can generate an overlay ourselves
        image = cv2.imread(str(image_file))
        if image is None:
            return {"success": False, "error": "Could not read image"}

        if progress_callback:
            progress_callback(10)

        if status_callback:
            status_callback("Loading SAM3 model...")

        # Build kwargs for model.predict
        predict_kwargs = {
            "source": str(image_file),
            "task": "segment",
            "verbose": False,
            "conf": conf,
            "device": device,
            "save": save,
        }
        if bboxes is not None:
            predict_kwargs["bboxes"] = bboxes
            # All bboxes belong to the same class (class 0) when provided as exemplars for a single concept
            # This ensures SAM3 treats all bboxes as the same class and returns them with consistent class_ids
            predict_kwargs["labels"] = [0] * len(bboxes)
        # Note: SAM3 does not support 'text' parameter - it uses bboxes as exemplars only
        if quantize is not None:
            predict_kwargs["quantize"] = quantize

        model = SAM(str(model_path))

        if progress_callback:
            progress_callback(40)

        if status_callback:
            status_callback("Segmenting...")

        print(f"Running SAM3 with concepts: {concepts} and bboxes: {bboxes}")
        results = model.predict(**predict_kwargs)
        print(f"SAM3 results: {results}")
        if progress_callback:
            progress_callback(70)

        # Handle both list and single result formats
        if not isinstance(results, list):
            results = [results]

        # Extract detections from results
        detections = []  # List of {bbox, label, confidence}
        masks_array = []

        for result in results:
            # Extract bounding boxes (primary detection output)
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()

                # Get class IDs to map to concepts
                if hasattr(result.boxes, 'cls'):
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                else:
                    class_ids = np.zeros(len(boxes), dtype=int)

                # Get confidence scores if available
                if hasattr(result.boxes, 'conf'):
                    confidences = result.boxes.conf.cpu().numpy()
                else:
                    confidences = np.ones(len(boxes))

                # When bboxes are provided as exemplars for a concept, all detections 
                # should use the primary concept name (first concept) to ensure consistent
                # coloring. SAM3 may return different class_ids for each bbox, but we
                # want all instances of the same class to have the same label.
                primary_concept = concepts[0] if concepts else "object"
                for j, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    # Use primary concept for all detections when bboxes are provided
                    concept = primary_concept if bboxes is not None else (concepts[class_id] if class_id < len(concepts) else f"class_{class_id}")
                    detections.append({
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "label": concept,
                        "confidence": float(conf),
                        "area": float((box[2] - box[0]) * (box[3] - box[1])),
                        "center": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
                    })

            # Extract masks if available
            if hasattr(result, 'masks') and result.masks is not None:
                mask_data = result.masks.data
                # Convert to numpy
                if hasattr(mask_data, "cpu"):
                    mask_data = mask_data.cpu().numpy()
                else:
                    mask_data = np.asarray(mask_data)
                # Each row is a binary mask
                for i in range(mask_data.shape[0]):
                    m = mask_data[i].astype(bool)
                    # Resize back to original image size if needed
                    if m.shape != image.shape[:2]:
                        m = cv2.resize(m.astype(np.uint8), (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    masks_array.append(m)

        if progress_callback:
            progress_callback(85)

        # Build segmented_regions from detections (primary source)
        segmented_regions = []
        for det in detections:
            segmented_regions.append({
                "bbox": det["bbox"],
                "label": det["label"],
                "confidence": det["confidence"],
                "area": det["area"],
                "center": det["center"],
            })

        # Combined mask + overlay
        if masks_array:
            combined_mask = np.zeros(image.shape[:2], dtype=bool)
            for m in masks_array:
                combined_mask |= m
            mask_overlay = create_mask_overlay(image, combined_mask.astype(np.uint8))
        elif segmented_regions:
            # No masks but we have detections - create overlay from bounding boxes
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_overlay = image.copy()
            # Draw bounding boxes on overlay (bboxes are in x1, y1, x2, y2 format)
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = det.get("label", "")
                cv2.putText(mask_overlay, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_overlay = image.copy()

        if log_callback:
            log_callback(f"SAM3 done: {len(detections)} detection(s), {len(masks_array)} mask(s).")
            if detections:
                labels_found = set(d["label"] for d in detections)
                log_callback(f"Labels found: {labels_found}")

        if progress_callback:
            progress_callback(100)

        return {
            "success": True,
            "detections": detections,
            "masks": masks_array,
            "combined_mask": combined_mask,
            "mask_overlay": mask_overlay,
            "segmented_regions": segmented_regions,
            "bboxes_used": bboxes or [],
            "concepts": concepts or [],
            "scores": [d["confidence"] for d in detections],
        }

    except ImportError:
        return {
            "success": False,
            "error": "ultralytics not installed. Install with: pip install ultralytics",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SAM3 inference failed: {str(e)}",
        }


def find_sam3_checkpoint(model_dir: Path) -> Optional[Path]:
    """Find SAM3 model checkpoint in directory."""
    if not model_dir.exists():
        return None
    
    # Common SAM3 checkpoint names
    checkpoint_names = [
        "sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth", 
        "sam_vit_b_01ec64.pth",
        "sam3_vit_h.pth",
        "sam3_vit_l.pth",
        "sam3_vit_b.pth",
    ]
    
    # Check for known checkpoint names
    for name in checkpoint_names:
        checkpoint = model_dir / name
        if checkpoint.exists():
            return checkpoint
    
    # Check for any .pth file
    pth_files = list(model_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]
    
    return None


def determine_sam3_model_type(checkpoint: Path) -> str:
    """Determine SAM3 model type from checkpoint filename."""
    name = checkpoint.name.lower()
    if "vit_h" in name or "huge" in name:
        return "vit_h"
    elif "vit_l" in name or "large" in name:
        return "vit_l"
    else:
        return "vit_b"


def create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of the mask on the original image."""
    overlay = image.copy()
    
    # Create color mask (green)
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [0, 255, 0]
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
    
    return overlay


def extract_segmented_regions(mask: np.ndarray, scores: np.ndarray) -> List[Dict]:
    """Extract segmented regions with contours from mask."""
    regions = []
    
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small regions
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                "contour": contour.tolist(),
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "center": [x + w // 2, y + h // 2],
            })
    
    return regions


def run_qwen(
    prompt: str,
    template_id: Optional[str] = None,
    output_format: str = "json",
    image_path: Optional[str | Path | List[str | Path]] = None,
    conditioning_images: Optional[List[Dict[str, str]]] = None,
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "qwen3.6",
    timeout: int = 120,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Run Qwen3.6 inference via Ollama API.
    
    Args:
        prompt: Text prompt for the model
        template_id: Optional template identifier for predefined prompts
        output_format: Desired output format ('json', 'yaml', 'bbox', 'text')
        image_path: Optional image path for multimodal input. Can be a single path
                    or a list of paths (for multiple images).
        conditioning_images: Optional list of conditioning/reference images for few-shot
                             visual prompting. Each item is a dict with 'label' (class name)
                             and 'path' (image path). These are prepended before the main image.
        ollama_base_url: Ollama API base URL
        model_name: Model name in Ollama
        timeout: Request timeout in seconds
        progress_callback: Callback for progress updates
        status_callback: Callback for status messages
        log_callback: Callback for log messages
        is_cancelled: Callback to check if operation should be cancelled
    
    Returns:
        Dictionary with inference results:
            - success: Boolean
            - response: Raw model response
            - parsed_output: Parsed output in requested format
            - format: Output format used
    """
    if status_callback:
        status_callback("Preparing Qwen3.6 request...")
    
    if log_callback:
        log_callback(f"Prompt: {prompt[:100]}...")
        log_callback(f"Model: {model_name}")
        log_callback(f"Output format: {output_format}")
    
    # Apply template if provided
    if template_id:
        prompt = apply_prompt_template(prompt, template_id)
    
    # Add format instruction to prompt
    format_instruction = get_format_instruction(output_format)
    full_prompt = f"{prompt}\n\n{format_instruction}"
    
    try:
        if progress_callback:
            progress_callback(10)
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
        }
        print(f"Sending request to Ollama at {ollama_base_url} with model {model_name}... prompt: {full_prompt}...")
        
        # Collect all images: conditioning images first, then main image
        all_images = []
        
        # Encode conditioning images
        if conditioning_images:
            for cond_img in conditioning_images:
                cond_path = Path(cond_img["path"])
                if not cond_path.exists():
                    return {"success": False, "error": f"Conditioning image not found: {cond_path}"}
                with open(cond_path, "rb") as f:
                    cond_data = base64.b64encode(f.read()).decode("utf-8")
                all_images.append(cond_data)
                if log_callback:
                    log_callback(f"Conditioning image: {cond_img['label']} -> {cond_path}")
        
        # Encode main image(s)
        if image_path is not None:
            # Handle both single path and list of paths
            if isinstance(image_path, list):
                for img_path in image_path:
                    image_file = Path(img_path)
                    if not image_file.exists():
                        return {"success": False, "error": f"Image not found: {image_file}"}
                    with open(image_file, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    all_images.append(image_data)
            else:
                image_file = Path(image_path)
                if not image_file.exists():
                    return {"success": False, "error": f"Image not found: {image_file}"}
                with open(image_file, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                all_images.append(image_data)
        
        if all_images:
            payload["images"] = all_images
        
        if status_callback:
            status_callback("Sending request to Ollama...")
        
        if progress_callback:
            progress_callback(30)
        
        # Make API request
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        
        if progress_callback:
            progress_callback(70)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Ollama API error: {response.status_code} - {response.text}",
            }
        
        result = response.json()
        raw_response = result.get("response", "")
        
        if status_callback:
            status_callback("Parsing response...")
        
        # Parse output based on format
        parsed_output = parse_output(raw_response, output_format)
        print(f"Parsed output: {parsed_output}")
        # Keep Qwen bounding boxes in x1,y1,x2,y2 format (pixel coordinates)
        # No conversion to xywh is performed.
        
        if log_callback:
            log_callback(f"Response received. Length: {len(raw_response)} chars")
        
        if progress_callback:
            progress_callback(100)
        
        return {
            "success": True,
            "response": raw_response,
            "parsed_output": parsed_output,
            "format": output_format,
            "model": model_name,
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Could not connect to Ollama at {ollama_base_url}. Is Ollama running?",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Qwen3.6 inference failed: {str(e)}",
        }


def run_qwen_api(
    prompt: str,
    template_id: Optional[str] = None,
    output_format: str = "json",
    image_path: Optional[str | Path | List[str | Path]] = None,
    conditioning_images: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    model_name: str = "qwen3.7-plus",
    timeout: int = 120,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Run Qwen VL Plus inference via Aliyun DashScope API (OpenAI-compatible).
    
    This function uses the OpenAI-compatible API to call Qwen vision-language models
    for object detection and image understanding tasks.
    
    Args:
        prompt: Text prompt for the model
        template_id: Optional template identifier for predefined prompts
        output_format: Desired output format ('json', 'yaml', 'bbox', 'text')
        image_path: Optional image path for multimodal input. Can be a single path
                    or a list of paths (for multiple images).
        conditioning_images: Optional list of conditioning/reference images for few-shot
                             visual prompting. Each item is a dict with 'label' (class name)
                             and 'path' (image path). These are prepended before the main image.
        api_key: API key (if None, loads from API_KEY environment variable)
        base_url: Aliyun DashScope API base URL
        model_name: Model name (e.g., 'qwen-vl-plus', 'qwen-vl-max')
        timeout: Request timeout in seconds
        progress_callback: Callback for progress updates
        status_callback: Callback for status messages
        log_callback: Callback for log messages
        is_cancelled: Callback to check if operation should be cancelled
    
    Returns:
        Dictionary with inference results:
            - success: Boolean
            - response: Raw model response
            - parsed_output: Parsed output in requested format
            - format: Output format used
            - model: Model name used
    """
    if status_callback:
        status_callback("Preparing Qwen API request...")
    
    if log_callback:
        log_callback(f"Prompt: {prompt[:100]}...")
        log_callback(f"Model: {model_name}")
        log_callback(f"Output format: {output_format}")
    
    # Apply template if provided
    if template_id:
        prompt = apply_prompt_template(prompt, template_id)
    
    # Add format instruction to prompt
    format_instruction = get_format_instruction(output_format)
    full_prompt = f"{prompt}\n\n{format_instruction}"
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("API_KEY")
    
    if not api_key:
        return {
            "success": False,
            "error": "API key not provided. Set API_KEY environment variable or pass api_key parameter.",
        }
    
    try:
        from openai import OpenAI
        
        if progress_callback:
            progress_callback(10)
        
        # Create OpenAI client with Aliyun base URL
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        if status_callback:
            status_callback("Building request...")
        
        # Build message content
        content_parts = []
        
        # Helper to encode an image and return (base64_data, mime_type)
        def _encode_image(img_path):
            img_file = Path(img_path)
            if not img_file.exists():
                return None, None
            with open(img_file, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            suffix = img_file.suffix.lower()
            mime_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
            }
            return img_data, mime_map.get(suffix, 'image/jpeg')
        
        # Add conditioning images with labels
        if conditioning_images:
            for cond_img in conditioning_images:
                cond_data, cond_mime = _encode_image(cond_img["path"])
                if cond_data is None:
                    return {"success": False, "error": f"Conditioning image not found: {cond_img['path']}"}
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{cond_mime};base64,{cond_data}"
                    }
                })
                content_parts.append({
                    "type": "text",
                    "text": f"Reference image for class: {cond_img['label']}"
                })
                if log_callback:
                    log_callback(f"Conditioning image: {cond_img['label']} -> {cond_img['path']}")
        
        # Add main image(s)
        if image_path is not None:
            if isinstance(image_path, list):
                for img_path in image_path:
                    img_data, img_mime = _encode_image(img_path)
                    if img_data is None:
                        return {"success": False, "error": f"Image not found: {img_path}"}
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_mime};base64,{img_data}"
                        }
                    })
            else:
                img_data, img_mime = _encode_image(image_path)
                if img_data is None:
                    image_file = Path(image_path)
                    return {"success": False, "error": f"Image not found: {image_file}"}
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img_mime};base64,{img_data}"
                    }
                })
        
        # Add text prompt
        content_parts.append({
            "type": "text",
            "text": full_prompt
        })
        
        # Build messages
        messages = [{
            "role": "user",
            "content": content_parts
        }]
        
        if status_callback:
            status_callback("Sending request to Qwen API...")
        
        if progress_callback:
            progress_callback(30)
        
        print(f"Sending request to Qwen API at {base_url} with model {model_name}...")
        
        # Make API request (non-streaming)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        
        if progress_callback:
            progress_callback(70)
        
        # Extract response
        raw_response = ""
        if completion.choices and len(completion.choices) > 0:
            message = completion.choices[0].message
            if message.content:
                raw_response = message.content
        
        if not raw_response:
            return {
                "success": False,
                "error": "No text content in Qwen API response",
                "raw_response": completion,
            }
        
        if status_callback:
            status_callback("Parsing response...")
        
        # Parse output based on format
        parsed_output = parse_output(raw_response, output_format)
        print(f"Parsed output: {parsed_output}")
        
        if log_callback:
            log_callback(f"Response received. Length: {len(raw_response)} chars")
        
        if progress_callback:
            progress_callback(100)
        
        return {
            "success": True,
            "response": raw_response,
            "parsed_output": parsed_output,
            "format": output_format,
            "model": model_name,
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "openai package not installed. Install with: pip install openai",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Qwen API inference failed: {str(e)}",
        }


def run_gemini(
    prompt: str,
    template_id: Optional[str] = None,
    output_format: str = "json",
    image_path: Optional[str | Path] = None,
    api_base_url: str = "https://api.hku.hk/gemini/student",
    deployment_id: str = "gemini-3.5-flash",
    timeout: int = 120,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    thinking_budget: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Run Gemini inference via HKU API proxy.
    
    Args:
        prompt: Text prompt for the model
        template_id: Optional template identifier for predefined prompts
        output_format: Desired output format ('json', 'yaml', 'bbox', 'text')
        image_path: Image path for multimodal input (required for object detection)
        api_base_url: Gemini API base URL
        deployment_id: Model deployment ID (e.g., 'gemini-3.5-flash')
        timeout: Request timeout in seconds
        system_instruction: Optional system instruction to set model role/style
        temperature: Controls randomness (0.0-1.0, higher = more creative)
        top_p: Controls nucleus sampling (0.0-1.0)
        top_k: Limits sampling pool to top K tokens
        thinking_budget: Max tokens for model's internal reasoning
        progress_callback: Callback for progress updates
        status_callback: Callback for status messages
        log_callback: Callback for log messages
        is_cancelled: Callback to check if operation should be cancelled
    
    Returns:
        Dictionary with inference results:
            - success: Boolean
            - response: Raw model response
            - parsed_output: Parsed output in requested format
            - format: Output format used
            - model: Model name used
    """
    if status_callback:
        status_callback("Preparing Gemini request...")
    
    if log_callback:
        log_callback(f"Prompt: {prompt[:100]}...")
        log_callback(f"Model: {deployment_id}")
        log_callback(f"Output format: {output_format}")
    
    # Apply template if provided
    if template_id:
        prompt = apply_prompt_template(prompt, template_id)
    
    # Add format instruction to prompt
    format_instruction = get_format_instruction(output_format)
    full_prompt = f"{prompt}\n\n{format_instruction}"
    
    try:
        if progress_callback:
            progress_callback(10)
        
        # Build the contents array
        contents_parts = []
        
        # Add image if provided
        if image_path is not None:
            image_file = Path(image_path)
            if not image_file.exists():
                return {"success": False, "error": f"Image not found: {image_file}"}
            
            # Read and encode image
            with open(image_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine MIME type from file extension
            suffix = image_file.suffix.lower()
            mime_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
            }
            mime_type = mime_map.get(suffix, 'image/jpeg')
            
            contents_parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            })
        
        # Add text prompt
        contents_parts.append({"text": full_prompt})
        
        # Build request payload
        payload = {
            "contents": [{
                "role": "user",
                "parts": contents_parts
            }]
        }
        
        # Add system instruction if provided
        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        # Build generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        if top_k is not None:
            generation_config["topK"] = top_k
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        if status_callback:
            status_callback("Sending request to Gemini API...")
        
        if progress_callback:
            progress_callback(30)
        
        # Make API request
        api_url = f"{api_base_url}/{deployment_id}:generateContent"
        print(f"Sending request to Gemini API at {api_url}...")
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
        )
        
        if progress_callback:
            progress_callback(70)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Gemini API error: {response.status_code} - {response.text}",
            }
        
        result = response.json()
        
        # Parse Gemini response format
        # Response structure: candidates[0].content.parts[0].text
        raw_response = ""
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                for part in parts:
                    if "text" in part:
                        raw_response += part["text"]
        
        if not raw_response:
            return {
                "success": False,
                "error": "No text content in Gemini response",
                "raw_response": result,
            }
        
        if status_callback:
            status_callback("Parsing response...")
        
        # Parse output based on format
        parsed_output = parse_output(raw_response, output_format)
        print(f"Parsed output: {parsed_output}")
        
        if log_callback:
            log_callback(f"Response received. Length: {len(raw_response)} chars")
        
        if progress_callback:
            progress_callback(100)
        
        return {
            "success": True,
            "response": raw_response,
            "parsed_output": parsed_output,
            "format": output_format,
            "model": deployment_id,
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Could not connect to Gemini API at {api_base_url}.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Gemini inference failed: {str(e)}",
        }


def apply_prompt_template(prompt: str, template_id: str) -> str:
    """Apply a predefined prompt template."""
    templates = {
        "object_detection": "Analyze this image and detect all objects. For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2]. x1, y1 are the pixel coordinates of the top left corner of the box and x2, y2 are the pixel coordinates of the bottom right corner of the box. Return the result as a JSON array like: [{\"label\": \"object_name\", \"bbox_2d\": [x1, y1, x2, y2]}].",
        "image_captioning": "Describe this image in detail, including all visible objects, their positions, and any actions occurring.",
        "scene_understanding": "Analyze the scene in this image. What is happening? What are the main elements? What is the context?",
        "counting": "Count all objects in this image by category. Provide a detailed breakdown.",
        "spatial_reasoning": "Analyze the spatial relationships between objects in this image. Describe relative positions and distances.",
        
    }
    
    template = templates.get(template_id, "")
    if template:
        return f"{template}\n\nUser request: {prompt}"
    return prompt


def get_format_instruction(output_format: str) -> str:
    """Get format instruction for the prompt."""
    instructions = {
        "json": "Please format your response as valid JSON.",
        "yaml": "Please format your response as valid YAML.",
        "bbox": "Please provide bounding box coordinates in the format: [class_name, x1, y1, x2, y2] for each detected object.",
        "text": "Please provide your response in plain text format.",
    }
    return instructions.get(output_format, "")


def parse_output(response: str, output_format: str) -> Any:
    """Parse model output based on requested format."""
    if output_format == "json":
        try:
            # Try to extract JSON from response
            # Look for JSON block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw": response, "parse_error": "Could not parse as JSON"}
    
    elif output_format == "yaml":
        try:
            import yaml
            if "```yaml" in response:
                yaml_str = response.split("```yaml")[1].split("```")[0]
            elif "```" in response:
                yaml_str = response.split("```")[1].split("```")[0]
            else:
                yaml_str = response
            
            return yaml.safe_load(yaml_str)
        except Exception:
            return {"raw": response, "parse_error": "Could not parse as YAML"}
    
    elif output_format == "bbox":
        # Try to extract bounding boxes
        bboxes = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Try to parse as [class, x1, y1, x2, y2]
                try:
                    parts = line.replace('[', '').replace(']', '').split(',')
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = [float(p.strip()) for p in parts[1:5]]
                        bboxes.append({
                            "class": parts[0].strip(),
                            "bbox": [x1, y1, x2, y2],
                        })
                except:
                    pass
        return bboxes if bboxes else {"raw": response}
    
    else:
        return response




def list_ollama_models(ollama_base_url: str = "http://localhost:11434") -> Dict:
    """List available models in Ollama."""
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "success": True,
                "models": [m.get("name") for m in models],
            }
        else:
            return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
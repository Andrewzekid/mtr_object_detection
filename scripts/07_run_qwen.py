#!/usr/bin/env python3
"""
Run Qwen3.6 inference via Ollama, llama.cpp, or the Aliyun DashScope API.

This script sends a prompt (with optional image) to Qwen3.6 and returns
structured output in the requested format.

BACKENDS:
    --backend ollama     Local Ollama server (default).
                         Prereq: ollama serve  +  ollama pull qwen3.6
    --backend llamacpp   Local llama.cpp server with an mmproj vision projector
                         (OpenAI-compatible /v1/chat/completions endpoint).
                         Prereq: llama-server -m <model.gguf> --mmproj <mmproj.gguf> --port 8089
    --use-api            Aliyun DashScope cloud API (overrides --backend).

USAGE:
    # Single image mode
    python scripts/07_run_qwen.py --prompt "Describe this image" --image ./sample.jpg
    python scripts/07_run_qwen.py --prompt "Detect all cars" --template object_detection --format json --vis-output ./result.jpg
    python scripts/07_run_qwen.py --prompt "Count objects" --template counting --format text

    # Batch mode (folder of images)
    python scripts/07_run_qwen.py --prompt "Detect all cars" --image-folder ./images/ --output ./output/qwen_results/
    python scripts/07_run_qwen.py --prompt "Describe scene" --image-folder ./photos/ --output ./results/ --vis-output ./vis/

    # llama.cpp backend (mmproj vision projector reads the image files)
    # First start the server:
    #   llama-server -m Qwen3.6-27B-Q5_K_M.gguf --mmproj mmproj.gguf --port 8089
    python scripts/07_run_qwen.py --backend llamacpp --llamacpp-url http://127.0.0.1:8089 \\
        --llamacpp-model ./Qwen3.6-27B-Q5_K_M.gguf \\
        --prompt "Detect all: Ceiling light, Exit Sign" --template object_detection \\
        --format json --image ./sample.jpg --vis-output ./vis.jpg

    EXAMPLE:
     python scripts/07_run_qwen.py \
        --prompt "Detect all objects: Ceiling light, Sign, Advertisement Board, Ticket Gate, Map. Signs are hanging lcd screens from the ceiling which show directions. They can contain arrows, characters like A B C, and X's. Do not categorize the X's on ticket gates as a sign. Do not classify posters as signs. Only hanging monitors can be classified as signs. Ceiling light are a flat and horizontal rectangular strip, do not detect reflections of lights in the glass or wall. If there are lights in the green wall, they are likely to be reflections. Consider carefully whether or not they are actually reflections. Detect individual ceiling lights and do not cluster them together. Advertisements are flat lcd screens on the green wall that only display commercial content, not directions. Also do not mistake them for posters. Maps are posters which show the directions in the MTR and which way to go. Ticket gates are turnstiles. Do not classify ticket vending machines as ticket gates." \
        --template object_detection \
        --image-folder ./qwen_test \
        --annotations-output ./output/annotations/ \
        --vis-output ./output/vis_qwen_batch/ \
        --split-by-class

    python scripts/07_run_qwen.py \
    --prompt "Detect any hanging overhead exit signage containing the standard Exit icon: a bright lime green square background displaying the white Chinese character '出' stacked above the white English word 'EXIT', there may or may not be adjacent directional arrows or exit letter identifiers (e.g., A, B, C). Do not classify normal hanging monitors or tvs without the lime '出' and EXIT text as exit signs. Do not classify posters as exit signs. Do not classify platform signs as exit sign. Do not classify advertisement boards as exit signs. Do not classify the fire extinguisher sign as exit sign." \
    --template object_detection \
    --image-folder ./Datasets/MTR/MTR_4k_dataset_exit_signs \
    --resume-from 381 \
    --output ./output/results/MTR_4k/ \
    --vis-output ./output/vis/MTR_4k/ \
    --split-by-class \
    --annotations-output ./output/annotations/MTR_4k/ \
    --format json \
    --per-class \
    --classes "Exit Sign" \
    --conditioning-images ./Datasets/MTR/ref_images/

    #API mode (Aliyun DashScope cloud API)
    python scripts/07_run_qwen.py \
        --use-api \
        --base-url "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" \
        --api-key "sk-ws-H.DMLLILY.FTdA.MEUCIEb-iEdMlxCgMR6KvcS2mymqAEpr8L010TOFDJOdPoL6AiEAjHxQe-XUDpUn1QhXnBJDi-tURMh4FumjXMI7t4C7lVI" \
        --prompt "Detect any hanging overhead exit signage containing the standard Exit icon: a bright lime green square background displaying the white Chinese character '出' stacked above the white English word 'EXIT', there may or may not be adjacent directional arrows or exit letter identifiers (e.g., A, B, C)." \
        --template object_detection \
        --image-folder ./Datasets/MTR/MTR_4k_dataset_exit_signs \
        --output ./output/results/MTR_4k/ \
        --vis-output ./output/vis/MTR_4k/ \
        --split-by-class \
        --resume-from 380 \
        --annotations-output ./output/annotations/MTR_4k/ \
        --format json \
        --per-class \
        --classes "Exit Sign" \
        --api-model "qwen3.8-max" \
        --conditioning-images ./Datasets/MTR/ref_images/

PREREQUISITES:
    - Ollama backend:  ollama serve  &&  ollama pull qwen3.6
    - llama.cpp backend: llama-server -m Qwen3.6-27B-Q5_K_M.gguf --mmproj mmproj.gguf --port 8089
    - API backend:     API_KEY env var (or --api-key)

    OUTPUT:
    - Raw model response
    - Parsed output in requested format (json/yaml/bbox/text)
      NOTE: bounding boxes are scaled to PIXEL coordinates (Qwen's raw 0-1000
      normalized coords are converted using --coord-scale and the image's real
      W/H) before being saved, so the JSON can be fed directly to SAM3.
    - Optional visualization image with bounding boxes drawn
    - For batch mode: individual JSON files per image + summary.json

    # llama.cpp batch mode (MTR 4k exit-sign dataset)
                          [--api-model API_MODEL] [--conditioning-images CONDITIONING_IMAGES] [--per-image-labels] [--per-class] [--classes [CLASSES ...]] [--bbox-order {auto,xyxy,yxyx}]
                      [--bbox-field {auto,bbox_2d,box_2d,bbox}] [--dedup-iou DEDUP_IOU]

   
"""

import argparse
import json
import os
import re
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

from core.models_inference import run_qwen, run_qwen_api, run_qwen_llamacpp, list_ollama_models


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
    "exit sign": "a hanging monitor/display showing the lime-green character 出 and text 'EXIT'. It is a hanging LCD screen, not a wall poster.",
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
    
    # Build the enhanced prompt. Framed anti-hallucination: the model must only
    # report boxes for the MAIN image, only for the listed classes, only when
    # confident, and may return an empty list. "Detect all instances of <every
    # class>" otherwise pressures the model into inventing absent classes and
    # detecting in the reference images.
    enhanced = (
        f"You will be given {main_image_num} images. "
        f"The first {num_refs} are REFERENCE examples (one per class) showing what "
        f"each class looks like. Image {main_image_num} is the MAIN scene to analyze.\n\n"
        f"REFERENCE IMAGES (for recognition only — do NOT report boxes for these):\n"
        f"{ref_text}\n\n"
        f"MAIN IMAGE (image {main_image_num}): the ONLY image you should report "
        f"boxes for.\n\n"
        f"Look at the MAIN image and detect instances of ONLY these classes that "
        f"are clearly and unambiguously present: {', '.join(class_names)}.\n"
        f"Use the reference images to recognize what each class looks like, then "
        f"compare objects in the MAIN image against those references.\n\n"
        f"Rules to avoid false detections:\n"
        f"- Only report a box if you are confident the object is really there and "
        f"clearly matches one of the classes above. If you are unsure, do NOT "
        f"report it.\n"
        f"- Do NOT invent objects to cover every class. If a class is absent from "
        f"the MAIN image, omit it. Returning fewer boxes — or an empty list — is "
        f"correct and expected when the scene is sparse.\n"
        f"- Report boxes ONLY for the MAIN image (image {main_image_num}). Never "
        f"report boxes located in a reference image.\n. If there is nothing to detect, return an empty list.\n"
        f"- Do not report reflections, shadows, glare, or partial/ambiguous matches.\n"
        f"- Do not output duplicate or heavily overlapping boxes for the same object; "
        f"one box per object.\n\n"
        f"{original_prompt}"
    )
    
    return enhanced


def build_per_class_prompt(class_name, args, has_conditioning=False):
    """Build a single-class detection prompt for ``class_name``.

    Used by ``--per-class`` mode, which runs Qwen once per class so the model
    can focus on a single concept at a time (typically better recall for
    confusable classes). The class description — including any negative guidance
    such as "do not detect reflections" — is pulled from
    ``MTR_CLASS_DESCRIPTIONS`` when available; the user's ``--prompt`` is kept
    as additional guidance so custom instructions are not lost.

    When ``has_conditioning`` is true the prompt explicitly tells the model
    that image 1 is a reference example of the class and image 2 is the scene to
    analyze — otherwise the reference image is delivered with no explanation and
    the model cannot tell it apart from the main image.
    """
    desc = MTR_CLASS_DESCRIPTIONS.get(class_name) or MTR_CLASS_DESCRIPTIONS.get(
        class_name.lower(), ""
    )
    bbox_order, bbox_field = resolve_bbox_format(args)
    if has_conditioning:
        parts = [
            f'You are given 2 images. Image 1 is a REFERENCE example of '
            f'"{class_name}" — use it only to recognize what {class_name} looks '
            f'like. Image 2 is the main scene to analyze. Detect all instances '
            f'of "{class_name}" in Image 2 (the main scene) only. Do NOT report '
            f'boxes for Image 1. ONLY DETECT objects in Image 2 that look very similar to Image 1.'
        ]
    else:
        parts = [f'Detect all instances of "{class_name}" in the main image.']
    if desc:
        parts.append(desc)
    parts.append(
        "Return ONLY bounding boxes for this class, as a JSON list of objects "
        f"each with {bbox_format_hint(bbox_order, bbox_field)}."
    )
    if args.prompt:
        parts.append(f"Additional guidance:\n{args.prompt}")
    return "\n\n".join(parts)


def parse_classes_from_prompt(prompt):
    """Best-effort extraction of a class list from a ``--prompt`` string.

    Handles the common ``"Detect all: A, B, C. <descriptions>"`` / ``"Detect all
    objects: A, B, C."`` phrasings: takes the comma-separated list between the
    ``Detect all`` clause and the first period. Returns ``[]`` if nothing that
    looks like a class list is found.
    """
    if not prompt:
        return []
    # "Detect all [objects [of]]: <list>." — capture up to the first period.
    m = re.search(r"detect\s+all[^.:：]*[:：]\s*([^.\n]+)", prompt, re.IGNORECASE)
    if not m:
        return []
    classes = []
    for raw in m.group(1).split(","):
        c = raw.strip()
        # Skip fragments that are clearly descriptions, not class names.
        if c and len(c) <= 40 and not c.endswith("."):
            classes.append(c)
    return classes


def resolve_per_class_list(args, conditioning_images):
    """Determine the class list to run one-by-one for ``--per-class``.

    Priority: explicit ``--classes`` > labels derived from
    ``--conditioning-images`` filenames > classes parsed from ``--prompt``.
    Raises ``ValueError`` if none of these yield a class list.
    """
    if args.classes:
        classes = [c.strip() for c in args.classes if c.strip()]
        if classes:
            return classes
    if conditioning_images:
        seen = set()
        classes = []
        for ci in conditioning_images:
            if ci["label"] not in seen:
                seen.add(ci["label"])
                classes.append(ci["label"])
        if classes:
            return classes
    classes = parse_classes_from_prompt(args.prompt)
    if classes:
        return classes
    raise ValueError(
        "--per-class could not determine the class list. Provide --classes, "
        "--conditioning-images, or a '--prompt' starting with "
        "'Detect all: A, B, C.'."
    )


def detect_vlm_family(model_name):
    """Return the bounding-box convention family for a model name.

    Returns ``"gemma"`` for Gemma / Gemini-family models (which emit
    ``box_2d`` in yxyx) and ``"qwen"`` otherwise (``bbox_2d`` in xyxy).
    Detection is by substring so Ollama tags like ``gemma4:31b`` resolve
    correctly.
    """
    name = (model_name or "").lower()
    if "gemma" in name or "gemini" in name or "paligemma" in name:
        return "gemma"
    return "qwen"


def resolve_bbox_format(args):
    """Determine the (bbox_order, bbox_field) pair for the configured model.

    Explicit ``--bbox-order`` / ``--bbox-field`` win; otherwise they are
    auto-detected from the model name via :func:`detect_vlm_family`.
    """
    family = detect_vlm_family(args.model)
    order = args.bbox_order if args.bbox_order != "auto" else (
        "yxyx" if family == "gemma" else "xyxy")
    field = args.bbox_field if args.bbox_field != "auto" else (
        "box_2d" if family == "gemma" else "bbox_2d")
    return order, field


def bbox_format_hint(bbox_order, bbox_field):
    """A short human-readable description of the bbox format for prompts."""
    if bbox_order == "yxyx":
        return (f'a "{bbox_field}" field in [y1, x1, y2, x2] format '
                '(top-left then bottom-right, normalized 0-1000)')
    return (f'a "{bbox_field}" field in [x1, y1, x2, y2] format '
            '(top-left then bottom-right, normalized 0-1000)')


def normalize_parsed_output_bboxes(parsed_output, bbox_order, bbox_field):
    """Normalize parsed boxes to ``bbox_2d`` in xyxy, in place on a copy.

    Different VLM families emit different keys/orders: Qwen uses
    ``bbox_2d`` in xyxy, Gemma/Gemini use ``box_2d`` in yxyx. Downstream
    code (``scale_parsed_output_to_pixels``, SAM3, COCO export) only reads
    ``bbox_2d``/``bbox`` in xyxy, so this folds any other convention into
    that canonical form before scaling. When the source key differs from
    ``bbox_2d`` it is removed to avoid leaving a stale transposed copy.
    """
    if parsed_output is None:
        return parsed_output

    def fix_item(item):
        if not isinstance(item, dict):
            return item
        item = dict(item)
        # Find the source box: explicit configured field, else common keys.
        src_key = None
        for k in (bbox_field, "bbox_2d", "box_2d", "bbox"):
            if k in item:
                src_key = k
                break
        if src_key is None:
            return item
        bbox = item[src_key]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return item
        vals = [float(v) for v in bbox]
        # Only the configured family is interpreted as yxyx; anything else
        # (including a stray box_2d when order==xyxy) is treated as xyxy.
        if bbox_order == "yxyx" and src_key in (bbox_field, "box_2d"):
            y1, x1, y2, x2 = vals
            vals = [x1, y1, x2, y2]
        item["bbox_2d"] = vals
        if src_key != "bbox_2d":
            item.pop(src_key, None)
        return item

    if isinstance(parsed_output, list):
        return [fix_item(it) for it in parsed_output]
    if isinstance(parsed_output, dict):
        return fix_item(parsed_output)
    return parsed_output


def _label_matches(model_label, target):
    """True if a model-emitted label refers to ``target`` class.

    Used in ``--per-class`` mode to decide which detections a single-class call
    should keep. Comparison is case-insensitive and fuzzy (substring either
    direction, plus token overlap) so "Ceiling light" / "ceiling light" /
    "ceiling_light" all match. A missing label is treated as a match: a call
    that asked for only one class returns unlabeled boxes for that class.
    """
    if model_label is None or str(model_label).strip() == "":
        return True
    ml = str(model_label).strip().lower().replace("_", " ")
    t = str(target).strip().lower().replace("_", " ")
    if ml == t:
        return True
    if t in ml or ml in t:
        return True
    return bool(set(ml.split()) & set(t.split()))


def _iou(a, b):
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_cross_class(parsed_output, iou_threshold):
    """Suppress same-position boxes that carry DIFFERENT labels.

    Two boxes at nearly the same location with different labels are almost
    always one object the model double-labeled (e.g. a ceiling light reported
    as both "Ceiling light" and "Exit Sign"). This keeps a single detection
    per such cluster — the one with an explicit score/confidence, else the
    first seen. Same-class boxes are left untouched: adjacent instances of
    one class (a row of ceiling lights) are legitimate and must not merge.

    Operates on the canonical ``bbox_2d`` xyxy form produced by
    :func:`normalize_parsed_output_bboxes`. ``iou_threshold <= 0`` disables.
    """
    if not parsed_output or iou_threshold <= 0:
        return parsed_output
    items = parsed_output if isinstance(parsed_output, list) else [parsed_output]
    items = [it for it in items if isinstance(it, dict)]

    def bbox_of(it):
        for k in ("bbox_2d", "box_2d", "bbox"):
            v = it.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                return [float(x) for x in v]
        return None

    def score_of(it):
        for k in ("score", "confidence", "prob"):
            v = it.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    kept = []
    for it in items:
        b = bbox_of(it)
        if b is None:
            kept.append(it)
            continue
        conflict_idx = None
        for i, k in enumerate(kept):
            kb = bbox_of(k)
            if kb is None or k.get("label") == it.get("label"):
                continue  # no box, or same class -> not a cross-class duplicate
            if _iou(b, kb) >= iou_threshold:
                conflict_idx = i
                break
        if conflict_idx is None:
            kept.append(it)
        else:
            # Same object, two labels: keep the more confident detection.
            si, sk = score_of(it), score_of(kept[conflict_idx])
            if si is not None and (sk is None or si > sk):
                kept[conflict_idx] = it
            # else drop `it` (keep the existing one)
    return kept


def _call_qwen(args, prompt, image_path, conditioning_images):
    """Dispatch a single Qwen call to the configured backend.

    Backends:
        - llamacpp: local llama.cpp server with mmproj vision projector
                    (OpenAI-compatible /v1/chat/completions)
        - ollama (default): local Ollama server (/api/generate or
                    /v1/chat/completions when --per-image-labels)
        - api: Aliyun DashScope cloud API (--use-api)
    """
    bbox_order, bbox_field = resolve_bbox_format(args)
    if args.use_api:
        api_key = get_api_key(args)
        return run_qwen_api(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=image_path,
            conditioning_images=conditioning_images,
            api_key=api_key,
            base_url=args.base_url,
            model_name=args.api_model,
            timeout=args.timeout,
            log_callback=lambda msg: print(f"  {msg}"),
            per_image_labels=args.per_image_labels,
            bbox_field=bbox_field,
            bbox_order=bbox_order,
        )
    if args.backend == "llamacpp":
        return run_qwen_llamacpp(
            prompt=prompt,
            template_id=args.template,
            output_format=args.format,
            image_path=image_path,
            conditioning_images=conditioning_images,
            llamacpp_base_url=args.llamacpp_url,
            model_name=args.llamacpp_model,
            api_key=args.llamacpp_api_key,
            timeout=args.timeout,
            log_callback=lambda msg: print(f"  {msg}"),
            bbox_field=bbox_field,
            bbox_order=bbox_order,
        )
    return run_qwen(
        prompt=prompt,
        template_id=args.template,
        output_format=args.format,
        image_path=image_path,
        conditioning_images=conditioning_images,
        ollama_base_url=args.ollama_url,
        model_name=args.model,
        timeout=args.timeout,
        log_callback=lambda msg: print(f"  {msg}"),
        per_image_labels=args.per_image_labels,
        bbox_field=bbox_field,
        bbox_order=bbox_order,
    )


def run_qwen_for_image(args, prompt, image_path, conditioning_images):
    """Run Qwen for one image and return a result dict with ``parsed_output``.

    When ``args.per_class`` is set, Qwen is run once per class — each call sees
    only that class's reference image (if any) and a focused single-class prompt
    — and the per-call bounding boxes are merged into one ``parsed_output``
    list. Only detections whose model label refers to the requested class are
    kept (and canonicalized to that class name); detections of other classes
    are dropped so they cannot be mislabeled and produce duplicate boxes at the
    same location with a different label.

    Otherwise a single call is made with the supplied (multi-class) prompt.
    """
    if not args.per_class:
        img_arg = str(image_path) if image_path is not None else None
        return _call_qwen(args, prompt, img_arg, conditioning_images)

    classes = resolve_per_class_list(args, conditioning_images)
    print(f"  Per-class mode: {len(classes)} class(es) -> {classes}")
    merged = []
    for class_name in classes:
        # Match the reference image for this class fuzzily (case-insensitive)
        # so --classes "Ceiling light" still pairs with filename "ceiling light".
        cond = [ci for ci in (conditioning_images or [])
                if _label_matches(ci["label"], class_name)] or None
        class_prompt = build_per_class_prompt(class_name, args,
                                              has_conditioning=bool(cond))
        r = _call_qwen(args, class_prompt, str(image_path), cond)
        if not r.get("success"):
            print(f"  [per-class] '{class_name}' failed: {r.get('error', '?')}")
            continue
        po = r.get("parsed_output")
        items = po if isinstance(po, list) else ([po] if isinstance(po, dict) else [])
        # This call was asked to detect ONLY class_name. Keep detections whose
        # model label refers to that class (canonicalized to class_name) and DROP
        # the rest — a non-matching box here is a different object that its own
        # class call will handle. Blindly forcing the label on every box is what
        # produced duplicate boxes at the same position with different labels
        # (e.g. a ceiling light detected during the exit-sign call, relabeled
        # "exit sign").
        kept = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            if _label_matches(it.get("label"), class_name):
                it["label"] = class_name  # canonicalize, don't guess
                merged.append(it)
                kept += 1
        print(f"  [per-class] '{class_name}': {kept}/{len(items)} box(es) kept")
    return {
        "success": True,
        "response": json.dumps(merged),
        "parsed_output": merged,
        "model": (args.api_model if args.use_api
                  else args.llamacpp_model if args.backend == "llamacpp"
                  else args.model),
        "format": args.format,
    }


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
    python scripts/07_run_qwen.py --prompt "Detect any directional or overhead exit signage containing the standard Exit icon: 
    a bright lime green square background displaying the white Chinese character '出' stacked above the white English word 'EXIT', 
    there may or may not be adjacent directional arrows or exit letter identifiers (e.g., A, B, C)." \\
        --template object_detection --image-folder ./MTR_4k_dataset_exit_signs \\
        --output ./output/results/MTR_4k/ --vis-output ./output/vis/MTR_4k/ --split-by-class --annotations-output ./output/annotations/MTR_4k/ --format json

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
        help="Ollama API base URL (default: http://localhost:11434). "
             "Only used when --backend=ollama.",
    )
    # --- llama.cpp backend ------------------------------------------------
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "llamacpp"],
        help="Inference backend to use (default: ollama). 'llamacpp' targets a "
             "local llama.cpp server (started with --mmproj for the vision "
             "projector) exposing an OpenAI-compatible /v1/chat/completions "
             "endpoint. --use-api overrides this.",
    )
    parser.add_argument(
        "--llamacpp-url",
        type=str,
        default="http://127.0.0.1:8089",
        help="Base URL of the llama.cpp server (no /v1 suffix). "
             "Default: http://127.0.0.1:8089",
    )
    parser.add_argument(
        "--llamacpp-model",
        type=str,
        default="./Qwen3.6-27B-Q5_K_M.gguf",
        help="Model identifier passed to the llama.cpp server (the .gguf path "
             "or alias it was started with). Default: ./Qwen3.6-27B-Q5_K_M.gguf",
    )
    parser.add_argument(
        "--llamacpp-api-key",
        type=str,
        default="local",
        help="API key for the llama.cpp server (any non-empty string for a "
             "local server; default: local).",
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
        help="Use Aliyun DashScope API (qwen-vl-plus) instead of a local "
             "backend. Overrides --backend.",
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
    parser.add_argument(
        "--per-image-labels",
        action="store_true",
        help="When using --conditioning-images, attach a 'Reference image for "
             "class: <name>' text part to each reference image so the model can "
             "ground each exemplar to its class. For the local Ollama backend this "
             "switches the call to the interleaved /v1/chat/completions endpoint "
             "(instead of /api/generate's flat image list). May improve detection "
             "accuracy at the cost of a few extra tokens.",
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Run Qwen once PER class on each image and merge the bounding boxes, "
             "instead of one multi-class call. Focusing on a single concept per "
             "call usually improves recall for confusable classes. The class list "
             "comes from --classes or, if omitted, from the --conditioning-images "
             "filenames. Costs N× more calls per image (one per class).",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Explicit class list for --per-class (space-separated), e.g. "
             "--classes 'Ceiling light' 'Exit Sign' 'Advertisement Board'. "
             "If omitted with --per-class, classes are derived from "
             "--conditioning-images filenames.",
    )
    parser.add_argument(
        "--bbox-order",
        choices=["auto", "xyxy", "yxyx"],
        default="auto",
        help="Bounding-box coordinate order the model emits. 'auto' (default) "
             "detects from --model: gemma/gemini -> yxyx, otherwise xyxy. "
             "Qwen uses xyxy; Gemma/Gemini use yxyx [y1,x1,y2,x2].",
    )
    parser.add_argument(
        "--bbox-field",
        choices=["auto", "bbox_2d", "box_2d", "bbox"],
        default="auto",
        help="JSON key the model emits bounding boxes under. 'auto' (default) "
             "detects from --model: gemma/gemini -> box_2d, otherwise bbox_2d. "
             "Boxes are normalized to the canonical 'bbox_2d' xyxy form before "
             "scaling to pixels, so downstream code is family-agnostic.",
    )
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.75,
        help="Cross-class duplicate suppression: when two boxes with DIFFERENT "
             "labels overlap by at least this IoU, they are treated as one object "
             "the model double-labeled and only the higher-confidence one is kept "
             "(default: 0.75). Same-class boxes are never merged. Set to 0 to "
             "disable. Applies in both single-call and --per-class modes.",
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
    
    # Build the prompt (with conditioning info if provided). Per-class mode
    # builds its own per-class prompts inside run_qwen_for_image, so the
    # multi-reference conditioning prompt is only used otherwise.
    prompt = args.prompt
    if conditioning_images and not args.per_class:
        prompt = build_conditioning_prompt(conditioning_images, args.prompt)
        print(f"  Conditioning images: {len(conditioning_images)} reference(s)")
        for ci in conditioning_images:
            print(f"    - {ci['label']}: {ci['path']}")

    # Run Qwen (once, or once per class when --per-class is set) and merge.
    result = run_qwen_for_image(args, prompt, image_path, conditioning_images)

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

    # Canonicalize labels to the requested --classes when provided.
    # In single-call (non --per-class) mode Qwen often emits verbose,
    # description-like labels (e.g. a full sentence describing "exit sign").
    # If the user passed an explicit --classes list, map each detection's
    # label to the single matching class name so downstream output (vis,
    # split-by-class, JSON) uses the short canonical name. Detections that
    # match no requested class are dropped.
    if not args.per_class and args.classes:
        requested = [c.strip() for c in args.classes if c.strip()]
        if isinstance(parsed_output, list):
            items = parsed_output
        elif isinstance(parsed_output, dict):
            items = [parsed_output]
        else:
            items = []
        canonicalized = []
        for it in items:
            if not isinstance(it, dict):
                canonicalized.append(it)
                continue
            model_label = it.get("label")
            matched = None
            for cls in requested:
                if _label_matches(model_label, cls):
                    matched = cls
                    break
            if matched is not None:
                it["label"] = matched
                canonicalized.append(it)
            else:
                # No requested class matched: drop the detection so an
                # unrelated verbose label cannot leak into the output.
                print(f"  Dropping detection with unmatched label: {str(model_label)[:80]}")
        parsed_output = canonicalized

    # Scale Qwen's normalized 0-1000 bounding boxes to pixel coordinates before
    # saving the JSON and before any downstream use (visualization, split-by-class,
    # SAM3). The saved JSON therefore contains pixel-space boxes. First fold any
    # model-family bbox convention (e.g. Gemma's box_2d yxyx) into the canonical
    # bbox_2d xyxy form the scaler expects.
    bbox_order, bbox_field = resolve_bbox_format(args)
    parsed_output = normalize_parsed_output_bboxes(
        parsed_output, bbox_order, bbox_field)
    parsed_output = dedup_cross_class(parsed_output, args.dedup_iou)
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
    seen_safe_names = {}
    for class_name, bboxes in class_bboxes.items():
        # Sanitize class name for filename (replace spaces/special chars with underscores)
        safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
        # Truncate to avoid OS "File name too long" errors (ext4 limit 255 bytes;
        # leave room for the .json suffix and a uniqueness counter).
        max_len = 120
        if len(safe_class_name) > max_len:
            import hashlib
            digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()[:8]
            safe_class_name = f"{safe_class_name[:max_len - 9]}_{digest}"
        # Disambiguate collisions after truncation.
        if safe_class_name in seen_safe_names:
            seen_safe_names[safe_class_name] += 1
            safe_class_name = f"{safe_class_name}_{seen_safe_names[safe_class_name]}"
        else:
            seen_safe_names[safe_class_name] = 1
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
        elif args.backend == "llamacpp":
            print(f"Running Qwen batch inference...")
            print(f"  Model: {args.llamacpp_model}")
            print(f"  Mode: llama.cpp (local, mmproj vision)")
            print(f"  Server: {args.llamacpp_url}")
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
    elif args.backend == "llamacpp":
        print(f"Running Qwen inference...")
        print(f"  Model: {args.llamacpp_model}")
        print(f"  Mode: llama.cpp (local, mmproj vision)")
        print(f"  Server: {args.llamacpp_url}")
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
    
    # Build the prompt (with conditioning info if provided). Per-class mode
    # builds its own per-class prompts inside run_qwen_for_image.
    prompt = args.prompt
    if conditioning_images and not args.per_class:
        prompt = build_conditioning_prompt(conditioning_images, args.prompt)

    # Run Qwen (once, or once per class when --per-class is set) and merge.
    result = run_qwen_for_image(args, prompt, args.image, conditioning_images)

    if not result.get("success"):
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Get outputs
    raw_response = result.get("response", "")
    parsed_output = result.get("parsed_output")

    # Scale Qwen's normalized 0-1000 bounding boxes to pixel coordinates before
    # saving the JSON and before any downstream use (visualization, SAM3). The
    # saved JSON therefore contains pixel-space boxes. First fold any model-family
    # bbox convention (e.g. Gemma's box_2d yxyx) into the canonical bbox_2d xyxy
    # form the scaler expects.
    bbox_order, bbox_field = resolve_bbox_format(args)
    parsed_output = normalize_parsed_output_bboxes(
        parsed_output, bbox_order, bbox_field)
    parsed_output = dedup_cross_class(parsed_output, args.dedup_iou)
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
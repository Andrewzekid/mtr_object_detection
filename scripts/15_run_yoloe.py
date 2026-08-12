#!/usr/bin/env python3
"""
Run YOLOE text-promptable segmentation/detection on a folder of images.

YOLOE (https://github.com/ultralytics/ultralytics) supports open-vocabulary
detection driven by a free-form text prompt, e.g. "exit signs". This script:

  1. Loads a YOLOE checkpoint (e.g. yoloe-26l-seg.pt).
  2. Encodes the user-supplied text prompt into text-prompt embeddings (TPE)
     via `model.get_text_pe(...)` and registers them with `model.set_classes(...)`.
  3. Runs inference over every image in `--image-folder`.
  4. Saves, for each image:
       - A per-image result JSON (bbox + optional segmentation polygon + conf)
         under `--output`, mirroring 07_run_qwen.py's per-image `_result.json`.
       - A visualization JPG under `--vis-output` (bboxes + class labels).
  5. Splits annotations by class into `--annotations-output`, using the same
     per-image subfolder + per-class JSON layout as 07_run_qwen.py's
     `--split-by-class` mode, so downstream tooling (SAM3, COCO export, etc.)
     can consume both Qwen and YOLOE outputs interchangeably.

The annotation JSON schema matches 07_run_qwen.py exactly:
    {
        "class_name": "<text prompt>",
        "image": "<absolute image path>",
        "bboxes": [[x1, y1, x2, y2], ...]   # pixel coordinates
    }

USAGE:
    # Basic
    python scripts/15_run_yoloe.py \\
        --model yoloe-26l-seg.pt \\
        --text-prompt "exit signs" \\
        --image-folder ./Datasets/MTR/MTR_4k_images \\
        --output ./output/results_yoloe/MTR_4k \\
        --vis-output ./output/vis_yoloe/MTR_4k \\
        --annotations-output ./output/annotations_yoloe/MTR_4k \\
        --split-by-class

    # Resume from image N (1-indexed, same semantics as 07_run_qwen.py)
    python scripts/15_run_yoloe.py \\
        --model yoloe-26l-seg.pt \\
        --text-prompt "exit signs" \\
        --image-folder ./Datasets/MTR/MTR_4k_images \\
        --output ./output/results_yoloe/MTR_4k \\
        --vis-output ./output/vis_yoloe/MTR_4k \\
        --annotations-output ./output/annotations_yoloe/MTR_4k \\
        --split-by-class \\
        --resume-from 381

PREREQUISITES:
    - ultralytics package with YOLOE support (pip install -U ultralytics)
    - The YOLOE checkpoint .pt file accessible locally or downloadable by
      ultralytics (passing a bare name like "yoloe-26l-seg.pt" will trigger
      an automatic download on first use).

OUTPUT:
    - <output>/<stem>_result.json   per-image detection result
    - <vis-output>/<stem>_vis.jpg   visualization with bboxes + labels
    - <annotations-output>/<stem>/<safe_class>.json
        per-class bbox files (when --split-by-class is set)
    - <output>/summary.json         batch summary
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so project utilities can be imported if needed.
sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def find_image_files(folder: Path):
    """Return a sorted list of image file Paths in ``folder``."""
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def sanitize_class_name(class_name: str, max_len: int = 120) -> str:
    """Make a class name safe to use as a filename component.

    Mirrors the sanitization in 07_run_qwen.py.split_annotations_by_class so
    that YOLOE and Qwen produce identically-named annotation files for the
    same class string.
    """
    safe = "".join(c if c.isalnum() else "_" for c in class_name)
    if len(safe) > max_len:
        digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:max_len - 9]}_{digest}"
    return safe


def split_annotations_by_class(detections, image_path, annotations_dir, img_w, img_h):
    """Split detections by class into per-image, per-class JSON files.

    Reproduces the layout produced by 07_run_qwen.py's
    ``split_annotations_by_class``:
        <annotations_dir>/<image_stem>/<safe_class>.json
    each containing ``{"class_name", "image", "bboxes"}`` with pixel-space
    [x1, y1, x2, y2] boxes clamped to image bounds.
    """
    image_folder = annotations_dir / image_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)

    class_bboxes = {}
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(int(x1), img_w))
        y1 = max(0, min(int(y1), img_h))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))
        class_bboxes.setdefault(label, []).append([x1, y1, x2, y2])

    saved_files = {}
    seen_safe_names = {}
    for class_name, bboxes in class_bboxes.items():
        safe_class_name = sanitize_class_name(class_name)
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


def draw_visualization(image_path, vis_path, detections):
    """Draw bboxes + labels on the image and save to ``vis_path``."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read image for visualization: {image_path}")
        return False
    img_h, img_w = img.shape[:2]

    for det in detections:
        x1 = max(0, min(int(det["bbox"][0]), img_w))
        y1 = max(0, min(int(det["bbox"][1]), img_h))
        x2 = max(0, min(int(det["bbox"][2]), img_w))
        y2 = max(0, min(int(det["bbox"][3]), img_h))
        label = det["label"]
        conf = det.get("confidence")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}" + (f" {conf:.2f}" if conf is not None else "")
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), img)
    return True


def process_single_image(model, image_path, args, output_dir, vis_dir,
                         visual_prompts=None, visual_predictor=None):
    """Run YOLOE on one image; return a result dict and a list of detections.

    ``detections`` is a list of ``{"label", "bbox": [x1,y1,x2,y2], "confidence",
    "segmentation": <optional polygon>}`` with pixel-space boxes.
    """
    print(f"\nProcessing: {image_path.name}")
    print(f"  Image: {image_path}")
    print(f"  Prompt mode: {args.prompt_mode}")
    if args.prompt_mode == "text":
        print(f"  Text prompt: {args.text_prompt}")

    if args.prompt_mode == "visual":
        results = model.predict(
            source=str(image_path),
            visual_prompts=visual_prompts,
            refer_image=args.visual_prompt_image,
            predictor=visual_predictor,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            save=False,
        )
    else:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            save=False,
        )
    if not results:
        print("  No results returned")
        return {"image": str(image_path), "success": False, "error": "no results"}, []

    res = results[0]
    img_h, img_w = (res.orig_shape[0], res.orig_shape[1]) if hasattr(res, "orig_shape") else (None, None)
    if img_h is None:
        img = cv2.imread(str(image_path))
        if img is not None:
            img_h, img_w = img.shape[:2]
        else:
            img_h, img_w = None, None

    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4) in pixels
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [None] * len(boxes_xyxy)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else [0] * len(boxes_xyxy)
        # YOLOE with set_classes() populates res.names with our prompt(s).
        names = res.names if hasattr(res, "names") and res.names else {}

        # Optional segmentation polygons.
        segs = None
        if args.include_segmentation and res.masks is not None and len(res.masks) > 0:
            # masks.xy is a list of (N_i, 2) pixel-coordinate polygon arrays.
            segs = res.masks.xy if hasattr(res.masks, "xy") else None

        for i, (box, conf, cid) in enumerate(zip(boxes_xyxy, confs, cls_ids)):
            # In visual-prompt mode res.names is {0: 'object0'}; use the
            # user-supplied --visual-prompt-label instead. In text-prompt mode
            # res.names is the registered class list (our prompt(s)).
            if args.prompt_mode == "visual":
                label = args.visual_prompt_label
            else:
                label = names.get(int(cid), args.text_prompt or "object")
            det = {
                "label": label,
                "bbox": [float(v) for v in box.tolist()],
                "confidence": float(conf) if conf is not None else None,
            }
            if segs is not None and i < len(segs) and segs[i] is not None and len(segs[i]) > 0:
                det["segmentation"] = [[float(p[0]), float(p[1])] for p in segs[i]]
            detections.append(det)

    print(f"  Detected {len(detections)} object(s)")

    # Prepare per-image result JSON (mirror 07_run_qwen.py schema).
    output_data = {
        "image": str(image_path),
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "text_prompt": args.text_prompt if args.prompt_mode == "text" else None,
        "visual_prompt_image": args.visual_prompt_image if args.prompt_mode == "visual" else None,
        "visual_prompt_label": args.visual_prompt_label if args.prompt_mode == "visual" else None,
        "parsed_output": [
            {
                "label": d["label"],
                "bbox_2d": d["bbox"],
                **({"score": d["confidence"]} if d["confidence"] is not None else {}),
                **({"segmentation": d["segmentation"]} if "segmentation" in d else {}),
            }
            for d in detections
        ],
    }

    if output_dir:
        json_path = output_dir / f"{image_path.stem}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to: {json_path}")

    if vis_dir and detections:
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"
        if draw_visualization(image_path, vis_path, detections):
            print(f"  Visualization saved to: {vis_path}")

    return {
        "image": str(image_path),
        "success": True,
        "parsed_output": output_data["parsed_output"],
        "num_detections": len(detections),
    }, detections


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOE text-promptable segmentation on a folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yoloe-26l-seg.pt",
        help="Path or name of the YOLOE checkpoint (default: yoloe-26l-seg.pt).",
    )
    parser.add_argument(
        "--text-prompt", "-p",
        type=str,
        default=None,
        help="Open-vocabulary text prompt for text-prompt mode, e.g. "
             "'hanging overhead green exit sign with EXIT text'. "
             "Required when --prompt-mode=text (the default). A richer, "
             "descriptive phrase generally works better than a bare noun "
             "because YOLOE encodes prompts via MobileCLIP, which is trained "
             "on natural-language image captions.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["text", "visual"],
        default="text",
        help="Prompting mode: 'text' (default) uses an open-vocabulary text "
             "prompt via MobileCLIP embeddings; 'visual' uses a reference "
             "image + bounding box (visual prompt embeddings, VPE), which is "
             "typically much more accurate for domain-specific objects.",
    )
    parser.add_argument(
        "--visual-prompt-image",
        type=str,
        default=None,
        help="Path to the reference image for visual-prompt mode. The image "
             "should contain at least one instance of the target object; the "
             "bounding box is specified by --visual-prompt-bbox (or auto-set "
             "to the full image if the reference is already a tight crop).",
    )
    parser.add_argument(
        "--visual-prompt-bbox",
        type=int,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box of the target object in the reference image "
             "(pixel coordinates). If omitted, the full reference image is "
             "used as the prompt (appropriate when the reference is a tight "
             "crop of just the object).",
    )
    parser.add_argument(
        "--visual-prompt-label",
        type=str,
        default="object",
        help="Class label to assign to detections in visual-prompt mode "
             "(default: 'object'). Only affects output labeling, not the "
             "embeddings.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Folder of images to process (batch mode).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Directory for per-image result JSON files and summary.json.",
    )
    parser.add_argument(
        "--vis-output",
        type=str,
        default=None,
        help="Directory for visualization JPGs.",
    )
    parser.add_argument(
        "--annotations-output",
        type=str,
        default=None,
        help="Base directory for split-by-class annotation JSONs. "
             "Used only with --split-by-class.",
    )
    parser.add_argument(
        "--split-by-class",
        action="store_true",
        help="Split annotations by class into per-image/per-class JSON files, "
             "matching the layout produced by 07_run_qwen.py.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. '0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Resume from this image number (1-indexed) in batch mode. "
             "Images before this number are skipped.",
    )
    parser.add_argument(
        "--include-segmentation",
        action="store_true",
        help="If the model is a seg variant (e.g. yoloe-26l-seg.pt), include "
             "segmentation polygons in the per-image JSON and annotations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    folder_path = Path(args.image_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Image folder not found: {folder_path}")
        sys.exit(1)

    image_files = find_image_files(folder_path)
    if not image_files:
        print(f"Error: No image files found in {folder_path}")
        print(f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)

    total_images = len(image_files)
    print(f"Found {total_images} image(s) in {folder_path}")

    # Resume support (1-indexed, same semantics as 07_run_qwen.py)
    start_index = 0
    if args.resume_from is not None:
        start_index = max(0, args.resume_from - 1)
        if start_index > 0:
            print(f"Resuming from image {args.resume_from} (skipping first {start_index} images)")
            image_files = image_files[start_index:]
            print(f"Images to process: {len(image_files)}")

    # Setup output dirs
    output_dir = Path(args.output) if args.output else folder_path / "yoloe_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.vis_output:
        vis_dir = Path(args.vis_output)
    elif args.split_by_class or args.output:
        vis_dir = output_dir / "visualizations"
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = None
    if args.split_by_class:
        annotations_dir = Path(args.annotations_output) if args.annotations_output else output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotations output directory (split-by-class): {annotations_dir}")

    print(f"Output directory: {output_dir}")
    if vis_dir:
        print(f"Visualization directory: {vis_dir}")

    # Load YOLOE model
    print(f"\nLoading YOLOE model: {args.model}")
    try:
        from ultralytics import YOLOE
    except ImportError:
        print("Error: ultralytics is not installed or does not expose YOLOE.")
        print("Install/upgrade with: pip install -U ultralytics")
        sys.exit(1)

    model = YOLOE(args.model)
    if args.device:
        # ultralytics' .to() expects a torch device spec; normalize common
        # shorthand like '0' to 'cuda:0'.
        dev = args.device
        if dev.isdigit():
            dev = f"cuda:{dev}"
        model.to(dev)

    # Validate prompt-mode-specific args.
    if args.prompt_mode == "text":
        if not args.text_prompt:
            print("Error: --text-prompt is required when --prompt-mode=text")
            sys.exit(1)
    elif args.prompt_mode == "visual":
        if not args.visual_prompt_image:
            print("Error: --visual-prompt-image is required when --prompt-mode=visual")
            sys.exit(1)
        if not Path(args.visual_prompt_image).exists():
            print(f"Error: visual prompt image not found: {args.visual_prompt_image}")
            sys.exit(1)

    # Prepare prompt embeddings / predictor.
    visual_prompts = None
    visual_predictor = None
    if args.prompt_mode == "text":
        # YOLOE text-prompt path: compute text prompt embeddings (TPE) for the
        # supplied class names, then set_classes(names, tpe) so the detector
        # treats them as its vocabulary. A single class (the prompt) is used.
        class_names = [args.text_prompt]
        print(f"Setting classes: {class_names}")
        tpe = model.get_text_pe(class_names)
        model.set_classes(class_names, tpe)
    else:
        # Visual-prompt path: build the {"bboxes", "cls"} prompt dict from the
        # reference image + bbox. The VPE itself is computed inside
        # model.predict(refer_image=...) on the first call; we just supply the
        # bbox + class index here. The seg predictor is used because the
        # checkpoint is a -seg model; fall back to the detect predictor for
        # non-seg checkpoints.
        from ultralytics.models.yolo.yoloe.predict import (
            YOLOEVPDetectPredictor, YOLOEVPSegPredictor,
        )
        ref_path = Path(args.visual_prompt_image)
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"Error: could not read reference image: {ref_path}")
            sys.exit(1)
        ref_h, ref_w = ref_img.shape[:2]
        if args.visual_prompt_bbox:
            bbox = list(args.visual_prompt_bbox)
        else:
            # Auto-use the full reference image as the prompt (appropriate
            # when the reference is already a tight crop of the object).
            bbox = [0, 0, ref_w, ref_h]
        print(f"Visual prompt image: {ref_path}  bbox={bbox}  label='{args.visual_prompt_label}'")
        visual_prompts = {
            "bboxes": np.array([bbox]),
            "cls": np.array([0]),
        }
        # Prefer the seg predictor for -seg checkpoints; the task is inferred
        # from the model file name.
        visual_predictor = YOLOEVPSegPredictor if "seg" in args.model.lower() else YOLOEVPDetectPredictor

    # Process images
    print(f"\nRunning YOLOE batch inference...")
    print(f"  Model: {args.model}")
    print(f"  Prompt mode: {args.prompt_mode}")
    if args.prompt_mode == "text":
        print(f"  Text prompt: {args.text_prompt}")
    else:
        print(f"  Reference image: {args.visual_prompt_image}")
        print(f"  Reference bbox: {bbox}")
        print(f"  Class label: {args.visual_prompt_label}")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    results = []
    successful = 0
    failed = 0
    total_classes_found = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        result, detections = process_single_image(
            model, image_path, args, output_dir, vis_dir,
            visual_prompts=visual_prompts,
            visual_predictor=visual_predictor,
        )

        if args.split_by_class and result["success"] and detections:
            print(f"  Splitting annotations by class...")
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Warning: could not read image size, skipping split: {image_path}")
                result["annotation_files"] = {}
            else:
                img_h, img_w = img.shape[:2]
                saved_files = split_annotations_by_class(
                    detections, image_path, annotations_dir, img_w, img_h
                )
                result["annotation_files"] = saved_files
                total_classes_found += len(saved_files)

        results.append(result)
        if result["success"]:
            successful += 1
        else:
            failed += 1

    # Summary
    summary = {
        "folder": str(folder_path),
        "total_images": len(image_files),
        "successful": successful,
        "failed": failed,
        "model": args.model,
        "text_prompt": args.text_prompt,
        "split_by_class": args.split_by_class,
        "total_classes_found": total_classes_found,
        "results": results,
    }
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


if __name__ == "__main__":
    main()
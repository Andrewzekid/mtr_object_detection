#!/usr/bin/env python3
"""
Unified YOLO tracking script.

Supports multiple trackers and model tasks:
    --tracker bytetrack         ByteTrack (default for segmentation models)
    --tracker botsort           BoT-SORT with configurable CMC/ReID
    --tracker detect-then-sam3  YOLO detect + SAM3 segment per tracked box

Works with both YOLO detection and segmentation models; segmentation masks are
exported as COCO polygons when available.

USAGE:
    # ByteTrack on segmentation model
    python scripts/track.py --tracker bytetrack \\
        --model runs/segment/.../weights/best.pt \\
        --data MTR_dataset --output output/tracking/bytetrack

    # BoT-SORT on detection model with camera-motion compensation
    python scripts/track.py --tracker botsort \\
        --model runs/detect/.../weights/best.pt \\
        --data MTR_metacam_right --output output/tracking/botsort \\
        --with-cmc --cmc-method sparseOptFlow

    # YOLO detect + SAM3 segment
    python scripts/track.py --tracker detect-then-sam3 \\
        --yolo-model runs/detect/.../weights/best.pt \\
        --sam3-model core/sam3/models/sam3-model/sam3.pt \\
        --data MTR_dataset --output output/tracking/detect_then_sam3
"""

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Suppress noisy Ultralytics deprecation warnings for the still-functional
# ``half`` argument so benchmark output stays readable.
warnings.filterwarnings("ignore", message=".*half.*deprecated.*", category=FutureWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from scripts.tracking_utils import (
    find_ultralytics_trackers_dir,
    build_runtime_tracker_yaml,
    find_image_files,
    create_tracking_video,
    bbox_iou,
    mask_to_polygon,
    polygon_area,
)


# =============================================================================
# Standard Tracking (ByteTrack / BoT-SORT)
# =============================================================================

def process_frame(result, img_h, img_w, image_id, annotation_id, class_names):
    """Extract COCO annotations and annotated frame from a tracking result."""
    annotated_frame = result.plot() if result is not None else None
    annotations = []

    if (
        result is not None
        and result.boxes is not None
        and hasattr(result.boxes, "id")
        and result.boxes.id is not None
    ):
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        has_masks = (
            hasattr(result, "masks")
            and result.masks is not None
            and len(result.masks) > 0
        )

        for i in range(len(track_ids)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            bbox_width = float(x2 - x1)
            bbox_height = float(y2 - y1)
            bbox_coco = [float(x1), float(y1), bbox_width, bbox_height]
            area = bbox_width * bbox_height
            class_id = int(class_ids[i])
            track_id = int(track_ids[i])
            confidence = float(confidences[i])

            segmentation = []
            if has_masks and i < len(result.masks.data):
                mask = result.masks.data[i].cpu().numpy()
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                segmentation_polygons = []
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = contour.flatten().tolist()
                        segmentation_polygons.append(polygon)

                if segmentation_polygons:
                    segmentation = segmentation_polygons
                    area = float(np.sum(mask_binary))

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox_coco,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation,
                "track_id": track_id,
                "confidence": confidence,
            })
            annotation_id += 1

        unique_ids = np.unique(track_ids)
        info_text = f"Tracker: result.boxes.id | Objects: {len(unique_ids)} | Tracks: {len(track_ids)}"
        cv2.putText(
            annotated_frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return annotated_frame, annotations, annotation_id


def _build_tracker_yaml(args, output_dir: Path):
    """Return the path to a runtime tracker YAML if BoT-SORT is requested."""
    if args.tracker == "botsort":
        trackers_dir = find_ultralytics_trackers_dir()
        base_yaml = trackers_dir / "botsort.yaml"
        if not base_yaml.exists():
            print(f"Error: botsort.yaml not found at {base_yaml}")
            sys.exit(1)
        tracker_yaml = build_runtime_tracker_yaml(
            base_yaml,
            tracker_type="botsort",
            with_cmc=args.with_cmc,
            cmc_method=args.cmc_method,
            track_buffer=args.track_buffer,
            track_high_thresh=args.track_high_thresh,
            with_reid=args.with_reid,
            output_dir=output_dir,
        )
        print(f"BoT-SORT: cmc={args.with_cmc} ({args.cmc_method}), buffer={args.track_buffer}, reid={args.with_reid}")
        return tracker_yaml
    return None


def run_standard_tracking(args):
    """Run ByteTrack or BoT-SORT tracking."""
    model_path = Path(args.model)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        sys.exit(1)

    print(f"Tracker:   {args.tracker}")
    print(f"Model:     {model_path}")
    print(f"Data:      {data_path}")
    print(f"Output:    {output_dir}")
    print(f"conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    # Resolve tracker configuration
    tracker_yaml = _build_tracker_yaml(args, output_dir)

    # Load model
    model = YOLO(str(model_path))
    class_names = model.names if hasattr(model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    print(f"Found {len(image_files)} images to process")

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    valid_paths = []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Warning: Could not read {image_path}, skipping")
            continue
        valid_paths.append(image_path)

        image_id = idx + 1
        img_h, img_w = frame.shape[:2]
        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_w,
            "height": img_h,
        })

        track_kwargs = {
            "source": frame,
            "persist": True,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "verbose": False,
            "device": args.device,
        }
        if tracker_yaml is not None:
            track_kwargs["tracker"] = str(tracker_yaml)

        results = model.track(**track_kwargs)
        result = results[0] if results and len(results) > 0 else None

        annotated_frame, annotations, annotation_id = process_frame(
            result, img_h, img_w, image_id, annotation_id, class_names
        )
        coco_annotations.extend(annotations)

        if annotated_frame is None:
            annotated_frame = frame
        output_path = output_dir / f"tracked_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"  Saved frame {image_id}/{len(image_files)}: {output_path.name}")

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": f"{args.tracker.capitalize()} tracking results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Tracking JSON saved to: {json_path}")

    print(f"\nTracking complete! Results saved to: {output_dir}")
    print(f"Total images: {len(coco_images)}")
    print(f"Total annotations: {len(coco_annotations)}")
    if coco_annotations:
        unique_track_ids = {a["track_id"] for a in coco_annotations}
        print(f"Unique track IDs: {len(unique_track_ids)}")

    # Summary video
    create_tracking_video(output_dir, valid_paths, args.fps)


# =============================================================================
# Benchmark tracking + segmentation without saving outputs
# =============================================================================

def run_benchmark_tracking(args):
    """Run YOLO tracking+segmentation and only report timing/throughput.

    No images, videos, or JSON results are written. This is useful for
    measuring the inference speed of a model + tracker combination on a
    folder of frames.
    """
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("TRACKING + SEGMENTATION BENCHMARK")
    print("=" * 60)
    tracker_yaml = _build_tracker_yaml(args, data_path)

    print(f"Model:     {model_path}")
    print(f"Tracker:   {args.tracker}")
    print(f"Data:      {data_path}")
    print(f"conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print(f"Half precision: {args.half}")
    print(f"Skip masks: {args.no_masks}")
    print(f"Warmup frames: {args.warmup_frames}")
    print("Note: no images, videos, or JSON will be saved.\n")

    model = YOLO(str(model_path))

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[: args.max_frames]
    total = len(image_files)
    if total == 0:
        print("Error: no images found in data directory")
        sys.exit(1)
    print(f"Found {total} images to process")

    model_times = []
    result_times = []
    total_tracks = 0
    total_masks = 0
    processed = 0

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Warning: Could not read {image_path}, skipping")
            continue

        track_kwargs = {
            "source": frame,
            "persist": True,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "verbose": False,
            "device": args.device,
            "half": args.half,
        }
        if tracker_yaml is not None:
            track_kwargs["tracker"] = str(tracker_yaml)

        # Time the YOLO model + tracker inference.
        t0 = time.perf_counter()
        results = model.track(**track_kwargs)
        t1 = time.perf_counter()

        # Time the post-processing of the tracking result.
        result = results[0] if results and len(results) > 0 else None
        if result is not None and result.boxes is not None and hasattr(result.boxes, "id"):
            track_ids = result.boxes.id
            if track_ids is not None:
                n_tracks = len(track_ids)
                total_tracks += n_tracks
                if not args.no_masks and hasattr(result, "masks") and result.masks is not None:
                    total_masks += len(result.masks.data)
        t2 = time.perf_counter()

        # Discard warmup frames from timing stats.
        if idx >= args.warmup_frames:
            model_times.append(t1 - t0)
            result_times.append(t2 - t1)

        processed += 1
        if processed % 50 == 0 or processed == total:
            print(f"  processed {processed}/{total} frames ...")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total images processed: {processed}")
    print(f"Total tracked objects:  {total_tracks}")
    print(f"Total segmentation masks: {total_masks}")

    def _print_stats(name, times):
        n = len(times)
        if n == 0:
            print(f"\n{name}: no frames timed (warmup covered all frames)")
            return
        mean_s = sum(times) / n
        min_s = min(times)
        max_s = max(times)
        p50 = float(np.percentile(times, 50))
        p95 = float(np.percentile(times, 95))
        p99 = float(np.percentile(times, 99))
        print(f"\n{name} (excluding first {args.warmup_frames} warmup frames):")
        print(f"  Frames timed:     {n}")
        print(f"  Mean per frame:   {mean_s * 1000:.2f} ms ({1.0 / mean_s:.2f} FPS)")
        print(f"  Min per frame:    {min_s * 1000:.2f} ms")
        print(f"  Max per frame:    {max_s * 1000:.2f} ms")
        print(f"  Median (p50):     {p50 * 1000:.2f} ms")
        print(f"  p95 per frame:    {p95 * 1000:.2f} ms")
        print(f"  p99 per frame:    {p99 * 1000:.2f} ms")
        print(f"  Total wall time:  {sum(times):.2f} s")

    _print_stats("YOLO model + tracker time", model_times)
    _print_stats("Tracking result processing time", result_times)

    if model_times:
        # Combined per-frame time = model + result post-processing.
        combined = [m + r for m, r in zip(model_times, result_times)]
        _print_stats("Combined per-frame time", combined)

    print("=" * 60)


# =============================================================================
# Detect-then-SAM3 Tracking
# =============================================================================

def match_sam3_outputs_to_boxes(sam_detections, sam_masks, input_boxes):
    """Match SAM3 output detections/masks back to the input boxes by bbox IoU."""
    if not sam_detections:
        return [(None, None) for _ in input_boxes]

    matches = []
    used = set()
    for in_box in input_boxes:
        best_idx = -1
        best_iou = 0.0
        for idx, det in enumerate(sam_detections):
            if idx in used:
                continue
            det_box = det.get("bbox")
            if det_box is None:
                continue
            iou = bbox_iou(in_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou > 0.1:
            used.add(best_idx)
            mask = sam_masks[best_idx] if best_idx < len(sam_masks) else None
            matches.append((sam_detections[best_idx], mask))
        else:
            matches.append((None, None))
    return matches


def run_sam3_for_boxes(sam_model, image_path, boxes, class_id, class_name, sam_conf, device):
    """Run SAM3 once for a group of same-class boxes and return matched masks."""
    if not boxes:
        return []

    predict_kwargs = {
        "source": str(image_path),
        "task": "segment",
        "verbose": False,
        "conf": sam_conf,
        "device": device,
        "save": False,
        "bboxes": boxes,
        "labels": [class_id] * len(boxes),
        "imgsz": 1024,
    }

    results = sam_model.predict(**predict_kwargs)
    if not isinstance(results, list):
        results = [results]
    result = results[0]

    detections = []
    masks = []

    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else np.ones(len(boxes_xyxy))
        for box, conf in zip(boxes_xyxy, confs):
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "label": class_name,
                "confidence": float(conf),
            })

    if hasattr(result, "masks") and result.masks is not None:
        mask_data = result.masks.data
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()
        else:
            mask_data = np.asarray(mask_data)

        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        for i in range(mask_data.shape[0]):
            m = mask_data[i].astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks.append(m)

    matches = match_sam3_outputs_to_boxes(detections, masks, boxes)
    return matches


def draw_detect_sam3_overlay(image, tracked_objects, matched_masks):
    """Draw masks, bounding boxes, track IDs, and class labels."""
    overlay = image.copy()
    np.random.seed(42)

    for obj, mask in zip(tracked_objects, matched_masks):
        track_id = obj["track_id"]
        class_name = obj["class_name"]
        x1, y1, x2, y2 = obj["bbox"]

        # deterministic color per track ID
        color = tuple(int(c) for c in np.random.RandomState(track_id).randint(80, 255, size=3))

        if mask is not None:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name} ID:{track_id}"
        cv2.putText(overlay, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return overlay


def run_detect_then_sam3(args):
    """Run YOLO detect-track + SAM3 segmentation pipeline."""
    from ultralytics import SAM

    project_dir = Path(__file__).parent.parent
    yolo_model_path = Path(args.yolo_model)
    sam3_model_path = Path(args.sam3_model)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not yolo_model_path.exists():
        print(f"Error: YOLO model not found at {yolo_model_path}")
        sys.exit(1)
    if not sam3_model_path.exists():
        print(f"Error: SAM3 model not found at {sam3_model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        sys.exit(1)

    print(f"YOLO model: {yolo_model_path}")
    print(f"SAM3 model: {sam3_model_path}")
    print(f"Data:       {data_path}")
    print(f"Output:     {output_dir}")
    print(f"YOLO conf={args.conf}, IoU={args.iou}, imgsz={args.imgsz}")
    print(f"SAM3 conf={args.sam3_conf}, device={args.sam3_device}")

    # Build runtime BoT-SORT yaml
    trackers_dir = find_ultralytics_trackers_dir()
    base_yaml = trackers_dir / "botsort.yaml"
    if not base_yaml.exists():
        print(f"Error: botsort.yaml not found at {base_yaml}")
        sys.exit(1)
    tracker_yaml = build_runtime_tracker_yaml(
        base_yaml,
        tracker_type="botsort",
        with_cmc=args.with_cmc,
        cmc_method=args.cmc_method,
        track_buffer=args.track_buffer,
        track_high_thresh=args.track_high_thresh,
        with_reid=args.with_reid,
        output_dir=output_dir,
    )

    # Load models
    print("Loading YOLO detection model...")
    yolo_model = YOLO(str(yolo_model_path))

    print("Loading SAM3 model...")
    sam_model = SAM(str(sam3_model_path))

    # Image list
    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    print(f"Found {len(image_files)} images to process")

    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Warning: could not read {image_path}, skipping")
            continue

        img_h, img_w = frame.shape[:2]
        image_id = idx + 1
        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_w,
            "height": img_h,
        })

        # 1) YOLO tracking on a single frame
        track_results = yolo_model.track(
            source=frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=str(tracker_yaml),
            verbose=False,
            device=args.yolo_device,
        )
        result = track_results[0] if track_results and len(track_results) > 0 else None

        tracked_objects = []
        if result is not None and result.boxes is not None and hasattr(result.boxes, "id") and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(track_ids)):
                tracked_objects.append({
                    "track_id": int(track_ids[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": class_names.get(int(class_ids[i]), f"class_{int(class_ids[i])}"),
                    "bbox": [float(v) for v in boxes_xyxy[i]],
                    "confidence": float(confidences[i]),
                })

        # 2) Group tracked boxes by class and run SAM3 once per class
        masks_per_object = [None] * len(tracked_objects)
        if tracked_objects:
            by_class = defaultdict(list)
            for obj_idx, obj in enumerate(tracked_objects):
                by_class[obj["class_id"]].append((obj_idx, obj))

            for class_id, obj_items in by_class.items():
                obj_indices = [idx for idx, _ in obj_items]
                boxes = [obj["bbox"] for _, obj in obj_items]
                class_name = class_names.get(class_id, f"class_{class_id}")

                matches = run_sam3_for_boxes(
                    sam_model, image_path, boxes, class_id, class_name,
                    args.sam3_conf, args.sam3_device,
                )

                for obj_idx, (det, mask) in zip(obj_indices, matches):
                    masks_per_object[obj_idx] = mask

        # 3) Build annotations and visualization
        overlay = draw_detect_sam3_overlay(frame, tracked_objects, masks_per_object)

        for obj, mask in zip(tracked_objects, masks_per_object):
            x1, y1, x2, y2 = obj["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            segmentation = []

            if mask is not None:
                poly = mask_to_polygon(mask)
                if poly:
                    segmentation = [poly]
                    area = polygon_area(poly)

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["class_id"],
                "bbox": [x1, y1, w, h],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": segmentation,
                "track_id": obj["track_id"],
                "confidence": obj["confidence"],
            }
            coco_annotations.append(annotation)
            annotation_id += 1

        output_path = output_dir / f"tracked_{image_path.name}"
        cv2.imwrite(str(output_path), overlay)
        print(f"  Frame {image_id}/{len(image_files)}: {len(tracked_objects)} tracks, "
              f"{sum(1 for m in masks_per_object if m is not None)} masks -> {output_path.name}")

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": "YOLO track + SAM3 per-box segmentation results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"\nTracking JSON saved to: {json_path}")

    print(f"Total images: {len(coco_images)}")
    print(f"Total annotations: {len(coco_annotations)}")
    if coco_annotations:
        unique_track_ids = {a["track_id"] for a in coco_annotations}
        print(f"Unique track IDs: {len(unique_track_ids)}")
        masks_count = sum(1 for a in coco_annotations if a["segmentation"])
        print(f"Annotations with segmentation masks: {masks_count}")

    # Summary video
    create_tracking_video(output_dir, image_files, args.fps)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Unified YOLO tracking script (ByteTrack / BoT-SORT / Detect-then-SAM3)"
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort", "detect-then-sam3", "benchmark"],
        help="Tracker to use (default: bytetrack). Use 'benchmark' to run YOLO tracking+segmentation without saving any outputs and report timing.",
    )

    # Benchmark tracking arguments
    parser.add_argument(
        "--benchmark-tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Underlying tracker used when --tracker benchmark (default: bytetrack)",
    )

    # Standard tracking arguments
    parser.add_argument(
        "--model",
        type=str,
        default=str(project_dir / "runs" / "segment" / "output" / "training" / "yolo_training" / "weights" / "best.pt"),
        help="Path to trained YOLO model weights (for bytetrack/botsort)",
    )

    # Detect-then-SAM3 arguments
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=str(project_dir / "runs" / "detect" / "output" / "training" / "mtr_detection_yolo26l" / "weights" / "best.pt"),
        help="Path to YOLO detection model (for detect-then-sam3)",
    )
    parser.add_argument(
        "--sam3-model",
        type=str,
        default=str(project_dir / "core" / "sam3" / "models" / "sam3-model" / "sam3.pt"),
        help="Path to SAM3 weights (for detect-then-sam3)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(project_dir / "MTR_dataset"),
        help="Directory containing images to track",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_dir / "output" / "tracking" / "bytetrack"),
        help="Directory to save tracking results",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=3,
        help="Number of frames to discard before benchmarking (default: 3).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Inference device: cuda, cpu, or auto")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half-precision inference (faster on CUDA, default: False)",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="In benchmark mode, skip reading segmentation masks from results (default: False)",
    )

    # BoT-SORT knobs
    parser.add_argument("--with-cmc", dest="with_cmc", action="store_true", default=True)
    parser.add_argument("--no-cmc", dest="with_cmc", action="store_false")
    parser.add_argument(
        "--cmc-method",
        type=str,
        default="sparseOptFlow",
        choices=["sparseOptFlow", "orb", "sift", "ecc", "none"],
        help="Global motion compensation method (BoT-SORT only)",
    )
    parser.add_argument("--track-buffer", type=int, default=30, help="Lost-track buffer")
    parser.add_argument("--track-high-thresh", type=float, default=0.5, help="First-stage association threshold")
    parser.add_argument("--with-reid", action="store_true", default=False, help="Enable ReID (BoT-SORT only)")

    # SAM3 knobs (for detect-then-sam3)
    parser.add_argument("--sam3-conf", type=float, default=0.25, help="SAM3 mask confidence threshold")
    parser.add_argument("--sam3-device", type=str, default="cuda", help="Device for SAM3")
    parser.add_argument("--yolo-device", type=str, default="cuda", help="Device for YOLO")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.tracker == "detect-then-sam3":
        print("Arguments:", args)
        run_detect_then_sam3(args)
    elif args.tracker == "benchmark":
        # Benchmark mode ignores the 'benchmark' tracker name and uses the
        # user-specified underlying tracker (bytetrack or botsort).
        args.tracker = args.benchmark_tracker
        run_benchmark_tracking(args)
    else:
        print("Arguments:", args)
        run_standard_tracking(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Unified SAM3VideoPredictor tracking/segmentation pipeline.

Provides three operating modes:
    --mode single     YOLO seed on frame 0 + SAM3 propagation through the whole video.
    --mode reseed     YOLO seed + periodic re-detection to add newly appearing objects.
    --mode chunks     Overlapping chunks, each seeded independently, with cross-chunk ID stitching.

USAGE:
    # Single-seed mode
    python scripts/track_sam3_video.py --mode single \\
        --yolo-model runs/detect/.../weights/best.pt \\
        --sam3-model core/sam3/models/sam3-model/sam3.pt \\
        --data MTR_metacam_right --output output/tracking/sam3_video

    # Periodic re-seed mode
    python scripts/track_sam3_video.py --mode reseed \\
        --data MTR_undistorted_right --output output/tracking/sam3_video_reseed \\
        --reseed-every 15

    # Overlapping-chunk mode
    python scripts/track_sam3_video.py --mode chunks \\
        --data MTR_undistorted_right --output output/tracking/sam3_video_chunks \\
        --chunk-size 30 --chunk-step 15
"""

import argparse
import gc
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import SAM3VideoPredictor

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tracking_utils import (
    bbox_iou,
    find_image_files,
    mask_to_polygon,
    polygon_area,
    create_tracking_video,
)


# =============================================================================
# Shared Utilities
# =============================================================================

def images_to_video(image_paths, video_path, fps):
    """Convert sorted image list to MP4."""
    if not image_paths:
        return None
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in image_paths:
        f = cv2.imread(str(p))
        if f is not None:
            out.write(f)
    out.release()
    return video_path


def mask_to_bbox(mask):
    """Tight bbox from binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def draw_overlay_single(image, boxes, masks, class_ids, class_names):
    """Draw colored masks, bounding boxes, labels, and object IDs (single mode)."""
    overlay = image.copy()
    np.random.seed(42)

    for obj_idx, (box, mask, cls_id) in enumerate(zip(boxes, masks, class_ids)):
        color = tuple(int(c) for c in np.random.RandomState(obj_idx + 1).randint(80, 255, size=3))
        x1, y1, x2, y2 = [int(v) for v in box]

        if mask is not None:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        label = f"{cls_name} ID:{obj_idx + 1}"
        cv2.putText(overlay, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return overlay


def draw_overlay_reseed(image, bboxes, masks, class_ids, class_names, track_ids):
    """Draw masks, boxes, labels, and track IDs (reseed mode)."""
    overlay = image.copy()
    np.random.seed(42)
    for obj_idx, (box, mask, cls_id, tid) in enumerate(zip(bboxes, masks, class_ids, track_ids)):
        color = tuple(int(c) for c in np.random.RandomState(int(tid) + 1).randint(80, 255, size=3))
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names.get(cls_id, f'class_{cls_id}')} ID:{tid}"
            cv2.putText(overlay, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if mask is not None:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    return overlay


def draw_overlay_chunks(image, objects):
    """Draw masks, boxes, labels, and IDs (chunks mode)."""
    overlay = image.copy()
    np.random.seed(42)
    for obj in objects:
        tid = int(obj["track_id"])
        cls_id = obj["class_id"]
        color = tuple(int(c) for c in np.random.RandomState(tid + 1).randint(80, 255, size=3))
        box = obj["bbox"]
        mask = obj["mask"]
        if mask is not None:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid}"
            cv2.putText(overlay, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return overlay


def run_yolo_on_frame(yolo_model, frame, conf, iou, imgsz, device):
    """Run YOLO detection on a single frame."""
    result = yolo_model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        cls = result.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = result.boxes.conf.cpu().tolist()
        for b, c, cf in zip(boxes, cls, confs):
            detections.append({"bbox": b, "class_id": c, "confidence": cf})
    return detections


# =============================================================================
# Mode: single - YOLO seed on frame 0 + SAM3 propagation
# =============================================================================

def run_single_mode(args):
    """Run single-seed tracking mode."""
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

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    print(f"Found {len(image_files)} images to process")

    if not image_files:
        print("No images found")
        sys.exit(1)

    # Create input video
    if args.temp_video:
        video_path = Path(args.temp_video)
    else:
        video_path = output_dir / "_temp_input_video.mp4"
    video_path = images_to_video(image_files, video_path, args.fps)
    if video_path is None:
        print("Failed to create input video")
        sys.exit(1)
    print(f"Input video: {video_path}")

    print("Loading YOLO detection model...")
    yolo_model = YOLO(str(yolo_model_path))
    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    print("Running YOLO on frame 0 to seed objects...")
    frame0 = cv2.imread(str(image_files[0]))
    yolo_results = yolo_model.predict(
        source=frame0,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.yolo_device,
        verbose=False,
    )[0]

    if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
        print("No detections on frame 0; nothing to track.")
        sys.exit(1)

    seed_boxes = yolo_results.boxes.xyxy.cpu().numpy().tolist()
    seed_class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int).tolist()
    seed_confidences = yolo_results.boxes.conf.cpu().tolist()

    print(f"Seeded {len(seed_boxes)} objects from frame 0")

    print("Initializing SAM3VideoPredictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=str(sam3_model_path),
        device=args.sam3_device,
        imgsz=args.sam3_imgsz,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)

    print("Propagating seeded objects through video...")
    results = predictor(source=str(video_path), bboxes=seed_boxes, stream=True)

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for frame_idx, (image_path, result) in enumerate(zip(image_files, results)):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        img_h, img_w = frame.shape[:2]
        coco_images.append({
            "id": frame_idx + 1,
            "file_name": image_path.name,
            "width": img_w,
            "height": img_h,
        })

        num_objects = 0
        if result.masks is not None and len(result.masks) > 0:
            mask_data = result.masks.data
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu().numpy()
            else:
                mask_data = np.asarray(mask_data)

            masks = [mask_data[i].astype(bool) for i in range(mask_data.shape[0])]
            boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []

            while len(boxes) < len(masks):
                boxes.append(None)

            overlay = draw_overlay_single(frame, boxes, masks, seed_class_ids, class_names)

            for obj_idx, (mask, box) in enumerate(zip(masks, boxes)):
                track_id = obj_idx + 1
                cls_id = seed_class_ids[obj_idx] if obj_idx < len(seed_class_ids) else 0
                conf = seed_confidences[obj_idx] if obj_idx < len(seed_confidences) else 1.0

                if box is None:
                    ys, xs = np.where(mask)
                    if len(xs) == 0:
                        continue
                    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

                x1, y1, x2, y2 = box
                w = float(x2 - x1)
                h = float(y2 - y1)
                area = w * h
                segmentation = []

                poly = mask_to_polygon(mask)
                if poly:
                    segmentation = [poly]
                    area = polygon_area(poly)

                annotation = {
                    "id": annotation_id,
                    "image_id": frame_idx + 1,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "area": float(area),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "track_id": int(track_id),
                    "confidence": float(conf),
                }
                coco_annotations.append(annotation)
                annotation_id += 1
                num_objects += 1
        else:
            overlay = frame.copy()

        output_path = output_dir / f"tracked_{image_path.name}"
        cv2.imwrite(str(output_path), overlay)
        print(f"  Frame {frame_idx + 1}/{len(image_files)}: {num_objects} objects -> {output_path.name}")

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": "YOLO seed + SAM3VideoPredictor tracking results",
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
        unique_ids = {a["track_id"] for a in coco_annotations}
        masked = sum(1 for a in coco_annotations if a["segmentation"])
        print(f"Unique track IDs: {len(unique_ids)}")
        print(f"Annotations with masks: {masked}")

    create_tracking_video(output_dir, image_files, args.fps)

    if args.temp_video is None:
        try:
            video_path.unlink()
            print(f"Cleaned up temporary video: {video_path}")
        except Exception as e:
            print(f"Could not remove temp video {video_path}: {e}")


# =============================================================================
# Mode: reseed - YOLO seed + periodic re-detection
# =============================================================================

def get_mask_for_new_object(predictor, im, box):
    """Run single-frame SAM3 inference to get a mask from a YOLO box."""
    masks, _ = predictor.inference(im, bboxes=[box])
    high_res = masks[0].cpu().numpy().astype(bool)
    lr_h, lr_w = predictor._bb_feat_sizes[0]
    low_res = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(),
        size=(lr_h, lr_w),
        mode="bilinear",
        align_corners=False,
    )
    low_res = (low_res > 0.5).float()
    return high_res, low_res


def run_reseed_mode(args):
    """Run periodic re-seed tracking mode."""
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
    print(f"Re-seed every {args.reseed_every} frames, match IoU >= {args.match_iou}")

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    print(f"Found {len(image_files)} images to process")

    if not image_files:
        print("No images found")
        sys.exit(1)

    # Create input video
    if args.temp_video:
        video_path = Path(args.temp_video)
    else:
        video_path = output_dir / "_temp_input_video.mp4"
    video_path = images_to_video(image_files, video_path, args.fps)
    if video_path is None:
        print("Failed to create input video")
        sys.exit(1)
    print(f"Input video: {video_path}")

    print("Loading YOLO detection model...")
    yolo_model = YOLO(str(yolo_model_path))
    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    print("Loading SAM3VideoPredictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=str(sam3_model_path),
        device=args.sam3_device,
        imgsz=args.sam3_imgsz,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)
    predictor.setup_model()
    predictor.setup_source(str(video_path))
    predictor.init_state(predictor)
    num_frames = predictor.dataset.frames
    print(f"SAM3 initialized for {num_frames} frames")

    def get_frame_batch(frame_idx):
        predictor.dataset.frame = frame_idx
        for batch in predictor.dataset:
            return batch
        return None

    print("Seeding objects on frame 0...")
    batch0 = get_frame_batch(0)
    predictor.batch = batch0
    im0 = predictor.preprocess(batch0[1])

    seed_detections = run_yolo_on_frame(
        yolo_model, batch0[1][0], args.conf, args.iou, args.imgsz, args.yolo_device
    )
    if not seed_detections:
        print("No detections on frame 0; nothing to track.")
        sys.exit(1)

    object_meta = {}
    next_obj_id = 0

    seed_boxes = [d["bbox"] for d in seed_detections]
    seed_masks, _ = predictor.inference(im0, bboxes=seed_boxes)
    for i, det in enumerate(seed_detections):
        obj_id = next_obj_id
        next_obj_id += 1
        object_meta[obj_id] = {
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "active": True,
        }
        lr_h, lr_w = predictor._bb_feat_sizes[0]
        low_res = torch.nn.functional.interpolate(
            seed_masks[[i]].unsqueeze(1).float(),
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        )
        low_res = (low_res > 0.5).float()
        predictor.add_new_prompts(obj_id=obj_id, masks=low_res, frame_idx=0)

    print(f"Seeded {len(seed_detections)} objects")

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for frame_idx in range(num_frames):
        try:
            obj_ids, pred_masks, obj_scores = predictor.propagate_in_video(
                predictor.inference_state, frame_idx
            )
        except Exception as e:
            print(f"  Warning: propagation failed at frame {frame_idx}: {e}")
            continue

        masks_np = pred_masks.cpu().numpy() if hasattr(pred_masks, "cpu") else np.asarray(pred_masks)
        frame_h, frame_w = batch0[1][0].shape[:2]

        active_objects = []
        for i, obj_id in enumerate(obj_ids):
            mask = masks_np[i].astype(bool)
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            box = mask_to_bbox(mask)
            active_objects.append({
                "obj_id": obj_id,
                "mask": mask,
                "bbox": box,
                "class_id": object_meta[obj_id]["class_id"],
                "confidence": object_meta[obj_id]["confidence"],
            })

        # Re-seed periodically
        if frame_idx > 0 and frame_idx % args.reseed_every == 0:
            print(f"  Re-seeding at frame {frame_idx}...")
            batch = get_frame_batch(frame_idx)
            frame_img = batch[1][0]
            im = predictor.preprocess(batch[1])
            new_dets = run_yolo_on_frame(
                yolo_model, frame_img, args.conf, args.iou, args.imgsz, args.yolo_device
            )

            for det in new_dets:
                det_box = det["bbox"]
                best_iou = 0.0
                best_obj = None
                for obj in active_objects:
                    if obj["bbox"] is None:
                        continue
                    iou = bbox_iou(det_box, obj["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_obj = obj

                if best_iou < args.match_iou:
                    obj_id = next_obj_id
                    next_obj_id += 1
                    object_meta[obj_id] = {
                        "class_id": det["class_id"],
                        "confidence": det["confidence"],
                        "active": True,
                    }
                    high_res_mask, low_res_mask = get_mask_for_new_object(predictor, im, det_box)
                    low_res_mask = low_res_mask[[0]]
                    predictor.add_new_prompts(obj_id=obj_id, masks=low_res_mask, frame_idx=frame_idx)
                    print(f"    Added new object {obj_id}: {class_names.get(det['class_id'], det['class_id'])}")

            obj_ids, pred_masks, obj_scores = predictor.propagate_in_video(
                predictor.inference_state, frame_idx
            )
            masks_np = pred_masks.cpu().numpy() if hasattr(pred_masks, "cpu") else np.asarray(pred_masks)
            active_objects = []
            for i, obj_id in enumerate(obj_ids):
                mask = masks_np[i].astype(bool)
                if mask.shape != (frame_h, frame_w):
                    mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                box = mask_to_bbox(mask)
                active_objects.append({
                    "obj_id": obj_id,
                    "mask": mask,
                    "bbox": box,
                    "class_id": object_meta[obj_id]["class_id"],
                    "confidence": object_meta[obj_id]["confidence"],
                })

        frame_img = batch0[1][0] if frame_idx == 0 else get_frame_batch(frame_idx)[1][0]
        coco_images.append({
            "id": frame_idx + 1,
            "file_name": image_files[frame_idx].name,
            "width": frame_w,
            "height": frame_h,
        })

        track_ids = [obj["obj_id"] + 1 for obj in active_objects]
        class_ids = [obj["class_id"] for obj in active_objects]
        boxes = [obj["bbox"] for obj in active_objects]
        masks = [obj["mask"] for obj in active_objects]
        overlay = draw_overlay_reseed(frame_img, boxes, masks, class_ids, class_names, track_ids)

        for obj, tid in zip(active_objects, track_ids):
            box = obj["bbox"]
            mask = obj["mask"]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            segmentation = []
            poly = mask_to_polygon(mask)
            if poly:
                segmentation = [poly]
                area = polygon_area(poly)

            annotation = {
                "id": annotation_id,
                "image_id": frame_idx + 1,
                "category_id": obj["class_id"],
                "bbox": [x1, y1, w, h],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": segmentation,
                "track_id": int(tid),
                "confidence": obj["confidence"],
            }
            coco_annotations.append(annotation)
            annotation_id += 1

        output_path = output_dir / f"tracked_{image_files[frame_idx].name}"
        cv2.imwrite(str(output_path), overlay)
        print(f"  Frame {frame_idx + 1}/{num_frames}: {len(active_objects)} objects -> {output_path.name}")

    # Save COCO JSON
    coco_output = {
        "info": {
            "description": "YOLO periodic seed + SAM3VideoPredictor reseed tracking results",
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
        unique_ids = {a["track_id"] for a in coco_annotations}
        masked = sum(1 for a in coco_annotations if a["segmentation"])
        print(f"Unique track IDs: {len(unique_ids)}")
        print(f"Annotations with masks: {masked}")

    create_tracking_video(output_dir, image_files, args.fps)

    if args.temp_video is None:
        try:
            video_path.unlink()
            print(f"Cleaned up temporary video: {video_path}")
        except Exception as e:
            print(f"Could not remove temp video {video_path}: {e}")


# =============================================================================
# Mode: chunks - Overlapping chunks with cross-chunk ID stitching
# =============================================================================

def process_chunk(yolo_model, sam3_model_path, image_files, start_idx, chunk_size, args):
    """Process one chunk and return {frame_global_idx: list of object dicts}."""
    chunk_files = image_files[start_idx:start_idx + chunk_size]
    if not chunk_files:
        return {}, []

    temp_dir = Path(args.output) / "_temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / f"chunk_{start_idx}_{start_idx + len(chunk_files) - 1}.mp4"
    images_to_video(chunk_files, video_path, args.fps)

    first_frame = cv2.imread(str(chunk_files[0]))
    seed_dets = run_yolo_on_frame(yolo_model, first_frame, args.conf, args.iou, args.imgsz, args.yolo_device)
    seed_boxes = [d["bbox"] for d in seed_dets]

    if not seed_boxes:
        print(f"  Chunk [{start_idx}, {start_idx + len(chunk_files) - 1}]: no seed detections")
        return {}, []

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=str(sam3_model_path),
        device=args.sam3_device,
        imgsz=args.sam3_imgsz,
        quantize=16,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)
    results = predictor(source=str(video_path), bboxes=seed_boxes, stream=True)

    chunk_results = {}
    for local_frame_idx, result in enumerate(results):
        global_frame_idx = start_idx + local_frame_idx
        frame = cv2.imread(str(chunk_files[local_frame_idx]))
        if frame is None:
            continue
        frame_h, frame_w = frame.shape[:2]

        objects = []
        if result.masks is not None and len(result.masks) > 0:
            mask_data = result.masks.data
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu().numpy()
            else:
                mask_data = np.asarray(mask_data)

            for obj_idx in range(mask_data.shape[0]):
                mask = mask_data[obj_idx].astype(bool)
                if mask.shape != (frame_h, frame_w):
                    mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                box = mask_to_bbox(mask)
                det = seed_dets[obj_idx] if obj_idx < len(seed_dets) else seed_dets[-1]
                objects.append({
                    "local_id": obj_idx,
                    "bbox": box,
                    "mask": mask,
                    "class_id": det["class_id"],
                    "confidence": det["confidence"],
                    "frame_idx": global_frame_idx,
                })

        chunk_results[global_frame_idx] = objects

    del predictor
    del results
    torch.cuda.empty_cache()
    gc.collect()

    return chunk_results, seed_dets


def run_chunks_mode(args):
    """Run overlapping-chunks tracking mode."""
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
    print(f"Chunk size: {args.chunk_size}, chunk step: {args.chunk_step}")

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    total_frames = len(image_files)
    print(f"Found {total_frames} images to process")

    print("Loading YOLO detection model...")
    yolo_model = YOLO(str(yolo_model_path))
    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    chunk_starts = list(range(0, total_frames, args.chunk_step))
    print(f"Number of chunks: {len(chunk_starts)}")

    global_results = {}
    next_global_id = 1

    for chunk_idx, start_idx in enumerate(chunk_starts):
        end_idx = min(start_idx + args.chunk_size, total_frames) - 1
        print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunk_starts)}: frames {start_idx}-{end_idx}")

        chunk_results, seed_dets = process_chunk(
            yolo_model, sam3_model_path, image_files, start_idx, args.chunk_size, args
        )

        if not chunk_results:
            print(f"  Chunk {chunk_idx + 1} produced no results; skipping")
            continue

        local_to_global = {}

        if chunk_idx == 0:
            for obj_idx in range(len(seed_dets)):
                local_to_global[obj_idx] = next_global_id
                next_global_id += 1
        else:
            match_frame = start_idx
            if match_frame in global_results and match_frame in chunk_results:
                prev_objects = global_results[match_frame]
                curr_objects = chunk_results[match_frame]

                candidates = []
                for curr_obj in curr_objects:
                    curr_box = curr_obj["bbox"]
                    if curr_box is None:
                        continue
                    best_iou = 0.0
                    best_prev_track_id = None
                    for prev_obj in prev_objects:
                        prev_box = prev_obj["bbox"]
                        if prev_box is None:
                            continue
                        iou = bbox_iou(curr_box, prev_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_prev_track_id = prev_obj["track_id"]
                    candidates.append((curr_obj["local_id"], best_iou, best_prev_track_id))

                candidates.sort(key=lambda x: x[1], reverse=True)

                used_prev = set()
                match_count = 0
                new_id_count = 0
                for curr_local_id, best_iou, best_prev_track_id in candidates:
                    if best_iou >= args.match_iou and best_prev_track_id not in used_prev:
                        local_to_global[curr_local_id] = best_prev_track_id
                        used_prev.add(best_prev_track_id)
                        match_count += 1
                    else:
                        local_to_global[curr_local_id] = next_global_id
                        next_global_id += 1
                        new_id_count += 1

                orphan_count = len(seed_dets) - len(local_to_global)
                print(f"  Cross-chunk ID match: {match_count} matched, "
                      f"{new_id_count} new IDs, {orphan_count} orphans")

                for obj_idx in range(len(seed_dets)):
                    if obj_idx not in local_to_global:
                        local_to_global[obj_idx] = next_global_id
                        next_global_id += 1
            else:
                for obj_idx in range(len(seed_dets)):
                    local_to_global[obj_idx] = next_global_id
                    next_global_id += 1

        for frame_idx, objects in chunk_results.items():
            global_objects = []
            for obj in objects:
                if obj["local_id"] not in local_to_global:
                    continue
                obj["track_id"] = local_to_global[obj["local_id"]]
                global_objects.append(obj)
            global_results[frame_idx] = global_objects

    # Build COCO output
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for frame_idx in sorted(global_results.keys()):
        image_path = image_files[frame_idx]
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        frame_h, frame_w = frame.shape[:2]

        coco_images.append({
            "id": frame_idx + 1,
            "file_name": image_path.name,
            "width": frame_w,
            "height": frame_h,
        })

        objects = global_results[frame_idx]
        overlay = draw_overlay_chunks(frame, objects)

        for obj in objects:
            box = obj["bbox"]
            mask = obj["mask"]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h
            segmentation = []
            poly = mask_to_polygon(mask)
            if poly:
                segmentation = [poly]
                area = polygon_area(poly)

            annotation = {
                "id": annotation_id,
                "image_id": frame_idx + 1,
                "category_id": obj["class_id"],
                "bbox": [x1, y1, w, h],
                "area": float(area),
                "iscrowd": 0,
                "segmentation": segmentation,
                "track_id": int(obj["track_id"]),
                "confidence": obj["confidence"],
            }
            coco_annotations.append(annotation)
            annotation_id += 1

        output_path = output_dir / f"tracked_{image_path.name}"
        cv2.imwrite(str(output_path), overlay)
        print(f"  Saved frame {frame_idx + 1}/{total_frames}: {len(objects)} objects")

    coco_output = {
        "info": {
            "description": "Overlapping-chunk YOLO + SAM3VideoPredictor tracking results",
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
        unique_ids = {a["track_id"] for a in coco_annotations}
        masked = sum(1 for a in coco_annotations if a["segmentation"])
        print(f"Unique track IDs: {len(unique_ids)}")
        print(f"Annotations with masks: {masked}")

    create_tracking_video(output_dir, image_files, args.fps)

    temp_dir = output_dir / "_temp_chunks"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary chunk videos: {temp_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Unified SAM3VideoPredictor tracking/segmentation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "reseed", "chunks"],
        help="Tracking strategy (default: single)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=str(project_dir / "runs" / "detect" / "output" / "training" / "mtr_detection_yolo26l" / "weights" / "best.pt"),
        help="Path to YOLO detection model",
    )
    parser.add_argument(
        "--sam3-model",
        type=str,
        default=str(project_dir / "core" / "sam3" / "models" / "sam3-model" / "sam3.pt"),
        help="Path to SAM3 weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_dir / "MTR_metacam_right"),
        help="Directory containing image sequence",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_dir / "output" / "tracking" / "sam3_video"),
        help="Directory to save results",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument("--yolo-device", type=str, default="cuda", help="Device for YOLO")
    parser.add_argument("--sam3-device", type=str, default="cuda", help="Device for SAM3")
    parser.add_argument("--sam3-imgsz", type=int, default=1024, help="SAM3 input image size")
    parser.add_argument("--temp-video", type=str, default=None,
                        help="Path to write temporary video (default: auto-generated in output dir)")

    # Reseed mode options
    parser.add_argument(
        "--reseed-every",
        type=int,
        default=15,
        help="(reseed mode) Run YOLO re-detection every N frames",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.3,
        help="(reseed/chunks modes) Min bbox IoU to match a detection to an existing object",
    )

    # Chunk mode options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="(chunks mode) Number of frames per chunk",
    )
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=15,
        help="(chunks mode) Frame step between consecutive chunks",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "single":
        run_single_mode(args)
    elif args.mode == "reseed":
        run_reseed_mode(args)
    elif args.mode == "chunks":
        run_chunks_mode(args)


if __name__ == "__main__":
    main()
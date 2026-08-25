#!/usr/bin/env python3
"""
Unified YOLO tracking script.

Supports multiple trackers and model tasks:
    --tracker bytetrack         ByteTrack (default for segmentation models)
    --tracker botsort           BoT-SORT with configurable CMC/ReID
    --tracker detect-then-sam3  YOLO detect + SAM3 segment per tracked box

Works with both YOLO detection and segmentation models; segmentation masks are
exported as COCO polygons when available.

An extra class-agnostic NMS pass (--nms-iou, default 0.5) runs on the raw
detections BEFORE the tracker: boxes overlapping a higher-confidence box with
IoU above the threshold are dropped regardless of class, so the same object
can't be tracked twice (ultralytics' own --iou NMS is class-aware and can
leave cross-class duplicates). Pass --nms-iou 1.0 to disable.

CPU post-processing (contour extraction, tracked-frame writes) runs in a
thread pool (--postprocess-workers, default 4) overlapping the next frame's
GPU inference; --no-masks skips polygon extraction entirely when only
boxes/track-ids are needed.

Speed knobs, in order of impact (profiled: raw GPU forward is only ~25% of
per-frame time — ultralytics' per-call Python/pre/post overhead and CPU
post-processing dominate):
    --no-vis            skip plotting/frames entirely (biggest win)
    --detect-batch N    one batched forward per N frames (4-8); tracking
                        stays sequential/stateful, results match N=1
    --mask-max-dim      contour resolution cap (masks are pre-scaled on the
                        GPU before the D2H transfer; TRK_NO_GPU_PRESCALE=1
                        reverts to CPU-side resizing)
    --half / --trt      faster model stage

--trt exports the --model .pt to a TensorRT engine once (FP16, at --imgsz)
and tracks with the engine; the cached .engine next to the checkpoint is
reused on later runs. It only accelerates the model-inference stage, so the
end-to-end gain depends on how inference-bound the run is (measured on
HKU_GH, yolo26l-seg @ 768, light scenes: ~7% wall-clock; bigger on
detection-heavy scenes). Same settings as scripts/15_export_trt.py. Delete
the .engine after changing --imgsz.

USAGE:
    python scripts/11_run_tracking.py \\
    --tracker botsort \\
    --model runs/segment/output/training/iw_segmentation/weights/best.pt \\
    --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \\
    --conf 0.5 --device 0 \\
    --warmup-frames 3 \\
    --output output/tracking/iw/IWrun2

    # TensorRT engine (export once, reuse after)
    python scripts/11_run_tracking.py --tracker deepocsort \\
        --model runs/segment/.../weights/best.pt \\
        --data HKU_GH_left --output output/tracking/hku_gh_left \\
        --trt --imgsz 768 --no-vis
    
    # ByteTrack on segmentation model
    python scripts/11_run_tracking.py --tracker bytetrack \\
        --model runs/segment/.../weights/best.pt \\
        --data MTR_dataset --output output/tracking/bytetrack

    # BoT-SORT on detection model with camera-motion compensation
    python scripts/11_run_tracking.py --tracker botsort \\
        --model runs/detect/.../weights/best.pt \\
        --data MTR_metacam_right --output output/tracking/botsort \\
        --with-cmc --cmc-method sparseOptFlow

    # YOLO detect + SAM3 segment
    python scripts/11_run_tracking.py --tracker detect-then-sam3 \\
        --yolo-model runs/detect/.../weights/best.pt \\
        --sam3-model core/sam3/models/sam3-model/sam3.pt \\
        --data MTR_dataset --output output/tracking/detect_then_sam3

    # Per-class confidence: 0.4 for all classes except Sprinkler (floor 0.1)
    python scripts/11_run_tracking.py --tracker deepocsort \\
        --model runs/segment/.../weights/best.pt \\
        --data HKU_GH_left --output output/tracking/hku_gh_left \\
        --conf 0.4 --conf-exempt-class 'Sprinkler -on the ceiling-' \\
        --conf-exempt-min 0.1 --imgsz 768
"""

import argparse
import os
import json
import queue
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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
    DinoReIDEncoder,
    build_dino_reid_encoder,
)


# =============================================================================
# Standard Tracking (ByteTrack / BoT-SORT)
# =============================================================================

class _FramePrefetcher:
    """Background cv2.imread loader so JPEG decode overlaps GPU inference."""

    _SENTINEL = object()

    def __init__(self, paths, buffer=8):
        self._q = queue.Queue(maxsize=buffer)
        self._t = threading.Thread(target=self._run, args=(paths,), daemon=True)
        self._t.start()

    def _run(self, paths):
        for p in paths:
            self._q.put((p, cv2.imread(str(p))))
        self._q.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


def _result_is_cpu(result) -> bool:
    """True when the result's tensors already live on the CPU."""
    try:
        if result.boxes is not None and result.boxes.xyxy.is_cuda:
            return False
        if getattr(result, "masks", None) is not None \
                and result.masks.data.is_cuda:
            return False
    except Exception:
        return False
    return True


def _mask_scale_for(img_h: int, img_w: int, mask_max_dim: int):
    """Shared contour-extraction downscale math: (scale, target_w, target_h).

    Used both by the GPU pre-downscale fast path (_prescale_masks_on_gpu)
    and process_frame's CPU fallback, so both agree on the target size.
    """
    scale = 1.0
    if mask_max_dim and max(img_h, img_w) > mask_max_dim:
        scale = mask_max_dim / max(img_h, img_w)
    return (scale, max(1, round(img_w * scale)),
            max(1, round(img_h * scale)))


def _prescale_masks_on_gpu(result, target_w: int, target_h: int) -> bool:
    """Downscale the result's mask stack ON THE GPU to (target_h, target_w).

    The mask tensor can dominate the GPU->CPU payload in dense scenes
    (N x orig-H x orig-W floats), so shrinking on-device before the .cpu()
    transfer cuts both PCIe traffic and the later CPU resize by scale^2.
    Polygon coordinates are unaffected (process_frame scales them back by
    1/scale). Set TRK_NO_GPU_PRESCALE=1 to disable (CPU-transfer fallback).

    Returns True when masks were left at (or moved to) the target size.
    """
    try:
        if result is None or getattr(result, "masks", None) is None:
            return False
        data = result.masks.data
        if not data.is_cuda or data.ndim != 3:
            return False
        if tuple(data.shape[1:]) == (target_h, target_w):
            return True
        import torch.nn.functional as F
        x = data[:, None].float()  # (N,H,W) -> (N,1,H,W)
        x = F.interpolate(x, size=(target_h, target_w),
                          mode="bilinear", align_corners=False)
        result.masks.data = x[:, 0]
        return True
    except Exception:
        return False


def process_frame(result, img_h, img_w, image_id, annotation_id, class_names,
                  vis=True, mask_max_dim=1024, no_masks=False):
    """Extract COCO annotations and annotated frame from a tracking result.

    With vis=False, skip result.plot() entirely (returns annotated_frame=None).
    mask_max_dim caps the mask resolution used for contour extraction — the
    polygon coordinates are scaled back to full image size, so accuracy loss
    is negligible while findContours runs far below 4K cost. 0 disables.
    no_masks skips mask contour extraction entirely (boxes/track-ids only).
    """
    annotated_frame = None
    annotations = []

    if result is not None and vis:
        # Move result tensors to CPU before plotting to avoid GPU OOM from
        # the mask rendering path in result.plot() (which does cumulative
        # alpha compositing on the GPU and can exhaust memory on large frames
        # or when another process shares the GPU). Skip the transfer when the
        # caller already moved the result off the GPU (postprocess-pool path).
        if not _result_is_cpu(result):
            result = result.cpu()
        annotated_frame = result.plot()

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
            not no_masks
            and hasattr(result, "masks")
            and result.masks is not None
            and len(result.masks) > 0
        )

        # Contours are extracted at (img_w, img_h) scaled down so the largest
        # side is <= mask_max_dim; polygon coords are divided back by `scale`.
        # When the main loop already pre-scaled the masks on the GPU
        # (_prescale_masks_on_gpu), the shape check below is a no-op.
        scale, target_w, target_h = _mask_scale_for(img_h, img_w,
                                                    mask_max_dim)

        # Vectorized mask preparation: ONE GPU->CPU transfer for all masks,
        # one batched resize of the full stack (cv2.resize treats trailing
        # dims as channels), and one vectorized binarization + area sum.
        masks_np = None
        areas_vec = None
        if has_masks:
            data = result.masks.data
            n_masks = min(len(track_ids), len(data))
            m = data[:n_masks].cpu().numpy().astype(np.float32)
            if m.ndim == 2:
                m = m[None]
            if m.shape[1:] != (target_h, target_w):
                # (N,H,W) -> (H,W,N) channels-last for a single resize call.
                m = cv2.resize(
                    m.transpose(1, 2, 0), (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                if m.ndim == 2:
                    m = m[:, :, None]
                m = m.transpose(2, 0, 1)
            masks_np = m
            mask_binary_all = (masks_np > 0.5)
            # Per-mask pixel counts in one pass; scaled back to full res.
            areas_vec = mask_binary_all.sum(axis=(1, 2)).astype(np.float64) \
                / (scale * scale)

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
            if masks_np is not None and i < len(masks_np):
                mask_binary = masks_np[i] > 0.5

                contours, _ = cv2.findContours(
                    mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                segmentation_polygons = []
                inv_scale = 1.0 / scale
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = (contour.flatten() * inv_scale).tolist()
                        segmentation_polygons.append(polygon)

                if segmentation_polygons:
                    segmentation = segmentation_polygons
                    area = float(areas_vec[i])

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

        if annotated_frame is not None:
            unique_ids = np.unique(track_ids)
            info_text = f"Tracker: result.boxes.id | Objects: {len(unique_ids)} | Tracks: {len(track_ids)}"
            cv2.putText(
                annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    return annotated_frame, annotations, annotation_id


def _install_per_class_conf_filter(model, exempt_ids, conf_default):
    """Filter detections per class BEFORE the tracker sees them.

    Inserts a callback at the front of ``on_predict_postprocess_end`` (i.e.
    before ultralytics' tracker update) that drops detections with
    conf < conf_default unless their class is in ``exempt_ids`` — those are
    kept down to the model-level confidence floor. Boxes and masks are
    filtered together so indices stay aligned.
    """
    import torch
    from ultralytics.engine.results import Masks

    exempt = torch.tensor(sorted(exempt_ids), dtype=torch.float32)

    def _filter(predictor):
        for r in predictor.results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            keep = r.boxes.conf >= conf_default
            if len(exempt_ids):
                keep = keep | torch.isin(r.boxes.cls, exempt.to(r.boxes.cls.device))
            if bool(keep.all()):
                continue
            r.boxes = r.boxes[keep]
            if r.masks is not None and len(r.masks.data):
                r.masks = Masks(r.masks.data[keep], r.orig_shape)

    model.callbacks["on_predict_postprocess_end"].insert(0, _filter)
    print(f"Per-class conf filter: conf>={conf_default} for all classes except "
          f"ids {sorted(exempt_ids)} (kept down to model floor)")


def _install_post_detection_nms(model, iou_thr):
    """Extra class-agnostic NMS on raw detections BEFORE the tracker sees them.

    Ultralytics' own NMS (--iou) is class-aware, so the same object can be
    detected twice under different classes (or survive with near-duplicate
    boxes just under the IoU cut). This pass drops any box overlapping a
    higher-confidence box with IoU > iou_thr, regardless of class. Same
    insertion point as the per-class conf filter: front of
    ``on_predict_postprocess_end``, i.e. before the tracker update — so
    duplicate detections cannot spawn duplicate tracks. Boxes and masks are
    filtered together so indices stay aligned.
    """
    import torchvision
    from ultralytics.engine.results import Masks

    def _nms(predictor):
        for r in predictor.results:
            if r.boxes is None or len(r.boxes) < 2:
                continue
            keep = torchvision.ops.nms(r.boxes.xyxy, r.boxes.conf, iou_thr)
            if len(keep) == len(r.boxes):
                continue
            r.boxes = r.boxes[keep]
            if r.masks is not None and len(r.masks.data):
                r.masks = Masks(r.masks.data[keep], r.orig_shape)

    model.callbacks["on_predict_postprocess_end"].insert(0, _nms)
    print(f"Post-detection NMS: class-agnostic, IoU > {iou_thr} keeps the "
          f"higher-confidence box")


def _resolve_trt_model(model_path: Path, args) -> Path:
    """With --trt, export a .pt checkpoint to a TensorRT engine (or reuse the
    cached one next to the checkpoint) and return the engine path.

    Same export settings as scripts/15_export_trt.py: FP16, fixed --imgsz,
    ONNX-simplified. Engines are GPU- and imgsz-specific — the cached engine
    is reused blindly, so the user must delete it after changing --imgsz (a
    note is printed either way).
    """
    if not args.trt or model_path.suffix != ".pt":
        return model_path
    engine = model_path.with_suffix(".engine")
    if engine.exists():
        print(f"TensorRT: reusing cached engine {engine}\n"
              f"  (built for a fixed imgsz — delete it to re-export after "
              f"changing --imgsz)")
        return engine
    print(f"TensorRT: exporting {model_path} -> {engine}\n"
          f"  (FP16, imgsz={args.imgsz}, workspace={args.trt_workspace} GiB; "
          f"one-time, a few minutes)…")
    t0 = time.perf_counter()
    YOLO(str(model_path)).export(
        format="engine", half=True, imgsz=args.imgsz, device=0,
        workspace=args.trt_workspace, simplify=True, verbose=False)
    if not engine.exists():
        raise RuntimeError(f"TensorRT export did not produce {engine}")
    print(f"TensorRT: export done in {time.perf_counter() - t0:.0f}s")
    return engine


def _build_tracker_yaml(args, output_dir: Path):
    """Return the path to a runtime tracker YAML for the requested tracker.

    Supports BoT-SORT, OC-SORT, and Deep OC-SORT. ByteTrack uses its own base
    yaml and is handled by ultralytics directly (no runtime override needed).
    """
    # Map tracker name -> base yaml filename.
    tracker_yamls = {
        "botsort": "botsort.yaml",
        "ocsort": "ocsort.yaml",
        "deepocsort": "deepocsort.yaml",
    }
    if args.tracker not in tracker_yamls:
        return None

    trackers_dir = find_ultralytics_trackers_dir()
    base_yaml = trackers_dir / tracker_yamls[args.tracker]
    if not base_yaml.exists():
        print(f"Error: {base_yaml.name} not found at {base_yaml}")
        sys.exit(1)

    tracker_yaml = build_runtime_tracker_yaml(
        base_yaml,
        tracker_type=args.tracker,
        with_cmc=args.with_cmc,
        cmc_method=args.cmc_method,
        track_buffer=args.track_buffer,
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=(args.conf_exempt_min
                          if getattr(args, "use_byte", False)
                          and getattr(args, "conf_exempt_class", None)
                          else None),
        with_reid=args.with_reid,
        output_dir=output_dir,
        reid_model=getattr(args, "reid_model", None),
        match_thresh=getattr(args, "match_thresh", None),
        new_track_thresh=getattr(args, "new_track_thresh", None),
        inertia=getattr(args, "inertia", None),
        delta_t=getattr(args, "delta_t", None),
        proximity_thresh=getattr(args, "proximity_thresh", None),
        appearance_thresh=getattr(args, "appearance_thresh", None),
        use_byte=getattr(args, "use_byte", None) or None,
    )
    reid_str = f", reid={args.with_reid}" if args.tracker in ("botsort", "deepocsort") else ""
    print(f"{args.tracker.upper()}: cmc={args.with_cmc} ({args.cmc_method}), buffer={args.track_buffer}{reid_str}")
    if args.with_reid and args.reid_model:
        print(f"  ReID model: {args.reid_model}")
    return tracker_yaml


def _install_dino_reid(args):
    """Patch ultralytics' ReID builder to return a DINOv2/DINOv3 encoder.

    Ultralytics' BoT-SORT calls ``build_encoder(with_reid, model)`` at tracker
    construction time. We monkey-patch that function so that when
    ``--reid-model dinov2*`` / ``dinov3*`` is requested, a ``DinoReIDEncoder``
    is returned instead of the built-in ``ReID`` class (which only handles YOLO
    ``.pt`` checkpoints or ONNX backends).

    Must be called *before* the first ``model.track()`` invocation so the patch
    is in place when ``on_predict_start`` builds the tracker.
    """
    if not args.with_reid or not args.reid_model:
        return
    if not args.reid_model.startswith(("dinov2", "dinov3")) and not Path(args.reid_model).is_dir():
        return  # leave the default behavior for "auto" / .pt / .onnx paths

    from scripts.tracking_utils import build_dino_reid_encoder
    import ultralytics.trackers.utils.reid as reid_mod

    original_build_encoder = reid_mod.build_encoder

    def patched_build_encoder(with_reid, model):
        if not with_reid:
            return None
        if model in (None, "auto"):
            # Fall back to the default "auto" path (detector backbone features).
            return original_build_encoder(with_reid, model)
        if model == args.reid_model:
            # Normalize device shorthand: '0' -> 'cuda:0', 'auto' -> None
            dev = args.device
            if dev and dev != "auto":
                if dev.isdigit():
                    dev = f"cuda:{dev}"
            else:
                dev = None
            encoder = build_dino_reid_encoder(
                model_name=args.reid_model,
                device=dev,
                imgsz=args.reid_imgsz,
            )
            if encoder is None:
                print(f"[ReID] DINO encoder load failed; falling back to '{model}' default path")
                return original_build_encoder(with_reid, model)
            return encoder
        return original_build_encoder(with_reid, model)

    reid_mod.build_encoder = patched_build_encoder
    # Also patch the reference imported into bot_sort.py at import time.
    import ultralytics.trackers.bot_sort as bot_sort_mod
    bot_sort_mod.build_encoder = patched_build_encoder
    # And the deep_oc_sort.py reference (used by DeepOCSORT).
    import ultralytics.trackers.deep_oc_sort as deep_oc_sort_mod
    deep_oc_sort_mod.build_encoder = patched_build_encoder
    # And the track_tracker.py reference (used by TRACKTRACK).
    import ultralytics.trackers.track_tracker as track_tracker_mod
    track_tracker_mod.build_encoder = patched_build_encoder
    print(f"[ReID] Patched build_encoder to use DINO model '{args.reid_model}'")


def _use_fp16(args) -> bool:
    """FP16 (quantize=16) is a CUDA-only win; keep FP32 on CPU."""
    return bool(args.half) and str(args.device) not in ("cpu",)


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
    model_path = _resolve_trt_model(model_path, args)

    print(f"Tracker:   {args.tracker}")
    print(f"Model:     {model_path}")
    print(f"Data:      {data_path}")
    print(f"Output:    {output_dir}")
    print(f"conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    # Load model
    model = YOLO(str(model_path))
    class_names = model.names if hasattr(model, "names") else {}
    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]

    # Extra class-agnostic NMS on raw detections before the tracker, so
    # duplicate boxes of the same object can't spawn duplicate tracks.
    # Installed before the conf filter so the conf filter lands at index 0
    # and runs first (fewer boxes for NMS).
    if args.nms_iou < 1.0:
        _install_post_detection_nms(model, args.nms_iou)

    # Per-class confidence exemption: keep low-confidence detections of one
    # class (e.g. Sprinkler) down to --conf-exempt-min while all other classes
    # need conf >= --conf. Tracker spawn/association thresholds are dropped to
    # the floor so exempted low-conf detections can also start new tracks.
    if args.conf_exempt_class:
        exempt_ids = {i for i, n in class_names.items() if n == args.conf_exempt_class}
        if not exempt_ids:
            print(f"WARNING: class {args.conf_exempt_class!r} not in model names "
                  f"{list(class_names.values())} — exemption ignored")
        else:
            conf_default = args.conf
            args.conf = min(args.conf, args.conf_exempt_min)
            if args.use_byte:
                # BYTE second association: exempt low-conf detections extend
                # EXISTING tracks (track_high_thresh..track_low_thresh band)
                # but never spawn new ones — keeps flickery small objects
                # (e.g. sprinklers) on one stable ID instead of churning.
                args.track_high_thresh = conf_default
                args.new_track_thresh = conf_default
                byte_low = args.conf_exempt_min
            else:
                byte_low = None
                args.track_high_thresh = min(args.track_high_thresh, args.conf)
                args.new_track_thresh = min(args.new_track_thresh, args.conf)
            _install_per_class_conf_filter(model, exempt_ids, conf_default)
            print(f"Exempt class {args.conf_exempt_class!r}: floor {args.conf}, "
                  f"others conf>={conf_default}; "
                  f"tracker track_high/new_track thresh -> "
                  f"{args.track_high_thresh}/{args.new_track_thresh}"
                  + (f" (BYTE band {args.track_high_thresh}-{byte_low})"
                     if byte_low else ""))

    # Resolve tracker configuration
    tracker_yaml = _build_tracker_yaml(args, output_dir)

    # If a DINO ReID model was requested, patch ultralytics' build_encoder so
    # BoT-SORT picks up our DinoReIDEncoder. Must happen before model.track().
    _install_dino_reid(args)

    image_files = find_image_files(data_path)
    if args.max_frames is not None:
        image_files = image_files[:args.max_frames]
    print(f"Found {len(image_files)} images to process")

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    valid_paths = []

    def _process_and_write(result, frame, img_h, img_w, image_id, image_path):
        """CPU-side post-processing (contour extraction + tracked_*.jpg
        write). Runs in a worker thread so it overlaps the NEXT frame's GPU
        inference. Annotation ids are local (0-based); the drain loop
        reassigns them sequentially, in frame order."""
        annotated_frame, annotations, _ = process_frame(
            result, img_h, img_w, image_id, 0, class_names,
            vis=not args.no_vis, mask_max_dim=args.mask_max_dim,
            no_masks=args.no_masks,
        )
        if not args.no_vis:
            if annotated_frame is None:
                annotated_frame = frame
            cv2.imwrite(str(output_dir / f"tracked_{image_path.name}"),
                        annotated_frame)
        return annotations

    # Post-processing pool: with N > 0, contour extraction and image writes
    # of previous frames run while the GPU tracks the current one. The
    # window is bounded (2×N in flight) and drained strictly in order.
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pp_pool = (ThreadPoolExecutor(max_workers=args.postprocess_workers)
               if args.postprocess_workers > 0 else None)
    pending = deque()

    def _collect(annotations):
        nonlocal annotation_id
        for ann in annotations:
            ann["id"] = annotation_id
            annotation_id += 1
            coco_annotations.append(ann)

    def _handle_result(result, frame, img_h, img_w, image_id, image_path):
        """GPU-prescale + hand off one tracked frame to the postprocess
        pool (or run it inline when the pool is disabled)."""
        if result is not None and not args.no_masks \
                and not os.environ.get("TRK_NO_GPU_PRESCALE"):
            # Shrink the D2H payload while the masks are still on the GPU:
            # the transfer is the main thread's only GPU-adjacent work, so
            # this keeps PCIe (not compute) from gating the loop.
            _, tw, th = _mask_scale_for(img_h, img_w, args.mask_max_dim)
            _prescale_masks_on_gpu(result, tw, th)
        if pp_pool is not None:
            # Move tensors off the GPU in the main thread so worker threads
            # never touch CUDA objects.
            result = result.cpu() if result is not None else None
            pending.append(pp_pool.submit(_process_and_write, result, frame,
                                          img_h, img_w, image_id, image_path))
            if len(pending) > 2 * args.postprocess_workers:
                _collect(pending.popleft().result())
        else:
            _collect(_process_and_write(result, frame, img_h, img_w,
                                        image_id, image_path))
        if args.no_vis:
            if image_id % 50 == 0 or image_id == len(image_files):
                print(f"  Processed frame {image_id}/{len(image_files)}")
        else:
            print(f"  Saved frame {image_id}/{len(image_files)}: "
                  f"tracked_{image_path.name}")

    track_kwargs = {
        "persist": True,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
        "device": args.device,
        "quantize": 16 if _use_fp16(args) else None,
    }
    if tracker_yaml is not None:
        track_kwargs["tracker"] = str(tracker_yaml)

    detect_batch = max(1, getattr(args, "detect_batch", None) or 1)

    def _track_batch(buf):
        """Track a list of [(image_path, frame, image_id)] in ONE model call.

        A list source goes through LoadPilAndNumpy, which yields all frames
        at once -> a single batched forward pass (amortizing ultralytics'
        fixed per-call Python/pre/post overhead). Tracking stays sequential
        and stateful: ultralytics' postprocess callback updates the SAME
        tracker instance over the batch's results in order.
        """
        frames = [f for _, f, _ in buf]
        track_kwargs["source"] = frames[0] if len(frames) == 1 else frames
        results = model.track(**track_kwargs)
        if results is None or len(results) != len(buf):
            raise RuntimeError(
                f"tracker returned {0 if not results else len(results)} "
                f"results for {len(buf)} frames")
        for (image_path, frame, image_id), res in zip(buf, results):
            img_h, img_w = frame.shape[:2]
            _handle_result(res, frame, img_h, img_w, image_id, image_path)

    batch_buf = []
    prefetcher = _FramePrefetcher(image_files)
    for idx, (image_path, frame) in enumerate(prefetcher):
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
        batch_buf.append((image_path, frame, image_id))
        if len(batch_buf) >= detect_batch:
            _track_batch(batch_buf)
            batch_buf = []
    if batch_buf:
        _track_batch(batch_buf)

    # Drain remaining in-flight post-processing, in order.
    while pending:
        _collect(pending.popleft().result())
    if pp_pool is not None:
        pp_pool.shutdown(wait=True)

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
    if not args.no_vis:
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
    model_path = _resolve_trt_model(model_path, args)

    print("=" * 60)
    print("TRACKING + SEGMENTATION BENCHMARK")
    print("=" * 60)
    tracker_yaml = _build_tracker_yaml(args, data_path)
    _install_dino_reid(args)

    print(f"Model:     {model_path}")
    print(f"Tracker:   {args.tracker}")
    print(f"Data:      {data_path}")
    print(f"conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print(f"Half precision: {args.half}")
    print(f"Skip masks: {args.no_masks}")
    print(f"Warmup frames: {args.warmup_frames}")
    print("Note: no images, videos, or JSON will be saved.\n")

    model = YOLO(str(model_path))

    if args.nms_iou < 1.0:
        _install_post_detection_nms(model, args.nms_iou)

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
            "quantize": 16 if _use_fp16(args) else None,
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
    yolo_model_path = _resolve_trt_model(yolo_model_path, args)
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
        track_low_thresh=(args.conf_exempt_min
                          if getattr(args, "use_byte", False)
                          and getattr(args, "conf_exempt_class", None)
                          else None),
        with_reid=args.with_reid,
        output_dir=output_dir,
        reid_model=getattr(args, "reid_model", None),
        match_thresh=getattr(args, "match_thresh", None),
        new_track_thresh=getattr(args, "new_track_thresh", None),
        inertia=getattr(args, "inertia", None),
        delta_t=getattr(args, "delta_t", None),
        proximity_thresh=getattr(args, "proximity_thresh", None),
        appearance_thresh=getattr(args, "appearance_thresh", None),
    )
    _install_dino_reid(args)

    # Load models
    print("Loading YOLO detection model...")
    yolo_model = YOLO(str(yolo_model_path))

    if args.nms_iou < 1.0:
        _install_post_detection_nms(yolo_model, args.nms_iou)

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

        if args.no_vis:
            if image_id % 50 == 0 or image_id == len(image_files):
                print(f"  Frame {image_id}/{len(image_files)}: {len(tracked_objects)} tracks, "
                      f"{sum(1 for m in masks_per_object if m is not None)} masks")
        else:
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
    if not args.no_vis:
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
        choices=["bytetrack", "botsort", "ocsort", "deepocsort", "detect-then-sam3", "benchmark"],
        help="Tracker to use (default: bytetrack). 'ocsort' = OC-SORT "
             "(robust to non-linear motion/sudden turns). 'deepocsort' = "
             "OC-SORT + appearance features (ReID). Use 'benchmark' to run "
             "YOLO tracking+segmentation without saving any outputs and "
             "report timing.",
    )

    # Benchmark tracking arguments
    parser.add_argument(
        "--benchmark-tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort", "ocsort", "deepocsort"],
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
    parser.add_argument(
        "--conf-exempt-class",
        type=str,
        default=None,
        help="Class name exempt from --conf: its detections are kept down to "
             "--conf-exempt-min and can spawn tracks at any confidence. "
             "E.g. --conf 0.4 --conf-exempt-class 'Sprinkler -on the ceiling-'",
    )
    parser.add_argument(
        "--conf-exempt-min",
        type=float,
        default=0.1,
        help="Confidence floor for --conf-exempt-class detections (default: 0.1)",
    )
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="Extra class-agnostic NMS on raw detections BEFORE the tracker: "
             "boxes overlapping with IoU above this keep only the "
             "higher-confidence one, so the same object can't be tracked "
             "twice (default: 0.5; >= 1.0 disables)",
    )
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
        dest="half",
        action="store_true",
        default=True,
        help="Use FP16 inference (maps to quantize=16; faster on CUDA, "
             "default: True)",
    )
    parser.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Disable FP16 inference (use FP32).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Export --model .pt to a TensorRT engine once (FP16, at --imgsz) "
             "and run tracking with it; a cached .engine next to the "
             "checkpoint is reused. Speeds up the model-inference stage "
             "(end-to-end gain is modest when tracker/post-processing "
             "dominate; ~7%% measured on HKU_GH yolo26l-seg @ 768). Delete "
             "the .engine to force re-export after changing --imgsz. Also "
             "applies to --yolo-model in detect-then-sam3 mode.",
    )
    parser.add_argument(
        "--trt-workspace",
        type=int,
        default=4,
        help="TensorRT build workspace in GiB (default: 4).",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Skip segmentation-mask extraction (boxes/track-ids only in "
             "results.json; benchmark mode also skips reading masks). "
             "Faster when you don't need polygons.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=4,
        help="Threads for CPU post-processing (contour extraction, tracked "
             "frame writes) so it overlaps GPU inference of the next frame "
             "(default: 4; 0 = serial, pre-pool behavior). Only the "
             "detection+tracking stage itself is inherently sequential.",
    )
    parser.add_argument(
        "--detect-batch",
        type=int,
        default=1,
        help="Frames per model.track() call (default: 1). Batching runs one "
             "GPU forward per N frames, amortizing ultralytics' fixed "
             "per-call Python/pre/post overhead — the dominant cost at "
             "imgsz<=768. Tracking stays sequential and stateful (same "
             "tracker instance, frames in order), so results match "
             "--detect-batch 1. Try 4-8.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip all visualization output: no per-frame tracked_*.jpg and no "
             "tracking_result.mp4 (results.json is still written). Much faster "
             "on large image sets.",
    )
    parser.add_argument(
        "--mask-max-dim",
        type=int,
        default=1024,
        help="Max mask resolution (largest side) used for polygon contour "
             "extraction; coords are scaled back to full size (default: 1024, "
             "0 = full resolution, slower).",
    )

    # BoT-SORT / Deep OC-SORT knobs
    parser.add_argument("--with-cmc", dest="with_cmc", action="store_true", default=True)
    parser.add_argument("--no-cmc", dest="with_cmc", action="store_false")
    parser.add_argument(
        "--cmc-method",
        type=str,
        default="sparseOptFlow",
        choices=["sparseOptFlow", "orb", "sift", "ecc", "none"],
        help="Global motion compensation method (BoT-SORT / Deep OC-SORT)",
    )
    parser.add_argument("--track-buffer", type=int, default=60, help="Lost-track buffer (frames to keep lost tracks alive)")
    parser.add_argument("--track-high-thresh", type=float, default=0.5, help="First-stage association threshold")
    parser.add_argument("--with-reid", action="store_true", default=False, help="Enable ReID (BoT-SORT / Deep OC-SORT)")
    parser.add_argument(
        "--use-byte",
        action="store_true",
        default=False,
        help="Enable ByteTrack-style second association pass so detections "
             "between track_low_thresh and track_high_thresh can extend "
             "existing tracks (OC-SORT / Deep OC-SORT). Useful with a low "
             "--conf floor: new tracks still spawn only from "
             ">= --track-high-thresh detections.",
    )
    # Deep OC-SORT specific knobs
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="Association similarity threshold (IoU/cost). Lower = more tolerant "
             "of prediction errors during turns. (BoT-SORT / Deep OC-SORT)",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.3,
        help="Minimum confidence to start a new track. Higher = harder to spawn "
             "new tracks (helps prevent the same object getting a new ID after a "
             "turn). (Deep OC-SORT)",
    )
    parser.add_argument(
        "--inertia",
        type=float,
        default=0.2,
        help="Weight of velocity consistency cost in association (OC-SORT / Deep "
             "OC-SORT). Lower = less penalty for direction changes, better for "
             "sudden turns. Default 0.2; try 0.05 for sharp turns.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        default=3,
        help="Temporal window for velocity direction computation (OC-SORT / Deep "
             "OC-SORT). Shorter = faster adaptation to direction changes. "
             "Default 3; try 1 for sharp turns.",
    )
    parser.add_argument(
        "--proximity-thresh",
        type=float,
        default=0.5,
        help="Min IoU to consider tracks proximate for ReID. Lower = allow ReID "
             "to compensate for poor IoU during turns. (BoT-SORT / Deep OC-SORT)",
    )
    parser.add_argument(
        "--appearance-thresh",
        type=float,
        default=0.9,
        help="Min appearance similarity for ReID. Lower = more aggressive ReID "
             "re-acquisition after track loss. (BoT-SORT / Deep OC-SORT)",
    )
    parser.add_argument(
        "--reid-model",
        type=str,
        default=None,
        help="ReID model source. 'auto' uses detector backbone features. "
             "DINO models: 'dinov2_vits14' (public), 'dinov3_vits16' (requires "
             "HF access). Loads the ViT via torch.hub and wraps it as a BoT-SORT "
             "ReID encoder. Overrides the tracker YAML's 'model' field. Only "
             "used with --tracker botsort --with-reid.",
    )
    parser.add_argument(
        "--reid-imgsz",
        type=int,
        default=224,
        help="Input size for DINO ReID crops (default: 224). Only used with "
             "--reid-model dinov2*/dinov3*.",
    )

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
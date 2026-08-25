#!/usr/bin/env python3
"""
Run a YOLO seg model over an image folder and write COCO-style results
(same schema as scripts/11_run_tracking.py's results.json: images /
categories / annotations with bbox, segmentation polygons, confidence).

Post-processing:
- Class-agnostic NMS: detections with box IoU > --nms-iou keep only the
  higher-confidence one (yolo26 is NMS-free by design; this re-applies
  explicit NMS on top of its output).
- Per-class confidence filter: detections below --conf-default are dropped
  for every class EXCEPT --keep-low-conf-class (e.g. Sprinkler), which is
  kept down to --conf-min.

USAGE:
    python scripts/14_predict_to_coco.py \
        --model runs/segment/.../yolo26n/.../best.pt \
        --source Datasets/HKU_GH/HKU_GH_left \
        --output-json Datasets/HKU_GH/results_yolo26n/HKU_GH_left_results.json \
        --imgsz 768
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="YOLO weights (.pt)")
    p.add_argument("--source", required=True, help="Image folder")
    p.add_argument("--output-json", required=True, help="Output COCO JSON path")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", default="0")
    p.add_argument("--conf-min", type=float, default=0.1,
                   help="Model-level confidence floor (default 0.1)")
    p.add_argument("--conf-default", type=float, default=0.4,
                   help="Confidence threshold for all classes except "
                        "--keep-low-conf-class (default 0.4)")
    p.add_argument("--keep-low-conf-class", default="Sprinkler -on the ceiling-",
                   help="Class name exempt from --conf-default")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="Class-agnostic NMS IoU threshold (default 0.5)")
    return p.parse_args()


def box_iou(a, b):
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def nms_by_confidence(dets, iou_thr):
    """Class-agnostic NMS: on IoU > iou_thr keep the higher-confidence det.

    `dets`: list of dicts with 'xyxy' and 'confidence'. Returns kept list.
    """
    order = sorted(dets, key=lambda d: -d["confidence"])
    kept = []
    for det in order:
        if all(box_iou(det["xyxy"], k["xyxy"]) <= iou_thr for k in kept):
            kept.append(det)
    return kept


def predict_folder(model_path, source, imgsz, device, conf_min):
    """Yield (file_name, width, height, dets) per image. Each det is a dict
    with xyxy, confidence, class_id, polygons (list of flat xy lists)."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    files = sorted(p for p in Path(source).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise RuntimeError(f"No images found in {source}")
    for i, f in enumerate(files, 1):
        r = model.predict(str(f), imgsz=imgsz, conf=conf_min, device=device,
                          verbose=False)[0]
        h, w = r.orig_shape
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            polys_per_det = r.masks.xy if r.masks is not None else [[]] * len(xyxy)
            for j in range(len(xyxy)):
                poly = polys_per_det[j] if len(polys_per_det) > j else []
                # masks.xy gives one (N, 2) polygon array per detection
                polygons = ([np.round(poly).astype(int).flatten().tolist()]
                            if len(poly) >= 3 else [])
                dets.append({
                    "xyxy": xyxy[j],
                    "confidence": float(confs[j]),
                    "class_id": int(clss[j]),
                    "polygons": polygons,
                })
        if i % 500 == 0 or i == len(files):
            print(f"  {i}/{len(files)} images", flush=True)
        yield f.name, int(w), int(h), dets


def main():
    args = parse_args()
    from ultralytics import YOLO
    names = YOLO(str(args.model)).names  # {id: name}
    categories = [{"id": int(i), "name": n} for i, n in sorted(names.items())]
    exempt_ids = {i for i, n in names.items()
                  if n == args.keep_low_conf_class}
    if not exempt_ids:
        print(f"WARNING: class {args.keep_low_conf_class!r} not in model "
              f"names {list(names.values())} — conf filter applies to ALL classes")

    images, annotations = [], []
    ann_id = 1
    n_raw = n_after_filter = 0
    for file_name, w, h, dets in predict_folder(
            args.model, args.source, args.imgsz, args.device, args.conf_min):
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": file_name,
                       "width": w, "height": h})
        n_raw += len(dets)
        # Per-class confidence filter
        dets = [d for d in dets
                if d["confidence"] >= args.conf_default
                or d["class_id"] in exempt_ids]
        n_after_filter += len(dets)
        # Class-agnostic NMS (keep higher confidence on overlap)
        for d in nms_by_confidence(dets, args.nms_iou):
            x1, y1, x2, y2 = [float(v) for v in d["xyxy"]]
            bw, bh = x2 - x1, y2 - y1
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": d["class_id"],
                "bbox": [x1, y1, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
                "confidence": d["confidence"],
            }
            if d["polygons"]:
                ann["segmentation"] = d["polygons"]
            annotations.append(ann)
            ann_id += 1

    out = {
        "info": {"description": f"YOLO seg predictions ({Path(args.model).name}, "
                                f"imgsz={args.imgsz}, nms_iou={args.nms_iou}, "
                                f"conf>={args.conf_default} except "
                                f"{args.keep_low_conf_class!r})",
                 "version": "1.0", "year": datetime.now().year,
                 "date_created": datetime.now().isoformat()},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"✅ {out_path}: {len(images)} images, {len(annotations)} detections "
          f"(raw {n_raw} → conf-filtered {n_after_filter} → NMS {len(annotations)})")


if __name__ == "__main__":
    main()

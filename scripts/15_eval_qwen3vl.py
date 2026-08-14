#!/usr/bin/env python3
"""
Evaluate Qwen3-VL object detection against ground truth annotations.

Runs Qwen3-VL (via Ollama API) on each image in the eval dataset, collects
bounding box predictions, and computes precision, recall, F1, mAP@0.5,
and per-class metrics against the ground truth.

USAGE:
    set -a && . ./.env && set +a && \
    python scripts/15_eval_qwen3vl.py \
        --eval-dir Datasets/MTR/MTR_keyframes_eval \
        --target-class "Exit Sign" \
        --workers 4

    # Evaluate all classes
    python scripts/15_eval_qwen3vl.py --eval-dir Datasets/MTR/MTR_keyframes_eval --workers 4

    # Use existing predictions (skip inference)
    python scripts/15_eval_qwen3vl.py \
        --eval-dir Datasets/MTR/MTR_keyframes_eval \
        --skip-inference
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_URL = os.getenv("QWEN_VL_API_URL", "https://ollamaapi.ianlo.site/api/chat")
API_KEY = os.getenv("IW_OLLAMA_API_KEY", "")
MODEL = "qwen3-vl:235b-a22b-instruct"

CLASSES = ["Exit Sign"]

CLASS_ALIASES = {
    "exit sign": "Exit Sign",
}

EXIT_SIGN_PROMPT = """Analyze this image and detect all objects.
For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner, normalized to the 0-1000 range relative to image width/height. Return the result as a JSON array like: [{"label": "object_name", "bbox_2d": [x1, y1, x2, y2]}]. Only report objects you are confident are actually present in the image; do not invent objects, and do not output duplicate or heavily overlapping boxes for the same object.

Detect ONLY "Exit Sign" objects.
"Exit Sign" - A hanging monitor/display showing the lime-green character 出 and text 'EXIT'. It is a hanging LCD screen, not a wall poster. Exit signs MUST CONTAIN 'EXIT' text and the '出' character and be an OVERHEAD HANGING DISPLAY. Do not detect normal hanging monitors, TVs, posters, or advertisement boards without the lime '出' and EXIT text.

Return ONLY bounding boxes for "Exit Sign" as a JSON list of objects each with a "label" and "bbox_2d" field in [x1, y1, x2, y2] format (top-left then bottom-right, normalized 0-1000). If no exit signs are present, return an empty list []."""


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def normalize_class(label: str) -> str:
    key = label.strip().lower()
    return CLASS_ALIASES.get(key, label.strip())


def parse_predictions(content: str, img_w: int, img_h: int):
    """Parse Qwen3-VL response into list of {label, bbox} in pixel coords."""
    predictions = []

    # Strip markdown code fences
    content = re.sub(r'```(?:json)?\s*', '', content).strip()

    # Extract the outermost JSON array by finding the first '[' and
    # its matching ']' (handles nested arrays like bbox_2d: [x1,y1,x2,y2])
    start = content.find('[')
    if start == -1:
        return predictions

    depth = 0
    end = -1
    for i, c in enumerate(content[start:], start):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return predictions

    try:
        items = json.loads(content[start:end])
    except json.JSONDecodeError:
        return predictions

    for item in items:
        label = item.get("label", item.get("class", item.get("name", "")))
        bbox = item.get("bbox_2d", item.get("bbox", item.get("box_2d", [])))

        if not label or not bbox or len(bbox) != 4:
            continue

        # Normalize label
        label = normalize_class(label)

        # Convert from 0-1000 normalized to pixel coords
        x1 = int(bbox[0] / 1000 * img_w)
        y1 = int(bbox[1] / 1000 * img_h)
        x2 = int(bbox[2] / 1000 * img_w)
        y2 = int(bbox[3] / 1000 * img_h)

        # Ensure valid bbox
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        predictions.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": 1.0,
        })

    return predictions


def run_inference(image_path: Path, retries=3) -> list:
    """Run Qwen3-VL on a single image, return predictions."""
    for attempt in range(retries):
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": EXIT_SIGN_PROMPT,
                        "images": [encode_image(image_path)],
                    }
                ],
                "stream": False,
            }

            headers = {
                "IW-Ollama-API-Key": API_KEY,
                "Content-Type": "application/json",
            }

            resp = requests.post(API_URL, headers=headers, json=payload)

            if resp.status_code != 200:
                time.sleep(2)
                continue

            content = resp.json().get("message", {}).get("content", "")

            with Image.open(image_path) as img:
                w, h = img.size

            return parse_predictions(content, w, h)

        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {image_path.name}: {e}", file=sys.stderr)
            time.sleep(2)

    return []


def compute_iou(box1, box2):
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def draw_annotated_image(image_path, gt_bboxes, pred_bboxes, target_class, iou_threshold=0.5):
    """Draw GT (green) and predictions (red/blue) on an image.
    Green = GT matched (TP), Red = GT unmatched (FN), Blue = pred unmatched (FP)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Filter by target class
    gt = [b for b in gt_bboxes if b["label"] == target_class]
    pred = [b for b in pred_bboxes if b["label"] == target_class]

    # Match predictions to GT
    matched_gt = set()
    matched_pred = set()

    for pi, p in enumerate(pred):
        best_iou = 0
        best_gi = -1
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            iou = compute_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    # Draw GT: green if matched (TP), yellow if unmatched (FN)
    for gi, g in enumerate(gt):
        x1, y1, x2, y2 = g["bbox"]
        color = (0, 255, 0) if gi in matched_gt else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = "GT" if gi in matched_gt else "GT-FN"
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw predictions: blue if matched (TP), red if unmatched (FP)
    for pi, p in enumerate(pred):
        x1, y1, x2, y2 = p["bbox"]
        color = (255, 0, 0) if pi in matched_pred else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = "TP" if pi in matched_pred else "FP"
        cv2.putText(img, label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def save_annotated_images(eval_dir, img_dir, gt_data, pred_data, target_class, iou_threshold=0.5):
    """Save annotated images with GT + predictions overlaid."""
    vis_dir = eval_dir / "annotated"
    vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for stem, gt_bboxes in sorted(gt_data.items()):
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        pred_bboxes = pred_data.get(stem, [])
        annotated = draw_annotated_image(img_path, gt_bboxes, pred_bboxes, target_class, iou_threshold)
        if annotated is not None:
            cv2.imwrite(str(vis_dir / f"{stem}_annotated.jpg"), annotated)
            count += 1

    return count


def evaluate(gt_data: dict, pred_data: dict, iou_threshold=0.5, target_class=None):
    """
    Compute precision, recall, F1, and per-class metrics.

    gt_data: {stem: [{"label", "bbox"}]}
    pred_data: {stem: [{"label", "bbox", "confidence"}]}
    target_class: if set, only evaluate this class
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_image_results = []

    for stem, gt_bboxes in gt_data.items():
        pred_bboxes = pred_data.get(stem, [])

        # Filter by target class
        if target_class:
            gt_bboxes = [b for b in gt_bboxes if b["label"] == target_class]
            pred_bboxes = [b for b in pred_bboxes if b["label"] == target_class]

        # Match predictions to GT using greedy IoU
        matched_gt = set()
        matched_pred = set()

        # Sort predictions by confidence (descending)
        pred_sorted = sorted(enumerate(pred_bboxes), key=lambda x: -x[1].get("confidence", 1.0))

        for pred_idx, pred in pred_sorted:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_bboxes):
                if gt_idx in matched_gt:
                    continue
                if gt["label"] != pred["label"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                all_tp += 1
                per_class[pred["label"]]["tp"] += 1
            else:
                all_fp += 1
                per_class[pred["label"]]["fp"] += 1

        # Unmatched GT are false negatives
        for gt_idx, gt in enumerate(gt_bboxes):
            if gt_idx not in matched_gt:
                all_fn += 1
                per_class[gt["label"]]["fn"] += 1

        per_image_results.append({
            "image": f"{stem}.jpg",
            "gt_count": len(gt_bboxes),
            "pred_count": len(pred_bboxes),
            "tp": len(matched_gt),
            "fp": len(pred_bboxes) - len(matched_pred),
            "fn": len(gt_bboxes) - len(matched_gt),
        })

    # Compute overall metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-class metrics
    class_metrics = {}
    for cls, counts in per_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        class_metrics[cls] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": all_tp,
            "fp": all_fp,
            "fn": all_fn,
            "total_gt": all_tp + all_fn,
            "total_pred": all_tp + all_fp,
        },
        "per_class": class_metrics,
        "per_image": per_image_results,
        "iou_threshold": iou_threshold,
        "target_class": target_class,
    }


def compute_ap(gt_data, pred_data, class_name, iou_threshold=0.5):
    """Compute AP@0.5 for a single class using all confidence thresholds."""
    # Collect all (confidence, TP/FP) pairs
    scores = []
    tp_fp = []
    total_gt = 0

    for stem, gt_bboxes in gt_data.items():
        gt_class = [b for b in gt_bboxes if b["label"] == class_name]
        pred_class = [b for b in pred_data.get(stem, []) if b["label"] == class_name]

        total_gt += len(gt_class)
        matched_gt = set()

        # Sort predictions by confidence
        pred_sorted = sorted(pred_class, key=lambda x: -x.get("confidence", 1.0))

        for pred in pred_sorted:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_class):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                scores.append(pred.get("confidence", 1.0))
                tp_fp.append(1)
            else:
                scores.append(pred.get("confidence", 1.0))
                tp_fp.append(0)

    if total_gt == 0:
        return 0.0

    # Sort by confidence descending
    sorted_pairs = sorted(zip(scores, tp_fp), key=lambda x: -x[0])
    
    # Compute precision-recall curve
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for conf, is_tp in sorted_pairs:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    if not precisions:
        return 0.0

    # Compute AP using 11-point interpolation
    ap = 0
    for t in [i / 10 for i in range(11)]:
        max_p = 0
        for i, r in enumerate(recalls):
            if r >= t:
                max_p = max(max_p, precisions[i])
        ap += max_p / 11

    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True,
                        help="Path to eval dataset (images/, predictions/)")
    parser.add_argument("--gt-coco", default=None,
                        help="Path to COCO-format GT JSON (overrides eval_dir/ground_truth/)")
    parser.add_argument("--target-class", default="Exit Sign",
                        help="Only evaluate this class (default: Exit Sign)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel inference workers (default: 4)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, use existing predictions")
    parser.add_argument("--vis-output", default=None,
                        help="Save annotated images to this dir (default: eval_dir/annotated/)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving annotated images")
    parser.add_argument("--output", default=None,
                        help="Output report JSON path (default: eval_dir/report.json)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    img_dir = eval_dir / "images"
    gt_dir = eval_dir / "ground_truth"
    pred_dir = eval_dir / "predictions"
    report_path = Path(args.output) if args.output else eval_dir / "report.json"

    if not img_dir.is_dir():
        sys.exit(f"Eval dir not valid: {eval_dir}")

    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_data = {}

    if args.gt_coco:
        gt_coco_path = Path(args.gt_coco)
        if not gt_coco_path.is_file():
            sys.exit(f"GT COCO file not found: {gt_coco_path}")
        coco = json.load(open(gt_coco_path))
        cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        img_id_to_name = {im["id"]: im["file_name"] for im in coco.get("images", [])}
        eval_stems = {p.stem for p in img_dir.glob("*.jpg")}
        for ann in coco.get("annotations", []):
            fname = img_id_to_name.get(ann["image_id"])
            if not fname:
                continue
            stem = Path(fname).stem
            if stem not in eval_stems:
                continue
            x, y, w, h = ann["bbox"]
            label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            gt_data.setdefault(stem, []).append({
                "label": label,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            })
        # Ensure all eval images have an entry (even if empty)
        for stem in eval_stems:
            gt_data.setdefault(stem, [])
        print(f"Loaded GT from COCO: {gt_coco_path}")
    else:
        gt_dir = eval_dir / "ground_truth"
        if not gt_dir.is_dir():
            sys.exit(f"GT dir not found: {gt_dir} (use --gt-coco for COCO format)")
        for gt_file in sorted(gt_dir.glob("*.json")):
            data = json.load(open(gt_file))
            stem = gt_file.stem
            gt_data[stem] = data.get("annotations", [])
        print(f"Loaded GT from per-image files in {gt_dir}")

    print(f"Loaded {len(gt_data)} ground truth files")
    total_gt_bboxes = sum(len(v) for v in gt_data.values())
    print(f"Total GT bboxes: {total_gt_bboxes}")

    # Load or run predictions
    pred_data = {}

    if args.skip_inference:
        for pred_file in sorted(pred_dir.glob("*.json")):
            data = json.load(open(pred_file))
            stem = pred_file.stem
            pred_data[stem] = data.get("predictions", [])
        print(f"Loaded {len(pred_data)} existing predictions")
    else:
        if not API_KEY:
            sys.exit("IW_OLLAMA_API_KEY not set. Source .env first.")

        image_files = sorted(img_dir.glob("*.jpg"))
        print(f"\nRunning Qwen3-VL on {len(image_files)} images with {args.workers} workers...")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_inference, img): img.stem
                for img in image_files
            }
            for i, fut in enumerate(as_completed(futures), 1):
                stem = futures[fut]
                try:
                    preds = fut.result()
                except Exception as e:
                    print(f"  ERROR {stem}: {e}", file=sys.stderr)
                    preds = []

                pred_data[stem] = preds

                # Save prediction
                with open(pred_dir / f"{stem}.json", "w") as f:
                    json.dump({"image": f"{stem}.jpg", "predictions": preds}, f, indent=2)

                if i % 20 == 0 or i == len(image_files):
                    dt = time.time() - t0
                    rate = i / dt if dt > 0 else 0
                    print(f"  [{i}/{len(image_files)}] {dt:.1f}s elapsed ({rate:.1f} img/s, {args.workers} workers)")

        print(f"Inference complete: {len(pred_data)} predictions in {time.time()-t0:.1f}s")

    # Evaluate
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (IoU={args.iou_threshold})")
    print(f"{'='*60}")

    results = evaluate(gt_data, pred_data, args.iou_threshold, args.target_class)

    overall = results["overall"]
    print(f"\nOverall:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}")
    print(f"  Total GT={overall['total_gt']}  Total Pred={overall['total_pred']}")

    print(f"\nPer-class:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print(f"  {'-'*85}")

    for cls, m in sorted(results["per_class"].items(), key=lambda x: -x[1]["support"]):
        print(f"  {cls:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['support']:>8}")

    # Compute AP@0.5 for each class
    print(f"\nmAP@0.5:")
    eval_classes = [args.target_class] if args.target_class else CLASSES
    aps = []
    for cls in eval_classes:
        ap = compute_ap(gt_data, pred_data, cls, args.iou_threshold)
        aps.append(ap)
        print(f"  {cls:<25} AP={ap:.4f}")

    mean_ap = sum(aps) / len(aps) if aps else 0
    print(f"  {'-'*40}")
    print(f"  {'mAP@0.5':<25} {mean_ap:.4f}")

    # Save report
    report = {
        "overall": results["overall"],
        "per_class": results["per_class"],
        "per_image": results["per_image"],
        "ap_50": {cls: ap for cls, ap in zip(eval_classes, aps)},
        "mAP_50": mean_ap,
        "iou_threshold": args.iou_threshold,
        "target_class": args.target_class,
        "num_images": len(gt_data),
        "num_predictions": sum(len(v) for v in pred_data.values()),
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")

    # Save annotated images
    if not args.no_vis:
        vis_dir = Path(args.vis_output) if args.vis_output else eval_dir / "annotated"
        print(f"\nSaving annotated images to {vis_dir}...")
        count = save_annotated_images(
            vis_dir.parent if args.vis_output else eval_dir,
            img_dir, gt_data, pred_data,
            args.target_class, args.iou_threshold,
        )
        if args.vis_output:
            # save_annotated_images uses eval_dir/annotated, override
            import shutil
            src = eval_dir / "annotated"
            if src.exists() and vis_dir != src:
                vis_dir.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    shutil.move(str(f), str(vis_dir / f.name))
                src.rmdir()
        print(f"Saved {count} annotated images")


if __name__ == "__main__":
    main()
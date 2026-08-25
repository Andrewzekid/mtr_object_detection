#!/usr/bin/env python3
"""
Propagate instance masks from one labeled *seed* image to a folder of target
images via DIFT (diffusion-feature correspondence) — a training-free way to turn
a single labeled image into labels for a whole folder, for faster annotation.

The seed image is labeled once (Qwen ``*_result.json`` boxes work directly: a
filled bbox is used as the seed mask). DIFT features are extracted for the seed
and every target, and each seed instance mask is transferred to each target by
nearest-neighbour matching in feature space (``core.dift_inference.propagate_instance``).
No per-target Qwen / SAM3 run is required — one forward pass per image plus a
cosine-similarity match.

This is the cheap counterpart to the SAM3 step (``06_run_sam3.py``): SAM3
re-segments every image from scratch using a text concept + box; DIFT instead
copies a known-good mask from one image to the rest, so the heavy labeling
happens once on the seed and is propagated.

USAGE:
    # Propagate labels from one seed image to every other image in a folder (CPU):
    python scripts/14_run_dift.py \
        --seed-image Datasets/MTR/MTR_new_10_images/1781167858589239000.jpg \
        --seed-annotations Datasets/MTR/MTR_new_10_images_annotations/1781167858589239000_result.json \
        --target-folder Datasets/MTR/MTR_new_10_images \
        --output output/MTR_new_10_images/dift \
        --device cpu

    # Point at the whole Qwen annotations folder; the matching *_result.json is
    # auto-selected by seed-image stem:
    python scripts/14_run_dift.py \
        --seed-image Datasets/MTR/MTR_new_10_images/1781167858589239000.jpg \
        --seed-annotations-folder Datasets/MTR/MTR_new_10_images_annotations \
        --target-folder Datasets/MTR/MTR_new_10_images \
        --output output/MTR_new_10_images/dift --device cpu

    # Manual single-instance seed (no Qwen file needed):
    python scripts/14_run_dift.py \
        --seed-image ./sample.jpg --seed-bbox 100 100 300 300 --label "Ceiling light" \
        --target-folder ./images --output ./out/dift --device cpu

    # Restrict the match to a window around the seed-instance center (helps when
    # targets are sequential video frames and the object doesn't move far):
    python scripts/14_run_dift.py ... --search-radius-frac 0.4

OUTPUT:
    - <output>/overlays/<target_stem>_dift.png   colored per-class mask overlays
    - <output>/masks/<target_stem>_<label>_<i>.png   per-instance binary masks
    - <output>/<target_stem>_dift.json           propagated boxes + scores
                                                  (Qwen ``parsed_output``-shaped,
                                                  so it drops into 08_click_review)
    - <output>/summary.json                      per-target / per-instance summary

NOTE:
    On CPU the SD1.5 UNet forward pass is fp32 and slow (~1-3 min/image on a
    multicore box). Use ``--limit`` to cap the number of targets for a quick
    smoke test, or ``--device cuda`` for full speed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dift_inference import DIFTModel, propagate_instance

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# BGR palette for overlays (mirrors 06_run_sam3.py).
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
]


def color_for(label, label_to_id):
    return CLASS_COLORS[label_to_id[label] % len(CLASS_COLORS)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Propagate seed-image masks to a folder via DIFT correspondence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    p.add_argument("--seed-image", required=True, type=str,
                   help="Path to the labeled seed image.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--seed-annotations", type=str, default=None,
                     help="Qwen *_result.json for the seed image.")
    src.add_argument("--seed-annotations-folder", type=str, default=None,
                     help="Folder of Qwen *_result.json; the file matching the "
                          "seed-image stem is auto-selected.")
    p.add_argument("--seed-bbox", type=float, nargs="+", default=None,
                   help="Manual seed bbox(es) [x1 y1 x2 y2 ...] (no Qwen file).")
    p.add_argument("--label", type=str, nargs="+", default=None,
                   help="Label(s) for --seed-bbox (one per box, or one shared).")
    p.add_argument("--target-folder", required=True, type=str,
                   help="Folder of target images to propagate masks to.")
    p.add_argument("--output", "-o", type=str, default="./output/dift",
                   help="Output directory.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Inference device (default: cuda).")
    p.add_argument("--model-id", type=str,
                   default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="HF model id for the SD UNet/VAE.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N target images (for quick tests).")
    p.add_argument("--min-score", type=float, default=0.30,
                   help="Drop propagated masks whose best cosine sim is below this.")
    p.add_argument("--rel-thresh", type=float, default=0.85,
                   help="threshold = max(0, rel_thresh * best_sim); kept pixels >= threshold. "
                        "Higher -> tighter mask (DIFT sim maps are smooth, so 0.85+ is usually "
                        "needed to avoid the mask bleeding across the whole object band).")
    p.add_argument("--erode-frac", type=float, default=0.06,
                   help="Erode the filled seed bbox by this fraction of its min side "
                        "so the exemplar feature is the object interior, not the border.")
    p.add_argument("--search-radius-frac", type=float, default=None,
                   help="If set, restrict the match to a disk of this fractional "
                        "radius around the seed-instance center in each target.")
    p.add_argument("--save-masks", action="store_true", default=True)
    p.add_argument("--no-save-masks", action="store_false", dest="save_masks")
    return p.parse_args()


def load_seed_instances(args):
    """Return a list of {label, bbox:[x1,y1,x2,y2]} for the seed image."""
    if args.seed_bbox:
        bboxes = args.seed_bbox
        if len(bboxes) % 4 != 0:
            print("Error: --seed-bbox must be groups of 4: x1 y1 x2 y2")
            sys.exit(1)
        boxes = [bboxes[i:i + 4] for i in range(0, len(bboxes), 4)]
        labels = args.label or ["object"] * len(boxes)
        if len(labels) == 1:
            labels = labels * len(boxes)
        if len(labels) != len(boxes):
            print("Error: number of --label entries must match --seed-bbox boxes")
            sys.exit(1)
        return [{"label": l, "bbox": b} for l, b in zip(labels, boxes)]

    # Locate the Qwen result.json.
    ann_path = None
    if args.seed_annotations:
        ann_path = Path(args.seed_annotations)
    elif args.seed_annotations_folder:
        stem = Path(args.seed_image).stem
        cand = Path(args.seed_annotations_folder) / f"{stem}_result.json"
        if cand.exists():
            ann_path = cand
        else:
            # Fall back to any *_result.json containing the stem.
            matches = list(Path(args.seed_annotations_folder).glob(f"{stem}*.json"))
            if matches:
                ann_path = matches[0]
    if ann_path is None or not ann_path.exists():
        print(f"Error: could not find seed annotations (use --seed-annotations or "
              f"--seed-annotations-folder, or --seed-bbox).")
        sys.exit(1)

    with open(ann_path) as f:
        data = json.load(f)
    parsed = data.get("parsed_output") if isinstance(data, dict) else data
    if not parsed:
        print(f"Error: no parsed_output in {ann_path}")
        sys.exit(1)
    instances = []
    for item in parsed:
        bbox = item.get("bbox_2d") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        instances.append({"label": item.get("label", "object"), "bbox": list(bbox)})
    if not instances:
        print(f"Error: no valid bboxes in {ann_path}")
        sys.exit(1)
    return instances


def bbox_to_seed_mask(bbox, h, w, erode_frac):
    """Filled bbox -> eroded bool mask at the seed image's (h, w)."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask.astype(bool)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    side = max(1, int(erode_frac * min(x2 - x1, y2 - y1)))
    if side > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side * 2 + 1, side * 2 + 1))
        mask = cv2.erode(mask, k)
    return mask.astype(bool)


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_overlay(image_bgr, instances, label_to_id):
    overlay = image_bgr.copy()
    for inst in instances:
        if inst["mask"] is None or not inst["mask"].any():
            continue
        m = inst["mask"].astype(bool)
        if m.shape[:2] != overlay.shape[:2]:
            m = cv2.resize(inst["mask"].astype(np.uint8),
                           (overlay.shape[1], overlay.shape[0])).astype(bool)
        color = color_for(inst["label"], label_to_id)
        overlay[m] = (overlay[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        bbox = inst.get("bbox")
        if bbox:
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(overlay, f'{inst["label"]} {inst["score"]:.2f}',
                        (bbox[0], max(15, bbox[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay


def main():
    args = parse_args()
    seed_path = Path(args.seed_image)
    if not seed_path.exists():
        print(f"Error: seed image not found: {seed_path}")
        sys.exit(1)

    seed_instances = load_seed_instances(args)
    print(f"Seed image: {seed_path}")
    print(f"Seed instances ({len(seed_instances)}):")
    for i, inst in enumerate(seed_instances):
        print(f"  [{i}] {inst['label']}  bbox={inst['bbox']}")

    seed_bgr = cv2.imread(str(seed_path))
    if seed_bgr is None:
        print(f"Error: could not read seed image: {seed_path}")
        sys.exit(1)
    h, w = seed_bgr.shape[:2]

    # Build seed masks (filled + eroded bbox) in seed-image pixel coords.
    for inst in seed_instances:
        inst["seed_mask"] = bbox_to_seed_mask(inst["bbox"], h, w, args.erode_frac)

    # Gather targets (exclude the seed itself by filename).
    target_folder = Path(args.target_folder)
    targets = sorted([f for f in target_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                      and f.name != seed_path.name])
    if args.limit:
        targets = targets[:args.limit]
    if not targets:
        print(f"Error: no target images in {target_folder}")
        sys.exit(1)
    print(f"\n{len(targets)} target image(s); device={args.device}")

    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Load DIFT model (once) and extract seed features.
    print(f"\nLoading DIFT model ({args.model_id}) on {args.device} ...")
    model = DIFTModel.get(model_id=args.model_id, device=args.device)
    print("Extracting seed features ...")
    seed_feat = model.extract_features(cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB))
    fr = seed_feat.shape[0]  # feature resolution (e.g. 64)

    # Seed masks are at image res; downsample to feat_res for correspondence.
    for inst in seed_instances:
        inst["seed_mask_fr"] = cv2.resize(
            inst["seed_mask"].astype(np.uint8), (fr, fr),
            interpolation=cv2.INTER_NEAREST).astype(bool)

    # Stable class colors.
    label_to_id = {}
    for inst in seed_instances:
        label_to_id.setdefault(inst["label"], len(label_to_id))

    # Optional search window: per-instance center in fraction coords.
    def center_frac(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (float(ys.mean()) / mask.shape[0], float(xs.mean()) / mask.shape[1])

    summary = {
        "seed_image": str(seed_path),
        "seed_instances": [{"label": i["label"], "bbox": i["bbox"]} for i in seed_instances],
        "device": args.device,
        "targets": [],
    }

    for ti, tpath in enumerate(targets, 1):
        print(f"\n[{ti}/{len(targets)}] {tpath.name}")
        tbgr = cv2.imread(str(tpath))
        if tbgr is None:
            print("  could not read; skipping")
            continue
        tfeat = model.extract_features(cv2.cvtColor(tbgr, cv2.COLOR_BGR2RGB))

        out_instances = []
        for si, inst in enumerate(seed_instances):
            prop_kwargs = dict(sim_floor=0.0, rel_thresh=args.rel_thresh,
                               min_score=args.min_score)
            if args.search_radius_frac is not None:
                c = center_frac(inst["seed_mask"])
                if c is not None:
                    prop_kwargs["search_center"] = c
                    prop_kwargs["search_radius_frac"] = args.search_radius_frac
            mask_fr, score = propagate_instance(
                seed_feat, inst["seed_mask_fr"], tfeat, **prop_kwargs)
            if mask_fr.any():
                mask = cv2.resize(mask_fr.astype(np.uint8),
                                  (tbgr.shape[1], tbgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = None
            bbox = mask_to_bbox(mask) if mask is not None else None
            out_instances.append({
                "label": inst["label"],
                "score": float(score),
                "bbox": bbox,
                "mask": mask,
                "seed_idx": si,
            })
            status = f"bbox={bbox} score={score:.3f}" if bbox else f"no match (score={score:.3f})"
            print(f"    {inst['label']:<20} {status}")

        # Save overlay.
        overlay = draw_overlay(tbgr, out_instances, label_to_id)
        cv2.imwrite(str(out_dir / "overlays" / f"{tpath.stem}_dift.png"), overlay)

        # Save masks + build JSON (Qwen parsed_output-shaped).
        parsed_output = []
        for j, oi in enumerate(out_instances):
            if oi["mask"] is None:
                continue
            if args.save_masks:
                safe_label = oi["label"].replace(" ", "_")
                mp = out_dir / "masks" / f"{tpath.stem}_{safe_label}_{j}.png"
                cv2.imwrite(str(mp), (oi["mask"].astype(np.uint8) * 255))
            parsed_output.append({"bbox_2d": oi["bbox"], "label": oi["label"],
                                  "score": oi["score"]})

        result = {
            "image": str(tpath),
            "model": "dift-propagation",
            "seed_image": str(seed_path),
            "parsed_output": parsed_output,
            "num_instances": len(parsed_output),
        }
        with open(out_dir / f"{tpath.stem}_dift.json", "w") as f:
            json.dump(result, f, indent=2)

        summary["targets"].append({
            "image": str(tpath),
            "num_instances": len(parsed_output),
            "instances": [{"label": o["label"], "bbox": o["bbox"],
                           "score": o["score"]} for o in out_instances],
        })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total = sum(t["num_instances"] for t in summary["targets"])
    print(f"\n{'=' * 60}")
    print(f"DIFT propagation complete.")
    print(f"  targets: {len(targets)}   propagated instances: {total}")
    print(f"  overlays: {out_dir / 'overlays'}")
    print(f"  per-target JSON + summary.json: {out_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fast parallel uploader for a YOLO seg dataset to a Roboflow project.

Reads ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT from env (.env)
and uploads every image + its YOLO label to the given split in parallel.

USAGE:
    set -a && . ./.env && set +a && \
    python scripts/14_upload_to_roboflow.py \
        --dataset-dir output/MTR_4k/yolo_seg_dataset \
        --project mtrexitsignonly \
        --workers 8
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from roboflow import Roboflow

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def upload_one(proj, img_path, label_path, split, retries=3):
    """Upload a single image+label with retry. Returns (name, ok, err)."""
    last_err = None
    for _ in range(retries):
        try:
            proj.upload(
                image_path=str(img_path),
                annotation_path=str(label_path) if label_path else None,
                split=split,
                num_retry_uploads=1,
            )
            return (img_path.name, True, None)
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0)
    return (img_path.name, False, last_err)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True,
                        help="YOLO dataset dir with images/{train,val,test}, "
                             "labels/{train,val,test}, data.yaml")
    parser.add_argument("--workspace", default=None,
                        help="Roboflow workspace (default: env ROBOFLOW_WORKSPACE "
                             "or the key's default)")
    parser.add_argument("--project", required=True,
                        help="Roboflow project id (e.g. mtrexitsignonly)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel upload workers (default: 8)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                        help="Splits to upload (default: train val test)")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set. Source .env first: set -a && . ./.env && set +a")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        sys.exit(f"Dataset dir not found: {dataset_dir}")

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace()
    if args.workspace:
        ws = rf.workspace(args.workspace)
    proj = ws.project(args.project)
    print(f"Uploading to: {ws.name}/{proj.name} ({proj.type})")

    total = ok = failed = 0
    jobs = []
    for split in args.splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.is_dir():
            print(f"  skip {split}: no image dir at {img_dir}")
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = lbl_dir / (img.stem + ".txt")
            label_path = str(label) if label.exists() else None
            jobs.append((split, img, label_path))

    total = len(jobs)
    print(f"Uploading {total} images with {args.workers} workers...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(upload_one, proj, img, lbl, split): (split, img.name)
                   for (split, img, lbl) in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            name, success, err = fut.result()
            if success:
                ok += 1
            else:
                failed += 1
                print(f"  FAIL {name}: {err}", file=sys.stderr)
            if i % 100 == 0 or i == total:
                dt = time.time() - t0
                print(f"  [{i}/{total}] ok={ok} fail={failed} ({dt:.1f}s, {i/dt:.1f}/s)")

    print(f"\nDone: {ok}/{total} ok, {failed} failed in {time.time()-t0:.1f}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
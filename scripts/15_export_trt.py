#!/usr/bin/env python3
"""Export trained YOLO checkpoints to TensorRT engines for fast tracking.

TensorRT FP16 engines typically give a 2-4x inference speedup over PyTorch
on desktop GPUs (e.g. RTX 4090) at fixed imgsz. Export once, then pass the
resulting .engine path to scripts/11_run_tracking.py --model.

USAGE:
    python scripts/15_export_trt.py \
        --model runs/segment/output/training/yolo_training/HKU_GH/yolo26lv3/yolo_training/weights/best.pt \
        --imgsz 768

    # Multiple models at once:
    python scripts/15_export_trt.py --imgsz 768 --model best1.pt best2.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_engine(model_path: str, imgsz: int, workspace: int) -> Path:
    model = YOLO(model_path)
    out = model.export(
        format="engine",
        half=True,
        imgsz=imgsz,
        device=0,
        workspace=workspace,
        simplify=True,
        verbose=False,
    )
    engine = Path(out)
    print(f"Exported {model_path} -> {engine}")
    return engine


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", nargs="+", required=True,
                   help="One or more YOLO .pt checkpoints")
    p.add_argument("--imgsz", type=int, default=768,
                   help="Engine input size; must match tracking --imgsz "
                        "(default: 768)")
    p.add_argument("--workspace", type=int, default=4,
                   help="TensorRT workspace size in GiB (default: 4)")
    args = p.parse_args()

    for m in args.model:
        if not Path(m).exists():
            print(f"Error: {m} not found, skipping")
            continue
        try:
            export_engine(m, args.imgsz, args.workspace)
        except Exception as exc:
            print(f"Error exporting {m}: {exc}")


if __name__ == "__main__":
    main()

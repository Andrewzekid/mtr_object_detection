#!/usr/bin/env python3
"""
Visualize YOLO segmentation model predictions on test images.

Runs the trained YOLO-seg model on test images and draws segmentation masks
with class labels and confidence scores.

USAGE:
    python scripts/14_visualize_seg_predictions.py \
        --model runs/segment/output/training/yolo_training/weights/best.pt \
        --images-dir train_yolo_seg_new/images/test \
        --output output/vis_seg_predictions \
        --conf 0.25
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO seg predictions")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--images-dir", "-i", type=str, required=True, help="Directory of test images")
    parser.add_argument("--output", "-o", type=str, default="./output/vis_seg_predictions", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--mask-alpha", type=float, default=0.4, help="Mask overlay alpha (0-1)")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        sys.exit(1)

    model_path = Path(args.model)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: Images dir not found: {images_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Get class names from model
    class_names = list(model.names.values()) if hasattr(model, 'names') and model.names else [f"class_{i}" for i in range(10)]
    print(f"Classes: {class_names}")

    # Find images
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Found {len(image_files)} images")
    print(f"Output: {output_dir}")
    print(f"Confidence: {args.conf}")

    successful = 0
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")

        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        if not results:
            print("  No results")
            continue

        result = results[0]
        img = result.orig_img.copy()

        # Draw masks
        if result.masks is not None and len(result.masks) > 0:
            mask_overlay = img.copy()
            boxes = result.boxes
            masks = result.masks

            for i in range(len(masks)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

                # Get mask as binary array
                mask = masks.data[i].cpu().numpy().astype(np.uint8)
                # Resize mask to original image size if needed
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Draw filled mask
                mask_bool = mask > 0
                mask_overlay[mask_bool] = color

                # Draw contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color, 2)

                # Draw label at centroid
                M = cv2.moments(contours[0]) if contours else None
                if M and M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (cx - tw//2 - 2, cy - th//2 - 2), (cx + tw//2 + 2, cy + th//2 + 2), color, -1)
                    cv2.putText(img, label, (cx - tw//2, cy + th//2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Blend mask overlay
            cv2.addWeighted(mask_overlay, args.mask_alpha, img, 1 - args.mask_alpha, 0, img)

        # Save
        out_path = output_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), img)
        successful += 1

    print(f"\nVisualization complete! {successful}/{len(image_files)} images processed.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
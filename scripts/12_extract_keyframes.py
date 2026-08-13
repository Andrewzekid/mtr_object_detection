#!/usr/bin/env python3
"""
Extract keyframes from a frame folder (or video) for the keyframe annotation
pipeline.

The annotation pipeline seeds/reviews boxes only on a sparse set of keyframes
(default every 20th frame) and propagates them to the remaining frames with
`13_interpolate_tracks.py`. This script selects the keyframes, materializes
them (symlink by default), and writes a manifest the interpolator consumes.

Frame order matters: frames are sorted lexicographically. For the MTR rosbag
extracts the filenames are fixed-width nanosecond timestamps, so lexicographic
order is chronological. A `--video` source is also supported, in which case the
frames are decoded in playback order.

USAGE:
    # Every 20th frame from a folder of images (default stride)
    python scripts/12_extract_keyframes.py \
        --image-folder Datasets/MTR/rosbags/MTR_metacam_right \
        --output-dir Datasets/MTR/MTR_keyframes \
        --every 20

    # From a video file instead of an image folder
    python scripts/12_extract_keyframes.py \
        --video source.mp4 \
        --output-dir Datasets/MTR/MTR_keyframes \
        --every 20 --mode copy

    # MTR 4k exit-sign dataset, stride 5 (relaunch with denser keyframes;
    # overwrites the previous stride-10 manifest, existing symlinks preserved)
    python scripts/12_extract_keyframes.py \
        --image-folder Datasets/MTR/MTR_4k_dataset_exit_signs \
        --output-dir Datasets/MTR/MTR_4k_keyframes \
        --every 5 --mode symlink

OUTPUT:
    - <output-dir>/*.jpg                 keyframe images (symlinked or copied)
    - <output-dir>/keyframe_manifest.json  manifest for the interpolator
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

# Add project root to path so the shared tracker helpers resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tracking_utils import find_image_files

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_frame_files(image_folder: Path):
    """Return image filenames in a folder, sorted (chronological for timestamps)."""
    files = sorted(
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return files


def extract_frames_from_video(video_path: Path, output_dir: Path, every: int,
                              mode: str, image_ext: str = ".jpg"):
    """Decode a video, keeping every Nth frame as a keyframe image."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_files = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every == 0:
            name = f"frame_{frame_idx:08d}{image_ext}"
            dst = output_dir / name
            if mode == "none":
                # Still record the (virtual) name so the manifest is consistent.
                pass
            elif mode == "copy":
                cv2.imwrite(str(dst), frame)
            dst_name = name
            if mode == "none":
                # No file written; record the source index for later lookup.
                dst_name = f"frame_{frame_idx:08d}{image_ext}"
            frame_files.append((frame_idx, dst_name))
        frame_idx += 1
    cap.release()
    return frame_files, frame_idx


def materialize(src: Path, dst: Path, mode: str):
    """Symlink, copy, or skip materializing a keyframe image."""
    if mode == "none":
        return
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Symlinks may fail on some filesystems / shares; fall back to copy.
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract keyframes (every Nth frame) and write a manifest "
                    "for the keyframe annotation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-folder",
                     help="Folder of chronologically ordered frames.")
    src.add_argument("--video",
                     help="Video file to decode keyframes from.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write keyframe images + manifest.")
    parser.add_argument("--every", type=int, default=20,
                        help="Keep every Nth frame (default: 20).")
    parser.add_argument("--start", type=int, default=0,
                        help="Index of the first keyframe (default: 0).")
    parser.add_argument("--mode", choices=["symlink", "copy", "none"],
                        default="symlink",
                        help="How to materialize keyframe images. 'none' only "
                             "writes the manifest and reads frames from the "
                             "source folder at interpolation time (default: symlink).")
    args = parser.parse_args()

    if args.every <= 0:
        parser.error("--every must be a positive integer")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_folder:
        image_folder = Path(args.image_folder)
        if not image_folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        all_files = list_frame_files(image_folder)
        total = len(all_files)
        if total == 0:
            raise RuntimeError(f"No images found in {image_folder}")

        # Keyframes: start, start+every, ... up to (but not past) the last frame.
        keyframe_indices = list(range(args.start, total, args.every))
        if not keyframe_indices:
            keyframe_indices = [args.start] if args.start < total else []

        keyframes = []
        for kidx, fidx in enumerate(keyframe_indices):
            src = all_files[fidx]
            if args.mode != "none":
                dst = output_dir / src.name
                materialize(src, dst, args.mode)
            keyframes.append({
                "keyframe_id": kidx,
                "frame_idx": fidx,
                "file_name": src.name,
            })
        frame_names = [p.name for p in all_files]
        source = str(image_folder.resolve())
    else:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        recorded, total = extract_frames_from_video(
            video_path, output_dir, args.every, args.mode)
        keyframes = [
            {"keyframe_id": kidx, "frame_idx": fidx, "file_name": name}
            for kidx, (fidx, name) in enumerate(recorded)
        ]
        frame_names = [f"frame_{i:08d}.jpg" for i in range(total)]
        source = str(video_path.resolve())

    manifest = {
        "source": source,
        "source_type": "image_folder" if args.image_folder else "video",
        "stride": args.every,
        "start": args.start,
        "total_frames": total,
        "num_keyframes": len(keyframes),
        "frames": frame_names,
        "keyframes": keyframes,
    }

    manifest_path = output_dir / "keyframe_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Extracted {len(keyframes)} keyframes (stride {args.every}) "
          f"from {total} frames")
    print(f"  Keyframes -> {output_dir}")
    print(f"  Manifest  -> {manifest_path}")


if __name__ == "__main__":
    main()
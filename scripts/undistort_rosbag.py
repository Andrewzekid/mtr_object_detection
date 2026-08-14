"""
Undistort Fisheye Images from a Normal Image Folder

This script processes a folder of images, applies fisheye undistortion
using calibration parameters, and saves the undistorted images to an output folder.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from tqdm import tqdm


# Global lock for thread-safe output writing
output_lock = Lock()


def load_calibration(calibration_path):
    """Load calibration parameters from JSON file."""
    with open(calibration_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_camera_params(calibration_data, camera_name, image_width, image_height):
    """Extract camera intrinsic and distortion parameters for a specific camera."""
    for cam in calibration_data.get('cameras', []):
        if cam.get('name') == camera_name:
            intrinsic = cam['intrinsic']
            distortion_params = cam['distortion']['params']

            calib_width = cam['width']
            calib_height = cam['height']
            scale_x = image_width / calib_width
            scale_y = image_height / calib_height

            camera_matrix = np.array([
                [intrinsic['fl_x'] * scale_x, 0, intrinsic['cx'] * scale_x],
                [0, intrinsic['fl_y'] * scale_y, intrinsic['cy'] * scale_y],
                [0, 0, 1]
            ], dtype=np.float64)

            distortion_coeffs = np.array([
                distortion_params['k1'],
                distortion_params['k2'],
                distortion_params['k3'],
                distortion_params['k4']
            ], dtype=np.float64)

            return camera_matrix, distortion_coeffs

    raise ValueError(f"Camera '{camera_name}' not found in calibration data")


def create_undistort_maps(camera_matrix, distortion_coeffs, img_width, img_height):
    """Pre-compute undistort maps for faster processing."""
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        np.eye(3),
        camera_matrix,
        (img_width, img_height),
        cv2.CV_32FC1,
    )
    return map1, map2


def undistort_fisheye_image_fast(image, map1, map2):
    """Apply undistortion with precomputed maps."""
    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def process_single_image(args):
    img_path, images_root, output_root, map1, map2 = args

    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return None, f"Could not read {img_path}"

        undistorted = undistort_fisheye_image_fast(image, map1, map2)
        rel_path = img_path.relative_to(images_root)
        output_file = output_root / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), undistorted)
        return output_file, None
    except Exception as e:
        return None, f"Error processing {img_path}: {e}"


def group_images_by_size(image_files):
    groups = {}
    for img_path in image_files:
        sample = cv2.imread(str(img_path))
        if sample is None:
            continue
        h, w = sample.shape[:2]
        groups.setdefault((w, h), []).append(img_path)
    return groups


def process_image_folder_parallel(images_root, output_root,
                                  camera_matrix, distortion_coeffs,
                                  image_extensions, num_workers=20,
                                  recursive=False):
    image_files = []
    if recursive:
        for ext in image_extensions:
            image_files.extend(images_root.rglob(f"*{ext}"))
    else:
        for ext in image_extensions:
            image_files.extend(images_root.glob(f"*{ext}"))

    if not image_files:
        return 0, 0

    image_files = sorted(image_files)
    total_images = len(image_files)

    size_to_maps = {}
    for img_path in tqdm(image_files, desc="Preparing maps", unit="img", leave=False):
        sample = cv2.imread(str(img_path))
        if sample is None:
            continue
        h, w = sample.shape[:2]
        if (w, h) not in size_to_maps:
            size_to_maps[(w, h)] = create_undistort_maps(
                camera_matrix, distortion_coeffs, w, h
            )

    total_processed = 0
    total_errors = 0

    with tqdm(total=total_images, desc="Processing", unit="img",
              bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for size, group in group_images_by_size(image_files).items():
            map1, map2 = size_to_maps[size]
            args_list = [
                (img_path, images_root, output_root, map1, map2)
                for img_path in group
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_single_image, arg)
                           for arg in args_list]
                for future in concurrent.futures.as_completed(futures):
                    result, error = future.result()
                    if result is not None:
                        total_processed += 1
                    else:
                        if error:
                            print(f"  {error}")
                        total_errors += 1
                    pbar.update(1)

    return total_processed, total_errors


def process_images_folder(images_root, output_root, calibration_path,
                          camera_name, num_workers=20, recursive=False):
    images_root = Path(images_root)
    output_root = Path(output_root)
    calibration_data = load_calibration(calibration_path)

    if not images_root.exists() or not images_root.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    if recursive:
        all_images = [
            p for ext in image_extensions for p in images_root.rglob(f"*{ext}")]
    else:
        all_images = [
            p for ext in image_extensions for p in images_root.glob(f"*{ext}")]

    if not all_images:
        raise RuntimeError(f"No images found in {images_root}")

    sample_img = cv2.imread(str(all_images[0]))
    if sample_img is None:
        raise RuntimeError(f"Unable to read sample image: {all_images[0]}")

    h, w = sample_img.shape[:2]
    camera_matrix, distortion_coeffs = get_camera_params(
        calibration_data, camera_name, w, h
    )

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Undistorting images in {images_root}")
    print(f"Output folder: {output_root}")
    print(f"Using camera: {camera_name}")

    processed, errors = process_image_folder_parallel(
        images_root,
        output_root,
        camera_matrix,
        distortion_coeffs,
        image_extensions,
        num_workers=num_workers,
        recursive=recursive,
    )

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total images processed: {processed}")
    print(f"Total errors: {errors}")
    print(f"Output saved to: {output_root}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Undistort fisheye images from a folder of images"
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Input image folder containing distorted images"
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory to save undistorted images"
    )
    parser.add_argument(
        "--calibration",
        default="/workspaces/detection_ws/config/calibration.json",
        help="Path to calibration.json file"
    )
    parser.add_argument(
        "--camera-name",
        default="left",
        help="Camera name in calibration.json to use"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=20,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process image subdirectories"
    )

    args = parser.parse_args()

    images_root = Path(args.images_root).resolve()
    output_root = Path(args.output_root).resolve(
    ) if args.output_root else images_root.parent / f"{images_root.name}_undistorted"

    print(f"Images root: {images_root}")
    print(f"Output root: {output_root}")
    print(f"Calibration file: {args.calibration}")
    print(f"Camera name: {args.camera_name}")
    print(f"Parallel workers: {args.workers}")
    print(f"Recursive: {args.recursive}")

    process_images_folder(
        images_root=images_root,
        output_root=output_root,
        calibration_path=args.calibration,
        camera_name=args.camera_name,
        num_workers=args.workers,
        recursive=args.recursive,
    )
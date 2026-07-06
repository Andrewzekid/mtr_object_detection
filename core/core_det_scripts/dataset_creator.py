#    core/dataset_creator.py - Undistort, random select, train/test/val split.
#
#    USAGE (OOP):
#        from core.dataset_creator import DatasetCreator
#        c = DatasetCreator()
#
#        # 1) Undistort wide-angle / fisheye images:
#        import numpy as np
#        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#        D = np.array([k1, k2, p1, p2, k3])
#        c.undistort_camera(
#            camera_matrix=K, dist_coeffs=D,
#            input_dir="./raw_images", output_dir="./undistorted_images",
#        )
#
#        # 2) Random subset of N images:
#        c.select_random_images(count=100, src="./all", dest="./sample", seed=42)
#
#        # 3) Train / test / val split:
#        c.split_dataset(
#            ratios=[0.7, 0.15, 0.15],
#            src="./images", output_dir="./split", seed=42,
#        )
#
#        # 4) Counts of an existing split:
#        print(c.get_split_statistics("./split"))
#
#    USAGE (BACKWARD-COMPATIBLE FUNCTION):
#        from core.dataset_creator import split_dataset
#        split_dataset("./images", "./split", ratios=[0.7, 0.15, 0.15])
#
#    RUN AS A ONE-LINER:
#        python -c "from core.dataset_creator import DatasetCreator; \
#            DatasetCreator().split_dataset(ratios=[0.7,0.15,0.15], \
#            src='./images', output_dir='./split', seed=42)"
#
#    ARGUMENTS:
#        undistort_camera(camera_matrix, dist_coeffs,
#                         input_dir, output_dir)         - OpenCV K matrix + 4-5
#                                                         distortion coefficients
#        select_random_images(count, src, dest, seed)   - integer count of images
#        split_dataset(ratios, src, output_dir, seed)   - ratios list sums to 1.0
#        get_split_statistics(split_dir)                - reads existing split
#
#    REQUIREMENTS:
#        pip install opencv-python-headless numpy

"""
Dataset creation utilities: undistortion, random selection, and train/test/val split.
OOP paradigm with DatasetCreator class.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
import random
import shutil
from collections import defaultdict


class DatasetCreator:
    """Class for handling dataset creation operations."""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(self, input_dir: Optional[str | Path] = None, output_dir: Optional[str | Path] = None):
        """Initialize with input and output directories."""
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None

    def set_paths(self, input_dir: str | Path, output_dir: str | Path):
        """Set input and output paths."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def undistort_camera(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        input_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Fix wide-angle/metacam distortion in images."""
        input_path = Path(input_dir) if input_dir else self.input_dir
        output_path = Path(output_dir) if output_dir else self.output_dir

        if not input_path or not output_path:
            return {"success": False, "error": "Input and output paths must be specified"}

        output_path.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]
        total = len(image_files)

        if total == 0:
            return {"success": False, "error": "No images found in input directory"}

        processed = 0
        errors = []

        for i, img_file in enumerate(image_files):
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True, "processed": processed}

            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    errors.append(f"Could not read {img_file.name}")
                    continue

                h, w = img.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                )
                undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

                x, y, w_roi, h_roi = roi
                if w_roi > 0 and h_roi > 0:
                    undistorted = undistorted[y:y+h_roi, x:x+w_roi]

                cv2.imwrite(str(output_path / img_file.name), undistorted)
                processed += 1

                if log_callback:
                    log_callback(f"Undistorted: {img_file.name}")
            except Exception as e:
                errors.append(f"Error processing {img_file.name}: {str(e)}")

            if progress_callback:
                progress_callback(int(((i + 1) / total) * 100))
            if status_callback:
                status_callback(f"Processing {i + 1}/{total}")

        return {
            "success": len(errors) == 0,
            "processed": processed,
            "total": total,
            "errors": errors,
            "output_dir": str(output_path),
        }

    def select_random_images(
        self,
        count: int,
        src: Optional[str | Path] = None,
        dest: Optional[str | Path] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Sub-sample dataset by selecting random images."""
        src_path = Path(src) if src else self.input_dir
        dest_path = Path(dest) if dest else self.output_dir

        if not src_path or not dest_path:
            return {"success": False, "error": "Source and destination paths must be specified"}

        dest_path.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in src_path.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]

        if len(image_files) == 0:
            return {"success": False, "error": "No images found in source directory"}

        if seed is not None:
            random.seed(seed)

        actual_count = min(count, len(image_files))
        selected = random.sample(image_files, actual_count)

        copied = 0
        errors = []

        for i, img_file in enumerate(selected):
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True, "copied": copied}

            try:
                dest_file = dest_path / img_file.name
                shutil.copy2(img_file, dest_file)

                # Copy labels if exist
                label_file = src_path / "labels" / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = dest_path / "labels" / f"{img_file.stem}.txt"
                    label_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_file, label_dest)

                copied += 1
                if log_callback:
                    log_callback(f"Copied: {img_file.name}")
            except Exception as e:
                errors.append(f"Error copying {img_file.name}: {str(e)}")

            if progress_callback:
                progress_callback(int(((i + 1) / actual_count) * 100))
            if status_callback:
                status_callback(f"Copying {i + 1}/{actual_count}")

        return {
            "success": len(errors) == 0,
            "copied": copied,
            "requested": count,
            "available": len(image_files),
            "errors": errors,
            "output_dir": str(dest_path),
        }

    def split_dataset(
        self,
        ratios: List[float] = [0.7, 0.15, 0.15],
        src: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Split dataset into Train/Test/Val sets."""
        src_path = Path(src) if src else self.input_dir
        out_path = Path(output_dir) if output_dir else self.output_dir

        if not src_path or not out_path:
            return {"success": False, "error": "Source and output paths must be specified"}

        if abs(sum(ratios) - 1.0) > 0.001:
            return {"success": False, "error": "Ratios must sum to 1.0"}

        image_files = [f for f in src_path.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]

        total = len(image_files)
        if total == 0:
            return {"success": False, "error": "No images found in source directory"}

        if seed is not None:
            random.seed(seed)
        random.shuffle(image_files)

        train_end = int(total * ratios[0])
        test_end = train_end + int(total * ratios[1])

        splits = {
            "train": image_files[:train_end],
            "test": image_files[train_end:test_end],
            "val": image_files[test_end:],
        }

        result = {"success": True, "splits": {}, "total": total}

        # Create directories
        for split_name in splits:
            split_dir = out_path / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        total_files = sum(len(f) for f in splits.values())
        processed_files = 0

        for split_name, files in splits.items():
            split_images_dir = out_path / split_name / "images"
            split_labels_dir = out_path / split_name / "labels"

            if log_callback:
                log_callback(f"Creating {split_name} split with {len(files)} images")

            for img_file in files:
                if is_cancelled and is_cancelled():
                    return {"success": False, "cancelled": True, "processed": processed_files}

                try:
                    shutil.copy2(img_file, split_images_dir / img_file.name)
                    label_file = src_path / "labels" / f"{img_file.stem}.txt"
                    if label_file.exists():
                        shutil.copy2(label_file, split_labels_dir / f"{img_file.stem}.txt")
                    processed_files += 1
                except Exception as e:
                    result["success"] = False
                    if "errors" not in result:
                        result["errors"] = []
                    result["errors"].append(f"Error copying {img_file.name}: {str(e)}")

                if progress_callback:
                    progress_callback(int((processed_files / total_files) * 100))
                if status_callback:
                    status_callback(f"Splitting: {processed_files}/{total_files}")

        for split_name, files in splits.items():
            result["splits"][split_name] = {
                "count": len(files),
                "ratio": len(files) / total if total > 0 else 0,
                "path": str(out_path / split_name),
            }

        return result

    def get_split_statistics(self, split_dir: str | Path) -> Dict:
        """Get statistics about a split dataset."""
        split_path = Path(split_dir)
        stats = {}

        for split_name in ["train", "test", "val"]:
            split_img_dir = split_path / split_name / "images"
            if split_img_dir.exists():
                image_files = [f for f in split_img_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                stats[split_name] = {
                    "image_count": len(image_files),
                }

        return stats


# Backward compatibility
def undistort_camera(input_dir, output_dir, camera_matrix, dist_coeffs,
                     progress_callback=None, status_callback=None,
                     log_callback=None, is_cancelled=None):
    creator = DatasetCreator()
    return creator.undistort_camera(camera_matrix, dist_coeffs, input_dir, output_dir,
                                    progress_callback, status_callback, log_callback, is_cancelled)


def select_random_images(src, dest, count, seed=None,
                         progress_callback=None, status_callback=None,
                         log_callback=None, is_cancelled=None):
    creator = DatasetCreator()
    return creator.select_random_images(count, src, dest, seed,
                                        progress_callback, status_callback, log_callback, is_cancelled)


def split_dataset(src, output_dir, ratios=[0.7, 0.15, 0.15], seed=None,
                  progress_callback=None, status_callback=None,
                  log_callback=None, is_cancelled=None):
    creator = DatasetCreator()
    return creator.split_dataset(ratios, src, output_dir, seed,
                                 progress_callback, status_callback, log_callback, is_cancelled)

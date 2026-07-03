#    core/data_processor.py - Image augmentation & dataset statistics.
#
#    USAGE (OOP):
#        from core.data_processor import DataProcessor
#        p = DataProcessor({
#            "input_dir": "./data/images",
#            "output_dir": "./output/augmented",
#            "augmentation_types": ["flip_horizontal", "rotate", "mosaic"],
#            "multiplier": 3,
#            "rotation_range": (-15, 15),
#        })
#        print(p.augment_dataset())
#        print(p.get_statistics("./data/images"))
#
#    USAGE (BACKWARD-COMPATIBLE FUNCTION):
#        from core.data_processor import augment_dataset, get_statistics
#        augment_dataset({"input_dir": "./in", "output_dir": "./out"})
#        print(get_statistics("./in"))
#
#    RUN AS A ONE-LINER:
#        python -c "from core.data_processor import DataProcessor; \
#            DataProcessor({'input_dir':'./data','output_dir':'./out'}).augment_dataset()"
#
#    ARGUMENTS (to DataProcessor.__init__ config_dict):
#        input_dir           - source directory (with images/ and labels/ subdirs)
#        output_dir          - destination for augmented images + labels
#        augmentation_types  - list of: flip_horizontal, flip_vertical, rotate,
#                              brightness, contrast, mosaic
#        multiplier          - how many augmented copies per image (1-10)
#        rotation_range      - (min_degrees, max_degrees) for rotate
#        brightness_range    - (min_factor, max_factor) for brightness/contrast
#
#    REQUIREMENTS:
#        pip install opencv-python-headless numpy
#
#    SEE ALSO:
#        core/dataset_creator.py - downstream split step
#        core/model_trainer.py   - downstream training step

"""
Data processing utilities: augmentation and dataset statistics.
OOP paradigm with DataProcessor class.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from collections import defaultdict
import random
import shutil


class DataProcessor:
    """Class for handling data augmentation and statistics."""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with configuration dictionary."""
        self.config = config_dict or {}
        self.input_dir = Path(self.config.get("input_dir", ""))
        self.output_dir = Path(self.config.get("output_dir", ""))
        self.augmentation_types = self.config.get("augmentation_types", ["flip_horizontal"])
        self.multiplier = self.config.get("multiplier", 3)
        self.rotation_range = self.config.get("rotation_range", (-15, 15))
        self.brightness_range = self.config.get("brightness_range", (0.7, 1.3))
    
    def get_image_files(self, directory: Optional[Path] = None) -> List[Path]:
        """Get all image files in a directory."""
        dir_path = directory or self.input_dir
        return [f for f in dir_path.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]
    
    def augment_dataset(
        self,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Apply augmentation transformations to dataset."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = self.output_dir / "images"
        output_labels_dir = self.output_dir / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = self.get_image_files()
        total = len(image_files)
        
        if total == 0:
            return {"success": False, "error": "No images found in input directory"}
        
        total_augmented = 0
        errors = []
        
        for i, img_file in enumerate(image_files):
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True, "augmented": total_augmented}
            
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    errors.append(f"Could not read {img_file.name}")
                    continue
                
                # Read labels
                label_file = self.input_dir / "labels" / f"{img_file.stem}.txt"
                labels = []
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        labels = [line.strip().split() for line in f.readlines()]
                
                # Apply mosaic augmentation if selected
                if "mosaic" in self.augmentation_types and i + 3 < total:
                    mosaic_img, mosaic_labels = self.apply_mosaic(
                        [img, cv2.imread(str(image_files[i+1])),
                         cv2.imread(str(image_files[i+2])), cv2.imread(str(image_files[i+3]))],
                        [labels] + [self._read_labels(image_files[k], self.input_dir) for k in range(i+1, i+4)]
                    )
                    aug_name = f"{img_file.stem}_mosaic{img_file.suffix}"
                    cv2.imwrite(str(output_images_dir / aug_name), mosaic_img)
                    self._save_labels(output_labels_dir, f"{img_file.stem}_mosaic", mosaic_labels)
                    total_augmented += 1
                    if log_callback:
                        log_callback(f"Augmented: {aug_name} (mosaic)")
                
                # Generate other augmented versions
                for aug_idx in range(self.multiplier):
                    aug_type = random.choice([t for t in self.augmentation_types if t != "mosaic"] or self.augmentation_types)
                    aug_img, aug_labels = self.apply_augmentation(
                        img, labels, aug_type, self.rotation_range, self.brightness_range
                    )
                    
                    aug_name = f"{img_file.stem}_aug{aug_idx}_{aug_type}{img_file.suffix}"
                    cv2.imwrite(str(output_images_dir / aug_name), aug_img)
                    self._save_labels(output_labels_dir, f"{img_file.stem}_aug{aug_idx}_{aug_type}", aug_labels)
                    
                    total_augmented += 1
                    if log_callback:
                        log_callback(f"Augmented: {aug_name} ({aug_type})")
                
                # Copy original
                shutil.copy2(img_file, output_images_dir / img_file.name)
                if label_file.exists():
                    shutil.copy2(label_file, output_labels_dir / f"{img_file.stem}.txt")
                
            except Exception as e:
                errors.append(f"Error processing {img_file.name}: {str(e)}")
            
            if progress_callback:
                progress_callback(int(((i + 1) / total) * 100))
            if status_callback:
                status_callback(f"Augmenting {i + 1}/{total}")
        
        return {
            "success": len(errors) == 0,
            "original_images": total,
            "augmented_images": total_augmented,
            "total_images": total + total_augmented,
            "errors": errors,
            "output_dir": str(self.output_dir),
        }
    
    def _read_labels(self, img_file: Path, base_dir: Path) -> List[List]:
        """Read labels for an image file."""
        label_file = base_dir / "labels" / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                return [line.strip().split() for line in f.readlines()]
        return []
    
    def _save_labels(self, output_dir: Path, name: str, labels: List[List]):
        """Save labels to file."""
        label_path = output_dir / f"{name}.txt"
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(str(x) for x in label) + '\n')
    
    def apply_augmentation(
        self,
        img: np.ndarray,
        labels: List[List],
        aug_type: str,
        rotation_range: Optional[Tuple[float, float]] = None,
        brightness_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, List[List]]:
        """Apply a single augmentation type to image and labels."""
        rot_range = rotation_range or self.rotation_range
        bright_range = brightness_range or self.brightness_range
        
        h, w = img.shape[:2]
        aug_img = img.copy()
        aug_labels = [label.copy() for label in labels]
        
        if aug_type == "flip_horizontal":
            aug_img = cv2.flip(img, 1)
            for label in aug_labels:
                label[1] = 1.0 - float(label[1])
        elif aug_type == "flip_vertical":
            aug_img = cv2.flip(img, 0)
            for label in aug_labels:
                label[2] = 1.0 - float(label[2])
        elif aug_type == "rotate":
            angle = random.uniform(*rot_range)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(img, M, (w, h))
        elif aug_type == "brightness":
            factor = random.uniform(*bright_range)
            aug_img = np.clip(img * factor, 0, 255).astype(np.uint8)
        elif aug_type == "contrast":
            factor = random.uniform(*bright_range)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            aug_img = np.clip(mean + factor * (img - mean), 0, 255).astype(np.uint8)
        
        return aug_img, aug_labels
    
    def apply_mosaic(
        self,
        images: List[np.ndarray],
        labels_list: List[List[List]],
    ) -> Tuple[np.ndarray, List[List]]:
        """Apply mosaic augmentation combining 4 images into a 2x2 grid."""
        if len(images) != 4 or len(labels_list) != 4:
            raise ValueError("Mosaic requires exactly 4 images and 4 label sets")
        
        h, w = images[0].shape[:2]
        mosaic_h, mosaic_w = h * 2, w * 2
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        
        # Place 4 images in 2x2 grid
        mosaic[0:h, 0:w] = images[0]
        mosaic[0:h, w:mosaic_w] = images[1]
        mosaic[h:mosaic_h, 0:w] = images[2]
        mosaic[h:mosaic_h, w:mosaic_w] = images[3]
        
        # Combine and adjust labels
        combined_labels = []
        
        # Top-left quadrant (image 0) - labels stay the same
        for label in labels_list[0]:
            new_label = label.copy()
            new_label[1] = float(label[1]) / 2.0  # x_center / 2
            new_label[2] = float(label[2]) / 2.0  # y_center / 2
            new_label[3] = float(label[3]) / 2.0  # width / 2
            new_label[4] = float(label[4]) / 2.0  # height / 2
            combined_labels.append(new_label)
        
        # Top-right quadrant (image 1)
        for label in labels_list[1]:
            new_label = label.copy()
            new_label[1] = 0.5 + float(label[1]) / 2.0
            new_label[2] = float(label[2]) / 2.0
            new_label[3] = float(label[3]) / 2.0
            new_label[4] = float(label[4]) / 2.0
            combined_labels.append(new_label)
        
        # Bottom-left quadrant (image 2)
        for label in labels_list[2]:
            new_label = label.copy()
            new_label[1] = float(label[1]) / 2.0
            new_label[2] = 0.5 + float(label[2]) / 2.0
            new_label[3] = float(label[3]) / 2.0
            new_label[4] = float(label[4]) / 2.0
            combined_labels.append(new_label)
        
        # Bottom-right quadrant (image 3)
        for label in labels_list[3]:
            new_label = label.copy()
            new_label[1] = 0.5 + float(label[1]) / 2.0
            new_label[2] = 0.5 + float(label[2]) / 2.0
            new_label[3] = float(label[3]) / 2.0
            new_label[4] = float(label[4]) / 2.0
            combined_labels.append(new_label)
        
        return mosaic, combined_labels
    
    def get_statistics(
        self,
        dataset_path: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Get dataset statistics."""
        dataset = Path(dataset_path) if dataset_path else self.input_dir
        images_dir = dataset / "images" if (dataset / "images").exists() else dataset
        labels_dir = dataset / "labels"
        
        image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]
        
        total = len(image_files)
        if total == 0:
            return {"success": False, "error": "No images found in dataset"}
        
        stats = {
            "total_images": total,
            "class_distribution": defaultdict(int),
            "total_annotations": 0,
            "image_dimensions": [],
            "images_with_labels": 0,
            "images_without_labels": 0,
        }
        
        for i, img_file in enumerate(image_files):
            if is_cancelled and is_cancelled():
                return {"success": False, "cancelled": True, "stats": stats}
            
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    stats["image_dimensions"].append({"file": img_file.name, "width": w, "height": h})
                
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    stats["images_with_labels"] += 1
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                stats["class_distribution"][class_id] += 1
                                stats["total_annotations"] += 1
                else:
                    stats["images_without_labels"] += 1
            except Exception:
                pass
            
            if progress_callback:
                progress_callback(int(((i + 1) / total) * 100))
            if status_callback:
                status_callback(f"Analyzing {i + 1}/{total}")
        
        if stats["image_dimensions"]:
            avg_width = sum(d["width"] for d in stats["image_dimensions"]) / len(stats["image_dimensions"])
            avg_height = sum(d["height"] for d in stats["image_dimensions"]) / len(stats["image_dimensions"])
            stats["avg_width"] = avg_width
            stats["avg_height"] = avg_height
        
        stats["class_distribution"] = dict(stats["class_distribution"])
        stats["success"] = True
        return stats


# Backward compatibility - expose legacy function signatures
def augment_dataset(config_dict, progress_callback=None, status_callback=None, log_callback=None, is_cancelled=None):
    processor = DataProcessor(config_dict)
    return processor.augment_dataset(progress_callback, status_callback, log_callback, is_cancelled)


def get_statistics(dataset_path, progress_callback=None, status_callback=None, log_callback=None, is_cancelled=None):
    processor = DataProcessor({"input_dir": str(dataset_path)})
    return processor.get_statistics(dataset_path, progress_callback, status_callback, log_callback, is_cancelled)


def apply_augmentation(img, labels, aug_type, rotation_range=(-15, 15), brightness_range=(0.7, 1.3)):
    processor = DataProcessor({})
    return processor.apply_augmentation(img, labels, aug_type, rotation_range, brightness_range)
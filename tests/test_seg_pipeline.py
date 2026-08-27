"""Tests for the post-annotation segmentation dataset pipeline:

    label-review COCO -> 01b_coco_to_yolo_seg.py -> 02_augment_data.py
    -> 03_split_dataset.py -> 04_train_model.py --task segment

Covers polygon-aware augmentations (core.data_processor), the COCO->YOLO-seg
converter, the chain runner's dataset validation, and an end-to-end run of
the whole chain on a tiny synthetic dataset.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_processor import DataProcessor


def _load_script(name):
    """Import a digit-named script from scripts/ via importlib."""
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


converter = _load_script("01b_coco_to_yolo_seg.py")
runner = _load_script("orchestrate_pipeline.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_coco(path, images, annotations, categories):
    path.write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": categories,
    }))
    return path


def _square_polygon(x, y, s):
    """Absolute-px square polygon with top-left corner (x, y), side s."""
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def _make_coco_session(tmp_path, stereo=False, n=2):
    """Create a small COCO json + image folder. Returns (coco_json, images_dir)."""
    from PIL import Image
    w, h = 64, 48
    images, annotations = [], []
    ann_id = 1
    if stereo:
        images_dir = tmp_path / "camera"
        (images_dir / "left").mkdir(parents=True)
        (images_dir / "right").mkdir(parents=True)
        for i in range(n):
            for side in ("left", "right"):
                fname = f"{1000 + i}.png"
                Image.fromarray(
                    np.full((h, w, 3), 100 + i, np.uint8)).save(images_dir / side / fname)
                images.append({"id": len(images), "file_name": fname,
                               "width": w, "height": h, "side": side})
                annotations.append({
                    "id": ann_id, "image_id": images[-1]["id"], "category_id": 1,
                    "bbox": [8, 8, 16, 16],
                    "segmentation": [_square_polygon(8, 8, 16)],
                })
                ann_id += 1
    else:
        images_dir = tmp_path / "imgs"
        images_dir.mkdir()
        for i in range(n):
            fname = f"{1000 + i}.png"
            Image.fromarray(
                np.full((h, w, 3), 100 + i, np.uint8)).save(images_dir / fname)
            images.append({"id": i, "file_name": fname, "width": w, "height": h})
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "segmentation": [_square_polygon(8, 8, 16)],
            })
            ann_id += 1
    categories = [{"id": 1, "name": "object"}]
    return _write_coco(tmp_path / "labels_coco.json", images, annotations,
                       categories), images_dir


# ---------------------------------------------------------------------------
# Polygon-aware augmentations
# ---------------------------------------------------------------------------

class TestSegAugmentations:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.poly = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
        self.det = ["0", "0.25", "0.5", "0.2", "0.4"]

    def test_is_seg_label(self):
        assert DataProcessor._is_seg_label(self.poly)
        assert not DataProcessor._is_seg_label(self.det)

    def test_flip_horizontal_polygon(self):
        dp = DataProcessor()
        _, labels = dp.apply_augmentation(self.img, [self.poly], "flip_horizontal")
        assert labels[0] == ["0", 0.9, "0.2", 0.7, "0.4", 0.5, "0.6"]

    def test_flip_vertical_polygon(self):
        dp = DataProcessor()
        _, labels = dp.apply_augmentation(self.img, [self.poly], "flip_vertical")
        assert labels[0] == ["0", "0.1", 0.8, "0.3", 0.6, "0.5", 0.4]

    def test_flip_detection_still_works(self):
        dp = DataProcessor()
        _, lh = dp.apply_augmentation(self.img, [self.det], "flip_horizontal")
        _, lv = dp.apply_augmentation(self.img, [self.det], "flip_vertical")
        assert lh[0][1] == 0.75 and lh[0][2] == "0.5"
        assert lv[0][1] == "0.25" and lv[0][2] == 0.5

    def test_mosaic_polygon_transform(self):
        dp = DataProcessor()
        imgs = [self.img] * 4
        labels = [[["1", "0.2", "0.4", "0.6", "0.8", "1.0", "1.0"]]]
        _, out = dp.apply_mosaic(imgs, labels * 4)
        assert out[0] == ["1", 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]        # top-left
        assert out[1] == ["1", 0.6, 0.2, 0.8, 0.4, 1.0, 0.5]        # top-right
        assert out[2] == ["1", 0.1, 0.7, 0.3, 0.9, 0.5, 1.0]        # bottom-left
        assert out[3] == ["1", 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]        # bottom-right

    def test_photometric_augs_leave_labels_unchanged(self):
        dp = DataProcessor({"hue_range": (20, 20), "blur_range": (5, 5)})
        for aug in ("hue", "blur", "brightness", "contrast"):
            _, labels = dp.apply_augmentation(self.img, [self.poly], aug)
            assert labels == [self.poly], aug

    def test_hue_blur_change_image(self):
        dp = DataProcessor({"hue_range": (20, 20), "blur_range": (5, 5)})
        hue_img, _ = dp.apply_augmentation(self.img, [], "hue")
        blur_img, _ = dp.apply_augmentation(self.img, [], "blur")
        assert not np.array_equal(hue_img, self.img)
        assert not np.array_equal(blur_img, self.img)

    def test_resize_changes_dims_not_labels(self):
        dp = DataProcessor({"resize": (32, 16)})
        out, labels = dp.apply_augmentation(self.img, [self.poly], "resize")
        assert out.shape[:2] == (16, 32)
        assert labels == [self.poly]

    def test_resize_without_config_is_noop(self):
        dp = DataProcessor()
        out, labels = dp.apply_augmentation(self.img, [self.poly], "resize")
        assert out.shape == self.img.shape
        assert labels == [self.poly]

    def test_detection_label_with_trailing_attrs_stays_detection(self):
        # 5 values + 1 trailing attribute (odd count) must remain detection
        det = ["0", "0.25", "0.5", "0.2", "0.4", "0.9"]
        assert not DataProcessor._is_seg_label(det)
        _, labels = DataProcessor().apply_augmentation(
            self.img, [det], "flip_horizontal")
        assert labels[0][1] == 0.75          # xc mirrored
        assert labels[0][5] == "0.9"         # trailing attr untouched


class TestMosaicCustomSubdir:
    """Mosaic must read neighbor labels from the configured subdir (#1)."""

    def _make_flat(self, root, n=6, images_subdir="images",
                   labels_subdir="labels"):
        img_dir = root / images_subdir
        lab_dir = root / labels_subdir
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(img_dir / f"{i}.png"),
                        np.full((32, 32, 3), 100 + i, np.uint8))
            (lab_dir / f"{i}.txt").write_text(
                "0 0.1 0.1 0.3 0.1 0.1 0.3 0.1 0.3\n")
        return img_dir, lab_dir

    def test_mosaic_reads_labels_from_custom_subdir(self, tmp_path):
        dp = DataProcessor({
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "augmentation_types": ["mosaic"],
            "multiplier": 0,
            "images_subdir": "train/images",
            "labels_subdir": "train/labels",
        })
        self._make_flat(tmp_path / "in", n=6, images_subdir="train/images",
                        labels_subdir="train/labels")
        result = dp.augment_dataset()
        assert result["success"], result
        mosaic_label = tmp_path / "out" / "train" / "labels" / "0_mosaic.txt"
        assert mosaic_label.exists()
        lines = [l.split() for l in mosaic_label.read_text().splitlines() if l.strip()]
        # 4 sources x 1 polygon each
        assert len(lines) == 4
        # quadrant transform: first source polygon in top-left quadrant
        first = [float(v) for v in lines[0]]
        assert first[1] == pytest.approx(0.05)   # 0.1/2 + 0
        assert first[2] == pytest.approx(0.05)

    def test_mosaic_skips_unreadable_neighbor(self, tmp_path):
        dp = DataProcessor({
            "input_dir": str(tmp_path / "in"),
            "output_dir": str(tmp_path / "out"),
            "augmentation_types": ["mosaic"],
            "multiplier": 0,
        })
        self._make_flat(tmp_path / "in", n=6)
        # sorted order 0..5; mosaic groups: 0->(1,2,3), 1->(2,3,4), 2->(3,4,5)
        (tmp_path / "in" / "images" / "4.png").write_bytes(b"not an image")
        result = dp.augment_dataset()
        # mosaic for 1 and 2 (which need image 4) is skipped, group 0 proceeds
        assert (tmp_path / "out" / "labels" / "0_mosaic.txt").exists()
        assert not (tmp_path / "out" / "labels" / "1_mosaic.txt").exists()
        assert not (tmp_path / "out" / "labels" / "2_mosaic.txt").exists()
        assert any("mosaic" in e for e in result["errors"])


class TestRotatePolygonAreaGuard:
    """Rotated polygons mostly outside the frame are dropped (#3)."""

    def test_mostly_out_of_frame_polygon_dropped(self):
        dp = DataProcessor({"rotation_range": (45, 45)})
        img = np.zeros((100, 100, 3), np.uint8)
        # square flush in the top-right corner; 45deg rotation around the
        # image center throws all of it above the top edge -> visible area
        # collapses to ~0 -> dropped
        poly = ["0", "0.9", "0.0", "1.0", "0.0", "1.0", "0.1", "0.9", "0.1"]
        _, out = dp.apply_augmentation(img, [poly], "rotate")
        assert out == []

    def test_centered_polygon_survives_rotation(self):
        dp = DataProcessor({"rotation_range": (15, 15)})
        img = np.zeros((100, 100, 3), np.uint8)
        # centered polygon stays inside the frame after a small rotation
        poly = ["0", "0.3", "0.3", "0.7", "0.3", "0.7", "0.7", "0.3", "0.7"]
        _, out = dp.apply_augmentation(img, [poly], "rotate")
        assert len(out) == 1
        # still a valid polygon: >= 6 even coords within [0, 1]
        coords = [float(v) for v in out[0][1:]]
        assert len(coords) % 2 == 0
        assert all(-1e-6 <= v <= 1 + 1e-6 for v in coords)


# ---------------------------------------------------------------------------
# COCO -> flat YOLO-seg converter
# ---------------------------------------------------------------------------

class TestConverter:
    def test_mono_roundtrip(self, tmp_path):
        coco_json, images_dir = _make_coco_session(tmp_path, stereo=False, n=2)
        out = tmp_path / "yolo_flat"
        summary = converter.convert(coco_json, images_dir, out)
        assert summary["images_written"] == 2
        assert summary["polygons_written"] == 2
        assert (out / "classes.txt").read_text().splitlines() == ["object"]
        stats = runner.validate_flat_dataset(out)
        assert stats == {"images": 2, "label_lines": 2}
        # Normalized coordinates: square 8..24 px on 64x48 image
        line = (out / "labels" / "1000.txt").read_text().split()
        assert line[0] == "0"
        coords = [float(v) for v in line[1:]]
        assert coords[:4] == pytest.approx([8 / 64, 8 / 48, 24 / 64, 8 / 48], abs=1e-6)

    def test_stereo_prefixes_avoid_collision(self, tmp_path):
        coco_json, images_dir = _make_coco_session(tmp_path, stereo=True, n=2)
        out = tmp_path / "yolo_flat"
        summary = converter.convert(coco_json, images_dir, out)
        assert summary["stereo"] is True
        assert summary["images_written"] == 4
        names = {p.name for p in (out / "images").iterdir()}
        assert "left_1000.png" in names and "right_1000.png" in names
        stats = runner.validate_flat_dataset(out)
        assert stats == {"images": 4, "label_lines": 4}

    def test_maskless_skipped_by_default(self, tmp_path):
        coco_json, images_dir = _make_coco_session(tmp_path, n=1)
        coco = json.loads(coco_json.read_text())
        coco["annotations"].append({
            "id": 99, "image_id": 0, "category_id": 1,
            "bbox": [30, 30, 10, 10], "segmentation": [],
        })
        coco_json.write_text(json.dumps(coco))
        out = tmp_path / "yolo_flat"
        summary = converter.convert(coco_json, images_dir, out)
        assert summary["maskless_annotations"] == 1
        assert summary["polygons_written"] == 1

    def test_bbox_as_rect_emits_maskless(self, tmp_path):
        coco_json, images_dir = _make_coco_session(tmp_path, n=1)
        coco = json.loads(coco_json.read_text())
        coco["annotations"].append({
            "id": 99, "image_id": 0, "category_id": 1,
            "bbox": [32, 24, 16, 12], "segmentation": [],
        })
        coco_json.write_text(json.dumps(coco))
        out = tmp_path / "yolo_flat"
        summary = converter.convert(coco_json, images_dir, out, bbox_as_rect=True)
        assert summary["polygons_written"] == 2
        line = (out / "labels" / "1000.txt").read_text().splitlines()[1].split()
        coords = [float(v) for v in line[1:]]
        # rect corners of bbox [32,24,16,12] on 64x48
        assert coords[:2] == pytest.approx([32 / 64, 24 / 48], abs=1e-6)
        assert coords[4:6] == pytest.approx([48 / 64, 36 / 48], abs=1e-6)


# ---------------------------------------------------------------------------
# Chain runner validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_label_file_rejected(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(ValueError, match="missing label file"):
            runner.validate_flat_dataset(tmp_path)

    def test_out_of_range_coordinate_rejected(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((8, 8, 3), np.uint8))
        (tmp_path / "labels" / "a.txt").write_text("0 0.1 0.2 1.5 0.4 0.5 0.6\n")
        with pytest.raises(ValueError, match="outside"):
            runner.validate_flat_dataset(tmp_path)

    def test_bad_class_id_rejected(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "classes.txt").write_text("only_one\n")
        cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((8, 8, 3), np.uint8))
        (tmp_path / "labels" / "a.txt").write_text("3 0.1 0.2 0.3 0.4 0.5 0.6\n")
        with pytest.raises(ValueError, match="out of range"):
            runner.validate_flat_dataset(tmp_path)

    def test_odd_coordinate_count_rejected(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        cv2.imwrite(str(tmp_path / "images" / "a.png"), np.zeros((8, 8, 3), np.uint8))
        (tmp_path / "labels" / "a.txt").write_text("0 0.1 0.2 0.3 0.4 0.5\n")
        with pytest.raises(ValueError, match="even number"):
            runner.validate_flat_dataset(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end chain
# ---------------------------------------------------------------------------

def test_full_chain_end_to_end(tmp_path):
    """convert -> augment -> split on a tiny synthetic stereo dataset."""
    coco_json, images_dir = _make_coco_session(tmp_path, stereo=True, n=6)
    out = tmp_path / "pipeline_out"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "orchestrate_pipeline.py"),
         "--coco-json", str(coco_json), "--images-dir", str(images_dir),
         "--output-root", str(out),
         "--augmentations", "flip_horizontal", "--multiplier", "1",
         "--ratios", "0.5", "0.25", "0.25", "--split-seed", "42"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # 6 stereo pairs = 12 source images; multiplier 1 -> 12 originals + 12 augs
    flat_stats = runner.validate_flat_dataset(out / "yolo_flat")
    assert flat_stats["images"] == 12
    aug_stats = runner.validate_flat_dataset(out / "augmented")
    assert aug_stats["images"] == 24
    assert aug_stats["label_lines"] == 24
    # classes.txt passed through the augmentation step
    assert (out / "augmented" / "classes.txt").is_file()

    split = out / "dataset"
    split_stats = runner.validate_split_dataset(split)
    assert sum(s["images"] for s in split_stats.values()) == 24
    assert sum(s["label_lines"] for s in split_stats.values()) == 24

    # dataset.yaml generated from classes.txt fallback (no --class-names given)
    import yaml
    cfg = yaml.safe_load((split / "dataset.yaml").read_text())
    assert cfg["nc"] == 1
    assert cfg["names"] == ["object"]

    # Every augmented copy still carries a flipped polygon label
    for line in (split / "train" / "labels").glob("*_aug0_flip_horizontal.txt"):
        coords = [float(v) for v in line.read_text().split()[1:]]
        assert coords[0] == pytest.approx(1 - 8 / 64)  # mirrored x1


def test_chain_skip_augment(tmp_path):
    coco_json, images_dir = _make_coco_session(tmp_path, stereo=False, n=4)
    out = tmp_path / "pipeline_out"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "orchestrate_pipeline.py"),
         "--coco-json", str(coco_json), "--images-dir", str(images_dir),
         "--output-root", str(out), "--skip-augment",
         "--ratios", "0.5", "0.5", "0.0", "--split-seed", "1"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (out / "augmented").exists()
    split_stats = runner.validate_split_dataset(out / "dataset")
    assert sum(s["images"] for s in split_stats.values()) == 4

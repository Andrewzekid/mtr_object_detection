"""Tests for scripts/01a_dataset_statistics.py (dataset statistics) and the
per-class CSV helpers in scripts/05_evaluate_model.py."""

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script(name):
    """Import a digit-named script from scripts/ via importlib."""
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats_mod = _load_script("01a_dataset_statistics.py")
eval_mod = _load_script("05_evaluate_model.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_labels(labels_dir, name, lines):
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")


def _make_single_dataset(root, with_classes=True):
    """images/ + labels/ with 3 images: 2 labeled, 1 background."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    _write_labels(root / "labels", "img1", [
        "0 0.5 0.5 0.2 0.2",          # detection line, class 0
        "1 0.1 0.1 0.2 0.2 0.3 0.3",  # segmentation line, class 1
    ])
    _write_labels(root / "labels", "img2", ["1 0.5 0.5 0.4 0.4"])
    _write_labels(root / "labels", "img3", [])  # background
    if with_classes:
        (root / "classes.txt").write_text("cat\ndog\n")


def _make_split_dataset(root):
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    _write_labels(root / "train" / "labels", "a", ["0 0.5 0.5 0.2 0.2"])
    _write_labels(root / "train" / "labels", "b",
                  ["0 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.2 0.2"])
    _write_labels(root / "val" / "labels", "c", ["1 0.5 0.5 0.2 0.2"])
    _write_labels(root / "test" / "labels", "d", [])


# ---------------------------------------------------------------------------
# 01a_dataset_statistics.py
# ---------------------------------------------------------------------------

def test_find_splits_single_dataset(tmp_path):
    _make_single_dataset(tmp_path)
    splits = stats_mod.find_splits(tmp_path)
    assert list(splits) == ["all"]
    assert splits["all"] == tmp_path / "labels"


def test_find_splits_split_root(tmp_path):
    _make_split_dataset(tmp_path)
    splits = stats_mod.find_splits(tmp_path)
    assert set(splits) == {"train", "val", "test"}


def test_find_splits_yolo_training_layout(tmp_path):
    """images/<split> + labels/<split> layout (e.g. hku_gh_yolo_seg)."""
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    _write_labels(tmp_path / "labels" / "train", "a", ["0 0.5 0.5 0.2 0.2"])
    splits = stats_mod.find_splits(tmp_path)
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"] == tmp_path / "labels" / "train"
    stats = stats_mod.compute_split_stats(splits["train"])
    assert stats["image_count"] == 1
    assert stats["per_class"][0]["instances"] == 1


def test_find_splits_missing_labels(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        stats_mod.find_splits(tmp_path)


def test_load_class_names_from_classes_txt(tmp_path):
    _make_single_dataset(tmp_path)
    assert stats_mod.load_class_names(tmp_path) == {0: "cat", 1: "dog"}


def test_load_class_names_fallback(tmp_path):
    _make_single_dataset(tmp_path, with_classes=False)
    assert stats_mod.load_class_names(tmp_path) == {}


def test_compute_split_stats_counts(tmp_path):
    _make_single_dataset(tmp_path)
    stats = stats_mod.compute_split_stats(tmp_path / "labels")
    assert stats["image_count"] == 3
    assert stats["labeled_count"] == 2
    assert stats["background_count"] == 1
    assert stats["total_instances"] == 3
    assert stats["per_class"][0] == {"instances": 1, "images": 1}
    assert stats["per_class"][1] == {"instances": 2, "images": 2}


def test_build_rows_percentages(tmp_path):
    _make_single_dataset(tmp_path)
    stats = stats_mod.compute_split_stats(tmp_path / "labels")
    rows = stats_mod.build_rows("all", stats, {0: "cat", 1: "dog"})
    assert rows[0]["class_name"] == "all"
    assert rows[0]["instances"] == 3
    # split-level totals only on the "all" row
    assert rows[0]["total_images"] == 3
    assert rows[0]["labeled_images"] == 2
    assert rows[0]["background_images"] == 1
    assert "total_images" not in rows[1]
    by_name = {r["class_name"]: r for r in rows[1:]}
    assert abs(by_name["dog"]["pct_instances"] - 66.666) < 0.01
    assert abs(by_name["dog"]["pct_images"] - 66.666) < 0.01
    assert abs(by_name["cat"]["avg_instances_per_image"] - 1 / 3) < 0.001


def test_stats_csv_end_to_end(tmp_path):
    _make_split_dataset(tmp_path)
    csv_path = tmp_path / "stats.csv"
    splits = stats_mod.find_splits(tmp_path)
    rows = []
    for split, labels_dir in splits.items():
        rows.extend(stats_mod.build_rows(split,
                                         stats_mod.compute_split_stats(labels_dir),
                                         {}))
    stats_mod.write_csv(rows, str(csv_path))
    with open(csv_path) as f:
        parsed = list(csv.DictReader(f))
    assert set(r["split"] for r in parsed) == {"train", "val", "test"}
    train_all = next(r for r in parsed
                     if r["split"] == "train" and r["class_name"] == "all")
    assert train_all["instances"] == "3"
    # test split has a single background image -> no class rows, one all row
    test_rows = [r for r in parsed if r["split"] == "test"]
    assert len(test_rows) == 1
    assert test_rows[0]["instances"] == "0"


# ---------------------------------------------------------------------------
# 05_evaluate_model.py CSV helpers
# ---------------------------------------------------------------------------

def _fake_eval_result(task="segment"):
    result = {
        "success": True,
        "metrics": {
            "mAP50": 0.6, "mAP50_95": 0.4,
            "precision": 0.7, "recall": 0.5,
            "mask_mAP50": 0.55, "mask_mAP50_95": 0.35,
            "mask_precision": 0.65, "mask_recall": 0.45,
        },
        "per_class": {
            "cat": {"precision": 0.8, "recall": 0.6, "f1": 0.6857,
                    "AP50": 0.7, "AP50_95": 0.5,
                    "mask_precision": 0.75, "mask_recall": 0.55,
                    "mask_f1": 0.6353, "mask_AP50": 0.65, "mask_AP50_95": 0.45},
            "dog": {"precision": 0.6, "recall": 0.4, "f1": 0.48,
                    "AP50": 0.5, "AP50_95": 0.3,
                    "mask_precision": 0.55, "mask_recall": 0.35,
                    "mask_f1": 0.429, "mask_AP50": 0.45, "mask_AP50_95": 0.25},
        },
        "class_names": {0: "cat", 1: "dog"},
    }
    return result


def test_build_csv_rows_structure():
    rows = eval_mod.build_csv_rows("test", _fake_eval_result(), {0: 3, 1: 5})
    assert len(rows) == 3  # all + 2 classes
    assert rows[0]["class_name"] == "all"
    assert rows[0]["instances"] == 8
    assert rows[1]["class_id"] == 0 and rows[1]["class_name"] == "cat"
    assert rows[2]["class_id"] == 1
    # overall F1 computed from overall precision/recall
    assert abs(rows[0]["f1"] - 2 * 0.7 * 0.5 / 1.2) < 1e-6
    assert abs(rows[0]["mask_f1"] - 2 * 0.65 * 0.45 / 1.1) < 1e-6


def test_build_csv_rows_detection_blanks_mask_columns():
    result = _fake_eval_result()
    result["metrics"] = {k: v for k, v in result["metrics"].items()
                         if not k.startswith("mask_")}
    for m in result["per_class"].values():
        for k in list(m):
            if k.startswith("mask_"):
                del m[k]
    rows = eval_mod.build_csv_rows("train", result, {})
    assert rows[0]["mask_ap50"] == ""
    assert rows[1]["mask_precision"] == ""
    assert rows[0]["instances"] == ""


def test_write_metrics_csv_formats_floats(tmp_path):
    rows = eval_mod.build_csv_rows("test", _fake_eval_result(), {0: 3, 1: 5})
    csv_path = tmp_path / "out" / "metrics.csv"
    eval_mod.write_metrics_csv(rows, str(csv_path))
    with open(csv_path) as f:
        parsed = list(csv.DictReader(f))
    assert parsed[0]["split"] == "test"
    assert parsed[1]["class_name"] == "cat"
    assert parsed[1]["precision"] == "0.8000"
    assert parsed[1]["mask_ap50"] == "0.6500"


def test_count_gt_instances(tmp_path):
    _make_split_dataset(tmp_path)
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        f"path: {tmp_path}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        "names: [cat, dog]\n")
    counts = eval_mod.count_gt_instances(str(yaml_path), "train")
    assert counts == {0: 2, 1: 1}
    assert eval_mod.count_gt_instances(str(yaml_path), "test") == {}
    assert eval_mod.count_gt_instances(str(yaml_path), "nope") == {}

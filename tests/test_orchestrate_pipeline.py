import argparse
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orchestrate_pipeline import (
    CamPaths,
    check_llamacpp_server,
    get_project_root,
    iter_images,
    link_or_copy,
    merge_coco_sides,
    parse_args,
    parse_classes_from_prompt,
    read_marker,
    run_pipeline,
    write_marker,
)


def _args(**kw):
    defaults = dict(
        rosbag=None, images=None, camera="left", output_root=None,
        sample_size=None, copy=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_get_project_root():
    root = get_project_root()
    assert (root / "scripts" / "orchestrate_pipeline.py").exists()


def test_parse_args_defaults():
    args = parse_args(["--images", "/tmp/fake_images"])
    assert args.camera == "left"
    assert args.keyframe_stride == 10
    assert args.sample_size is None
    assert args.gui_on == "keyframes"
    assert args.ratios == [0.7, 0.15, 0.15]
    assert args.augmentations == ["flip_horizontal", "rotate", "brightness"]
    assert args.multiplier == 2
    assert args.task == "segment"
    assert args.model_type == "yolo26n"
    assert args.tracker == "deepocsort"
    assert args.eval_splits == ["test", "train"]


def test_run_pipeline_requires_input(tmp_path):
    args = parse_args(["--images", str(tmp_path / "nope")])
    with pytest.raises(FileNotFoundError):
        run_pipeline(args)


def test_parse_classes_from_prompt():
    prompt = (
        "Sofa: Detect any upholstered ...\n\n"
        "Wooden Door: Detect dark brown ...\n\n"
        "Overhead Signage: Detect large rectangular ...\n\n"
        "Sprinkler (on the ceiling): Detect small ..."
    )
    classes = parse_classes_from_prompt(prompt)
    assert classes == ["Sofa", "Wooden Door", "Overhead Signage", "Sprinkler"]


def test_stage_marker(tmp_path):
    marker = tmp_path / "marker.json"
    assert not read_marker(marker)
    write_marker(marker, {"stage": "test"})
    data = read_marker(marker)
    assert data["stage"] == "test"
    assert "completed_at" in data


def test_check_llamacpp_server_ok():
    with patch("orchestrate_pipeline.urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        assert check_llamacpp_server("http://127.0.0.1:8089") is True


def test_check_llamacpp_server_fail():
    with patch("orchestrate_pipeline.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        assert check_llamacpp_server("http://127.0.0.1:8089") is False


def test_iter_images_missing_dir(tmp_path):
    assert iter_images(tmp_path / "nope") == []


def test_link_or_copy(tmp_path):
    src = tmp_path / "src.jpg"
    src.write_text("x")
    dst = tmp_path / "out" / "src.jpg"
    link_or_copy(src, dst, copy=False)
    assert dst.is_symlink()
    dst2 = tmp_path / "out2" / "src.jpg"
    link_or_copy(src, dst2, copy=True)
    assert dst2.is_file() and not dst2.is_symlink()


def test_campaths_mono_images(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    paths = CamPaths(_args(images=str(imgs)), tmp_path / "out")
    assert paths.cameras == ["left"]
    assert paths.undistorted("left") == imgs  # no rosbag -> used as-is
    assert paths.working("left", False) == imgs
    assert paths.working("left", True) == tmp_path / "out" / "sampled" / "left"


def test_campaths_stereo_images(tmp_path):
    for side in ("left", "right"):
        (tmp_path / "imgs" / side).mkdir(parents=True)
    paths = CamPaths(_args(images=str(tmp_path / "imgs"), camera="both"),
                     tmp_path / "out")
    assert paths.cameras == ["left", "right"]
    assert paths.raw["right"] == tmp_path / "imgs" / "right"


def test_campaths_rosbag_requires_calibration(tmp_path):
    rosbag = tmp_path / "bag"
    (rosbag / "camera" / "left").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        CamPaths(_args(rosbag=str(rosbag)), tmp_path / "out")


def test_merge_coco_sides(tmp_path):
    def _coco(side, img_id, cat_id, cat_name):
        return {
            "images": [{"id": img_id, "file_name": "111.jpg", "side": side,
                        "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": img_id,
                             "category_id": cat_id,
                             "bbox": [0, 0, 5, 5]}],
            "categories": [{"id": cat_id, "name": cat_name}],
        }

    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    # Same category name with different ids on the two sides -> merged.
    left.write_text(json.dumps(_coco("left", 1, 0, "door")))
    right.write_text(json.dumps(_coco("right", 1, 7, "door")))
    out = tmp_path / "merged.json"
    merge_coco_sides({"left": left, "right": right}, out)
    merged = json.loads(out.read_text())
    assert len(merged["categories"]) == 1
    assert merged["categories"][0]["name"] == "door"
    assert {img["side"] for img in merged["images"]} == {"left", "right"}
    # ids renumbered without collision
    assert sorted(img["id"] for img in merged["images"]) == [1, 2]
    for ann in merged["annotations"]:
        assert ann["category_id"] == 0
    assert sorted(ann["image_id"] for ann in merged["annotations"]) == [1, 2]

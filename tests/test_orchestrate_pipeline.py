import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from orchestrate_pipeline import (
    check_llamacpp_server,
    export_yolo_from_coco,
    get_project_root,
    parse_args,
    parse_classes_from_prompt,
    read_marker,
    time_based_split,
    write_marker,
)


def test_get_project_root():
    root = get_project_root()
    assert (root / "scripts" / "orchestrate_pipeline.py").exists()


def test_parse_args_defaults():
    args = parse_args(["--rosbag", "/tmp/fake_rosbag"])
    assert args.camera == "left"
    assert args.keyframe_stride == 10
    assert args.splits == [0.8, 0.1, 0.1]
    assert args.propagation_method == "interpolation+sam3"


def test_parse_classes_from_prompt():
    prompt = (
        "Sofa: Detect any upholstered ...\n\n"
        "Wooden Door: Detect dark brown ...\n\n"
        "Overhead Signage: Detect large rectangular ...\n\n"
        "Sprinkler (on the ceiling): Detect small ..."
    )
    classes = parse_classes_from_prompt(prompt)
    assert classes == ["Sofa", "Wooden Door", "Overhead Signage", "Sprinkler"]


def test_time_based_split(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        (src / f"{i:04d}.jpg").write_text("")
    out = tmp_path / "out"
    result = time_based_split(src, out, [0.8, 0.1, 0.1])
    assert result["train"] == 8
    assert result["val"] == 1
    assert result["test"] == 1
    assert (out / "images" / "train" / "0000.jpg").exists()
    assert (out / "images" / "val" / "0008.jpg").exists()
    assert (out / "images" / "test" / "0009.jpg").exists()


def test_stage_marker(tmp_path):
    marker = tmp_path / "marker.json"
    assert not read_marker(marker)
    write_marker(marker, {"stage": "test"})
    assert read_marker(marker)["stage"] == "test"


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


def test_export_yolo_from_coco(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "0001.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10.0, 10.0, 20.0, 20.0]}
        ],
        "categories": [{"id": 0, "name": "Sofa"}],
    }
    coco_path = tmp_path / "coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "0001.jpg").write_text("")
    labels_dir = tmp_path / "labels"
    export_yolo_from_coco(coco_path, img_dir, labels_dir, ["Sofa"])
    label_file = labels_dir / "0001.txt"
    assert label_file.exists()
    assert label_file.read_text().strip() == "0 0.200000 0.200000 0.200000 0.200000"

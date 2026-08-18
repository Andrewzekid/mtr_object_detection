"""CLI tests for scripts/09b_split_and_combine.py (split / combine),
run as real subprocesses against tmp folders."""

import json
import subprocess
import sys

from conftest import ROOT, make_image_folder

SCRIPT = ROOT / "scripts" / "09b_split_and_combine.py"

# timestamp-style names: list_images sorts them numerically, not by name
NAMES = ["1783995353161694000.jpg", "100.jpg", "2.jpg", "50.jpg", "7.jpg"]
SORTED = ["2.jpg", "7.jpg", "50.jpg", "100.jpg", "1783995353161694000.jpg"]


def run_cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          cwd=str(cwd), capture_output=True, text=True,
                          timeout=60)


def test_split_range_mode_copy(tmp_path):
    src = make_image_folder(tmp_path / "src", NAMES)
    r = run_cli("split", src, "--ranges", "1-2", "--copy")
    assert r.returncode == 0, r.stderr
    rd = src / "range_000001-000002"
    # indices 1-2 of the numerically sorted list, names preserved
    assert sorted(p.name for p in rd.iterdir()) == ["50.jpg", "7.jpg"]
    # --copy keeps the originals in place
    assert sorted(p.name for p in src.glob("*.jpg")) == sorted(NAMES)


def test_split_range_mode_move_and_multiple_ranges(tmp_path):
    src = make_image_folder(tmp_path / "src", NAMES)
    r = run_cli("split", src, "--ranges", "0-1", "3-3")
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (src / "range_000000-000001").iterdir()) \
        == ["2.jpg", "7.jpg"]
    assert [p.name for p in (src / "range_000003-000003").iterdir()] \
        == ["100.jpg"]
    # default is MOVE: covered originals leave the source folder
    # (sorted() here is plain lexicographic, unlike the splitter's
    # numeric-aware ordering)
    assert sorted(p.name for p in src.glob("*.jpg")) == \
        sorted(["50.jpg", "1783995353161694000.jpg"])


def test_split_even_divide(tmp_path):
    src = make_image_folder(tmp_path / "src", NAMES)
    r = run_cli("split", src, "--divide-evenly", "2")
    assert r.returncode == 0, r.stderr
    # 5 files, 2 ways → 3 + 2 (earlier part gets the remainder)
    assert sorted(p.name for p in (src / "split_00").iterdir()) == \
        ["2.jpg", "50.jpg", "7.jpg"]
    assert sorted(p.name for p in (src / "split_01").iterdir()) == \
        ["100.jpg", "1783995353161694000.jpg"]
    assert list(src.glob("*.jpg")) == []  # moved


def test_split_output_dir(tmp_path):
    src = make_image_folder(tmp_path / "src", NAMES)
    dest = tmp_path / "elsewhere"
    r = run_cli("split", src, "--divide-evenly", "5", "--copy",
                "--output-dir", dest)
    assert r.returncode == 0, r.stderr
    subs = sorted(p.name for p in dest.iterdir())
    assert subs == [f"split_0{i}" for i in range(5)]
    # one image per subfolder, all names preserved
    found = sorted(p.name for d in dest.iterdir() for p in d.iterdir())
    assert found == sorted(NAMES)


def test_split_rejects_bad_ranges(tmp_path):
    src = make_image_folder(tmp_path / "src", NAMES)
    for spec in ("3-1", "0-5", "a-b"):
        r = run_cli("split", src, "--ranges", spec)
        assert r.returncode != 0, spec
    # overlapping ranges rejected
    r = run_cli("split", src, "--ranges", "0-2", "2-4")
    assert r.returncode != 0
    assert "overlap" in (r.stderr + r.stdout).lower()


def _write_coco(path, imgs, anns, cats):
    json.dump({"images": imgs, "annotations": anns, "categories": cats},
              open(path, "w"))


def test_combine_merges_with_remapped_ids(tmp_path):
    j1, j2 = tmp_path / "p1.json", tmp_path / "p2.json"
    _write_coco(
        j1,
        [{"id": 1, "file_name": "a.jpg", "width": 4, "height": 4}],
        [{"id": "1", "image_id": "1", "category_id": "3",
          "bbox": [0, 0, 2, 2], "track_id": "7", "confidence": "0.9"}],
        [{"id": 3, "name": "box"}])
    _write_coco(
        j2,
        [{"id": 1, "file_name": "b.jpg", "width": 4, "height": 4},
         {"id": 2, "file_name": "c.jpg", "width": 4, "height": 4}],
        [{"id": 5, "image_id": 1, "category_id": 0, "bbox": [1, 1, 1, 1],
          "segmentation": [[0, 0, 1, 0, 1, 1]]},
         {"id": 6, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        [{"id": 0, "name": "box"},      # same name as j1's cat → merged
         {"id": 1, "name": "lid"}])

    out = tmp_path / "out.json"
    r = run_cli("combine", j1, j2, "--output", out)
    assert r.returncode == 0, r.stderr
    d = json.load(open(out))

    assert [c["name"] for c in d["categories"]] == ["box", "lid"]
    assert [c["id"] for c in d["categories"]] == [0, 1]
    assert [i["id"] for i in d["images"]] == [1, 2, 3]
    assert [i["file_name"] for i in d["images"]] == ["a.jpg", "b.jpg",
                                                     "c.jpg"]
    anns = sorted(d["annotations"], key=lambda a: a["id"])
    assert [a["id"] for a in anns] == [1, 2, 3]
    assert [a["image_id"] for a in anns] == [1, 2, 3]
    # j1's cat id 3 ("box") → merged id 0; j2's cat 1 ("lid") → merged id 1
    assert [a["category_id"] for a in anns] == [0, 0, 1]
    # extras preserved
    assert anns[0]["track_id"] == "7" and anns[0]["confidence"] == "0.9"
    assert anns[1]["segmentation"] == [[0, 0, 1, 0, 1, 1]]


def test_combine_rejects_duplicate_image(tmp_path):
    j1, j3 = tmp_path / "p1.json", tmp_path / "p3.json"
    _write_coco(j1, [{"id": 1, "file_name": "a.jpg"}], [], [])
    _write_coco(j3, [{"id": 9, "file_name": "a.jpg"}], [], [])
    r = run_cli("combine", j1, j3, "--output", tmp_path / "dup.json")
    assert r.returncode != 0
    assert "duplicate" in (r.stderr + r.stdout).lower()

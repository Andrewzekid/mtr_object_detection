"""Pure unit tests: CocoState, UndoStack, mask/segmentation helpers,
ImageFolderIndex, import_coco, _iou_xyxy, _resolve_device, category
bootstrapping. No Qt windows here (only the session QApplication)."""

import base64
import json
import os

import numpy as np
import pytest

from conftest import FakeIdx, make_image_folder


# ---------------------------------------------------------------------------
# ImageFolderIndex
# ---------------------------------------------------------------------------

def test_image_folder_index_sorting_and_decode(lr, tmp_path):
    folder = make_image_folder(tmp_path / "imgs",
                               [("b.jpg", 60), ("a.png", 120),
                                ("c.jpg", 200)], size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    assert len(idx) == 3
    assert [f["file_name"] for f in idx.frames] == ["a.png", "b.jpg", "c.jpg"]
    arr = idx.decode_image(0)
    assert arr.shape == (10, 12, 3) and arr[0, 0, 0] == 120
    assert not idx.timestamps_real
    assert idx.find_idx_by_timestamp(1_500_000) == 2  # snaps to nearest
    assert idx.frame_at(1)["existing_boxes"] == []


def test_image_folder_index_mixed_file_and_folder(lr, tmp_path):
    folder = make_image_folder(tmp_path / "imgs2",
                               ["a.png", "b.jpg", "c.jpg"])
    single = tmp_path / "single.bmp"
    make_image_folder(tmp_path, [("single.bmp", 0)], size=(5, 6))
    assert single.exists()
    idx = lr.ImageFolderIndex([str(single), str(folder)])
    assert len(idx) == 4


# ---------------------------------------------------------------------------
# StereoIndex
# ---------------------------------------------------------------------------

def test_stereo_index_pairs_positionally(lr, tmp_path):
    left = make_image_folder(tmp_path / "left", ["f0.png", "f1.png"])
    right = make_image_folder(tmp_path / "right", ["f0.png", "f1.png"])
    idx = lr.StereoIndex([str(left)], [str(right)])
    assert len(idx) == 2
    assert idx.pairing_warning is None


def test_stereo_index_warns_on_filename_mismatch(lr, tmp_path):
    left = make_image_folder(tmp_path / "left", ["a.png", "b.png"])
    right = make_image_folder(tmp_path / "right", ["a.png", "c.png"])
    idx = lr.StereoIndex([str(left)], [str(right)])
    assert len(idx) == 2
    assert idx.pairing_warning is not None
    assert "b.png" in idx.pairing_warning
    assert "c.png" in idx.pairing_warning


def test_image_folder_index_timestamp_named_series(lr, tmp_path):
    folder = make_image_folder(
        tmp_path / "ts",
        ["3000000000.jpg", "1000000000.png", "2000000000.jpg"])
    idx = lr.ImageFolderIndex([str(folder)])
    assert idx.timestamps_real
    # sorted by timestamp, not name
    assert [f["timestamp_ns"] for f in idx.frames] == [10**9, 2 * 10**9,
                                                       3 * 10**9]
    assert idx.frames[0]["file_name"] == "1000000000.png"
    assert idx.find_idx_by_timestamp(2_600_000_000) == 2  # snaps to 3e9
    assert idx.find_idx_by_timestamp(100) == 0
    # mixing in a non-numeric name turns timestamp mode off
    make_image_folder(folder, [("other.jpg", 0)])
    idx2 = lr.ImageFolderIndex([str(folder)])
    assert not idx2.timestamps_real
    assert idx2.find_idx_by_timestamp(2_000_000) == 2


# ---------------------------------------------------------------------------
# UndoStack
# ---------------------------------------------------------------------------

def test_undostack_push_pop(lr):
    st = lr.UndoStack()
    log = []
    st.push("a", lambda: log.append("undo-a"), lambda: log.append("redo-a"))
    assert st.can_undo() and not st.can_redo()
    desc, undo, redo = st.pop_undo()
    assert desc == "a"
    undo()
    assert log == ["undo-a"]
    assert st.can_redo() and not st.can_undo()
    _desc, _undo, redo = st.pop_redo()
    assert st.can_undo() and not st.can_redo()
    redo()
    assert log == ["undo-a", "redo-a"]
    assert st.pop_redo() is None


def test_undostack_push_clears_redo(lr):
    st = lr.UndoStack()
    st.push("a", lambda: None, lambda: None)
    st.pop_undo()
    assert st.can_redo()
    st.push("b", lambda: None, lambda: None)  # new mutation kills redo
    assert not st.can_redo()


def test_undostack_group_coalesces(lr):
    st = lr.UndoStack()
    log = []
    with st.group("batch"):
        st.push("1", lambda: log.append("u1"), lambda: log.append("r1"))
        st.push("2", lambda: log.append("u2"), lambda: log.append("r2"))
    assert len(st._undo) == 1  # one composite entry
    st.pop_undo()[1]()
    assert log == ["u2", "u1"]  # undo runs in reverse order
    st.pop_redo()[2]()
    assert log == ["u2", "u1", "r1", "r2"]  # redo in original order


def test_undostack_nested_group_merges(lr):
    st = lr.UndoStack()
    with st.group("outer"):
        st.push("1", lambda: None, lambda: None)
        with st.group("inner"):
            st.push("2", lambda: None, lambda: None)
    assert len(st._undo) == 1


def test_undostack_mute_drops_pushes_but_clears_redo(lr):
    st = lr.UndoStack()
    st.push("a", lambda: None, lambda: None)
    st.pop_undo()
    with st.mute():
        st.push("b", lambda: None, lambda: None)
    assert not st.can_undo() and not st.can_redo()


def test_undostack_max_depth(lr):
    st = lr.UndoStack()
    for i in range(lr.UndoStack.MAX_DEPTH + 10):
        st.push(str(i), lambda: None, lambda: None)
    assert len(st._undo) == lr.UndoStack.MAX_DEPTH


# ---------------------------------------------------------------------------
# CocoState: boxes, categories, track ids
# ---------------------------------------------------------------------------

def _mkframe(i):
    return {"timestamp_ns": i * 10**9, "log_time_ns": i * 10**9,
            "frame_idx": i, "existing_boxes": [], "file_name": f"f{i}.png"}


def _tid(coco, ann_id):
    return next(a.get("track_id") for a in coco.annotations
                if a["id"] == ann_id)


def test_add_box_and_global_track_ids(lr, make_coco):
    cats = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    coco = make_coco(cats)
    coco.sticky_track_ids = False  # global auto-increment mode
    img1 = coco.ensure_image(_mkframe(0), 10, 10)
    img2 = coco.ensure_image(_mkframe(1), 10, 10)
    coco.add_box(img1, 0, 0, 2, 2, 0)
    coco.add_box(img1, 0, 0, 2, 2, 1)
    coco.add_box(img2, 0, 0, 2, 2, 0)
    assert [a["track_id"] for a in coco.annotations] == [1, 2, 3]
    assert coco.dirty
    a = coco.annotations[0]
    assert a["bbox"] == [0.0, 0.0, 2.0, 2.0] and a["area"] == 4.0


def test_add_box_sticky_ids_default(lr, make_coco):
    """Sticky ids default ON: the k-th box on a frame inherits the k-th
    track id from the nearest earlier annotated frame."""
    cats = [{"id": 0, "name": "a"}]
    coco = make_coco(cats)
    assert coco.sticky_track_ids is True
    img1 = coco.ensure_image(_mkframe(0), 10, 10)
    img2 = coco.ensure_image(_mkframe(1), 10, 10)
    coco.add_box(img1, 0, 0, 2, 2, 0)   # no earlier frame → fresh id
    coco.add_box(img1, 0, 0, 2, 2, 0)   # second box on the frame → fresh id
    coco.add_box(img2, 0, 0, 2, 2, 0)   # inherits k-th id from img1
    coco.add_box(img2, 0, 0, 2, 2, 0)
    assert [a["track_id"] for a in coco.annotations] == [1, 2, 1, 2]


def test_track_id_counter_rebuilt_after_load(lr, tmp_path):
    cats = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    path = str(tmp_path / "rebuild.json")
    coco = lr.CocoState(path, cats)
    img = coco.ensure_image(_mkframe(0), 10, 10)
    coco.add_box(img, 0, 0, 2, 2, 0)
    coco.add_box(img, 0, 0, 2, 2, 1)
    coco.add_box(img, 0, 0, 2, 2, 0)
    coco.save(is_final=True)
    coco2 = lr.CocoState(path, cats)
    coco2.load_existing()
    coco2.add_box(img, 0, 0, 2, 2, 1)
    assert coco2.annotations[-1]["track_id"] == 4  # saved max 3 → next is 4


def test_sticky_track_id_inheritance(lr, make_coco):
    coco = make_coco([{"id": 0, "name": "a"}, {"id": 1, "name": "b"}])
    coco.sticky_track_ids = True
    img0 = coco.ensure_image(_mkframe(0), 10, 10)
    img1 = coco.ensure_image(_mkframe(1), 10, 10)
    img2 = coco.ensure_image(_mkframe(2), 10, 10)

    # frame 0 (no earlier frame): global counter 1, 2
    a1 = coco.add_box(img0, 0, 0, 2, 2, 0)
    a2 = coco.add_box(img0, 4, 4, 2, 2, 1)
    assert (_tid(coco, a1), _tid(coco, a2)) == (1, 2)

    # frame 1: box 1 → track 1, box 2 → track 2 (inherit from frame 0)
    b1 = coco.add_box(img1, 1, 1, 2, 2, 0)
    b2 = coco.add_box(img1, 5, 5, 2, 2, 1)
    assert (_tid(coco, b1), _tid(coco, b2)) == (1, 2)
    # third box exceeds the reference frame's box count → fresh global id
    b3 = coco.add_box(img1, 2, 2, 1, 1, 0)
    assert _tid(coco, b3) == 3

    # frame 2: nearest earlier annotated frame is frame 1 → inherit its ids
    c1 = coco.add_box(img2, 1, 1, 2, 2, 0)
    c2 = coco.add_box(img2, 5, 5, 2, 2, 1)
    assert (_tid(coco, c1), _tid(coco, c2)) == (1, 2)

    # delete frame-1 box 1, then draw a new one on frame 1: live count is
    # 2 (b2, b3) → k=2, reference (frame 0) has only 2 → fresh id
    coco.remove_box(b1)
    b4 = coco.add_box(img1, 3, 3, 1, 1, 0)
    assert _tid(coco, b4) == 4


def test_seed_and_interp_box_track_ids(lr, make_coco):
    coco = make_coco([{"id": 0, "name": "a"}, {"id": 1, "name": "b"}])
    coco.sticky_track_ids = True  # even sticky mode: seeds stay global
    img0 = coco.ensure_image(_mkframe(0), 10, 10)
    img1 = coco.ensure_image(_mkframe(1), 10, 10)
    coco.add_box(img0, 0, 0, 2, 2, 0)  # track 1
    # seed on a later frame with an earlier annotated frame present
    coco.seed_box(img1, 5, 5, 1, 1, "a")
    seed_ann = coco.annotations[-1]
    assert seed_ann["seed"] and seed_ann["track_id"] == 2
    # interpolation boxes take their explicit track id
    interp_id = coco.add_interp_box(img1, 0, 0, 2, 2, 0, 7, "flow", 0.9)
    assert _tid(coco, interp_id) == 7


def test_remove_box_undo_redo(lr, make_coco):
    coco = make_coco([{"id": 0, "name": "a"}])
    img = coco.ensure_image(_mkframe(0), 10, 10)
    ann_id = coco.add_box(img, 1, 1, 3, 3, 0)
    assert len(coco.anns_for_image(img)) == 1
    coco.undo_stack.pop_undo()[1]()  # undo add
    assert len(coco.anns_for_image(img)) == 0
    coco.undo_stack.pop_redo()[2]()  # redo add
    assert len(coco.anns_for_image(img)) == 1

    coco.remove_box(ann_id)
    assert ann_id in coco.removed_ids
    coco.undo_stack.pop_undo()[1]()  # undo remove restores the box
    assert ann_id not in coco.removed_ids
    assert coco.get_box(ann_id) is not None


def test_set_cat_and_set_track_id(lr, make_coco):
    coco = make_coco([{"id": 0, "name": "a"}, {"id": 1, "name": "b"}])
    img = coco.ensure_image(_mkframe(0), 10, 10)
    ann_id = coco.add_box(img, 1, 1, 3, 3, 0)
    assert coco.set_cat(ann_id, 1) is True
    assert coco.get_box(ann_id)["category_id"] == 1
    coco.undo_stack.pop_undo()[1]()
    assert coco.get_box(ann_id)["category_id"] == 0

    assert coco.set_track_id(ann_id, 42) is True
    assert coco.get_box(ann_id)["track_id"] == 42
    assert coco.set_track_id(ann_id, None) is True
    assert "track_id" not in coco.get_box(ann_id)
    coco.undo_stack.pop_undo()[1]()  # undo the unset
    assert coco.get_box(ann_id)["track_id"] == 42


# ---------------------------------------------------------------------------
# Mask encode/decode + polygon segmentation
# ---------------------------------------------------------------------------

def test_mask_png_roundtrip(lr):
    mask = np.zeros((30, 40), bool)
    mask[5:20, 10:35] = True
    blob = lr._encode_mask_png(mask)
    assert isinstance(blob, bytes) and blob
    back = lr._decode_mask_png(blob)
    assert back.shape == mask.shape and (back == mask).all()
    assert lr._encode_mask_png(None) is None
    assert lr._decode_mask_png(b"") is None
    assert lr._decode_mask_png(b"not a png") is None


def test_mask_to_polygons_speck_filter(lr):
    # one big 20x20 blob (area ~361) + one tiny 3x3 speck (area ~4)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    mask[50:53, 50:53] = True
    assert len(lr._mask_to_polygons(mask)) == 1          # default min_area=100
    assert len(lr._mask_to_polygons(mask, 0)) == 2       # keep everything
    # specks dropped but the main contour is ALWAYS kept, even below
    # min_area — otherwise small-object masks vanish from the saved file
    assert len(lr._mask_to_polygons(mask, 1000)) == 1
    assert lr._mask_to_polygons(np.zeros((8, 8), bool)) == []


def test_segmentation_save_load_roundtrip(lr, make_coco, tmp_path):
    path = str(tmp_path / "seg.json")
    coco = lr.CocoState(path, [{"id": 0, "name": "a"}])
    # Shape test, not speck filtering — keep every contour (the default
    # min_polygon_area=100 would drop the 3x3 blob).
    coco.min_polygon_area = 0
    img_id = coco.ensure_image({"timestamp_ns": 0, "log_time_ns": 0,
                                "frame_idx": 0}, 100, 100)
    ann_id = coco.add_box(img_id, 10, 10, 40, 40, 0)
    mask = np.zeros((100, 100), bool)
    mask[20:60, 20:60] = True   # filled square
    mask[5:8, 5:8] = True       # small disjoint blob → second polygon
    coco.set_mask(ann_id, mask)
    coco.save(is_final=True)

    d = json.load(open(path))
    ann = d["annotations"][0]
    assert "mask" not in ann, "legacy mask field should not be written"
    seg = ann.get("segmentation")
    assert isinstance(seg, list) and len(seg) == 2  # two disjoint regions
    assert all(isinstance(p, list) and len(p) >= 6 and
               all(isinstance(v, int) for v in p) for p in seg)

    # round-trip: load back, mask restored approximately (rasterized lazily)
    coco2 = lr.CocoState(path, [{"id": 0, "name": "a"}])
    coco2.load_existing()
    m2 = coco2.ensure_mask(coco2.annotations[0])
    assert m2 is not None and m2.shape == (100, 100)
    iou = (m2 & mask).sum() / (m2 | mask).sum()
    assert iou > 0.95, f"mask round-trip IoU too low: {iou}"


def test_legacy_base64_mask_migrates_to_segmentation(lr, tmp_path):
    mask = np.zeros((100, 100), bool)
    mask[20:60, 20:60] = True
    path = str(tmp_path / "legacy.json")
    png = lr._encode_mask_png(mask)
    json.dump({
        "images": [{"id": 1, "timestamp_ns": 0, "frame_idx": 0,
                    "file_name": "x.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [10, 10, 40, 40], "area": 1600,
                         "iscrowd": 0,
                         "mask": base64.b64encode(png).decode("ascii")}],
        "categories": [{"id": 0, "name": "a"}],
    }, open(path, "w"))
    coco = lr.CocoState(path, [{"id": 0, "name": "a"}])
    coco.load_existing()
    # Legacy base64 masks load as lazy PNG bytes (_mask_png) and are
    # materialized on demand — eager decoding of a whole dataset would
    # exhaust RAM.
    assert coco.annotations[0].get("_mask") is None
    m3 = coco.ensure_mask(coco.annotations[0])
    assert m3 is not None and m3.shape == (100, 100) and m3[30, 30]
    # re-saving a legacy-loaded file writes segmentation, not mask
    coco.save(is_final=True)
    d3 = json.load(open(path))
    assert "mask" not in d3["annotations"][0]
    assert d3["annotations"][0].get("segmentation")


def test_set_mask_muted_skips_undo_snapshot(lr, tmp_path):
    """While the undo stack is muted (propagation runs), set_mask still
    attaches the mask but pushes no undo entry and makes no snapshot
    copies."""
    path = str(tmp_path / "muted.json")
    coco = lr.CocoState(path, [{"id": 0, "name": "a"}])
    img = coco.ensure_image({"timestamp_ns": 0, "log_time_ns": 0,
                            "frame_idx": 0}, 32, 32)
    ann_id = coco.add_box(img, 1, 1, 10, 10, 0)
    n_undo = len(coco.undo_stack._undo)
    mask = np.zeros((32, 32), bool)
    mask[2:8, 2:8] = True
    with coco.undo_stack.mute():
        coco.set_mask(ann_id, mask)
    assert coco.get_box(ann_id)["_mask"] is mask  # attached, no copy
    assert len(coco.undo_stack._undo) == n_undo   # nothing pushed
    assert coco.dirty


def test_save_honors_min_polygon_area(lr, tmp_path):
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    mask[50:53, 50:53] = True
    path = str(tmp_path / "minpoly.json")
    coco = lr.CocoState(path, [{"id": 0, "name": "a"}])
    assert coco.min_polygon_area == 100.0  # default
    img = coco.ensure_image({"timestamp_ns": 1, "log_time_ns": 1,
                             "frame_idx": 0}, 64, 64)
    ann_id = coco.add_box(img, 0, 0, 60, 60, 0)
    coco.set_mask(ann_id, mask)
    coco.save(is_final=True)
    seg = json.load(open(path))["annotations"][0].get("segmentation", [])
    assert len(seg) == 1
    coco.min_polygon_area = 0
    coco.save(is_final=True)
    seg = json.load(open(path))["annotations"][0].get("segmentation", [])
    assert len(seg) == 2


def test_save_load_roundtrip_fields(lr, tmp_path):
    path = str(tmp_path / "rt.json")
    cats = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    coco = lr.CocoState(path, cats)
    img = coco.ensure_image(_mkframe(3), 100, 80)
    coco.add_box(img, 5, 6, 7, 8, 1)
    coco.mark_reviewed(3)
    coco.annotated_marks.add(5)  # "Mark as annotated" on an empty frame
    coco.save(is_final=True)
    assert not coco.dirty

    d = json.load(open(path))
    assert d["categories"] == cats
    assert d["images"][0]["timestamp_ns"] == 3 * 10**9
    assert d["annotations"][0]["bbox"] == [5.0, 6.0, 7.0, 8.0]
    # annotated_image_ids = image ids with boxes ∪ explicit marks
    # (frame 5 has no image record so the mark contributes nothing)
    assert d["annotated_image_ids"] == [img]
    # annotated_timestamps mirrors the annotated image's timestamp_ns.
    assert d["annotated_timestamps"] == [3 * 10**9]
    # progress sidecar
    prog = json.load(open(path.replace(".json", ".progress")))
    assert prog["reviewed"] == [3]
    assert prog["annotated_marks"] == [5]

    coco2 = lr.CocoState(path, cats)
    coco2.load_existing()
    assert len(coco2.annotations) == 1
    assert coco2.annotations[0]["category_id"] == 1


# ---------------------------------------------------------------------------
# load_existing categories merge rules (resume behavior)
# ---------------------------------------------------------------------------

_STALE_CATS = [{"id": 0, "name": "Ceiling light"}, {"id": 1, "name": "Monitor"},
               {"id": 2, "name": "3D printer"}, {"id": 3, "name": "Ticket Gate"}]


def test_empty_previous_session_does_not_resurrect_cats(lr, tmp_path):
    p = str(tmp_path / "empty.json")
    json.dump({"images": [], "annotations": [], "categories": _STALE_CATS},
              open(p, "w"))
    coco = lr.CocoState(p, [])  # fresh start, empty categories
    coco.load_existing()
    assert coco.categories == []
    assert coco.cat_map == {}


def test_content_session_inherits_categories(lr, tmp_path):
    p = str(tmp_path / "content.json")
    json.dump({
        "images": [{"id": 1, "timestamp_ns": 1, "frame_idx": 0,
                    "file_name": "a.jpg", "width": 8, "height": 8}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "Monitor"}],
    }, open(p, "w"))
    coco = lr.CocoState(p, [])
    coco.load_existing()
    assert coco.cat_map == {1: "Monitor"}


def test_seeded_categories_merge_with_content_session(lr, tmp_path):
    p = str(tmp_path / "content2.json")
    json.dump({
        "images": [{"id": 1, "timestamp_ns": 1, "frame_idx": 0,
                    "file_name": "a.jpg", "width": 8, "height": 8}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "Monitor"}],
    }, open(p, "w"))
    coco = lr.CocoState(p, [{"id": 0, "name": "New"}])
    coco.load_existing()
    assert coco.cat_map == {1: "Monitor", 0: "New"}


def test_seed_categories(lr, tmp_path):
    assert lr._seed_categories(None) == []
    seed = tmp_path / "seed.json"
    json.dump({"categories": [{"id": 0, "name": "x"}, {"id": 5, "name": "y"}]},
              open(seed, "w"))
    cats = lr._seed_categories(str(seed))
    assert [c["name"] for c in cats] == ["x", "y"]


# ---------------------------------------------------------------------------
# import_coco
# ---------------------------------------------------------------------------

def test_import_coco_merge_dedup_masks(lr, tmp_path):
    folder = make_image_folder(tmp_path / "src", ["a.png", "b.png", "c.png"],
                               size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    coco = lr.CocoState(str(tmp_path / "out.json"), [{"id": 0, "name": "known"}])

    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:9] = True
    polys = lr._mask_to_polygons(mask, 0)
    src = {
        "images": [
            {"id": 10, "file_name": "a.png", "width": 12, "height": 10},
            {"id": 11, "file_name": "b.png", "width": 12, "height": 10},
            {"id": 12, "file_name": "zzz_foreign.png", "width": 4,
             "height": 4},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 7, "bbox": [1, 1, 4, 4],
             "track_id": 41},                                    # known cat
            {"id": 2, "image_id": 11, "category_id": 8, "bbox": [2, 2, 5, 5],
             "segmentation": polys},                             # new cat + mask
            {"id": 3, "image_id": 12, "category_id": 7,
             "bbox": [0, 0, 1, 1]},                              # foreign image
            {"id": 4, "image_id": 11, "category_id": 99,
             "bbox": [0, 0, 1, 1]},                              # dangling cat
        ],
        "categories": [{"id": 7, "name": "known"},
                       {"id": 8, "name": "imported_cat"}],
    }
    file_to_frame = {os.path.basename(fp): i for i, fp in enumerate(idx.files)}
    n_frames, n_ok, n_skip, n_merged = coco.import_coco(src, file_to_frame,
                                                        idx)
    assert n_frames == 2
    assert n_ok == 2
    assert n_skip == 1  # dangling category 99
    assert n_merged == 0
    assert coco.cat_name_to_id.get("imported_cat") is not None

    # box lands on the right frame's image with remapped ids + kept track id
    img_a = next(i for i in coco.images if i["file_name"] == "a.png")
    anns_a = [a for a in coco.annotations if a["image_id"] == img_a["id"]]
    assert len(anns_a) == 1
    assert anns_a[0]["category_id"] == 0  # "known" merged by name, not id 7
    assert anns_a[0]["track_id"] == 41    # source track id preserved
    img_b = next(i for i in coco.images if i["file_name"] == "b.png")
    ann_b = [a for a in coco.annotations if a["image_id"] == img_b["id"]][0]
    assert isinstance(coco.ensure_mask(ann_b), np.ndarray)  # mask restored

    # round-trip through save: segmentation comes back out
    coco.save(is_final=True)
    saved = json.load(open(coco.output_json))
    s_b = next(a for a in saved["annotations"] if a["image_id"] == img_b["id"])
    assert s_b.get("segmentation"), "imported mask must save as segmentation"

    # importing the same file again is a no-op (duplicate guard)
    n2 = coco.import_coco(src, file_to_frame, idx)
    assert n2[1] == 0 and n2[2] >= 2


def test_import_coco_merges_masks_into_duplicate_boxes(lr, tmp_path):
    """Re-loading a SAM3-annotated file over already-drawn boxes must merge
    the masks onto the existing boxes instead of silently dropping them
    (the duplicate guard used to skip the whole annotation)."""
    folder = make_image_folder(tmp_path / "src", ["a.png"], size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    coco = lr.CocoState(str(tmp_path / "out.json"), [{"id": 0, "name": "k"}])

    # the box already exists in the session (drawn by the user), no mask
    frame = idx.frame_at(0)
    img_id = coco.ensure_image(frame, 12, 10)
    coco.add_box(img_id, 1, 1, 4, 4, 0)

    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:9] = True
    polys = lr._mask_to_polygons(mask, 0)
    src = {
        "images": [{"id": 1, "file_name": "a.png", "width": 12,
                    "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [1, 1, 4, 4], "segmentation": polys}],
        "categories": [{"id": 0, "name": "k"}],
    }
    file_to_frame = {os.path.basename(fp): i for i, fp in enumerate(idx.files)}
    n_frames, n_ok, n_skip, n_merged = coco.import_coco(src, file_to_frame,
                                                        idx)
    assert (n_ok, n_skip, n_merged) == (0, 1, 1)
    assert len(coco.annotations) == 1  # still just the original box
    merged = coco.ensure_mask(coco.annotations[0])
    assert isinstance(merged, np.ndarray) and merged.any()

    # second import: the box already has a mask — nothing to merge
    n2 = coco.import_coco(src, file_to_frame, idx)
    assert n2[3] == 0


def test_import_coco_zero_dim_source_decodes_frame(lr, tmp_path):
    """Source image records without width/height must not produce 0-dim
    image records (which silently drop every mask on reload) — the frame
    is decoded for the real dims."""
    folder = make_image_folder(tmp_path / "src", ["a.png"], size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    coco = lr.CocoState(str(tmp_path / "out.json"), [{"id": 0, "name": "k"}])
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:9] = True
    src = {
        "images": [{"id": 1, "file_name": "a.png"}],  # no width/height
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [1, 1, 4, 4],
                         "segmentation": lr._mask_to_polygons(mask, 0)}],
        "categories": [{"id": 0, "name": "k"}],
    }
    file_to_frame = {os.path.basename(fp): i for i, fp in enumerate(idx.files)}
    coco.import_coco(src, file_to_frame, idx)
    img = coco.images[0]
    assert (img["width"], img["height"]) == (12, 10)
    assert coco.ensure_mask(coco.annotations[0]) is not None


def test_ensure_image_never_clobbers_dims_with_zero(lr, make_coco):
    coco = make_coco([])
    frame = {"timestamp_ns": 5, "log_time_ns": 5, "frame_idx": 0,
             "existing_boxes": [], "file_name": "a.png"}
    img_id = coco.ensure_image(frame, 12, 10)
    coco.ensure_image(frame, 0, 0)  # e.g. import from a dim-less source
    img = next(i for i in coco.images if i["id"] == img_id)
    assert (img["width"], img["height"]) == (12, 10)


# ---------------------------------------------------------------------------
# discard frames (excluded from the final JSON)
# ---------------------------------------------------------------------------

def test_discarded_frames_excluded_from_final_only(lr, tmp_path):
    folder = make_image_folder(tmp_path / "src", ["a.png", "b.png"],
                               size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    out = str(tmp_path / "out.json")
    coco = lr.CocoState(out, [{"id": 0, "name": "k"}])
    id_a = coco.ensure_image(idx.frame_at(0), 12, 10)
    id_b = coco.ensure_image(idx.frame_at(1), 12, 10)
    coco.add_box(id_a, 0, 0, 2, 2, 0)
    coco.add_box(id_b, 1, 1, 2, 2, 0)
    coco.annotated_marks.add(1)
    coco.discarded_frames.add(1)

    # tmp save keeps everything (discard stays reversible)
    coco.save(is_final=False)
    tmp = json.load(open(out.replace(".json", "_tmp.json")))
    assert len(tmp["images"]) == 2 and len(tmp["annotations"]) == 2

    # final save drops the discarded frame's image, boxes and marks
    coco.save(is_final=True)
    final = json.load(open(out))
    assert [i["id"] for i in final["images"]] == [id_a]
    assert {a["image_id"] for a in final["annotations"]} == {id_a}
    assert final["annotated_image_ids"] == [id_a]
    # annotated_timestamps reflects only the surviving (non-discarded) frame
    assert final["annotated_timestamps"] == [
        next(i["timestamp_ns"] for i in final["images"]
             if i["id"] == id_a)]

    # the discard set persists in the .progress sidecar
    progress = json.load(open(out.replace(".json", ".progress")))
    assert progress["discarded"] == [1]
    coco2 = lr.CocoState(out, [{"id": 0, "name": "k"}])
    coco2.load_progress(2)
    assert coco2.discarded_frames == {1}


def test_add_interp_box_invalidates_ann_cache(lr, tmp_path):
    """Regression: add_interp_box must invalidate the anns_for_image cache.

    Without _invalidate_ann_caches() the interpolated boxes were appended to
    self.annotations but anns_for_image() kept serving the stale cached
    list — the UI reported the interpolation done while the canvas drew
    nothing until some other action rebuilt the cache."""
    out = str(tmp_path / "interp.json")
    coco = lr.CocoState(out, [{"id": 0, "name": "k"}])
    idx = FakeIdx(3)
    id_0 = coco.ensure_image(idx.frame_at(0), 12, 10)
    id_1 = coco.ensure_image(idx.frame_at(1), 12, 10)
    coco.add_box(id_0, 0, 0, 2, 2, 0)
    assert len(coco.anns_for_image(id_0)) == 1  # cache built

    coco.add_interp_box(id_1, 1, 1, 2, 2, 0, track_id=None,
                        source="linear", confidence=0.5)
    # The freshly added interp box must be visible immediately.
    anns = coco.anns_for_image(id_1)
    assert len(anns) == 1 and anns[0].get("interp") is True
    assert anns[0]["source"] == "linear" and anns[0]["confidence"] == 0.5
    assert coco.frame_has_boxes(1)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def test_iou_xyxy(lr):
    assert lr._iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert lr._iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    iou = lr._iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 25 / 175) < 1e-9
    # zero-area boxes → 0, no crash
    assert lr._iou_xyxy([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_resolve_device(lr):
    assert lr._resolve_device("cuda") == "cuda"
    assert lr._resolve_device("cpu") == "cpu"
    import torch
    assert lr._resolve_device("auto") == ("cuda" if torch.cuda.is_available()
                                          else "cpu")


def test_empty_index(lr):
    idx = lr.EmptyIndex()
    assert len(idx) == 0
    assert idx.find_idx_by_timestamp(123) == -1
    with pytest.raises(IndexError):
        idx.frame_at(0)


def test_fake_idx_interface():
    # sanity: the conftest fake actually satisfies the index interface
    idx = FakeIdx(3)
    assert len(idx) == 3
    f = idx.frame_at(1)
    assert f["frame_idx"] == 1 and f["file_name"] == "f1.png"
    assert idx.decode_image(0).shape == (80, 100, 3)
    assert idx.find_idx_by_timestamp(0) == -1


def test_save_dirty_gate(lr, tmp_path, capsys):
    """Navigation-only saves refresh only the .progress sidecar — the full
    JSON dump (which re-polygonizes every mask) happens only when the
    annotations actually changed."""
    import json
    out = tmp_path / "gate.json"
    tmp_json = tmp_path / "gate_tmp.json"
    prog = tmp_path / "gate.progress"
    coco = lr.CocoState(str(out), [{"id": 0, "name": "a"}])

    # First save: tmp doesn't exist yet → full dump even though not dirty.
    coco.current_idx = 0
    coco.save(is_final=False)
    assert tmp_json.exists() and prog.exists()
    first_bytes = tmp_json.read_bytes()

    # Navigation-only save: tmp untouched, progress updated, no print.
    coco.current_idx = 5
    capsys.readouterr()
    coco.save(is_final=False)
    assert capsys.readouterr().out == ""
    assert tmp_json.read_bytes() == first_bytes
    assert json.loads(prog.read_text())["last_index"] == 6

    # Review marks alone don't force a full dump either.
    coco.mark_reviewed(5)
    coco.save(is_final=False)
    assert tmp_json.read_bytes() == first_bytes
    assert json.loads(prog.read_text())["reviewed"] == [5]

    # A real mutation flips dirty → full dump on next save.
    frame = {"frame_idx": 0, "timestamp_ns": 1000, "log_time_ns": 1000,
             "existing_boxes": [], "file_path": "/img/f0.png",
             "file_name": "f0.png"}
    img = coco.ensure_image(frame, 100, 80)
    coco.add_box(img, 1, 2, 3, 4, 0)
    coco.save(is_final=False)
    assert tmp_json.read_bytes() != first_bytes
    assert "Saved progress" in capsys.readouterr().out
    assert not coco.dirty

    # Final save always dumps, dirty or not.
    coco.save(is_final=True)
    assert out.exists()
    assert "Saved final" in capsys.readouterr().out

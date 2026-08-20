"""Stereo dual-view: StereoIndex pairing, side-aware CocoState (ensure_image
/ save / import_coco), and the two-canvas ReviewWindow (offscreen)."""

import json
import os

import pytest

from conftest import FakePropagateWorker, make_image_folder


def _mkframe(i, ts=None, name=None):
    ts = i * 10**9 if ts is None else ts
    return {"timestamp_ns": ts, "log_time_ns": ts, "frame_idx": i,
            "existing_boxes": [], "file_name": name or f"f{i}.png"}


# ---------------------------------------------------------------------------
# StereoIndex
# ---------------------------------------------------------------------------

def test_stereo_index_pairing_and_decode(lr, tmp_path):
    left = make_image_folder(tmp_path / "L", [("b.png", 10), ("a.png", 20)])
    right = make_image_folder(tmp_path / "R", [("b.png", 200),
                                               ("a.png", 210)])
    idx = lr.StereoIndex([str(left)], [str(right)])
    assert idx.stereo
    assert len(idx) == 2
    # positional pairing after per-side name sorting
    assert idx.frame_at(0, "left")["file_name"] == "a.png"
    assert idx.frame_at(0, "right")["file_name"] == "a.png"
    assert idx.files == idx.files_left
    assert idx.files_left[0].endswith(os.path.join("L", "a.png"))
    assert idx.files_right[0].endswith(os.path.join("R", "a.png"))
    # default side is left; per-side decode reads the right folder
    assert idx.frame_at(1)["file_name"] == "b.png"
    assert idx.decode_image(1, "left")[0, 0, 0] == 10
    assert idx.decode_image(1, "right")[0, 0, 0] == 200
    # mono indices are flagged non-stereo
    assert not lr.ImageFolderIndex([str(left)]).stereo
    assert not lr.EmptyIndex().stereo


def test_stereo_index_uneven_lengths_warns(lr, tmp_path, capsys):
    left = make_image_folder(tmp_path / "L2", ["a.png", "b.png", "c.png"])
    right = make_image_folder(tmp_path / "R2", ["a.png", "b.png"])
    idx = lr.StereoIndex([str(left)], [str(right)])
    assert len(idx) == 2  # shorter side wins
    assert len(idx.files_left) == 2 and len(idx.files_right) == 2
    assert "differ in length" in capsys.readouterr().out


def test_stereo_side_index_worker_interface(lr, tmp_path):
    left = make_image_folder(tmp_path / "L3", [("a.png", 30)])
    right = make_image_folder(tmp_path / "R3", [("a.png", 220)])
    idx = lr.StereoIndex([str(left)], [str(right)])
    sv = idx.side_index("right")
    # exactly the worker-facing mono interface, scoped to one side
    assert len(sv) == len(idx) == 1
    assert sv.frame_at(0)["file_path"] == idx.files_right[0]
    assert sv.decode_image(0)[0, 0, 0] == 220
    assert sv.files == idx.files_right
    assert sv.timestamps_real == idx.timestamps_real
    assert not sv.stereo  # looks mono to its consumers
    with pytest.raises(ValueError):
        idx.side_index("middle")


# ---------------------------------------------------------------------------
# CocoState: side-aware image identity
# ---------------------------------------------------------------------------

def test_ensure_image_same_ts_different_sides(lr, make_coco):
    coco = make_coco([])
    frame = _mkframe(0)
    left_id = coco.ensure_image(frame, 10, 10)
    right_id = coco.ensure_image(frame, 10, 10, side="right")
    assert left_id != right_id
    assert len(coco.images) == 2
    assert coco.images[0]["side"] == "left"
    assert coco.images[1]["side"] == "right"
    # idempotent per (ts, side)
    assert coco.ensure_image(frame, 10, 10) == left_id
    assert coco.ensure_image(frame, 10, 10, side="right") == right_id
    # bare-key lookups still hit the left record (mono-era compat)
    assert coco._img_id_by_idx[0] == left_id
    assert coco._img_id_by_idx[(0, "right")] == right_id
    assert coco._img_id_by_ts[(frame["timestamp_ns"], "right")] == right_id


def test_save_emits_side_and_unioned_annotated_idxs(lr, tmp_path):
    path = str(tmp_path / "stereo.json")
    coco = lr.CocoState(path, [{"id": 0, "name": "a"}])
    l0 = coco.ensure_image(_mkframe(0), 10, 10)
    coco.ensure_image(_mkframe(0), 10, 10, side="right")  # visited, no boxes
    r1 = coco.ensure_image(_mkframe(1), 10, 10, side="right")
    coco.add_box(l0, 0, 0, 2, 2, 0)      # frame 0 annotated on the LEFT
    coco.add_box(r1, 1, 1, 2, 2, 0)      # frame 1 annotated on the RIGHT
    coco.annotated_marks.add(2)          # frame 2 explicitly marked
    coco.save(is_final=True)

    d = json.load(open(path))
    keys = {(img["frame_idx"], img["side"]) for img in d["images"]}
    assert keys == {(0, "left"), (0, "right"), (1, "right")}
    # image ids annotated on either side; frame 2 was explicitly marked
    # but has no image record, so it contributes nothing.
    assert d["annotated_image_ids"] == [l0, r1]

    # reload: side-keyed lookup tables are rebuilt from the file
    coco2 = lr.CocoState(path, [{"id": 0, "name": "a"}])
    coco2.load_existing()
    assert coco2._img_id_by_idx[(1, "right")] == r1
    assert coco2.frame_has_boxes(1)          # either side
    assert coco2.frame_has_boxes(1, "right")
    assert not coco2.frame_has_boxes(1, "left")
    assert coco2.labeled_frame_idxs() == [0, 1]
    assert coco2.labeled_frame_idxs("right") == [1]


def test_import_coco_matches_by_basename_and_side(lr, tmp_path):
    left = make_image_folder(tmp_path / "Li", ["a.png", "b.png"],
                             size=(10, 12))
    right = make_image_folder(tmp_path / "Ri", ["a.png", "b.png"],
                              size=(10, 12))
    idx = lr.StereoIndex([str(left)], [str(right)])
    coco = lr.CocoState(str(tmp_path / "out.json"), [{"id": 0, "name": "k"}])
    src = {
        "images": [
            # no "side" field → goes to the left side
            {"id": 1, "file_name": "a.png", "width": 12, "height": 10},
            {"id": 2, "file_name": "a.png", "width": 12, "height": 10,
             "side": "right"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [3, 3, 2, 2]},
        ],
        "categories": [{"id": 0, "name": "k"}],
    }
    file_to_frame = {}
    for side, files in (("left", idx.files_left), ("right", idx.files_right)):
        for i, fp in enumerate(files):
            file_to_frame[(os.path.basename(fp), side)] = i
    n_frames, n_ok, n_skip = coco.import_coco(src, file_to_frame, idx)
    # frames_matched counts distinct frame indices — both annotations land
    # on frame 0 (one per side)
    assert (n_frames, n_ok, n_skip) == (1, 2, 0)
    by_key = {(img["frame_idx"], img["side"]): img["id"]
              for img in coco.images}
    assert {a["image_id"] for a in coco.annotations} == {
        by_key[(0, "left")], by_key[(0, "right")]}
    left_ann = next(a for a in coco.annotations
                    if a["image_id"] == by_key[(0, "left")])
    right_ann = next(a for a in coco.annotations
                     if a["image_id"] == by_key[(0, "right")])
    assert left_ann["bbox"] == [1.0, 1.0, 2.0, 2.0]
    assert right_ann["bbox"] == [3.0, 3.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# ReviewWindow smoke (offscreen)
# ---------------------------------------------------------------------------

def _stereo_setup(lr, make_coco, make_window, tmp_path, name="st"):
    left = make_image_folder(tmp_path / f"{name}_L",
                             ["a.png", "b.png", "c.png"], size=(16, 16))
    right = make_image_folder(tmp_path / f"{name}_R",
                              ["a.png", "b.png", "c.png"], size=(16, 16))
    idx = lr.StereoIndex([str(left)], [str(right)])
    coco = make_coco([{"id": 0, "name": "a"}])
    win = make_window(idx, coco)
    return win, coco, idx, str(left), str(right)


def test_mono_window_single_canvas(lr, make_coco, make_window, tmp_path):
    folder = make_image_folder(tmp_path / "mono", ["a.png", "b.png"])
    win = make_window(lr.ImageFolderIndex([str(folder)]), make_coco([]))
    assert list(win.canvases) == ["left"]
    assert win.canvas.side == "left"
    assert win._active_canvas is win.canvas
    assert not win._stereo


def test_stereo_window_two_canvases_nav(lr, make_coco, make_window,
                                        tmp_path):
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path)
    assert win._stereo
    assert set(win.canvases) == {"left", "right"}
    assert win.canvas.side == "left" and win.canvases["right"].side == "right"
    # both canvases got their own image records for the same frame
    assert win.canvas._image_id is not None
    assert win.canvases["right"]._image_id is not None
    assert win.canvas._image_id != win.canvases["right"]._image_id
    assert win.canvas._info_text.startswith("LEFT")
    assert win.canvases["right"]._info_text.startswith("RIGHT")

    # navigation advances BOTH canvases (index-based, lockstep)
    left0, right0 = win.canvas._image_id, win.canvases["right"]._image_id
    win._on_frame_nav(+1)
    assert win._current_idx == 1
    assert win.canvas._image_id not in (None, left0)
    assert win.canvases["right"]._image_id not in (None, right0)

    # four image records so far: frames 0+1 on both sides
    assert {(img["frame_idx"], img["side"]) for img in coco.images} == {
        (0, "left"), (0, "right"), (1, "left"), (1, "right")}


def test_stereo_active_canvas_and_side_aware_save(lr, make_coco, make_window,
                                                  tmp_path):
    win, coco, idx, left_dir, right_dir = _stereo_setup(
        lr, make_coco, make_window, tmp_path, name="act")
    # default active = left; the compat attr tracks the active side
    assert win._current_image_id == win.canvas._image_id

    # switch the active side and draw on the right
    win._set_active_canvas(win.canvases["right"])
    assert win._current_image_id == win.canvases["right"]._image_id
    win._on_box_added(win.canvases["right"]._image_id, 2, 2, 4, 4, 0)
    assert len(coco.annotations) == 1
    ann_img = coco.annotations[0]["image_id"]
    assert ann_img == win.canvases["right"]._image_id

    # per-side tmp image paths point at the matching folder's file
    assert win._write_tmp_image("left").startswith(left_dir)
    assert win._write_tmp_image("right").startswith(right_dir)

    # save: the box lands on a "right"-sided image record
    out = str(tmp_path / "act_out.json")
    coco.output_json = out
    coco.progress_file = out.replace(".json", ".progress")
    coco.save(is_final=True)
    d = json.load(open(out))
    img = next(i for i in d["images"] if i["id"] == ann_img)
    assert img["side"] == "right"
    assert d["annotated_image_ids"] == [ann_img]


def test_stereo_discard_all_hits_active_side_only(lr, make_coco, make_window,
                                                  tmp_path):
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="x")
    win._skip_discard_confirm = True  # skip the modal confirm dialog
    left_id, right_id = win.canvas._image_id, win.canvases["right"]._image_id
    coco.add_box(left_id, 0, 0, 2, 2, 0)
    coco.add_box(right_id, 1, 1, 2, 2, 0)
    win._refresh_boxes()

    # X with the right side active discards only the right side's boxes
    win._set_active_canvas(win.canvases["right"])
    win._on_discard_all()
    assert len(coco.anns_for_image(right_id)) == 0
    assert len(coco.anns_for_image(left_id)) == 1


def test_sam3_all_runs_both_sides(lr, make_coco, make_window, tmp_path,
                                  fake_sam3, auto_yes):
    """SAM3 ALL in stereo chains both sides: left batch, then right."""
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="sam3all")
    # one maskless box on each side of frame 0
    coco.add_box(win.canvas._image_id, 0, 0, 4, 4, 0)
    coco.add_box(win.canvases["right"]._image_id, 1, 1, 4, 4, 0)
    win._refresh_boxes()

    win._on_sam3_all_frames()
    assert len(fake_sam3.any.instances) == 1  # left batch started
    assert win._batch_side == "left"
    assert win._sam3_all_pending == ["right"]

    # finishing the left batch chains the right side
    fake_sam3.any.instances[0].finished_signal.emit(1, 0)
    assert len(fake_sam3.any.instances) == 2
    assert win._batch_side == "right"
    assert win._sam3_all_pending == []

    # finishing the right batch completes the whole run
    fake_sam3.any.instances[1].finished_signal.emit(1, 0)
    assert win._sam3_all_ok == 2
    assert win._sam3_all_fail == 0


def test_sam3_all_cancel_stops_the_chain(lr, make_coco, make_window,
                                         tmp_path, fake_sam3, auto_yes):
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="sam3cancel")
    coco.add_box(win.canvas._image_id, 0, 0, 4, 4, 0)
    coco.add_box(win.canvases["right"]._image_id, 1, 1, 4, 4, 0)
    win._refresh_boxes()

    win._on_sam3_all_frames()
    fake_sam3.any.instances[0].cancelled_signal.emit()
    # cancel must NOT chain the right side
    assert len(fake_sam3.any.instances) == 1
    assert win._sam3_all_pending == []


def test_sam3_all_mono_single_batch(lr, make_coco, make_window, tmp_path,
                                    fake_sam3, auto_yes):
    folder = make_image_folder(tmp_path / "sam3mono", ["a.png", "b.png"])
    win = make_window(lr.ImageFolderIndex([str(folder)]), make_coco([]))
    win.coco.add_box(win.canvas._image_id, 0, 0, 4, 4, 0)

    win._on_sam3_all_frames()
    assert len(fake_sam3.any.instances) == 1
    assert win._batch_side == "left"
    fake_sam3.any.instances[0].finished_signal.emit(1, 0)
    assert win._sam3_all_ok == 1
    assert win._sam3_all_pending == []


def test_run_sam3_all_segments_both_sides(lr, make_coco, make_window,
                                          tmp_path, fake_sam3):
    """'Run SAM3' on a stereo frame segments BOTH sides' boxes: the left
    job starts immediately, the right one is queued behind it — regardless
    of which canvas has focus."""
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="sam3frame")
    left_ann = coco.add_box(win.canvas._image_id, 0, 0, 4, 4, 0)
    right_ann = coco.add_box(win.canvases["right"]._image_id, 1, 1, 4, 4, 0)
    win._refresh_boxes()
    # the user has the RIGHT side focused (the reported bug scenario)
    win._set_active_canvas(win.canvases["right"])

    win._on_run_sam3_all()
    assert len(fake_sam3.any.instances) == 1       # left job running
    assert fake_sam3.any.instances[0].kw["ann_ids"] == [left_ann]
    assert len(win._sam3_queue) == 1               # right job queued
    assert win._sam3_queue[0]["ann_ids"] == [right_ann]

    # when the left run finishes, the queued right job starts
    fake_sam3.any.instances[0].stop()
    win._start_next_queued_sam3()
    assert len(fake_sam3.any.instances) == 2
    assert fake_sam3.any.instances[1].kw["ann_ids"] == [right_ann]
    win._sam3_queue.clear()


def test_propagate_multiselect_both_sides(lr, make_coco, make_window,
                                          tmp_path, fake_sam3, auto_yes):
    """A cross-side multi-select propagates BOTH seeds, each on its own
    side: the first runs immediately, the second waits in the queue."""
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="propboth")
    coco.add_box(win.canvas._image_id, 0, 0, 4, 4, 0)
    coco.add_box(win.canvases["right"]._image_id, 1, 1, 4, 4, 0)
    win._refresh_boxes()
    win.canvas._multi_selected = {0}
    win.canvas._selected_idx = 0
    win.canvases["right"]._multi_selected = {0}
    win.canvases["right"]._selected_idx = 0
    win._set_active_canvas(win.canvases["right"])  # right focused, as reported

    win._on_propagate_track()
    assert len(FakePropagateWorker.instances) == 1  # left seed running
    assert win._propagate_meta()["side"] == "left"
    assert len(win._sam3_queue) == 1                # right seed queued
    assert win._sam3_queue[0]["kind"] == "propagate"
    assert win._sam3_queue[0]["side"] == "right"
    # each side's job carries exactly its own side's seed
    left_tid = coco.anns_for_image(win.canvas._image_id)[0]["track_id"]
    right_tid = coco.anns_for_image(win.canvases["right"]._image_id)[0]["track_id"]
    assert [s["track_id"] for s in
            FakePropagateWorker.instances[0].kw["seeds"]] == [left_tid]
    assert [s["track_id"] for s in
            win._sam3_queue[0]["seeds"]] == [right_tid]
    win._sam3_queue.clear()

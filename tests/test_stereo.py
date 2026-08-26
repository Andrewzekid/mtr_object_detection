"""Stereo dual-view: StereoIndex pairing, side-aware CocoState (ensure_image
/ save / import_coco), and the two-canvas ReviewWindow (offscreen)."""

import json
import os

import pytest

from conftest import (FakeAutolabelSingle, FakePropagateWorker,
                      FakeWorker, make_image_folder)


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


def test_stereo_index_pairs_by_timestamp_skipping_unmatched(lr, tmp_path,
                                                            capsys):
    """Timestamp-named folders are paired by EXACT timestamp; images with
    no counterpart on the other side are skipped (a dropped frame on one
    camera must not shift every later pair)."""
    left = make_image_folder(tmp_path / "Lts",
                             ["100.png", "200.png", "300.png", "400.png"])
    right = make_image_folder(tmp_path / "Rts",
                              ["200.png", "300.png", "400.png", "500.png"])
    idx = lr.StereoIndex([str(left)], [str(right)])
    assert idx.timestamps_real
    assert len(idx) == 3  # 100 left-only, 500 right-only — both skipped
    for i, ts in enumerate((200, 300, 400)):
        assert idx.frame_at(i, "left")["timestamp_ns"] == ts
        assert idx.frame_at(i, "right")["timestamp_ns"] == ts
    assert "skipped" in idx.pairing_warning
    assert "1 left-only" in idx.pairing_warning
    # timestamp lookup lands on the PAIR index
    assert idx.find_idx_by_timestamp(300) == 1
    assert idx.find_idx_by_timestamp(0) == 0  # snaps to nearest pair
    # the worker-facing side view follows the same pairing
    sv = idx.side_index("right")
    assert len(sv) == 3
    assert sv.frame_at(0)["timestamp_ns"] == 200
    assert sv.files[0].endswith(os.path.join("Rts", "200.png"))
    capsys.readouterr()


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
    # annotated_timestamps: unique timestamp_ns of annotated images.
    # _mkframe(0) → ts=0, _mkframe(1) → ts=1*10**9.
    assert d["annotated_timestamps"] == [0, 10**9]

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
    n_frames, n_ok, n_skip, n_merged = coco.import_coco(src, file_to_frame,
                                                        idx)
    # frames_matched counts distinct frame indices — both annotations land
    # on frame 0 (one per side)
    assert (n_frames, n_ok, n_skip, n_merged) == (1, 2, 0, 0)
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
    # the right-side frame 0 has timestamp 0 (FakeIdx default 1000+idx, but
    # StereoIndex uses the file's timestamp; make_image_folder → 0). Just
    # check the field exists and matches the annotated image's timestamp.
    assert d["annotated_timestamps"] == [img["timestamp_ns"]]


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


def test_autolabel_frame_runs_both_sides_in_stereo(lr, make_coco, make_window,
                                                   tmp_path, fake_sam3,
                                                   monkeypatch):
    """'Autolabel frame' in stereo autolabels BOTH sides: the active side's
    job starts immediately, the other waits in the SAM3 queue."""
    from gui.label_review.ui import main_window as mw
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="alframe")
    # Patch the generic autolabel single-frame worker (falcon/grounding_dino/
    # florence2 share it) so we can observe both dispatches.
    monkeypatch.setattr(mw, "GenericAutolabelWorker", FakeAutolabelSingle)
    FakeAutolabelSingle.instances.clear()
    win.autolabel_detector = "falcon"
    win.falcon_model = "tiiuae/Falcon-Perception"
    # RIGHT canvas focused — the reported bug scenario (only right got labelled)
    win._set_active_canvas(win.canvases["right"])

    win._on_autolabel_frame()
    # One worker is running, the other side's job is queued behind it.
    assert len(FakeAutolabelSingle.instances) == 1
    running = FakeAutolabelSingle.instances[0]
    assert running.isRunning()
    assert running.kw["detector"] == "falcon"
    assert len(win._sam3_queue) == 1
    queued = win._sam3_queue[0]
    assert queued["kind"] == "autolabel"
    assert queued["detector"] == "falcon"
    # The two jobs target different image ids (one per side).
    left_id = win.canvas._image_id
    right_id = win.canvases["right"]._image_id
    started_ids = {running.kw["image_id"], queued["image_id"]}
    assert started_ids == {left_id, right_id}
    win._sam3_queue.clear()
    FakeAutolabelSingle.instances[-1].stop()


def test_autolabel_all_runs_both_sides_in_stereo(lr, make_coco, make_window,
                                                tmp_path, fake_sam3,
                                                monkeypatch, auto_yes):
    """'Autolabel ALL frames' in stereo chains both sides: left batch, then
    right batch, with the accumulated box count reported at the end."""
    from gui.label_review.ui import main_window as mw
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="alall")
    monkeypatch.setattr(mw, "GenericAutolabelBatchWorker", FakeWorker)
    FakeWorker.instances.clear()
    win.autolabel_detector = "falcon"
    win.falcon_model = "tiiuae/Falcon-Perception"

    win._on_autolabel_all()
    # Left batch started, right pending.
    assert len(FakeWorker.instances) == 1
    assert win._autolabel_batch_side == "left"
    assert win._autolabel_all_pending == ["right"]
    # GenericAutolabelBatchWorker receives `detector` as the first positional.
    assert FakeWorker.instances[0].args[0] == "falcon"

    # Finishing the left batch chains the right batch.
    FakeWorker.instances[0].finished_signal.emit(0)
    assert len(FakeWorker.instances) == 2
    assert win._autolabel_batch_side == "right"
    assert win._autolabel_all_pending == []

    # Finishing the right batch completes the run; no extra workers spawned.
    FakeWorker.instances[1].finished_signal.emit(0)
    assert len(FakeWorker.instances) == 2


def test_autolabel_all_cancel_stops_the_chain(lr, make_coco, make_window,
                                              tmp_path, fake_sam3,
                                              monkeypatch, auto_yes):
    """Cancelling autolabel-all mid-side must NOT chain the other side."""
    from gui.label_review.ui import main_window as mw
    win, coco, idx, _l, _r = _stereo_setup(lr, make_coco, make_window,
                                           tmp_path, name="alcancel")
    monkeypatch.setattr(mw, "GenericAutolabelBatchWorker", FakeWorker)
    FakeWorker.instances.clear()
    win.autolabel_detector = "falcon"
    win.falcon_model = "tiiuae/Falcon-Perception"

    win._on_autolabel_all()
    FakeWorker.instances[0].cancelled_signal.emit()
    assert len(FakeWorker.instances) == 1
    assert win._autolabel_all_pending == []


def test_autolabel_all_mono_single_batch(lr, make_coco, make_window, tmp_path,
                                        fake_sam3, monkeypatch, auto_yes):
    """In mono, autolabel-all still runs exactly one batch (no stereo
    chaining)."""
    from gui.label_review.ui import main_window as mw
    folder = make_image_folder(tmp_path / "almono", ["a.png", "b.png"])
    monkeypatch.setattr(mw, "GenericAutolabelBatchWorker", FakeWorker)
    FakeWorker.instances.clear()
    win = make_window(lr.ImageFolderIndex([str(folder)]),
                     make_coco([{"id": 0, "name": "a"}]))
    win.autolabel_detector = "falcon"
    win.falcon_model = "tiiuae/Falcon-Perception"

    win._on_autolabel_all()
    assert len(FakeWorker.instances) == 1
    assert win._autolabel_batch_side == "left"
    assert getattr(win, "_autolabel_all_pending", []) == []
    FakeWorker.instances[0].finished_signal.emit(0)
    assert len(FakeWorker.instances) == 1  # no chained second batch


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


# ---------------------------------------------------------------------------
# Synced-only save (stereo)
# ---------------------------------------------------------------------------

def _stereo_ts_index(lr, tmp_path, left_names, right_names, tag):
    left = make_image_folder(tmp_path / f"L_{tag}", left_names)
    right = make_image_folder(tmp_path / f"R_{tag}", right_names)
    return lr.StereoIndex([str(left)], [str(right)])


def test_paired_timestamps_sorted(lr, tmp_path):
    idx = _stereo_ts_index(lr, tmp_path,
                           ["300.png", "100.png", "200.png"],
                           ["400.png", "200.png", "300.png"], "pts")
    assert idx.paired_timestamps == [200, 300]


def test_save_excludes_unsynced_images_and_sorts(lr, make_coco, tmp_path):
    """Only timestamps present in BOTH folders are written to the JSON;
    the images list is sorted by timestamp (left before right)."""
    idx = _stereo_ts_index(lr, tmp_path,
                           ["300.png", "100.png", "200.png"],
                           ["400.png", "200.png", "300.png"], "save")
    coco = make_coco()
    coco.set_synced_timestamps(idx.paired_timestamps)
    # Visit the synced frames out of order.
    id300l = coco.ensure_image(idx.frame_at(1, "left"), 12, 10, side="left")
    id200l = coco.ensure_image(idx.frame_at(0, "left"), 12, 10, side="left")
    id300r = coco.ensure_image(idx.frame_at(1, "right"), 12, 10, side="right")
    # Stale unsynced record (e.g. loaded from an older save file): ts 100
    # exists on the left only.
    unsynced_id = coco.ensure_image(_mkframe(99, ts=100, name="100.png"),
                                    12, 10, side="left")
    coco.add_box(unsynced_id, 1, 1, 2, 2, 0)
    coco.add_box(id200l, 1, 1, 2, 2, 0)

    coco.save(is_final=True)
    data = json.loads(open(coco.output_json).read())
    saved = [(i["timestamp_ns"], i.get("side")) for i in data["images"]]
    assert saved == [(200, "left"), (300, "left"), (300, "right")]
    saved_ids = {i["id"] for i in data["images"]}
    assert saved_ids == {id200l, id300l, id300r}
    assert all(a["image_id"] in saved_ids for a in data["annotations"])

    # The _tmp progress save excludes unsynced records too (they are never
    # wanted, unlike discards which stay reversible).
    coco.dirty = True
    coco.save(is_final=False)
    tmp = json.loads(
        open(coco.output_json.replace(".json", "_tmp.json")).read())
    assert all(i["timestamp_ns"] in (200, 300) for i in tmp["images"])


def test_save_unfiltered_without_synced_timestamps(make_coco):
    """Mono sessions (no set_synced_timestamps call) keep every image."""
    coco = make_coco()
    id_b = coco.ensure_image(_mkframe(0, ts=200, name="b.png"), 12, 10)
    id_a = coco.ensure_image(_mkframe(1, ts=100, name="a.png"), 12, 10)
    coco.add_box(id_b, 1, 1, 2, 2, 0)
    coco.save(is_final=True)
    data = json.loads(open(coco.output_json).read())
    # both kept; sorted by timestamp even without the filter
    assert [i["timestamp_ns"] for i in data["images"]] == [100, 200]
    assert {i["id"] for i in data["images"]} == {id_a, id_b}


def test_frame_at_reports_pair_index(lr, tmp_path):
    """With leading unpaired frames, frame_idx is the PAIR index (not the
    side folder's own position) so discard marks / lookups stay aligned."""
    idx = _stereo_ts_index(lr, tmp_path,
                           ["100.png", "200.png", "300.png"],
                           ["200.png", "300.png"], "pairidx")
    assert len(idx) == 2  # 100 is left-only
    fr = idx.frame_at(0, "left")
    assert fr["timestamp_ns"] == 200
    assert fr["frame_idx"] == 0  # pair idx, not folder position 1
    assert idx.frame_at(0, "right")["frame_idx"] == 0


def test_register_all_frames_saves_all_synced(lr, make_coco, tmp_path):
    """Saving without visiting a single frame still writes every synced
    image (both sides), sorted by timestamp, and no unsynced ones."""
    idx = _stereo_ts_index(lr, tmp_path,
                           ["100.png", "200.png", "300.png"],
                           ["200.png", "300.png", "400.png"], "regall")
    coco = make_coco()
    coco.set_synced_timestamps(idx.paired_timestamps)
    created = coco.register_all_frames(idx)
    assert created == 4  # 2 pairs x 2 sides

    coco.save(is_final=True)
    data = json.loads(open(coco.output_json).read())
    saved = [(i["timestamp_ns"], i.get("side")) for i in data["images"]]
    assert saved == [(200, "left"), (200, "right"),
                     (300, "left"), (300, "right")]

    # Re-registering is idempotent (no duplicate records).
    assert coco.register_all_frames(idx) == 0
    assert len(coco.images) == 4


def test_discard_drops_both_sides_despite_skips(lr, make_coco, tmp_path):
    """Discard marks pair indices; with leading unpaired frames the image
    records' frame_idx must match them (pair index, not folder position)."""
    idx = _stereo_ts_index(lr, tmp_path,
                           ["100.png", "200.png", "300.png"],
                           ["200.png", "300.png"], "disc")
    coco = make_coco()
    coco.set_synced_timestamps(idx.paired_timestamps)
    coco.register_all_frames(idx)
    coco.discarded_frames.add(0)  # pair 0 = ts 200 on both sides

    coco.save(is_final=True)
    data = json.loads(open(coco.output_json).read())
    assert [i["timestamp_ns"] for i in data["images"]] == [300, 300]

    # _tmp keeps the discarded pair (reversible until final save)
    coco.dirty = True
    coco.save(is_final=False)
    tmp = json.loads(
        open(coco.output_json.replace(".json", "_tmp.json")).read())
    assert sorted(i["timestamp_ns"] for i in tmp["images"]) == \
        [200, 200, 300, 300]

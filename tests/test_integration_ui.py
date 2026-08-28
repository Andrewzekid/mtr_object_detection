"""Window-level integration tests for gui/label_review (was scripts/09_label_review.py),
ported from the standalone /tmp test scripts. Everything runs offscreen
with fake SAM3 workers — no real inference, no network."""

import json
import os
import sys
import time

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QEvent, QPointF
from PyQt6.QtCore import Qt

from conftest import (FakeIdx, FakeSig, FakeWorker, FakeAutolabelSingle,
                      FakePropagateWorker, make_image_folder)

NM = Qt.KeyboardModifier.NoModifier
CATS = [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mouse(win_canvas, etype, pt, button=Qt.MouseButton.LeftButton,
           mods=NM):
    buttons = button if etype != QEvent.Type.MouseButtonRelease else button
    ev = QtGui.QMouseEvent(etype, pt, button, buttons, mods)
    if etype == QEvent.Type.MouseButtonPress:
        win_canvas.mousePressEvent(ev)
    elif etype == QEvent.Type.MouseMove:
        win_canvas.mouseMoveEvent(ev)
    else:
        win_canvas.mouseReleaseEvent(ev)


def draw_box(win, x1=1.0, y1=1.0, x2=4.0, y2=4.0):
    """Drive the canvas draw path with synthetic mouse events."""
    c = win.canvas
    c._edit_mode = "draw"
    c._drawing = True
    c._draw_start = (x1, y1)
    c._draw_current = (x2, y2)
    c.mouseReleaseEvent(QtGui.QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(50, 50),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, NM))


def click_box(win, cx, cy, shift=False):
    c = win.canvas
    pt = c._img_to_widget(cx, cy)
    mods = Qt.KeyboardModifier.ShiftModifier if shift else NM
    _mouse(c, QEvent.Type.MouseButtonPress, pt, mods=mods)
    rel = QtGui.QMouseEvent(QEvent.Type.MouseButtonRelease, pt,
                            Qt.MouseButton.LeftButton,
                            Qt.MouseButton.NoButton, mods)
    c.mouseReleaseEvent(rel)


def key(win_canvas, k, text=""):
    ev = QtGui.QKeyEvent(QEvent.Type.KeyPress, k, NM, text)
    win_canvas.keyPressEvent(ev)


# ---------------------------------------------------------------------------
# drawing / categories / clamping
# ---------------------------------------------------------------------------

def test_draw_assigns_category_and_sticks(lr, make_coco, make_window):
    coco = make_coco(CATS)
    win = make_window(FakeIdx(2), coco)

    # First draw: no previous category → must wait for a pick.
    draw_box(win)
    assert win.canvas._waiting_cat
    key(win.canvas, Qt.Key.Key_1, "1")
    assert len(coco.annotations) == 1
    assert coco.annotations[0]["category_id"] == 1
    assert win._last_cat_id == 1  # sticky

    # Second draw: auto-assign previous category, no waiting state.
    draw_box(win)
    assert not win.canvas._waiting_cat
    assert len(coco.annotations) == 2
    assert coco.annotations[1]["category_id"] == 1

    # Preselect another category → overrides sticky for one draw only.
    win._on_preselect_cat(0)
    draw_box(win)
    assert len(coco.annotations) == 3
    assert coco.annotations[2]["category_id"] == 0
    assert win._last_cat_id == 0  # preselected draw updates sticky cat
    assert win._pending_cat_id is None  # pending preselect consumed


def test_draw_multi_digit_category_buffer(lr, make_coco, make_window):
    """10+ categories: typed digits accumulate and commit as soon as no
    longer id can start with them (with 0..12, "1" waits, "12" commits
    immediately); Enter commits an ambiguous prefix early; Backspace
    erases; invalid ids keep the pick pending."""
    cats = [{"id": i, "name": f"c{i}"} for i in range(13)]
    coco = make_coco(cats)
    win = make_window(FakeIdx(2), coco)
    c = win.canvas

    draw_box(win)
    assert c._waiting_cat
    key(c, Qt.Key.Key_1, "1")
    assert c._cat_buffer == "1"  # waits: 10..12 could follow
    assert len(coco.annotations) == 0
    key(c, Qt.Key.Key_2, "2")
    # "12" auto-commits: no valid id extends it (120 > 12)
    assert not c._waiting_cat
    assert len(coco.annotations) == 1
    assert coco.annotations[0]["category_id"] == 12

    # Ambiguous prefix: with 0..102, typing "10" waits → Enter commits 10
    win2 = make_window(FakeIdx(2), make_coco(
        [{"id": i, "name": f"d{i}"} for i in range(103)]))
    c2 = win2.canvas
    draw_box(win2)
    key(c2, Qt.Key.Key_1, "1")
    key(c2, Qt.Key.Key_0, "0")
    assert c2._cat_buffer == "10"  # 100..102 still possible → wait
    assert len(win2.coco.annotations) == 0
    key(c2, Qt.Key.Key_Return)
    assert not c2._waiting_cat
    assert win2.coco.annotations[0]["category_id"] == 10

    # Backspace: fresh window (no sticky cat), type 13 → 1 → Enter commits 1
    win3 = make_window(FakeIdx(2), make_coco(cats))
    c3 = win3.canvas
    draw_box(win3)
    assert c3._waiting_cat
    key(c3, Qt.Key.Key_1, "1")
    key(c3, Qt.Key.Key_3, "3")  # 13 > 12 → invalid: stays buffered
    assert c3._cat_buffer == "13"
    key(c3, Qt.Key.Key_Backspace)
    assert c3._cat_buffer == "1"  # 1 is ambiguous again (10..12) → waits
    key(c3, Qt.Key.Key_Return)
    assert len(win3.coco.annotations) == 1
    assert win3.coco.annotations[0]["category_id"] == 1

    # Invalid id + Enter: warns and stays in pick mode; Esc cancels cleanly
    draw_box(win)
    key(c, Qt.Key.Key_9, "9")
    key(c, Qt.Key.Key_Return)
    c._stop_cat_wait()
    assert len(coco.annotations) == 2


def test_box_clamped_to_image_bounds(lr, make_coco, make_window):
    class FakeIdx8(FakeIdx):
        def decode_image(self, i):
            return np.zeros((8, 8, 3), np.uint8)

    win = make_window(FakeIdx8(2), make_coco(CATS))
    c = win.canvas

    # clamp helper: image is 8x8
    assert c._clamp_img_pt(-5, 100) == (0.0, 8.0)
    assert c._clamp_img_pt(3, 4) == (3.0, 4.0)

    # draw: press outside top-left corner clamps the start point
    c._drawing = True
    pt = c._img_to_widget(-5.0, -5.0)
    _mouse(c, QEvent.Type.MouseButtonPress, pt)
    assert c._draw_start == (0.0, 0.0)
    # move beyond bottom-right clamps the current point
    pt = c._img_to_widget(20.0, 20.0)
    _mouse(c, QEvent.Type.MouseMove, pt, button=Qt.MouseButton.NoButton)
    assert c._draw_current == (8.0, 8.0)
    _mouse(c, QEvent.Type.MouseButtonRelease, pt)
    assert c._waiting_cat
    assert c.get_pending_rect() == (0.0, 0.0, 8.0, 8.0)
    c.reset_state()

    # move: dragging a box outside top-left keeps it at 0,0
    c.set_boxes([{"id": 1, "bbox": [1, 1, 4, 4], "cat_id": 0,
                  "cat_name": "a"}])
    c._selected_idx = 0
    c._edit_mode = "move"
    c._edit_start_box = (1, 1, 4, 4)
    c._edit_start_cursor = (4.0, 4.0)
    _mouse(c, QEvent.Type.MouseMove, c._img_to_widget(-10.0, -10.0),
           button=Qt.MouseButton.NoButton)
    assert c._boxes[0]["bbox"] == [0.0, 0.0, 4, 4]
    # move beyond bottom-right: 4x4 box in 8x8 image stops at (4,4)
    _mouse(c, QEvent.Type.MouseMove, c._img_to_widget(100.0, 100.0),
           button=Qt.MouseButton.NoButton)
    assert c._boxes[0]["bbox"] == [4.0, 4.0, 4, 4]

    # resize: br corner dragged past the image edge stops at the edge
    c._edit_mode = "resize_br"
    c._edit_start_box = (0, 0, 4, 4)
    c._edit_start_cursor = (4.0, 4.0)
    _mouse(c, QEvent.Type.MouseMove, c._img_to_widget(100.0, 100.0),
           button=Qt.MouseButton.NoButton)
    assert c._boxes[0]["bbox"] == [0.0, 0.0, 8.0, 8.0]


def test_backspace_deletes_selected_box(lr, make_coco, make_window):
    coco = make_coco(CATS)
    win = make_window(FakeIdx(20), coco)
    ann_id = coco.add_box(win._current_image_id, 0, 0, 2, 2, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    key(win.canvas, Qt.Key.Key_Backspace)
    assert ann_id in coco.removed_ids


def test_jump_buttons(lr, make_coco, make_window):
    win = make_window(FakeIdx(20), make_coco(CATS))
    texts = [b.text() for b in win.side.jump_buttons]
    assert texts == ["-10", "-5", "+5", "+10"]
    win._current_idx = 10
    win.side.jump_buttons[1].click()   # -5
    assert win._current_idx == 5
    win.side.jump_buttons[0].click()   # -10 → out of range, stays
    assert win._current_idx == 5
    win.side.jump_buttons[2].click()   # +5
    assert win._current_idx == 10
    win.side.jump_buttons[3].click()   # +10 → 20 out of range, stays
    assert win._current_idx == 10


def test_shift_click_multiselect_and_delete(lr, make_coco, make_window):
    class FakeIdx200(FakeIdx):
        def decode_image(self, i):
            return np.zeros((200, 200, 3), np.uint8)

    coco = make_coco(CATS)
    win = make_window(FakeIdx200(2), coco)
    img_id = win._current_image_id
    a1 = coco.add_box(img_id, 10, 10, 40, 40, 0)
    a2 = coco.add_box(img_id, 100, 100, 40, 40, 1)
    a3 = coco.add_box(img_id, 150, 10, 30, 30, 0)
    win._refresh_boxes()
    c = win.canvas

    # plain click selects only that box
    click_box(win, 30, 30)
    assert c._selected_idx == 0 and c._multi_selected == {0}
    # shift+click adds a second box; clicked becomes primary
    click_box(win, 120, 120, shift=True)
    assert c._multi_selected == {0, 1}
    assert c._selected_idx == 1
    # shift+click a third
    click_box(win, 165, 25, shift=True)
    assert c._multi_selected == {0, 1, 2}
    assert c._selected_idx == 2
    # shift+click again toggles it off; primary falls back to a remaining
    click_box(win, 165, 25, shift=True)
    assert c._multi_selected == {0, 1}
    assert c._selected_idx in (0, 1)

    # delete removes all selected boxes
    key(c, Qt.Key.Key_Delete)
    assert a1 in coco.removed_ids and a2 in coco.removed_ids
    assert a3 not in coco.removed_ids

    # plain click resets the multi-selection
    click_box(win, 165, 25)
    assert c._multi_selected == {0}
    live = [a for a in coco.annotations if a["id"] not in coco.removed_ids]
    assert len(live) == 1


# ---------------------------------------------------------------------------
# runtime source switching / image folder window
# ---------------------------------------------------------------------------

def test_switch_to_images_at_runtime(lr, make_coco, make_window, tmp_path):
    folder = make_image_folder(tmp_path / "switch",
                               [("im1.jpg", 30), ("im2.jpg", 90)],
                               size=(6, 7))
    coco = make_coco(CATS)
    win = make_window(FakeIdx(2), coco)
    assert win._current_image_id is not None

    # menu has both open actions
    menus = [a.text() for a in win.menuBar().actions()]
    assert "&File" in menus
    acts = [a.text() for a in win.menuBar().actions()[0].menu().actions()]
    assert any("image file" in a for a in acts)
    assert any("folder" in a for a in acts)

    win._switch_to_images([str(folder)])
    assert isinstance(win.frame_index, lr.ImageFolderIndex)
    assert len(win.frame_index) == 2
    assert win.coco.output_json == os.path.join(str(folder),
                                                "labels_coco.json")
    assert win.side.coco is win.coco  # side panel follows the new session
    assert win._current_image_id is not None
    assert win.coco.images[0]["file_name"] == "im1.jpg"
    assert win.windowTitle().startswith("Computer Vision Label Review Tool")

    # drawing still works after the switch
    draw_box(win)
    assert win.canvas._waiting_cat
    win._assign_pending_cat(1)
    assert len(win.coco.annotations) == 1
    assert win.coco.annotations[0]["image_id"] == win._current_image_id


def test_image_folder_window(lr, make_coco, make_window, tmp_path):
    folder = make_image_folder(tmp_path / "folder",
                               [("b.jpg", 60), ("a.png", 120),
                                ("c.jpg", 200)], size=(10, 12))
    idx = lr.ImageFolderIndex([str(folder)])
    coco = make_coco(CATS)
    win = make_window(idx, coco)
    assert not hasattr(win, "web_view")
    assert win._current_image_id is not None
    img_rec = coco.images[0]
    assert img_rec["file_name"] == "a.png"
    assert (img_rec["width"], img_rec["height"]) == (12, 10)

    draw_box(win, 1.0, 1.0, 5.0, 6.0)
    assert win.canvas._waiting_cat
    win._assign_pending_cat(0)
    assert len(coco.annotations) == 1
    assert coco.annotations[0]["bbox"] == [1.0, 1.0, 4.0, 5.0]


def test_write_tmp_image_in_image_mode(lr, make_coco, make_window, tmp_path):
    folder = make_image_folder(tmp_path / "sam3img", ["frame.jpg"],
                               size=(8, 8))
    idx = lr.ImageFolderIndex([str(folder / "frame.jpg")])
    win = make_window(idx, make_coco([{"id": 0, "name": "a"}]))
    assert win._current_image_id is not None
    # image-folder frames have no blob → _write_tmp_image must still
    # produce a path SAM3 can read (the file itself or a decoded PNG copy)
    out = win._write_tmp_image()
    assert out is not None and os.path.exists(out)


def test_open_existing_project_restores_categories(lr, make_coco,
                                                   make_window, tmp_path):
    folder = make_image_folder(tmp_path / "proj",
                               ["1000000000.jpg", "2000000000.jpg"],
                               size=(16, 16))
    json.dump({
        "images": [{"id": 1, "timestamp_ns": 1000000000, "frame_idx": 0,
                    "file_name": "1000000000.jpg", "width": 16,
                    "height": 16}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "monitor"},
                       {"id": 1, "name": "ticket gate"}],
    }, open(folder / "labels_coco.json", "w"))

    # idle window with NO categories (fresh start)
    win = make_window(lr.EmptyIndex(), make_coco([]))
    assert win.side.cat_list.count() == 0

    win._switch_to_images([str(folder)])
    names = [win.side.cat_list.item(i).text()
             for i in range(win.side.cat_list.count())]
    assert len(names) == 2
    assert any("monitor" in n for n in names)
    assert any("ticket gate" in n for n in names)
    assert {c["name"] for c in win.coco.categories} == {"monitor",
                                                        "ticket gate"}
    assert len(win.coco.annotations) == 1


# ---------------------------------------------------------------------------
# category rename / delete
# ---------------------------------------------------------------------------

def test_rename_delete_category(lr, make_coco, make_window, monkeypatch):
    coco = make_coco(CATS)
    win = make_window(FakeIdx(2), coco)
    img_id = win._current_image_id
    coco.add_box(img_id, 0, 0, 2, 2, 0)
    coco.add_box(img_id, 1, 1, 3, 3, 1)
    win._refresh_boxes()

    # sidebar selection helper
    win.side.cat_list.setCurrentRow(1)  # cat id 1 ("b")
    assert win.side._selected_cat_id() == 1

    # --- rename ---
    monkeypatch.setattr(lr.QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("b-renamed", True)))
    win._on_rename_category(1)
    assert coco.cat_map[1] == "b-renamed"
    assert "b" not in coco.cat_name_to_id
    assert coco.cat_name_to_id["b-renamed"] == 1
    assert coco.dirty
    labels = [win.side.cat_list.item(i).text()
              for i in range(win.side.cat_list.count())]
    assert "1 — b-renamed" in labels
    assert any(b["cat_name"] == "b-renamed" for b in win.canvas._boxes)

    # rename to an existing name is rejected
    monkeypatch.setattr(lr.QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("a", True)))
    win._on_rename_category(1)
    assert coco.cat_map[1] == "b-renamed"

    # --- delete: cancel keeps everything ---
    monkeypatch.setattr(
        lr.QMessageBox, "question",
        staticmethod(lambda *a, **k: lr.QMessageBox.StandardButton.Cancel))
    win._on_delete_category(1)
    assert 1 in coco.cat_map
    assert len(coco.removed_ids) == 0

    # --- delete: confirmed removes category + its boxes ---
    monkeypatch.setattr(
        lr.QMessageBox, "question",
        staticmethod(lambda *a, **k: lr.QMessageBox.StandardButton.Yes))
    # reselect the "1 — b-renamed" row (rebuild cleared the selection)
    for i in range(win.side.cat_list.count()):
        if win.side.cat_list.item(i).data(lr.Qt.UserRole) == 1:
            win.side.cat_list.setCurrentRow(i)
    win._on_delete_category(1)
    assert 1 not in coco.cat_map
    assert "b-renamed" not in coco.cat_name_to_id
    assert all(c["id"] != 1 for c in coco.categories)
    kept = [a for a in coco.annotations if a["id"] not in coco.removed_ids]
    assert len(kept) == 1 and kept[0]["category_id"] == 0
    assert len(win.canvas._boxes) == 1
    labels = [win.side.cat_list.item(i).text()
              for i in range(win.side.cat_list.count())]
    assert labels == ["0 — a"]

    # --- pending preselection cleared when its category is deleted ---
    win._on_add_category("temp")  # preselected for the next draw
    temp_id = win._pending_cat_id
    assert temp_id is not None
    win._on_delete_category(temp_id)
    assert win._pending_cat_id is None
    assert win.side._preselected_cat_id is None


# ---------------------------------------------------------------------------
# track ids: edit flow, sticky-mode config, display option
# ---------------------------------------------------------------------------

def test_track_edit_flow(lr, make_coco, make_window, tmp_path):
    coco = make_coco(CATS)
    win = make_window(lr.EmptyIndex(), coco, show_track_ids=True)
    # idle empty index must not crash on load, then switch to images
    folder = make_image_folder(tmp_path / "trackedit",
                               [("b.jpg", 60), ("a.png", 120),
                                ("c.jpg", 200)], size=(10, 12))
    win._switch_to_images([str(folder)])
    assert len(win.frame_index) == 3
    win.coco.add_box(win._current_image_id, 1, 1, 3, 3, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    win._focus_track_edit()  # focuses the field
    assert win.side.track_edit.text() != ""  # prefilled
    win.side.track_edit.setText("42")
    win.side._on_track_entered()
    assert win.coco.annotations[-1]["track_id"] == 42
    # clear via empty field
    win.side.track_edit.setText("")
    win.side._on_track_entered()
    assert "track_id" not in win.coco.annotations[-1]


def test_sticky_ids_runtime_config(lr, make_coco, make_window):
    coco = make_coco(CATS)
    # Sticky ids default ON (what interpolation pairing expects).
    assert coco.sticky_track_ids is True
    win = make_window(lr.EmptyIndex(), coco)
    assert win.coco is coco
    win._apply_runtime_config({"tracking": {"sticky_ids": False}})
    assert coco.sticky_track_ids is False
    win._apply_runtime_config({"tracking": {"sticky_ids": True}})
    assert coco.sticky_track_ids is True

    dlg = lr.ConfigDialog(win)
    assert dlg.check_sticky_ids.isChecked() is True  # prefill from window
    cfg = dlg._collect()
    assert cfg["tracking"]["sticky_ids"] is True
    dlg._prefill_from_config({"tracking": {"sticky_ids": False}})
    assert dlg.check_sticky_ids.isChecked() is False
    assert dlg._collect()["tracking"]["sticky_ids"] is False
    dlg.close()


def test_track_id_display_option(lr, make_coco, make_window, qapp):
    """Track-id display: shown by default (current behavior), hidden via
    tracking.show_ids=False."""
    coco = make_coco([{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(2), coco)
    win.coco.add_box(win._current_image_id, 1, 1, 3, 3, 0)
    win._refresh_boxes()

    # default: track ids shown everywhere
    assert win.show_track_ids is True
    assert win.canvas.show_track_ids is True
    assert win.side.show_track_ids is True
    assert not win.side.track_edit.isHidden()
    tid = win.coco.annotations[-1].get("track_id")
    assert f" T{tid}" in win.side.box_list.item(0).text()
    # T key focuses the field
    win.canvas._selected_idx = 0
    win._focus_track_edit()
    qapp.processEvents()
    assert win.side.track_edit.hasFocus()

    # hide via runtime config
    win._apply_runtime_config({"tracking": {"show_ids": False}})
    assert win.show_track_ids is False
    assert win.canvas.show_track_ids is False
    assert win.side.track_edit.isHidden()
    assert " T" not in win.side.box_list.item(0).text()
    # T key is a no-op while hidden
    win._focus_track_edit()
    qapp.processEvents()
    assert not win.side.track_edit.hasFocus()

    # enable again
    win._apply_runtime_config({"tracking": {"show_ids": True}})
    assert not win.side.track_edit.isHidden()
    assert f" T{tid}" in win.side.box_list.item(0).text()

    # constructor kwarg: explicit False hides
    win2 = make_window(FakeIdx(2), make_coco([{"id": 0, "name": "a"}]),
                       show_track_ids=False)
    assert win2.canvas.show_track_ids is False
    assert win2.side.track_edit.isHidden()

    # config dialog: checkbox lives in the advanced Interpolation section
    dlg = lr.ConfigDialog(win)
    assert dlg.check_show_track_ids.isChecked()  # prefilled from window
    dlg.check_advanced.setChecked(True)
    assert not dlg.interp_box.isHidden()  # section only in advanced mode
    dlg.check_show_track_ids.setChecked(False)
    assert dlg._collect()["tracking"]["show_ids"] is False
    dlg2 = lr.ConfigDialog(win2)
    assert not dlg2.check_show_track_ids.isChecked()
    dlg.close()
    dlg2.close()


# ---------------------------------------------------------------------------
# config: constructor kwargs, runtime apply, dialog roundtrip, defaults
# ---------------------------------------------------------------------------

def test_config_constructor_settings(lr, make_coco, make_window):
    win = make_window(
        FakeIdx(2), make_coco(CATS),
        interp_flow_method="klt", interp_camera_model="global",
        interp_match_frac=0.35, interp_confirm_mismatch=False,
        ui_hide=["sam3_all_frames", "play", "interpolate"],
        mask_opacity=80, advanced_ui=True)

    assert win.interp_flow_method == "klt"
    assert win.interp_camera_model == "global"
    assert win.interp_match_frac == 0.35
    assert win.interp_confirm_mismatch is False

    s = win.side
    assert s.btn_sam3_all_frames.isHidden()
    assert s.btn_play.isHidden() and s.combo_speed.isHidden()
    assert s.btn_interpolate.isHidden() and s.btn_cancel_interp.isHidden()
    assert s.interp_status.isHidden()
    # untouched groups stay visible
    assert not s.btn_run_sam3.isHidden()
    assert not s.btn_keyframe.isHidden()
    assert not s.btn_masks.isHidden()
    assert not s.jump_buttons[0].isHidden()

    # mask opacity applied to slider + canvas
    assert s.opacity_slider.value() == 80
    assert s.opacity_value_label.text() == "80%"
    assert win.canvas.mask_alpha() == round(80 * 255 / 100)

    # unknown group warns but doesn't crash
    s.set_hidden_groups(["nonexistent_group"])


def test_runtime_config_apply(lr, make_coco, make_window):
    win = make_window(FakeIdx(2), make_coco(CATS))
    win._apply_runtime_config(
        {"ui": {"hide": ["sam3_run"], "mask_opacity": 10},
         "interpolation": {"flow_method": "klt", "confirm_mismatch": False},
         "sam3": {"device": "cpu", "conf": 0.5}})
    assert win.interp_flow_method == "klt" and win.interp_confirm_mismatch is False
    assert win.sam3_device == "cpu" and win.sam3_conf == 0.5
    assert win.side.btn_run_sam3.isHidden()
    assert win.canvas.mask_alpha() == round(10 * 255 / 100)
    # invalid values ignored
    win._apply_runtime_config({"interpolation": {"flow_method": "bogus"},
                               "sam3": {"device": "tpu"}})
    assert win.interp_flow_method == "klt" and win.sam3_device == "cpu"
    # un-hide again (symmetric hide)
    win._apply_runtime_config({"ui": {"hide": []}})
    assert not win.side.btn_run_sam3.isHidden()

    # menu has the config entry (and no rrd entry — rrd support removed)
    acts = [a.text() for a in win.menuBar().actions()[0].menu().actions()]
    low = [a.lower() for a in acts]
    assert not any("rrd" in a for a in low)
    assert any("config" in a for a in low)


def test_config_dialog_roundtrip(lr, make_coco, make_window, monkeypatch,
                                 tmp_path):
    win = make_window(FakeIdx(2), make_coco(CATS), sam3_device="cpu")

    # --- dialog prefill matches live window state ---
    dlg = lr.ConfigDialog(win)
    assert dlg.combo_flow.currentText() == win.interp_flow_method
    assert dlg.combo_cam.currentText() == win.interp_camera_model
    assert abs(dlg.spin_match_frac.value() - win.interp_match_frac) < 1e-9
    assert dlg.check_confirm_mismatch.isChecked() == win.interp_confirm_mismatch
    assert dlg.combo_device.currentText() == "cpu"
    assert dlg.check_auto_segment.isChecked() == win.auto_segment
    assert dlg.spin_opacity.value() == win.side.opacity_slider.value()
    # basic mode: hide checkboxes unchecked; interpolate/keyframe checkboxes
    # exist but are advanced-gated (the groups themselves stay visible).
    assert all(not cb.isChecked() for cb in dlg.hide_checks.values())
    # advanced sections hidden until the checkbox is ticked
    assert not dlg.check_advanced.isChecked()
    dlg.show()
    assert dlg.interp_box.isHidden() and not dlg.track_box.isHidden()
    assert dlg.hide_checks["interpolate"].isHidden()
    assert dlg.hide_checks["keyframe"].isHidden()
    dlg.check_advanced.setChecked(True)
    assert not dlg.interp_box.isHidden()
    assert not dlg.hide_checks["interpolate"].isHidden()
    dlg.check_advanced.setChecked(False)  # back to basic for the rest
    assert dlg.interp_box.isHidden()

    # --- Apply: hide groups, change device + opacity ---
    dlg.hide_checks["sam3_run"].setChecked(True)
    dlg.hide_checks["play"].setChecked(True)
    dlg.combo_device.setCurrentText("cuda")
    dlg.spin_sam3_conf.setValue(0.5)
    dlg.check_auto_segment.setChecked(True)
    dlg.spin_opacity.setValue(33)
    dlg.combo_flow.setCurrentText("klt")
    dlg._on_apply()

    s = win.side
    assert s.btn_run_sam3.isHidden() and s.btn_reseg.isHidden()
    assert s.btn_cancel_sam3.isHidden()
    assert s.btn_play.isHidden() and s.combo_speed.isHidden()
    # keyframe/interpolate are no longer advanced-gated — they stay visible
    assert not s.btn_keyframe.isHidden()
    assert not s.btn_interpolate.isHidden()
    assert win.sam3_device == "cuda"
    assert abs(win.sam3_conf - 0.5) < 1e-9
    assert win.auto_segment is True
    assert win.interp_flow_method == "klt"
    assert s.opacity_slider.value() == 33
    assert win.canvas.mask_alpha() == round(33 * 255 / 100)

    # --- Save: applies + writes JSON ---
    save_path = str(tmp_path / "saved_cfg.json")
    monkeypatch.setattr(lr.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (save_path, "")))
    dlg.spin_opacity.setValue(60)
    dlg._on_save()
    saved = json.load(open(save_path))
    assert saved["ui"]["mask_opacity"] == 60
    assert saved["ui"]["advanced"] is False
    # hide list only contains the explicitly checked groups
    assert set(saved["ui"]["hide"]) == {"sam3_run", "play"}
    assert saved["sam3"]["device"] == "cuda"
    assert saved["interpolation"]["flow_method"] == "klt"
    assert s.opacity_slider.value() == 60  # save also applies

    # --- Load from file: fills widgets and applies ---
    load_path = str(tmp_path / "load_cfg.json")
    json.dump({
        "interpolation": {"flow_method": "farneback", "camera_model": "global",
                          "match_max_dist_frac": 0.4, "confirm_mismatch": False},
        "sam3": {"device": "cpu", "conf": 0.1, "auto_segment": False},
        "ui": {"hide": ["keyframe", "sam3_run"], "mask_opacity": 10},
    }, open(load_path, "w"))
    monkeypatch.setattr(lr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (load_path, "")))
    dlg._on_load()

    assert dlg.combo_flow.currentText() == "farneback"
    assert dlg.combo_cam.currentText() == "global"
    assert abs(dlg.spin_match_frac.value() - 0.4) < 1e-9
    assert dlg.check_confirm_mismatch.isChecked() is False
    assert dlg.combo_device.currentText() == "cpu"
    assert dlg.hide_checks["keyframe"].isChecked()
    assert dlg.hide_checks["sam3_run"].isChecked()
    assert not dlg.hide_checks["masks"].isChecked()
    assert dlg.spin_opacity.value() == 10
    # applied to the window: previous hides un-hidden, new ones hidden
    assert win.interp_flow_method == "farneback"
    assert win.interp_camera_model == "global"
    assert win.interp_confirm_mismatch is False
    assert win.sam3_device == "cpu"
    assert s.btn_keyframe.isHidden()
    assert s.btn_run_sam3.isHidden()   # still hidden under the loaded config
    assert not s.btn_play.isHidden()   # un-hidden by the new config
    assert s.opacity_slider.value() == 10

    # --- Hide list explicitly controls keyframe/interpolate visibility ---
    win._apply_runtime_config({"ui": {"hide": []}})
    assert not s.btn_interpolate.isHidden() and not s.btn_keyframe.isHidden()
    win._apply_runtime_config({"ui": {"hide": ["interpolate", "keyframe"]}})
    assert s.btn_interpolate.isHidden() and s.btn_keyframe.isHidden()
    dlg.close()


def test_min_polygon_area_config(lr, make_coco, make_window):
    coco = make_coco([{"id": 0, "name": "a"}])
    win = make_window(lr.EmptyIndex(), coco)
    win._apply_runtime_config({"sam3": {"min_polygon_area": 250}})
    assert coco.min_polygon_area == 250.0
    dlg = lr.ConfigDialog(win)
    assert dlg.spin_min_poly_area.value() == 250
    assert dlg._collect()["sam3"]["min_polygon_area"] == 250
    dlg._prefill_from_config({"sam3": {"min_polygon_area": 42}})
    assert dlg.spin_min_poly_area.value() == 42
    dlg.close()


def test_config_dialog_buttons_not_enter_default(lr, make_coco, make_window):
    """Enter in a field must not trigger the Load/Apply/Save buttons."""
    win = make_window(lr.EmptyIndex(), make_coco(CATS))
    dlg = lr.ConfigDialog(win)
    for b in (dlg.btn_load, dlg.btn_apply, dlg.btn_save, dlg.btn_close):
        assert not b.autoDefault() and not b.isDefault()
    dlg.close()


def test_main_hides_advanced_groups_by_default(lr, qapp, monkeypatch,
                                               tmp_path):
    """main() with no config: autolabel hidden; interpolate + keyframe
    visible by default now (they were advanced-gated before)."""
    # main()'s idle mode uses ./untitled_labels_coco.json relative to the
    # cwd and saves progress on exit — run it from tmp_path so the repo
    # root stays untouched.
    monkeypatch.chdir(tmp_path)
    wins = []
    orig = lr.ReviewWindow

    class CaptureWin(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            wins.append(self)

    monkeypatch.setattr(lr, "ReviewWindow", CaptureWin)
    monkeypatch.setattr(sys, "argv", ["prog", "--sam3-device", "cpu"])
    QtCore.QTimer.singleShot(600, qapp.quit)
    try:
        lr.main()
    except SystemExit:
        pass

    assert wins, "main() did not construct a window"
    win = wins[0]
    win._quit_confirmed = True
    s = win.side
    s.show()
    # keyframe/interpolate are visible by default (not advanced-gated)
    assert not s.btn_interpolate.isHidden()
    assert not s.interp_status.isHidden()
    assert not s.btn_keyframe.isHidden()
    assert s.btn_autolabel_frame.isHidden(), "autolabel hidden by default"
    assert s.btn_autolabel_all.isHidden()
    assert s.autolabel_header.isHidden()
    # other groups still visible
    assert not s.btn_run_sam3.isHidden()
    assert not s.btn_masks.isHidden()
    assert not s.btn_play.isHidden()
    assert not hasattr(win, "btn_toggle_rerun")

    # config dialog prefill: only the explicitly hidden autolabel group is
    # checked — keyframe/interpolate are visible by default now
    dlg = lr.ConfigDialog(win)
    assert not dlg.hide_checks["interpolate"].isChecked()
    assert not dlg.hide_checks["keyframe"].isChecked()
    assert dlg.hide_checks["autolabel"].isChecked()
    assert not dlg.hide_checks["sam3_run"].isChecked()
    dlg.close()
    win.close()


# ---------------------------------------------------------------------------
# save / load / navigation persistence
# ---------------------------------------------------------------------------

def test_save_and_save_as(lr, make_coco, make_window, monkeypatch, tmp_path):
    out_a = str(tmp_path / "save_a.json")
    coco = lr.CocoState(out_a, [{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(2), coco)
    coco.add_box(win._current_image_id, 0, 0, 2, 2, 0)

    # menu has Save / Save as…
    acts = [a.text() for a in win.menuBar().actions()[0].menu().actions()]
    assert "Save" in acts and "Save as…" in acts

    # Save → writes final JSON to the current output path
    win._on_save()
    assert len(json.load(open(out_a))["annotations"]) == 1

    # Save as… → chosen path becomes the session output and gets the file
    target = str(tmp_path / "save_b.json")
    monkeypatch.setattr(lr.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (target, "")))
    win._on_save_as()
    assert coco.output_json == target
    assert len(json.load(open(target))["annotations"]) == 1
    assert coco.progress_file == target.replace(".json", ".progress")
    assert os.path.exists(coco.progress_file)

    # Save as… with no .json suffix gets one appended
    nosuffix = str(tmp_path / "save_c")
    monkeypatch.setattr(lr.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (nosuffix, "")))
    win._on_save_as()
    assert coco.output_json == nosuffix + ".json"
    assert os.path.exists(nosuffix + ".json")

    # cancel keeps the current path
    monkeypatch.setattr(lr.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    win._on_save_as()
    assert coco.output_json == nosuffix + ".json"


def test_slider_saves_on_release_only(lr, make_window, tmp_path):
    out = tmp_path / "slider.json"
    tmp_json = tmp_path / "slider_tmp.json"
    coco = lr.CocoState(str(out), [{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(5), coco)

    # Scrub across several frames: frames load, nothing written to disk.
    for i in (1, 2, 3, 4):
        win._on_slider_moved(i)
        assert win._current_idx == i
        assert win._current_image_id is not None
    assert not tmp_json.exists(), "per-tick save is back — scrub will lag"

    # Release once: progress + tmp json are written.
    win._on_slider_released()
    assert tmp_json.exists()
    assert (tmp_path / "slider.progress").exists()

    # Idle / out-of-range release doesn't write or crash.
    win._current_idx = -1
    tmp_json.unlink()
    win._on_slider_released()
    assert not tmp_json.exists()


def test_playback_speed_map_and_save(lr, make_window, tmp_path):
    out = tmp_path / "playback.json"
    tmp_json = tmp_path / "playback_tmp.json"
    coco = lr.CocoState(str(out), [{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(6), coco)

    # speed map: 1x = 30 fps (33 ms), index-based
    data = {win.side.combo_speed.itemText(i): win.side.combo_speed.itemData(i)
            for i in range(win.side.combo_speed.count())}
    assert data == {"0.25x": 133, "0.5x": 67, "1x": 33, "2x": 17, "5x": 7,
                    "10x": 3}
    assert win._play_interval_ms == 33  # default matches 1x
    # changing speed mid-play retargets the timer
    win._start_playback()
    win._on_play_speed_changed(7)
    assert win._play_interval_ms == 7
    assert win._play_timer.interval() == 7 and win._play_timer.isActive()
    win._stop_playback()
    if tmp_json.exists():
        tmp_json.unlink()

    # play ticks advance without per-tick saves
    win._start_playback()
    for _ in range(3):
        win._on_play_tick()
    assert win._current_idx == 3
    assert not tmp_json.exists(), "per-tick save during playback is back"
    # stopping saves progress once
    win._stop_playback()
    assert tmp_json.exists()
    # forward nav during play marked those frames reviewed
    assert {0, 1, 2} <= coco.reviewed

    # playing to the end stops and saves
    tmp_json.unlink()
    win._start_playback()
    win._on_play_tick()  # idx 4
    win._on_play_tick()  # idx 5 (last)
    win._on_play_tick()  # at last → stop
    assert not win._playing and not win._play_timer.isActive()
    assert tmp_json.exists()


def test_load_annotations_dialog(lr, make_coco, make_window, monkeypatch,
                                 tmp_path):
    folder = make_image_folder(tmp_path / "import_src", ["x.png", "y.png"],
                               size=(6, 6))
    win = make_window(lr.EmptyIndex(), make_coco([]))
    win._switch_to_images([str(folder)])
    src = tmp_path / "import_in.json"
    json.dump({"images": [{"id": 1, "file_name": "x.png", "width": 6,
                           "height": 6}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                                "bbox": [0, 0, 3, 3]}],
               "categories": [{"id": 0, "name": "thing"}]},
              open(src, "w"))
    monkeypatch.setattr(lr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(src), "")))
    win._load_annotations_dialog()
    assert len(win.coco.annotations) == 1
    assert win.coco.cat_name_to_id.get("thing") is not None
    # box visible on the canvas after navigating to frame 0
    win._on_slider_moved(0)
    assert len(win.canvas._boxes) == 1


# ---------------------------------------------------------------------------
# SAM3 queue / status / cancel
# ---------------------------------------------------------------------------

def test_sam3_fifo_queue_status_and_cancel(lr, make_coco, make_window,
                                           fake_sam3):
    win = make_window(lr.EmptyIndex(), make_coco([{"id": 0, "name": "a"}]))

    win._start_sam3_worker("/img/f1.png", [[0, 0, 4, 4]], ["a"], [1])
    assert len(FakeWorker.instances) == 1
    assert FakeWorker.instances[-1].isRunning()
    # the canvas stays usable (drawing/navigating) while SAM3 runs
    assert win.canvas.isEnabled()
    assert win.side.sam3_status.text() == "SAM3: running on 1 box(es)…"

    # second + third requests while busy → queued, status shows the count
    win._start_sam3_worker("/img/f2.png", [[1, 1, 5, 5]], ["a"], [2])
    assert len(win._sam3_queue) == 1 and len(FakeWorker.instances) == 1
    assert win.side.sam3_status.text() == \
        "SAM3 (1 in queue): running on 1 box(es)…"
    win._start_sam3_worker("/img/f3.png", [[2, 2, 6, 6]], ["a"], [3])
    assert win.side.sam3_status.text().startswith("SAM3 (2 in queue):")

    # buttons: per-frame run stays enabled (so it can queue), batch disabled
    assert win.side.btn_run_sam3.isEnabled()
    assert win.side.btn_reseg.isEnabled()
    assert not win.side.btn_sam3_all_frames.isEnabled()
    assert win.side.btn_cancel_sam3.isEnabled()

    # current worker ends → drain pops the oldest queued job
    FakeWorker.instances[-1].stop()
    win._start_next_queued_sam3()
    assert len(win._sam3_queue) == 1
    assert len(FakeWorker.instances) == 2
    assert FakeWorker.instances[-1].kw["image_path"] == "/img/f2.png"
    assert FakeWorker.instances[-1].kw["ann_ids"] == [2]
    assert win.side.sam3_status.text().startswith("SAM3 (1 in queue):")

    # cancel drops the running job AND the queue
    win._on_cancel_sam3()
    assert len(win._sam3_queue) == 0
    assert "cancelling" in win.side.sam3_status.text()
    assert "(1 in queue)" not in win.side.sam3_status.text()
    FakeWorker.instances[-1].stop()


def test_line_edit_defocus_on_outside_click(lr, make_coco, make_window,
                                            qapp):
    win = make_window(lr.EmptyIndex(), make_coco(CATS))
    edit = win.side.add_cat_edit
    edit.setFocus()
    qapp.processEvents()
    assert QtWidgets.QApplication.focusWidget() is edit
    press = QtGui.QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5),
                              Qt.MouseButton.LeftButton,
                              Qt.MouseButton.LeftButton, NM)
    QtWidgets.QApplication.sendEvent(win.canvas, press)
    qapp.processEvents()
    assert QtWidgets.QApplication.focusWidget() is not edit


def test_source_label(lr, make_coco, make_window, tmp_path):
    win = make_window(lr.EmptyIndex(), make_coco(CATS))
    folder = make_image_folder(tmp_path / "src_folder",
                               ["1000000000.jpg", "2000000000.jpg"],
                               size=(8, 8))
    win.frame_index = lr.ImageFolderIndex([str(folder)])
    win._update_source_label()
    assert win.side.source_label.toolTip() == str(folder)
    assert win.side.source_label.text().startswith("Source:")

    win.frame_index = lr.EmptyIndex()
    win._update_source_label()
    assert win.side.source_label.text() == "Source: —"


# ---------------------------------------------------------------------------
# reseg (R / Re-seg selected)
# ---------------------------------------------------------------------------

@pytest.fixture
def reseg_setup(lr, make_coco, make_window, fake_sam3, tmp_path):
    folder = make_image_folder(tmp_path / "reseg_q",
                               ["1000000000.jpg", "2000000000.jpg"],
                               size=(32, 32))
    idx = lr.ImageFolderIndex([str(folder)])
    coco = lr.CocoState(str(tmp_path / "reseg_out.json"),
                        [{"id": 0, "name": "a"}])
    win = make_window(idx, coco)
    img_id = coco._img_id_by_idx[0]
    ann_id = coco.add_box(img_id, 2, 2, 10, 10, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    return win, coco, idx, img_id, ann_id, str(folder)


def test_reseg_queues_while_busy_and_drains(lr, reseg_setup):
    win, coco, idx, img_id, ann_id, folder = reseg_setup
    img_path = os.path.join(folder, "1000000000.jpg")
    win._start_sam3_worker(img_path, [[0, 0, 4, 4]], ["a"], [999])
    assert FakeWorker.instances[-1].isRunning()

    # press reseg while busy → must queue
    win._on_resegment_selected()
    assert len(win._sam3_queue) == 1
    job = win._sam3_queue[0]
    assert job["ann_ids"] == [ann_id]
    assert job["bboxes_xyxy"] == [[2.0, 2.0, 12.0, 12.0]]
    assert win.side.sam3_status.text().startswith("SAM3 (1 in queue):")

    # drain: queued reseg job starts with the same ann_id
    FakeWorker.instances[-1].stop()
    win._start_next_queued_sam3()
    assert len(win._sam3_queue) == 0
    assert FakeWorker.instances[-1].kw["ann_ids"] == [ann_id]
    assert FakeWorker.instances[-1].isRunning()


def test_reseg_button_queues_while_busy(lr, reseg_setup):
    win, coco, idx, img_id, ann_id, folder = reseg_setup
    win._start_sam3_worker(os.path.join(folder, "1000000000.jpg"),
                           [[0, 0, 4, 4]], ["a"], [999])
    n_before = len(FakeWorker.instances)
    win.side.btn_reseg.click()
    assert len(win._sam3_queue) == 1
    assert len(FakeWorker.instances) == n_before  # queued, not started


def test_drawn_box_auto_selected_and_no_selection_noop(lr, make_coco,
                                                       make_window,
                                                       fake_sam3, tmp_path):
    folder = make_image_folder(tmp_path / "reseg2", ["1000000000.jpg"],
                               size=(32, 32))
    idx = lr.ImageFolderIndex([str(folder)])
    win = make_window(idx, lr.CocoState(str(tmp_path / "reseg_out2.json"),
                                        [{"id": 0, "name": "a"}]))
    img_id = win.coco._img_id_by_idx[0]
    win._on_box_added(img_id, 3, 3, 8, 8, 0)   # the shared draw path
    sel = win.canvas._selected_idx
    assert 0 <= sel < len(win.canvas._boxes)
    assert win.canvas._boxes[sel]["id"] == 1

    # reseg with no selection → visible no-op: no queue, no worker
    win.canvas._selected_idx = -1
    win.canvas._multi_selected = set()
    n_workers = len(FakeWorker.instances)
    win._on_resegment_selected()
    assert not win._sam3_queue
    assert len(FakeWorker.instances) == n_workers


def test_multiselect_reseg_covers_all_selected(lr, make_coco, make_window,
                                               fake_sam3, tmp_path):
    folder = make_image_folder(tmp_path / "reseg3", ["1000000000.jpg"],
                               size=(32, 32))
    idx = lr.ImageFolderIndex([str(folder)])
    win = make_window(idx, lr.CocoState(str(tmp_path / "reseg_out3.json"),
                                        [{"id": 0, "name": "a"}]))
    img_id = win.coco._img_id_by_idx[0]
    win._on_box_added(img_id, 3, 3, 8, 8, 0)
    win._on_box_added(img_id, 16, 16, 8, 8, 0)
    assert len(win.canvas._boxes) == 2
    win.canvas._multi_selected = {0, 1}   # shift-click selection
    win.canvas._selected_idx = 1          # primary = last clicked
    win._on_resegment_selected()
    # not busy → starts immediately, with BOTH boxes in one job
    w = FakeWorker.instances[-1]
    assert w.isRunning()
    assert w.kw["ann_ids"] == [1, 2]
    assert w.kw["bboxes_xyxy"] == [[3.0, 3.0, 11.0, 11.0],
                                   [16.0, 16.0, 24.0, 24.0]]
    w.stop()

    # while busy, a multi-select reseg queues as ONE job with both ann_ids
    win._start_sam3_worker(os.path.join(str(folder), "1000000000.jpg"),
                           [[0, 0, 4, 4]], ["a"], [999])
    win._on_resegment_selected()
    assert len(win._sam3_queue) == 1
    assert win._sam3_queue[0]["ann_ids"] == [1, 2]


def test_recat_updates_sticky_category(lr, make_coco, make_window, tmp_path):
    folder = make_image_folder(tmp_path / "sticky_recat", ["1000000000.jpg"],
                               size=(32, 32))
    idx = lr.ImageFolderIndex([str(folder)])
    cats = [{"id": 0, "name": "ceiling light"}, {"id": 1, "name": "monitor"}]
    win = make_window(idx, make_coco(cats))
    img_id = win.coco._img_id_by_idx[0]

    # draw a box as "ceiling light" → sticky = 0
    win._on_box_added(img_id, 2, 2, 10, 10, 0)
    assert win._last_cat_id == 0

    # recategorize it to "monitor" (C → type 1 → Enter path)
    assert win.canvas._selected_idx >= 0  # auto-selected after drawing
    win._on_recat_selected(1)
    assert win._last_cat_id == 1
    assert win.coco.annotations[-1]["category_id"] == 1

    # next draw: no explicit preselection → sticky (recat) category wins
    win._pending_cat_id = None
    pre = (win._pending_cat_id if win._pending_cat_id is not None
           else win._last_cat_id)
    assert pre == 1

    # an explicit sidebar preselection still overrides the sticky cat
    win._pending_cat_id = 0
    pre = (win._pending_cat_id if win._pending_cat_id is not None
           else win._last_cat_id)
    assert pre == 0


# ---------------------------------------------------------------------------
# autolabel
# ---------------------------------------------------------------------------

AL_CATS = [{"id": 0, "name": "ceiling light"}, {"id": 1, "name": "monitor"}]


def test_autolabel_buttons_and_hide_group(lr, make_coco, make_window,
                                          fake_sam3):
    win = make_window(lr.EmptyIndex(), make_coco(AL_CATS))
    for b in ("btn_autolabel_frame", "btn_autolabel_all"):
        assert hasattr(win.side, b), b
    assert not hasattr(win.side, "btn_autolabel_cat"), "sel-cat button removed"
    assert "autolabel" in lr.SidePanel._HIDEABLE
    assert set(lr.SidePanel._HIDEABLE["autolabel"]) == {
        "autolabel_header", "btn_autolabel_frame", "btn_autolabel_all"}


def test_autolabel_concepts_multiselect(lr, make_coco, make_window,
                                        fake_sam3):
    win = make_window(lr.EmptyIndex(), make_coco(AL_CATS))
    concepts, cat_ids = win._autolabel_concepts()
    assert concepts == ["ceiling light", "monitor"] and cat_ids == [0, 1]
    # Ctrl/Shift multi-select restricts the run to the highlighted
    # categories (setSelected() alone only preselects; _restrict_selection
    # is set by the Ctrl/Shift-click handler — simulate that here)
    win.side._rebuild_cat_list()
    win.side.cat_list.item(1).setSelected(True)  # monitor
    win.side._restrict_selection = True
    concepts, cat_ids = win._autolabel_concepts()
    assert concepts == ["monitor"] and cat_ids == [1]
    win.side.cat_list.item(0).setSelected(True)  # + ceiling light
    concepts, cat_ids = win._autolabel_concepts()
    assert concepts == ["ceiling light", "monitor"] and cat_ids == [0, 1]
    win.side.cat_list.clearSelection()
    win.side._restrict_selection = False
    # preselect alone does NOT restrict autolabel
    win._on_preselect_cat(1)
    win.side._preselected_cat_id = 1
    concepts, cat_ids = win._autolabel_concepts()
    assert concepts == ["ceiling light", "monitor"] and cat_ids == [0, 1]


def test_autolabel_concepts_plain_multiselect(lr, make_coco, make_window,
                                              fake_sam3):
    """A 2+-row highlight restricts autolabel even without a Ctrl/Shift
    click (e.g. rubber-band / drag selection)."""
    cats = [{"id": 0, "name": "ceiling light"}, {"id": 1, "name": "monitor"},
            {"id": 2, "name": "door"}]
    win = make_window(lr.EmptyIndex(), make_coco(cats))
    win.side._rebuild_cat_list()
    win.side.cat_list.item(0).setSelected(True)
    win.side.cat_list.item(2).setSelected(True)
    assert not win.side._restrict_selection  # no Ctrl/Shift click happened
    concepts, cat_ids = win._autolabel_concepts()
    assert concepts == ["ceiling light", "door"] and cat_ids == [0, 2]
    # a single plain (non-Ctrl/Shift) selection still does NOT restrict
    win.side.cat_list.item(2).setSelected(False)
    concepts, cat_ids = win._autolabel_concepts()
    assert cat_ids == [0, 1, 2]


def test_autolabel_header_follows_detector(lr, make_coco, make_window,
                                           fake_sam3):
    """The autolabel section header/tooltips name the active detector."""
    win = make_window(lr.EmptyIndex(), make_coco(AL_CATS),
                      autolabel_detector="grounding_dino")
    assert win.side.autolabel_header.text() == "Grounding DINO Autolabel:"
    assert "Grounding DINO" in win.side.btn_autolabel_frame.toolTip()
    win._apply_runtime_config({"autolabel": {"detector": "owlv2_exemplar"}})
    assert win.side.autolabel_header.text() == "OWLv2 exemplar Autolabel:"
    assert "visual query" in win.side.btn_autolabel_frame.toolTip()
    win._apply_runtime_config({"autolabel": {"detector": "owlv2"}})
    assert win.side.autolabel_header.text() == "OWLv2 Autolabel:"
    assert "with masks" not in win.side.btn_autolabel_frame.toolTip()


@pytest.fixture
def autolabel_win(lr, make_coco, make_window, fake_sam3):
    coco = make_coco(AL_CATS)
    win = make_window(lr.EmptyIndex(), coco)
    frame = {"frame_idx": 0, "timestamp_ns": 111, "log_time_ns": 111,
             "existing_boxes": [], "file_path": "/img/f0.png",
             "file_name": "f0.png"}
    image_id = coco.ensure_image(frame, 100, 80)
    return win, coco, image_id


def _al_mask():
    mask = np.zeros((80, 100), dtype=bool)
    mask[10:30, 10:40] = True
    return mask


def test_apply_autolabel_dets_dedup_and_undo(lr, autolabel_win):
    win, coco, image_id = autolabel_win
    mask = _al_mask()
    dets = [
        {"label": "ceiling light", "cat_id": 0,
         "bbox_xyxy": [10, 10, 40, 30], "mask": mask, "confidence": 0.9},
        {"label": "monitor", "cat_id": 1,
         "bbox_xyxy": [50, 20, 90, 60], "mask": None, "confidence": 0.8},
        {"label": "??", "cat_id": None,  # unknown cat → skipped
         "bbox_xyxy": [0, 0, 5, 5], "mask": None, "confidence": 0.5},
        {"label": "tiny", "cat_id": 0,  # degenerate box → skipped
         "bbox_xyxy": [0, 0, 1, 1], "mask": None, "confidence": 0.5},
    ]
    added, skipped = win._apply_autolabel_dets(image_id, dets)
    assert (added, skipped) == (2, 2)
    anns = coco.anns_for_image(image_id)
    assert len(anns) == 2
    a0 = [a for a in anns if a["category_id"] == 0][0]
    assert a0["bbox"] == [10, 10, 30, 20]
    assert a0.get("_mask") is not None, "mask not attached"
    a1 = [a for a in anns if a["category_id"] == 1][0]
    assert a1.get("_mask") is None

    # duplicate of an existing same-cat box (IoU > 0.7) → skipped
    dups = [{"label": "ceiling light", "cat_id": 0,
             "bbox_xyxy": [11, 11, 41, 31], "mask": mask, "confidence": 0.95}]
    added, skipped = win._apply_autolabel_dets(image_id, dups)
    assert (added, skipped) == (0, 1)
    assert len(coco.anns_for_image(image_id)) == 2

    # one undo step reverts the whole autolabel group
    win._on_undo()
    win._on_undo()
    assert len(coco.anns_for_image(image_id)) == 0
    win._on_redo()
    win._on_redo()
    assert len(coco.anns_for_image(image_id)) == 2


def test_apply_autolabel_dets_nms(lr, autolabel_win):
    win, coco, image_id = autolabel_win
    # two overlapping same-cat dets → highest confidence kept
    nms_dets = [
        {"label": "monitor", "cat_id": 1, "bbox_xyxy": [55, 5, 95, 25],
         "mask": None, "confidence": 0.9},
        {"label": "monitor", "cat_id": 1, "bbox_xyxy": [56, 6, 96, 26],
         "mask": None, "confidence": 0.6},
    ]
    added, skipped = win._apply_autolabel_dets(image_id, nms_dets)
    assert (added, skipped) == (1, 1)
    new_mon = [a for a in coco.anns_for_image(image_id)
               if a["category_id"] == 1 and a["bbox"][0] == 55.0]
    assert len(new_mon) == 1
    # same overlap but DIFFERENT category → both kept (class-aware NMS)
    cross = [
        {"label": "ceiling light", "cat_id": 0, "bbox_xyxy": [5, 50, 35, 70],
         "mask": None, "confidence": 0.9},
        {"label": "monitor", "cat_id": 1, "bbox_xyxy": [6, 51, 36, 71],
         "mask": None, "confidence": 0.6},
    ]
    added, skipped = win._apply_autolabel_dets(image_id, cross)
    assert (added, skipped) == (2, 0)
    # nms_iou = 1.0 disables the dedup
    win.sam3_nms_iou = 1.0
    added, skipped = win._apply_autolabel_dets(image_id, [
        {"label": "ceiling light", "cat_id": 0, "bbox_xyxy": [60, 60, 70, 70],
         "mask": None, "confidence": 0.9},
        {"label": "ceiling light", "cat_id": 0, "bbox_xyxy": [61, 61, 71, 71],
         "mask": None, "confidence": 0.6},
    ])
    assert (added, skipped) == (2, 0)


def test_autolabel_queue_dispatch_fifo(lr, autolabel_win):
    win, coco, image_id = autolabel_win
    mask = _al_mask()
    # a pre-existing box to reference
    a1 = coco.add_box(image_id, 50, 20, 40, 40, 1)

    win._start_sam3_worker("/img/f1.png", [[0, 0, 4, 4]], ["monitor"],
                           [a1])
    assert len(FakeWorker.instances) == 1
    assert FakeWorker.instances[-1].isRunning()

    # autolabel request while box worker runs → queued
    win._start_autolabel_worker("/img/f1.png", ["monitor"], [1], image_id)
    assert len(win._sam3_queue) == 1
    assert win._sam3_queue[0]["kind"] == "autolabel"
    assert len(FakeAutolabelSingle.instances) == 0
    assert "(1 in queue)" in win.side.sam3_status.text()

    # box request while busy also queues behind it
    win._start_sam3_worker("/img/f2.png", [[1, 1, 5, 5]], ["monitor"],
                           [a1])
    assert len(win._sam3_queue) == 2

    # box worker finishes → drain pops the autolabel job first (FIFO)
    FakeWorker.instances[-1].stop()
    win._start_next_queued_sam3()
    assert len(FakeAutolabelSingle.instances) == 1
    aw = FakeAutolabelSingle.instances[-1]
    assert aw.isRunning()
    assert aw.kw["image_path"] == "/img/f1.png"
    assert aw.kw["concepts"] == ["monitor"] and aw.kw["cat_ids"] == [1]
    assert aw.kw["image_id"] == image_id
    assert len(win._sam3_queue) == 1

    # autolabel worker finishes → drain starts the queued box job
    aw.stop()
    win._start_next_queued_sam3()
    assert len(FakeWorker.instances) == 2
    assert FakeWorker.instances[-1].kw["image_path"] == "/img/f2.png"
    assert len(win._sam3_queue) == 0
    FakeWorker.instances[-1].stop()


def test_autolabel_results_apply_by_image_id(lr, autolabel_win):
    win, coco, image_id = autolabel_win
    mask = _al_mask()
    win._start_autolabel_worker("/img/f0.png", ["ceiling light"], [0],
                                image_id)
    aw = FakeAutolabelSingle.instances[-1]
    assert aw.isRunning()
    n_before = len(coco.anns_for_image(image_id))
    det = [{"label": "ceiling light", "cat_id": 0,
            "bbox_xyxy": [60, 5, 80, 15], "mask": mask, "confidence": 0.7}]
    # simulate the worker finishing while the user is on another frame
    for slot in aw.finished_signal._slots:
        slot(image_id, det)
    aw.stop()
    assert len(coco.anns_for_image(image_id)) == n_before + 1
    assert "autolabel: +1 box(es)" in win.side.sam3_status.text()


def test_busy_check_covers_autolabel_workers(lr, autolabel_win):
    win, coco, image_id = autolabel_win
    a1 = coco.add_box(image_id, 5, 5, 10, 10, 0)
    win._start_autolabel_worker("/img/f0.png", ["monitor"], [1], image_id)
    assert win._sam3_busy()
    # a box request while autolabel runs must queue, not run concurrently
    win._start_sam3_worker("/img/f9.png", [[0, 0, 4, 4]], ["monitor"], [a1])
    assert len(win._sam3_queue) == 1
    assert len(FakeWorker.instances) == 0  # unchanged — nothing started
    FakeAutolabelSingle.instances[-1].stop()
    win._sam3_queue.clear()


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------

@pytest.fixture
def propagate_win(lr, make_coco, make_window, fake_sam3, auto_yes):
    coco = make_coco([{"id": 0, "name": "ceiling light"}])
    win = make_window(FakeIdx(6), coco)
    return win, coco


def test_propagate_headers_and_hide(lr, propagate_win):
    win, coco = propagate_win
    assert win.side.sam3_header.text() == "SAM3 segmentation:"
    assert win.side.autolabel_header.text() == "SAM3 Autolabel:"
    assert "sam3_header" in lr.SidePanel._HIDEABLE["sam3_run"]
    assert "btn_propagate" in lr.SidePanel._HIDEABLE["sam3_run"]
    assert "autolabel_header" in lr.SidePanel._HIDEABLE["autolabel"]
    win.side.set_hidden_groups(["sam3_run"])
    assert win.side.sam3_header.isHidden() or \
        not win.side.sam3_header.isVisible()
    assert not win.side.btn_propagate.isVisible()
    win.side.set_hidden_groups([])


def test_propagate_button_enable_rules(lr, propagate_win):
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    coco.add_box(image_id, 50, 40, 20, 20, 0)  # second box for multi-select
    win._refresh_boxes()
    assert not win.side.btn_propagate.isEnabled()  # nothing selected

    win.canvas._selected_idx = 0
    win.canvas._multi_selected = {0}
    win._update_propagate_button()
    assert win.side.btn_propagate.isEnabled()

    win.canvas._multi_selected = {0, 1}  # multi-select → also enabled
    win._update_propagate_button()
    assert win.side.btn_propagate.isEnabled()


@pytest.fixture
def propagate_seeded(lr, propagate_win):
    """Window with a selected seed box and the propagate worker started."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    seed_ann = coco.add_box(image_id, 10, 10, 30, 20, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    win.canvas._multi_selected = {0}
    seed_tid = coco.get_box(seed_ann)["track_id"]
    win._on_propagate_track()
    return win, coco, seed_ann, seed_tid


def test_propagate_seed_starts_worker(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    assert len(FakePropagateWorker.instances) == 1
    w = FakePropagateWorker.instances[-1]
    assert w.isRunning()
    assert w.kw["start_frame_idx"] == 0
    assert w.kw["end_frame_idx"] is None  # no keyframes → runs to the end
    assert w.kw["seeds"] == [{"track_id": seed_tid, "cat_id": 0,
                              "concept": "ceiling light",
                              "bbox_xyxy": [10.0, 10.0, 40.0, 30.0]}]
    assert win._propagate_meta()["seeds"][0]["track_id"] == seed_tid
    assert win.side.sam3_status.text().startswith(
        f"SAM3: propagate T{seed_tid}:")


def test_propagate_stops_at_next_keyframe(lr, propagate_win):
    """With a later ★ keyframe marked, propagation only covers the frames
    between the keyframes (up to and including the next keyframe)."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    win.canvas._multi_selected = {0}
    coco.keyframes.add(3)
    win._on_propagate_track()
    w = FakePropagateWorker.instances[-1]
    assert w.kw["end_frame_idx"] == 4  # next keyframe is 3, inclusive
    assert "0/3 frames" in win.side.sam3_status.text()


def test_propagate_ignores_past_keyframes(lr, propagate_win):
    """Keyframes at/before the current frame don't bound the run."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    win.canvas._multi_selected = {0}
    coco.keyframes.add(0)
    win._on_propagate_track()
    w = FakePropagateWorker.instances[-1]
    assert w.kw["end_frame_idx"] is None


def test_mark_every_nth(lr, make_coco, make_window, fake_sam3):
    """'★ mark keyframes': frames 0, N, 2N… get marked; existing marks
    are kept."""
    coco = make_coco([{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(21), coco)
    coco.keyframes.add(5)  # pre-existing mark must survive
    win._on_mark_every_nth(10)
    assert coco.keyframes == {0, 5, 10, 20}
    # narrower stride adds more marks, keeps old ones
    win._on_mark_every_nth(5)
    assert coco.keyframes == {0, 5, 10, 15, 20}


def test_propagate_all_keyframes_queues_per_gap(lr, propagate_win):
    """'⇉ Propagate all keyframes': one queued SAM3 job per keyframe gap
    per side, seeded from the boxes already on each keyframe and bounded
    at the next keyframe; a gap with no tracked boxes is skipped."""
    win, coco = propagate_win
    # Keyframes 0, 3 (boxes with track ids on 0 only) and 5 (no boxes).
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    coco.set_track_id(coco.annotations[-1]["id"], 1)
    coco.keyframes.update({0, 3, 5})
    win._on_propagate_all_keyframes()
    # Two gaps (0→3, 3→5); only gap 1 has seeds.
    assert len(win._sam3_queue) == 1
    job = win._sam3_queue[0]
    assert job["kind"] == "propagate" and job["side"] == "left"
    assert job["start_frame_idx"] == 0
    assert job["end_frame_idx"] == 4  # bounded at keyframe 3 (+1)
    assert job["seeds"][0]["track_id"] == 1
    win._sam3_queue.clear()


def test_propagate_all_keyframes_needs_two_keyframes(lr, propagate_win):
    win, coco = propagate_win
    coco.keyframes.add(2)
    win._on_propagate_all_keyframes()
    assert not win._sam3_queue  # one keyframe → nothing to do


def test_interpolate_all_keyframes_queues_gaps(lr, make_coco, make_window,
                                               fake_sam3, auto_yes,
                                               monkeypatch):
    """'⇉ Interpolate all keyframes' queues one span per adjacent labeled
    anchor pair (wrapping) and drains them through the interp worker;
    gaps already filled are skipped."""
    from gui.label_review.ui import main_window as mw
    coco = make_coco([{"id": 0, "name": "a"}])
    win = make_window(FakeIdx(9), coco)
    # Labeled anchors at frames 0 and 3 (and 7) → gaps (0,3), (3,8), (8,0
    # wraps to nothing since 0 is before 8 in the wrap scan? 8 is last
    # anchor; wrap starts list over → no gap from 8).
    for fidx in (0, 3, 8):
        img_id = win._ensure_image_id(fidx, "left")
        coco.add_box(img_id, 10.0 + fidx, 10.0, 20.0, 20.0, 0)
    # Fake the worker so no real optical flow runs.
    class FakeInterp:
        instances = []
        def __init__(self, *a, **kw):
            self.jobs = a[1] if len(a) > 1 else []
            for s in ("progress_signal", "finished_signal", "failed_signal",
                      "cancelled_signal"):
                setattr(self, s, FakeSig())
            self._running = False
            FakeInterp.instances.append(self)
        def start(self): self._running = True
        def isRunning(self): return self._running
        def cancel(self): self._running = False
        def stop(self): self._running = False
    monkeypatch.setattr(mw, "InterpBatchWorker", FakeInterp)
    FakeInterp.instances.clear()
    win._on_interpolate_all_keyframes()
    # Spans start from the first anchor AFTER the current frame (frame 0):
    # (3,8) runs first, (0,3) queued next.
    assert len(FakeInterp.instances) == 1
    assert FakeInterp.instances[0].jobs[0]["a"] == 3
    assert win._interp_all_pending == [(0, 3)]
    # Finish the first → the next gap starts. (Emitting finished marks
    # the fake torn down, mirroring a real QThread exiting run().)
    FakeInterp.instances[0].stop()
    FakeInterp.instances[0].finished_signal.emit([])
    assert len(FakeInterp.instances) == 2
    assert FakeInterp.instances[1].jobs[0]["a"] == 0
    # Finishing the last span drains the queue.
    FakeInterp.instances[1].stop()
    FakeInterp.instances[1].finished_signal.emit([])
    assert win._interp_all_pending == []


def test_rerun_markers_labeled_with_waypoint_ids(lr, make_coco,
                                                 make_window):
    """Annotated-frame markers on the rerun map are labeled with their
    waypoint ordinal (sorted by frame index), not just the frame name."""
    class FakePoseDb:
        def pose_for(self, file_name, timestamp_ns):
            return np.array([1.0, 2.0, 3.0])

    class FakeRerun:
        rrd_path = "map.rrd"
        win_id = None

        def __init__(self):
            self.markers = None

        def log_annotated_markers(self, markers):
            self.markers = markers
            return True

    coco = make_coco([{"id": 0, "name": "x"}])
    rerun = FakeRerun()
    win = make_window(FakeIdx(6), coco,
                      pose_db=FakePoseDb(), rerun_logger=rerun)
    coco.annotated_marks.update({1, 2, 4})
    coco.discarded_frames.add(2)  # discarded: excluded, no waypoint number
    win._on_show_annotated_rerun()
    labels = [lbl for _, lbl in rerun.markers]
    assert labels == ["Waypoint #1 (frame 2)", "Waypoint #2 (frame 5)"]


def test_clear_rerun_waypoints(lr, make_coco, make_window):
    """'✖ Clear waypoints' logs an empty marker list into the open .rrd."""
    class FakeRerun:
        rrd_path = "map.rrd"

        def __init__(self):
            self.calls = []

        def log_annotated_markers(self, markers):
            self.calls.append(markers)
            return True

    coco = make_coco([{"id": 0, "name": "x"}])
    rerun = FakeRerun()
    win = make_window(FakeIdx(6), coco, rerun_logger=rerun)
    win._on_clear_rerun_waypoints()
    assert rerun.calls == [[]]


def test_clear_rerun_waypoints_requires_recording(lr, make_coco, make_window):
    """Without an open .rrd the clear button only shows a status hint."""
    class FakeRerun:
        rrd_path = None

        def log_annotated_markers(self, markers):
            raise AssertionError("must not log without a recording")

    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(6), coco, rerun_logger=FakeRerun())
    win._on_clear_rerun_waypoints()  # no exception, no log call


def test_open_rerun_embeds_and_reveals_panel(lr, make_coco, make_window,
                                             monkeypatch):
    """Opening an .rrd embeds the viewer by default (embed=True) and the
    rerun map page replaces the image view — no external window."""
    class FakeRerun:
        def __init__(self):
            self.rrd_path = None
            self.win_id = None
            self.embed_modes = []

        def open_recording(self, path, embed=False):
            self.embed_modes.append(embed)
            self.rrd_path = path
            self.win_id = 0xABC if embed else None
            return True

    coco = make_coco([{"id": 0, "name": "x"}])
    rerun = FakeRerun()
    win = make_window(FakeIdx(6), coco, rerun_logger=rerun)
    assert win._stack.currentWidget() is win._splitter  # image view
    # Tests run on the offscreen platform — pretend X11 reparenting works.
    monkeypatch.setattr(win, "_native_embed_available", lambda: True)
    monkeypatch.setattr(lr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("map.rrd", "")))
    loaded = []
    monkeypatch.setattr(win, "_load_rerun_embedded",
                        lambda: loaded.append(win.rerun.win_id))
    win._open_rerun_file()
    assert rerun.embed_modes == [True]
    assert loaded == [0xABC]
    # The embedded map replaces the image view.
    assert win._stack.currentWidget() is win._rerun_page


def test_open_rerun_external_only_without_embed(lr, make_coco,
                                                make_window, monkeypatch):
    """When native embedding is unavailable the standalone windowed viewer
    is the fallback (embed=False, image view stays)."""
    class FakeRerun:
        def __init__(self):
            self.rrd_path = None
            self.win_id = None
            self.embed_modes = []

        def open_recording(self, path, embed=False):
            self.embed_modes.append(embed)
            self.rrd_path = path
            return True

    coco = make_coco([{"id": 0, "name": "x"}])
    rerun = FakeRerun()
    win = make_window(FakeIdx(6), coco, rerun_logger=rerun)
    monkeypatch.setattr(win, "_native_embed_available", lambda: False)
    monkeypatch.setattr(lr.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("map.rrd", "")))
    win._open_rerun_file()
    assert rerun.embed_modes == [False]
    assert win._stack.currentWidget() is win._splitter  # stays on images


def test_embedded_rerun_click_forwards_focus(lr, make_coco, make_window,
                                             monkeypatch):
    """Clicking the embedded rerun container forwards X11 keyboard focus to
    it (so WASD camera controls work); re-activating the app window while
    the map page is shown forwards it again."""
    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(6), coco)
    calls = []
    monkeypatch.setattr(win, "_focus_rerun_window", lambda: calls.append(1))
    win._rerun_embedded_wid = 0xABC
    container = QtWidgets.QWidget()
    win._rerun_container = container

    press = QtGui.QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 1),
                              Qt.MouseButton.LeftButton,
                              Qt.MouseButton.LeftButton, NM)
    win.eventFilter(container, press)
    assert calls == [1]

    # WindowActivate while the map page is shown forwards focus too.
    win._show_rerun_page()
    activate = QEvent(QEvent.Type.WindowActivate)
    win.eventFilter(win, activate)
    assert len(calls) == 2

    # On the image page, WindowActivate does NOT forward focus.
    win._show_image_page()
    win.eventFilter(win, QEvent(QEvent.Type.WindowActivate))
    assert len(calls) == 2


def test_rerun_view_menu_switch(lr, make_coco, make_window):
    """View → 'Switch to Rerun waypoint view' replaces the image view with
    the rerun map page; its label flips to 'Switch back to image view'
    while the map is shown."""
    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(6), coco)
    act = win._act_rerun_view
    assert act.text() == "Switch to Rerun waypoint view"
    assert win._stack.currentWidget() is win._splitter
    act.trigger()
    assert win._stack.currentWidget() is win._rerun_page
    assert act.text() == "Switch back to image view"
    act.trigger()
    assert win._stack.currentWidget() is win._splitter
    assert act.text() == "Switch to Rerun waypoint view"


def test_propagate_queues_behind_running_job(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    win._start_propagate_worker(
        1, [{"track_id": 99, "cat_id": 0, "bbox_xyxy": [0, 0, 5, 5]}],
        side="left")
    assert len(win._sam3_queue) == 1
    assert win._sam3_queue[0]["kind"] == "propagate"
    assert win._sam3_queue[0]["seeds"][0]["track_id"] == 99
    assert len(FakePropagateWorker.instances) == 1  # not started
    win._sam3_queue.clear()


def test_propagate_frame_results_apply_seed_track(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    mask = np.zeros((80, 100), dtype=bool)
    mask[12:32, 14:46] = True
    det1 = {"bbox_xyxy": [12, 12, 44, 32], "mask": mask, "confidence": 0.9}
    win._on_propagate_frame_done(1, [det1])
    img1 = coco._img_id_by_idx.get(1)
    anns1 = coco.anns_for_image(img1)
    assert len(anns1) == 1
    assert anns1[0]["track_id"] == seed_tid
    assert anns1[0]["category_id"] == 0
    assert anns1[0]["bbox"] == [12.0, 12.0, 32.0, 20.0]
    assert anns1[0].get("_mask") is not None
    assert anns1[0].get("propagated") is True

    # frame 2 skipped when a box with the same track id already exists
    img2 = coco.ensure_image(FakeIdx().frame_at(2), 100, 80)
    pre = coco.add_box(img2, 1, 1, 5, 5, 0)
    coco.set_track_id(pre, seed_tid)
    win._on_propagate_frame_done(2, [det1])
    assert len(coco.anns_for_image(img2)) == 1  # only the pre-existing box

    # det None (lost) adds nothing
    win._on_propagate_frame_done(3, [None])
    assert coco._img_id_by_idx.get(3) is None or \
        len(coco.anns_for_image(coco._img_id_by_idx[3])) == 0

    # finished: status + single undo group
    win._on_propagate_finished(1, {0: 3})  # lost from frame 3
    assert f"propagate T{seed_tid}: done — 1 box(es) added " \
           f"(T{seed_tid} lost at frame 4)" in win.side.sam3_status.text()
    win._on_undo()  # one undo reverts the whole propagate run…
    assert len(coco.anns_for_image(img1)) == 0
    assert coco.get_box(pre)["track_id"] == seed_tid  # …not the manual box
    win._on_redo()
    assert len(coco.anns_for_image(img1)) == 1


def test_propagate_cancel_wiring(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    w = FakePropagateWorker.instances[-1]
    assert w.isRunning()
    win._on_cancel_sam3()
    assert w._cancelled
    win._on_propagate_cancelled()
    assert "cancelled" in win.side.sam3_status.text()


def test_propagate_queued_dispatch_after_run_ends(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    w3 = FakePropagateWorker.instances[-1]
    assert w3.isRunning()
    win._sam3_queue.append({"kind": "propagate", "start_frame_idx": 2,
                            "seeds": [{"track_id": 7, "cat_id": 0,
                                       "bbox_xyxy": [0, 0, 5, 5]}],
                            "side": "left"})
    w3.stop()
    win._on_propagate_finished(0, {})  # ends the run, drain via direct call
    win._start_next_queued_sam3()
    w4 = FakePropagateWorker.instances[-1]
    assert w4 is not w3 and w4.isRunning()
    assert w4.kw["start_frame_idx"] == 2
    assert win._propagate_meta()["seeds"][0]["track_id"] == 7


def test_propagate_multi_select_single_job_all_seeds(lr, propagate_win):
    """Selecting two boxes of different tracks propagates both in ONE
    memory-bank session (no queue)."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    coco.add_box(image_id, 50, 40, 20, 20, 0)
    win._refresh_boxes()
    win.canvas._multi_selected = {0, 1}
    win.canvas._selected_idx = 1
    win._on_propagate_track()
    assert len(FakePropagateWorker.instances) == 1
    w = FakePropagateWorker.instances[-1]
    assert w.isRunning()
    assert len(w.kw["seeds"]) == 2
    assert len(win._sam3_queue) == 0  # nothing queued — one job, two seeds


def test_propagate_multi_select_dedupes_shared_track(lr, propagate_win):
    """Two selected boxes with the SAME track id seed that track once."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    a = coco.add_box(image_id, 10, 10, 30, 20, 0)
    b = coco.add_box(image_id, 50, 40, 20, 20, 0)
    tid = coco.get_box(a)["track_id"]
    coco.set_track_id(b, tid)
    win._refresh_boxes()
    win.canvas._multi_selected = {0, 1}
    win.canvas._selected_idx = 1
    win._on_propagate_track()
    assert len(FakePropagateWorker.instances) == 1
    assert len(FakePropagateWorker.instances[-1].kw["seeds"]) == 1
    assert len(win._sam3_queue) == 0  # nothing queued — one track only


def test_select_all_and_esc_clear(lr, propagate_win):
    """Ctrl+A (canvas.select_all) selects every box; Esc
    (canvas.clear_selection via keyPressEvent) clears the selection."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    coco.add_box(image_id, 50, 40, 20, 20, 0)
    win._refresh_boxes()
    win.canvas.select_all()
    assert win.canvas._multi_selected == {0, 1}
    assert win.side.btn_propagate.isEnabled()
    esc = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_Escape,
                          Qt.NoModifier)
    win.canvas.keyPressEvent(esc)
    assert win.canvas._multi_selected == set()
    assert win.canvas._selected_idx == -1
    assert not win.side.btn_propagate.isEnabled()


def test_paint_masks_tolerates_size_mismatch(lr, propagate_win, qapp):
    """A mask whose shape differs from the canvas image must be resized,
    not crash paintEvent (the old code built the RGBA array from the
    pre-resize shape → ValueError → masks silently never painted)."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    ann_id = coco.add_box(image_id, 10, 10, 30, 20, 0)
    mask = np.zeros((40, 50), dtype=bool)  # half the canvas image size
    mask[5:20, 5:30] = True
    coco.set_mask(ann_id, mask)
    win._refresh_boxes()
    win.canvas.show()
    win.canvas.repaint()
    qapp.processEvents()
    img = win.canvas.grab().toImage()
    pt = win.canvas._img_to_widget(20, 20)
    c = img.pixelColor(int(pt.x()), int(pt.y()))
    assert (c.red(), c.green(), c.blue()) != (0, 0, 0)  # mask overlay drawn


# ---------------------------------------------------------------------------
# review-fix regressions:
# #1 _shutdown_workers reaps ALL workers (incl. non-cancellable autolabel)
# #2 Cancel during single-frame autolabel reports instead of no-op
# #3 _switch_source clears the queue + drops stale-session results
# #4 failed SAM3 result never erases an existing mask
# #5a _on_sam3_finished is one undo group
# #5b queue-only cancel says "dropped", not "cancelling…"
# #5c autolabel-ALL cancel has its own status text
# ---------------------------------------------------------------------------

def test_failed_result_keeps_mask_and_single_undo_group(lr, make_coco,
                                                        make_window,
                                                        fake_sam3, auto_yes):
    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(4), coco)
    img0 = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    a_ok = coco.add_box(img0, 5, 5, 20, 20, 0)
    a_fail = coco.add_box(img0, 40, 40, 20, 20, 0)
    good_mask = np.zeros((80, 100), dtype=bool)
    good_mask[40:60, 40:60] = True
    coco.set_mask(a_fail, good_mask)
    new_mask = np.zeros((80, 100), dtype=bool)
    new_mask[5:25, 5:25] = True
    results = [
        {"ann_id": a_ok, "mask": new_mask, "success": True, "error": None},
        {"ann_id": a_fail, "mask": None, "success": False, "error": "boom"},
    ]
    win._on_sam3_finished(results)  # direct call: no sender → not stale
    assert coco.get_box(a_ok).get("_mask") is new_mask
    assert coco.get_box(a_fail).get("_mask") is good_mask, \
        "failed result erased the existing mask"
    assert "1 mask(s), 1 failed" in win.side.sam3_status.text()
    win._on_undo()  # single group → the whole run's mask change reverts
    assert coco.get_box(a_ok).get("_mask") is None
    assert coco.get_box(a_fail).get("_mask") is good_mask
    win._on_redo()
    assert coco.get_box(a_ok).get("_mask") is not None


def test_cancel_during_single_autolabel_reports(lr, make_coco, make_window,
                                                fake_sam3, auto_yes):
    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(4), coco)
    img0 = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    win._start_autolabel_worker("/img/f0.png", ["x"], [0], img0)
    alw = FakeAutolabelSingle.instances[-1]
    assert alw.isRunning()
    win._on_cancel_sam3()
    # not cancelled, honest message, no fake "cancelling…" status
    assert alw.isRunning()
    assert "cancelling" not in win.side.sam3_status.text()

    # queue-only cancel while autolabel runs: text says dropped
    win._sam3_queue.append({"kind": "autolabel", "img_path": "/img/f1.png",
                            "concepts": ["x"], "cat_ids": [0],
                            "image_id": img0})
    win._on_cancel_sam3()
    assert len(win._sam3_queue) == 0
    assert "1 queued job(s) dropped" in win.side.sam3_status.text()
    assert alw.isRunning()
    alw.stop()


def test_autolabel_batch_cancel_text(lr, make_coco, make_window, fake_sam3,
                                     auto_yes):
    win = make_window(FakeIdx(2), make_coco([{"id": 0, "name": "x"}]))
    win._on_autolabel_batch_cancelled()
    assert win.side.sam3_status.text() == "SAM3: autolabel all: cancelled"


def test_shutdown_reaps_all_worker_slots(lr, make_coco, make_window,
                                         fake_sam3, auto_yes):
    win = make_window(FakeIdx(2), make_coco([{"id": 0, "name": "x"}]))
    win._sam3_worker = FakeWorker()
    win._sam3_worker.start()
    win._sam3_batch_worker = FakeWorker()
    win._sam3_batch_worker.start()
    win._sam3_autolabel_worker = FakeAutolabelSingle()
    win._sam3_autolabel_worker.start()
    win._sam3_autolabel_batch_worker = FakeWorker()
    win._sam3_autolabel_batch_worker.start()
    win._sam3_propagate_worker = FakeWorker()
    win._sam3_propagate_worker.start()
    win._sam3_queue.append({"kind": "autolabel"})
    win._shutdown_workers()
    assert len(win._sam3_queue) == 0
    for w in (win._sam3_worker, win._sam3_batch_worker,
              win._sam3_autolabel_worker, win._sam3_autolabel_batch_worker,
              win._sam3_propagate_worker):
        assert not w.isRunning()
    # the non-cancellable autolabel worker refused wait() → terminate()
    assert win._sam3_autolabel_worker._terminated


def test_source_switch_session_guard(lr, make_coco, make_window, fake_sam3,
                                     auto_yes):
    coco = make_coco([{"id": 0, "name": "x"}])
    win = make_window(FakeIdx(4), coco)
    win._start_sam3_worker("/img/f0.png", [[0, 0, 4, 4]], ["x"], [1])
    w_old = FakeWorker.instances[-1]
    assert w_old.isRunning()
    win._sam3_queue.append({"img_path": "/img/f1.png",
                            "bboxes_xyxy": [[0, 0, 2, 2]],
                            "concepts": ["x"], "ann_ids": [2]})
    seq0 = win._session_seq
    out2 = str(__import__("pathlib").Path(coco.output_json).with_name(
        "switched.json"))
    win._switch_source(FakeIdx(), out2, "new")
    assert win._session_seq == seq0 + 1
    assert len(win._sam3_queue) == 0
    assert w_old._cancelled  # cancel requested on the in-flight worker

    # stale result signal is dropped: fake the Qt sender as the old worker
    new_mask = np.zeros((80, 100), dtype=bool)
    new_mask[5:25, 5:25] = True
    win.sender = lambda: w_old
    n_anns_before = len(win.coco.annotations)
    win._on_sam3_finished([{"ann_id": 1, "mask": new_mask, "success": True,
                            "error": None}])
    assert len(win.coco.annotations) == n_anns_before  # nothing applied
    # a fresh-session worker's results are applied
    w_new = FakeWorker()
    w_new._lr_session = win._session_seq
    win.sender = lambda: w_new
    img_new = win.coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    aid = win.coco.add_box(img_new, 1, 1, 3, 3, 0)
    win._on_sam3_finished([{"ann_id": aid, "mask": new_mask,
                            "success": True, "error": None}])
    assert win.coco.get_box(aid).get("_mask") is not None


# ---------------------------------------------------------------------------
# quit path reaps real QThreads
# ---------------------------------------------------------------------------

def test_quit_reaps_running_workers(lr, make_coco, make_window):
    class SlowThread(QtCore.QThread):
        """Stands in for a SAM3/interp worker with a long in-flight call."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def run(self):
            time.sleep(30)

    win = make_window(lr.EmptyIndex(), make_coco([]))
    w = SlowThread(win)
    w.start()
    win._sam3_worker = w
    assert w.isRunning()

    # Simulate the Q-key quit path with a dirty state, choosing Discard.
    win.coco.dirty = True
    win._confirm_quit = lambda: False   # discard without saving
    win._on_quit()                      # QApplication.quit() without exec

    assert w.cancelled, "worker was not asked to cancel"
    assert not w.isRunning(), "worker still running after quit — would crash Qt"


# ---------------------------------------------------------------------------
# discard image (excluded from the final JSON)
# ---------------------------------------------------------------------------

def test_discard_image_button(lr, make_coco, make_window, tmp_path):
    from conftest import make_image_folder
    folder = make_image_folder(tmp_path / "disc", ["a.png", "b.png"],
                               size=(16, 16))
    win = make_window(lr.ImageFolderIndex([str(folder)]), make_coco([]))
    btn = win.side.btn_discard_image

    # toggles the discard set + button state + HUD marker
    assert not btn.isChecked()
    win._on_toggle_discard()
    assert win.coco.discarded_frames == {0}
    assert btn.isChecked()
    assert "DISCARDED" in win.canvas._info_text

    # navigating away and back keeps the button in sync
    win._on_frame_nav(+1)
    assert not btn.isChecked()
    win._on_frame_nav(-1)
    assert btn.isChecked()

    # toggle again restores the frame
    win._on_toggle_discard()
    assert win.coco.discarded_frames == set()
    assert not btn.isChecked()
    assert "DISCARDED" not in win.canvas._info_text


# ---------------------------------------------------------------------------
# open-set autolabel backends (grounding_dino) and
# OWLv2 exemplar — routing, config roundtrip, exemplar crop extraction.
# ---------------------------------------------------------------------------

def test_autolabel_detector_config_roundtrip(lr, make_coco, make_window,
                                             fake_sam3):
    """Every autolabel backend roundtrips through the Config dialog and the
    window's apply_config; invalid values fall back to sam3."""
    win = make_window(lr.EmptyIndex(), make_coco(AL_CATS))
    dlg = lr.ConfigDialog(win)
    assert dlg._collect()["autolabel"]["detector"] == "sam3"
    for det in ("owlv2", "owlv2_exemplar", "grounding_dino"):
        dlg._prefill_from_config({"autolabel": {
            "detector": det, "owlv2_model": "m/ow", "owlv2_conf": 0.55,
            "gdino_model": "m/gd", "gdino_conf": 0.45}})
        out = dlg._collect()["autolabel"]
        assert out["detector"] == det
        assert out["owlv2_model"] == "m/ow" and out["owlv2_conf"] == 0.55
        assert out["gdino_model"] == "m/gd" and out["gdino_conf"] == 0.45
    # _apply_runtime_config on the live window
    win._apply_runtime_config({"autolabel": {"detector": "grounding_dino",
                                             "gdino_model": "m/gd2",
                                             "gdino_conf": 0.5}})
    assert win.autolabel_detector == "grounding_dino"
    assert win.gdino_model == "m/gd2" and win.gdino_conf == 0.5
    win._apply_runtime_config({"autolabel": {"detector": "bogus"}})
    assert win.autolabel_detector == "grounding_dino"  # unchanged


def test_autolabel_detector_ctor_validation(lr, make_coco, make_window,
                                            fake_sam3):
    win = make_window(lr.EmptyIndex(), make_coco(AL_CATS),
                      autolabel_detector="bogus")
    assert win.autolabel_detector == "sam3"
    win2 = make_window(lr.EmptyIndex(), make_coco(AL_CATS),
                       autolabel_detector="grounding_dino",
                       gdino_model="m/gd", gdino_conf=0.45)
    assert win2.autolabel_detector == "grounding_dino"
    assert win2.gdino_model == "m/gd" and win2.gdino_conf == 0.45


def test_generic_autolabel_routing(lr, autolabel_win, monkeypatch):
    """detector=grounding_dino routes to GenericAutolabelWorker and picks up
    the gdino model/conf."""
    from gui.label_review.ui import main_window as mw
    win, coco, image_id = autolabel_win
    monkeypatch.setattr(mw, "GenericAutolabelWorker", FakeAutolabelSingle)
    FakeAutolabelSingle.instances.clear()

    win.autolabel_detector = "grounding_dino"
    win.gdino_model = "m/gd"
    win.gdino_conf = 0.45
    win._start_autolabel_worker("/img/f0.png", ["monitor"], [1], image_id)
    aw = FakeAutolabelSingle.instances[-1]
    assert aw.isRunning()
    assert aw.kw["detector"] == "grounding_dino"
    assert aw.kw["model_id"] == "m/gd" and aw.kw["conf"] == 0.45
    assert aw.kw["image_path"] == "/img/f0.png"
    aw.stop()


def test_exemplar_crop_from_selection(lr, make_coco, make_window,
                                      fake_sam3):
    """The exemplar crop is the selected box's region of the decoded frame;
    its label comes from the box's category."""
    win = make_window(FakeIdx(4), make_coco(AL_CATS))
    frame = FakeIdx().frame_at(0)
    image_id = win.coco.ensure_image(frame, 100, 80)
    win.coco.add_box(image_id, 10, 20, 30, 15, 1)  # monitor
    win._current_idx = 0
    win._refresh_boxes()
    win.canvas._selected_idx = 0
    win.canvas._multi_selected = {0}
    out = win._exemplar_from_selection()
    assert out is not None
    crop, label, cat_id = out
    assert crop.shape == (15, 30, 3)
    assert label == "monitor" and cat_id == 1
    # no selection → None
    win.canvas._selected_idx = -1
    win.canvas._multi_selected = set()
    assert win._exemplar_from_selection() is None


def test_exemplar_autolabel_routing(lr, autolabel_win, monkeypatch):
    """detector=owlv2_exemplar routes to Owlv2ExemplarWorker carrying the
    crop/label/cat_id; without an exemplar no worker starts."""
    from gui.label_review.ui import main_window as mw
    win, coco, image_id = autolabel_win
    monkeypatch.setattr(mw, "Owlv2ExemplarWorker", FakeAutolabelSingle)
    FakeAutolabelSingle.instances.clear()
    win.autolabel_detector = "owlv2_exemplar"

    # no exemplar → refused, nothing started
    win._start_autolabel_worker("/img/f0.png", ["monitor"], [1], image_id)
    assert len(FakeAutolabelSingle.instances) == 0

    crop = np.zeros((10, 12, 3), np.uint8)
    win._start_autolabel_worker("/img/f0.png", ["monitor"], [1], image_id,
                                exemplar=(crop, "monitor", 1))
    aw = FakeAutolabelSingle.instances[-1]
    assert aw.isRunning()
    assert aw.kw["label"] == "monitor" and aw.kw["cat_id"] == 1
    assert aw.kw["exemplar"] is crop
    assert aw.kw["model_id"] == win.owlv2_model

    # the exemplar survives queueing (still busy → queued with the job)
    assert len(FakeAutolabelSingle.instances) == 1  # running
    win._start_autolabel_worker("/img/f1.png", ["monitor"], [1], image_id,
                                exemplar=(crop, "monitor", 1))
    assert len(FakeAutolabelSingle.instances) == 1  # queued, not started
    assert len(win._sam3_queue) == 1
    assert win._sam3_queue[0]["exemplar"] == (crop, "monitor", 1)
    win._sam3_queue.clear()
    FakeAutolabelSingle.instances[-1].stop()



def test_nav_annotated_buttons(lr, make_coco, make_window, fake_sam3,
                               auto_yes):
    """Prev/next annotated buttons jump between annotated frames (boxed or
    ✔-marked, discarded skipped), wrapping around at the ends; with none
    the view stays put."""
    coco = make_coco(CATS)
    win = make_window(FakeIdx(6), coco)

    # No annotated marks yet → nothing happens.
    win._on_nav_annotated(+1)
    assert win._current_idx == 0

    coco.annotated_marks.update({1, 4})
    win.side.btn_next_annotated.click()
    assert win._current_idx == 1  # nearest mark forward
    win.side.btn_next_annotated.click()
    assert win._current_idx == 4
    win.side.btn_next_annotated.click()
    assert win._current_idx == 1  # wraps to the first mark
    win.side.btn_prev_annotated.click()
    assert win._current_idx == 4  # wraps back to the last mark
    win.side.btn_prev_annotated.click()
    assert win._current_idx == 1

    # A single mark: both directions land on it from any other frame.
    coco.annotated_marks.discard(4)
    win._current_idx = 3
    win.side.btn_prev_annotated.click()
    assert win._current_idx == 1
    win._current_idx = 3
    win.side.btn_next_annotated.click()
    assert win._current_idx == 1  # wraps

    # Frames with boxes count as annotated too (same semantics as the
    # output JSON's annotated_image_ids) — even without a ✔ mark.
    coco.annotated_marks.clear()
    win._current_idx = 3
    win._load_current()  # creates frame 3's image record
    coco.add_box(win._current_image_id, 0, 0, 2, 2, 0)
    win._current_idx = 0
    win.side.btn_next_annotated.click()
    assert win._current_idx == 3  # boxed frame is a jump target

    # Discarded frames are skipped even when ✔-marked or boxed.
    coco.annotated_marks.update({1, 5})
    coco.discarded_frames.add(1)
    win._current_idx = 0
    win.side.btn_next_annotated.click()
    assert win._current_idx == 3  # frame 1 is discarded → skipped
    win.side.btn_next_annotated.click()
    assert win._current_idx == 5


def test_nav_keyframe_buttons(lr, make_coco, make_window, fake_sam3,
                              auto_yes):
    """Prev/next keyframe buttons jump between ★ keyframes, wrapping
    around at the ends; with no keyframes the view stays put."""
    coco = make_coco(CATS)
    win = make_window(FakeIdx(6), coco)

    # No keyframes yet → nothing happens.
    win._on_nav_keyframe(+1)
    assert win._current_idx == 0

    coco.keyframes.update({1, 4})
    win.side.btn_next_keyframe.click()
    assert win._current_idx == 1  # nearest keyframe forward
    win.side.btn_next_keyframe.click()
    assert win._current_idx == 4
    win.side.btn_next_keyframe.click()
    assert win._current_idx == 1  # wraps to the first keyframe
    win.side.btn_prev_keyframe.click()
    assert win._current_idx == 4  # wraps back to the last keyframe
    win.side.btn_prev_keyframe.click()
    assert win._current_idx == 1

    # A single keyframe: both directions land on it from any other frame.
    coco.keyframes.discard(4)
    win._current_idx = 3
    win.side.btn_prev_keyframe.click()
    assert win._current_idx == 1
    win._current_idx = 3
    win.side.btn_next_keyframe.click()
    assert win._current_idx == 1  # wraps

"""Side panel: category list + buttons + frame slider."""

from ..qt_compat import Qt, QtWidgets, pyqtSignal, _QT_HORZ  # enum shims
from ..qt_compat import (  # noqa: F401
    QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from typing import Any, Dict, List, Optional

from ..state.coco_state import CocoState  # noqa: F401  (type annotations)


# Display names for the autolabel detector backends (config
# "autolabel": {"detector": ...}). Used for the side-panel header, button
# tooltips and confirmation dialogs.
DETECTOR_LABELS = {
    "sam3": "SAM3",
    "owlv2": "OWLv2",
    "owlv2_exemplar": "OWLv2 exemplar",
    "grounding_dino": "Grounding DINO",
    "florence2": "Florence-2",
    "falcon": "Falcon Perception",
}

# Detectors whose detections carry segmentation masks.
DETECTORS_WITH_MASKS = ("sam3", "falcon")


# ---------------------------------------------------------------------------
# Side panel: category list + buttons + frame slider
# ---------------------------------------------------------------------------

class SidePanel(QWidget):

    cat_clicked = pyqtSignal(int)  # cat_id
    slider_moved = pyqtSignal(int)  # frame_idx
    slider_released = pyqtSignal()  # drag ended (mouse released)
    nav_delta = pyqtSignal(int)     # -10 / -5 / +5 / +10 frame jump buttons
    run_sam3_clicked = pyqtSignal()      # "Run SAM3 (all)" button
    toggle_masks_clicked = pyqtSignal()  # "Masks: on/off" button
    resegment_clicked = pyqtSignal()     # "Re-segment selected" button
    play_pause_clicked = pyqtSignal()     # "▶ / ⏸" button
    play_speed_changed = pyqtSignal(int)  # ms-per-frame
    box_selected = pyqtSignal(int)       # box list row clicked → canvas selection
    preselect_cat = pyqtSignal(int)      # category preselected for next draw
    mask_opacity_changed = pyqtSignal(int)  # 0-100 percent
    cancel_sam3_clicked = pyqtSignal()   # "Cancel SAM3" button
    recat_selected = pyqtSignal(int)     # new cat_id for the selected box
    sam3_all_frames_clicked = pyqtSignal()  # "SAM3 ALL frames" button
    autolabel_frame_clicked = pyqtSignal()     # "Autolabel frame" (all cats)
    autolabel_all_clicked = pyqtSignal()       # "Autolabel ALL frames"
    propagate_clicked = pyqtSignal()           # "Propagate →" button
    toggle_keyframe_clicked = pyqtSignal()   # "★ Keyframe" button (K)
    toggle_annotated_clicked = pyqtSignal()  # "✔ Mark as annotated" button
    toggle_discard_clicked = pyqtSignal()    # "🚫 Discard image" button
    point_seg_toggled = pyqtSignal(bool)     # "🎯 Add points" toggle
    segment_points_clicked = pyqtSignal()    # "▶ Segment points" button
    interpolate_clicked = pyqtSignal()       # "Interpolate" button (I)
    cancel_interp_clicked = pyqtSignal()     # "Stop" button (running interp)
    track_id_selected = pyqtSignal(object)   # new track id (int) or None
    add_cat_clicked = pyqtSignal(str)        # new category name
    rename_cat_clicked = pyqtSignal(int)     # cat_id to rename
    del_cat_clicked = pyqtSignal(int)        # cat_id to delete

    def __init__(self, coco: CocoState, parent=None):
        super().__init__(parent)
        self.coco = coco
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.source_label = QLabel("Source: —")
        self.source_label.setToolTip(
            "Current frame source (image file / folder).")
        layout.addWidget(self.source_label)

        self.cat_label = QLabel("Categories (click = preselect for next draw / 0-9; multi-select = autolabel only these):")
        layout.addWidget(self.cat_label)

        self.cat_list = QListWidget()
        # Highlighting 2+ rows (any multi-select) restricts an autolabel run
        # to those categories; a plain single click just preselects the
        # category for the next drawn box.
        self.cat_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.cat_list, 1)
        self.cat_list.itemClicked.connect(self._on_cat_clicked)
        self._rebuild_cat_list()
        self._preselected_cat_id: Optional[int] = None
        # True only while the highlight came from an explicit Ctrl/Shift
        # click. A plain single-click preselects a category for the next
        # drawn box but must NOT restrict an autolabel run to it (a 2+-row
        # highlight always restricts, however it was made).
        self._restrict_selection: bool = False

        # Add a new category: type a name, press Enter or click Add.
        add_cat_row = QHBoxLayout()
        self.add_cat_edit = QLineEdit()
        self.add_cat_edit.setPlaceholderText("new category name…")
        self.add_cat_edit.setToolTip(
            "Type a name and press Enter (or Add) to create a category. "
            "It gets the next free id and is preselected for the next draw.")
        self.add_cat_edit.returnPressed.connect(self._on_add_cat_entered)
        add_cat_row.addWidget(self.add_cat_edit, 1)
        self.btn_add_cat = QPushButton("Add")
        self.btn_add_cat.setToolTip("Create the category.")
        self.btn_add_cat.clicked.connect(self._on_add_cat_entered)
        add_cat_row.addWidget(self.btn_add_cat)
        layout.addLayout(add_cat_row)

        # Rename / delete the category selected in the list above.
        edit_cat_row = QHBoxLayout()
        self.btn_rename_cat = QPushButton("Rename")
        self.btn_rename_cat.setToolTip(
            "Rename the category selected in the list above.")
        self.btn_rename_cat.clicked.connect(self._on_rename_cat)
        edit_cat_row.addWidget(self.btn_rename_cat)
        self.btn_del_cat = QPushButton("Delete")
        self.btn_del_cat.setToolTip(
            "Delete the selected category (asks first; boxes using it "
            "are deleted too).")
        self.btn_del_cat.clicked.connect(self._on_del_cat)
        edit_cat_row.addWidget(self.btn_del_cat)
        layout.addLayout(edit_cat_row)

        # Boxes on current frame list.
        self.boxes_label = QLabel("Boxes on this frame:")
        layout.addWidget(self.boxes_label)
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(140)
        self.box_list.itemClicked.connect(self._on_box_list_clicked)
        layout.addWidget(self.box_list)

        # Recategorize the selected box: type a category id, press Enter.
        # Works for any id (not just 0-9 like the draw-time number keys).
        recat_row = QHBoxLayout()
        recat_row.addWidget(QLabel("Cat of selected:"))
        self.recat_edit = QLineEdit()
        self.recat_edit.setPlaceholderText("id, e.g. 13")
        self.recat_edit.setToolTip(
            "Select a box, type a category id, press Enter to reassign. "
            "Press C to focus this field.")
        self.recat_edit.returnPressed.connect(self._on_recat_entered)
        recat_row.addWidget(self.recat_edit, 1)
        layout.addLayout(recat_row)

        # Track id of the selected box: type a number + Enter to set it,
        # clear the field + Enter to unset. Auto-assigned by inheritance
        # from the nearest earlier annotated frame (global counter as
        # fallback); edit it when a track continues. The whole row is
        # hidden unless track ids are enabled (tracking.show_ids config).
        track_row = QHBoxLayout()
        self.track_row_label = QLabel("Track of selected:")
        track_row.addWidget(self.track_row_label)
        self.track_edit = QLineEdit()
        self.track_edit.setPlaceholderText("id, e.g. 2 (empty clears)")
        self.track_edit.setToolTip(
            "Select a box, type a track id, press Enter to set it. "
            "Clear the field and press Enter to unset. Press T to focus.")
        self.track_edit.returnPressed.connect(self._on_track_entered)
        track_row.addWidget(self.track_edit, 1)
        layout.addLayout(track_row)
        # Track ids are shown by default; the window applies the authoritative
        # value via set_track_ids_visible() from the tracking.show_ids config.
        self.show_track_ids: bool = True

        # Frame playback controls: play/pause + speed.
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Play / pause the frame timeline (Space).")
        self.btn_play.clicked.connect(self._on_play_clicked)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(QLabel("speed:"))
        self.combo_speed = QtWidgets.QComboBox()
        # Frame rate is index-based (one frame per tick), not timestamp-
        # based. 1x = 30 fps (33 ms/frame); the rest scale from there.
        for label, ms in [("0.25x", 133), ("0.5x", 67), ("1x", 33),
                          ("2x", 17), ("5x", 7), ("10x", 3)]:
            self.combo_speed.addItem(label, ms)
        self.combo_speed.setCurrentIndex(2)  # 1x default
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        play_row.addWidget(self.combo_speed, 1)
        layout.addLayout(play_row)

        # Big jumps, both directions.
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Jump:"))
        self.jump_buttons = []  # kept for config-driven hiding
        for delta in (-10, -5, 5, 10):
            sign = "+" if delta > 0 else ""
            btn = QPushButton(f"{sign}{delta}")
            btn.setToolTip(f"Jump {delta:+d} frames.")
            btn.clicked.connect(lambda _c=False, d=delta: self.nav_delta.emit(d))
            jump_row.addWidget(btn)
            self.jump_buttons.append(btn)
        jump_row.addStretch(1)
        layout.addLayout(jump_row)

        # Mark the current frame as annotated without drawing any boxes —
        # it is included in the output JSON's ``annotated_image_ids``.
        self.btn_mark_annotated = QPushButton("✔ Mark as annotated")
        self.btn_mark_annotated.setCheckable(True)
        self.btn_mark_annotated.setToolTip(
            "Count this frame as annotated even though it has no boxes: "
            "its index is written into the output JSON's "
            "annotated_image_ids list. Toggle again to unmark.")
        self.btn_mark_annotated.clicked.connect(
            self.toggle_annotated_clicked.emit)
        layout.addWidget(self.btn_mark_annotated)

        # Discard the current frame (both sides in stereo): its image
        # record(s) and annotations are excluded from the FINAL output
        # JSON. Toggle again to restore — reversible until the final save.
        self.btn_discard_image = QPushButton("🚫 Discard image")
        self.btn_discard_image.setCheckable(True)
        self.btn_discard_image.setToolTip(
            "Exclude this frame's image(s) and boxes from the FINAL "
            "output JSON (the _tmp progress file keeps everything, so "
            "toggling again restores it). Use for bad/misaligned frames.")
        self.btn_discard_image.clicked.connect(
            self.toggle_discard_clicked.emit)
        layout.addWidget(self.btn_discard_image)

        layout.addSpacing(12)

        # Interpolation controls.
        interp_row = QHBoxLayout()
        self.btn_keyframe = QPushButton("★ Keyframe")
        self.btn_keyframe.setCheckable(True)
        self.btn_keyframe.setToolTip(
            "Mark / unmark this frame as an interpolation keyframe (K). "
            "Keyframes with boxes become the preferred interpolation "
            "anchors.")
        self.btn_keyframe.clicked.connect(self.toggle_keyframe_clicked.emit)
        interp_row.addWidget(self.btn_keyframe)
        self.btn_interpolate = QPushButton("Interpolate (I)")
        self.btn_interpolate.setToolTip(
            "Fill the gap between the nearest labeled frames with "
            "optical-flow interpolated boxes. Frames that already have "
            "boxes are skipped. Ctrl+Z undoes the whole fill.")
        self.btn_interpolate.clicked.connect(self.interpolate_clicked.emit)
        interp_row.addWidget(self.btn_interpolate)
        self.btn_cancel_interp = QPushButton("Stop")
        self.btn_cancel_interp.setToolTip("Stop the running interpolation.")
        self.btn_cancel_interp.setEnabled(False)
        self.btn_cancel_interp.clicked.connect(
            self.cancel_interp_clicked.emit)
        interp_row.addWidget(self.btn_cancel_interp)
        layout.addLayout(interp_row)

        self.interp_status = QLabel("Interpolation: idle")
        self.interp_status.setWordWrap(True)
        layout.addWidget(self.interp_status)

        # SAM3 controls
        self.sam3_header = QLabel("SAM3 segmentation:")
        f = self.sam3_header.font()
        f.setBold(True)
        self.sam3_header.setFont(f)
        layout.addWidget(self.sam3_header)
        sam_layout = QHBoxLayout()
        self.btn_run_sam3 = QPushButton("Run SAM3 (all)")
        self.btn_run_sam3.setToolTip("Run SAM3 segmentation on every bbox on the current frame.")
        self.btn_run_sam3.clicked.connect(self.run_sam3_clicked.emit)
        sam_layout.addWidget(self.btn_run_sam3)
        self.btn_reseg = QPushButton("Re-seg sel (R)")
        self.btn_reseg.setToolTip("Re-run SAM3 on the selected bbox only.")
        self.btn_reseg.clicked.connect(self.resegment_clicked.emit)
        sam_layout.addWidget(self.btn_reseg)
        self.btn_cancel_sam3 = QPushButton("Cancel")
        self.btn_cancel_sam3.setToolTip(
            "Stop the running SAM3 job (finishes the current concept first).")
        self.btn_cancel_sam3.setEnabled(False)
        self.btn_cancel_sam3.clicked.connect(self.cancel_sam3_clicked.emit)
        sam_layout.addWidget(self.btn_cancel_sam3)
        layout.addLayout(sam_layout)

        point_row = QHBoxLayout()
        self.btn_add_points = QPushButton("🎯 Add points")
        self.btn_add_points.setCheckable(True)
        self.btn_add_points.setToolTip(
            "Point-prompt mode: while ON, left-click adds a positive point "
            "and right-click a negative point on the canvas. Points only "
            "accumulate — nothing runs until you press ▶ Segment points. "
            "Enter accepts the segmented object (next points start a new "
            "one), Esc cancels it. Toggle off to go back to draw/select.")
        self.btn_add_points.toggled.connect(self.point_seg_toggled.emit)
        point_row.addWidget(self.btn_add_points)
        self.btn_segment_points = QPushButton("▶ Segment points")
        self.btn_segment_points.setEnabled(False)  # needs ≥1 point
        self.btn_segment_points.setToolTip(
            "Run SAM3 once with ALL accumulated points (positive + "
            "negative). The selected category is passed as a text prompt: "
            "SAM3 detects all instances of that category and your points "
            "pick which one to keep (falls back to pure point prompting "
            "when nothing matches). Add more points and press again to "
            "refine the same mask. Category = preselected / last-used.")
        self.btn_segment_points.clicked.connect(
            self.segment_points_clicked.emit)
        point_row.addWidget(self.btn_segment_points)
        layout.addLayout(point_row)

        self.btn_sam3_all_frames = QPushButton("SAM3 ALL frames")
        self.btn_sam3_all_frames.setToolTip(
            "Background auto-annotate: run SAM3 on every box that has no "
            "mask yet, across ALL frames. Cancel anytime.")
        self.btn_sam3_all_frames.clicked.connect(self.sam3_all_frames_clicked.emit)
        layout.addWidget(self.btn_sam3_all_frames)

        self.btn_propagate = QPushButton("Propagate →")
        self.btn_propagate.setToolTip(
            "Propagate the selected boxes' tracks forward: SAM3 re-detects "
            "each object on every following frame (chained from the "
            "previous frame's box), keeping the same track id. Each track "
            "stops when its object is lost; multiple selected tracks run "
            "one at a time. Select one or more boxes first.")
        self.btn_propagate.setEnabled(False)  # needs a selected box
        self.btn_propagate.clicked.connect(self.propagate_clicked.emit)
        layout.addWidget(self.btn_propagate)

        layout.addSpacing(8)

        self.autolabel_header = QLabel("Autolabel:")
        f = self.autolabel_header.font()
        f.setBold(True)
        self.autolabel_header.setFont(f)
        layout.addWidget(self.autolabel_header)

        # Text-prompt autolabel: the detector finds objects by category name,
        # no drawn boxes needed; detections become editable boxes (with masks
        # for SAM3/Falcon). When categories are highlighted in the list above
        # (Ctrl/Shift, or any multi-select), only those are detected —
        # otherwise every category. set_autolabel_detector() rewrites the
        # header/tooltips with the active backend's name.
        self.btn_autolabel_frame = QPushButton("Autolabel frame")
        self.btn_autolabel_frame.clicked.connect(
            self.autolabel_frame_clicked.emit)
        layout.addWidget(self.btn_autolabel_frame)
        self.btn_autolabel_all = QPushButton("Autolabel ALL frames")
        self.btn_autolabel_all.clicked.connect(self.autolabel_all_clicked.emit)
        layout.addWidget(self.btn_autolabel_all)
        self.set_autolabel_detector("sam3")

        self.btn_masks = QPushButton("Masks: ON")
        self.btn_masks.setCheckable(True)
        self.btn_masks.setChecked(True)
        self.btn_masks.setToolTip("Toggle mask overlay visibility (M).")
        self.btn_masks.clicked.connect(self._on_masks_toggled)
        layout.addWidget(self.btn_masks)

        # Mask overlay opacity (percent).
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Mask opacity:"))
        self.opacity_slider = QSlider(_QT_HORZ)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(47)  # matches canvas default alpha 120/255
        self.opacity_slider.setToolTip("Mask overlay opacity (0-100%).")
        self.opacity_slider.valueChanged.connect(self.mask_opacity_changed.emit)
        op_row.addWidget(self.opacity_slider, 1)
        self.opacity_value_label = QLabel("47%")
        self.opacity_value_label.setMinimumWidth(36)
        op_row.addWidget(self.opacity_value_label)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_value_label.setText(f"{v}%"))
        layout.addLayout(op_row)

        self.sam3_status = QLabel("SAM3: idle")
        layout.addWidget(self.sam3_status)

        # Annotation coverage: how many frames have at least one box.
        self.annot_progress = QProgressBar()
        self.annot_progress.setMinimum(0)
        self.annot_progress.setMaximum(1)
        self.annot_progress.setValue(0)
        self.annot_progress.setFormat("Annotated: %v/%m (%p%)")
        self.annot_progress.setToolTip(
            "Frames with at least one box. Press U to jump to the next "
            "frame that still needs labels.")
        layout.addWidget(self.annot_progress)

        self.frame_slider = QSlider(_QT_HORZ)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_moved.emit)
        self.frame_slider.sliderReleased.connect(self.slider_released.emit)
        layout.addWidget(self.frame_slider)

        self.info_label = QLabel("Frame: -\nTimestamp: -")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.help_label = QLabel(
            "<b>Keys (work anywhere):</b><br>"
            "D = delete sel &nbsp; A = draw &nbsp; N = next &nbsp; B = back<br>"
            "X = discard all &nbsp; S = save+quit &nbsp; Q = quit<br>"
            "0-9 = pick cat (when drawing) &nbsp; + / - = zoom &nbsp; F = fit<br>"
            "M = toggle masks &nbsp; R = re-seg sel &nbsp; Space = play/pause<br>"
            "Z = zoom to sel &nbsp; Ctrl+Z = undo &nbsp; Ctrl+Shift+Z = redo<br>"
            "Ctrl+A = select all &nbsp; Esc = clear sel / quit<br>"
            "U = jump to next unlabeled frame &nbsp; C = focus cat-id field<br>"
            "T = focus track-id field &nbsp; K = toggle keyframe<br>"
            "I = interpolate between nearest labeled frames<br>"
            "<i>Click a category first to preselect it for the next draw.<br>"
            "New draws reuse the previous box's category automatically.</i>"
        )
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

    # ---- box list ---- #

    def set_boxes(self, boxes: List[Dict[str, Any]]) -> None:
        """Rebuild the box list display. boxes: list of dicts (see CanvasWidget.set_boxes)."""
        self.box_list.blockSignals(True)
        self.box_list.clear()
        for i, b in enumerate(boxes):
            x, y, w, h = b["bbox"]
            tid = b.get("track_id")
            ttxt = (f" T{tid}" if (tid is not None and self.show_track_ids)
                    else "")
            itxt = " ~interp" if b.get("interp") else ""
            txt = (f"{b.get('cat_name','?')}{ttxt}{itxt} (id: {b['id']})  "
                   f"[{int(x)},{int(y)},{int(w)},{int(h)}]")
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, i)  # store the box index, not ann_id
            self.box_list.addItem(it)
        self.box_list.blockSignals(False)
        self.boxes_label.setText(f"Boxes on this frame ({len(boxes)}):")

    def highlight_box_row(self, box_idx: int) -> None:
        """Sync the list's current row with the canvas selection. -1 = none."""
        self.box_list.blockSignals(True)
        if 0 <= box_idx < self.box_list.count():
            self.box_list.setCurrentRow(box_idx)
        else:
            self.box_list.setCurrentRow(-1)
        self.box_list.blockSignals(False)

    def _on_box_list_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self.box_selected.emit(int(idx))

    # ---- play/pause ---- #

    def _on_play_clicked(self) -> None:
        on = self.btn_play.isChecked()
        self.btn_play.setText("⏸ Pause" if on else "▶ Play")
        self.play_pause_clicked.emit()

    def _on_speed_changed(self) -> None:
        ms = self.combo_speed.currentData()
        if ms is not None:
            self.play_speed_changed.emit(int(ms))

    def set_playing(self, playing: bool) -> None:
        """Sync the play/pause button from external triggers (e.g. Space key)."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(playing)
        self.btn_play.setText("⏸ Pause" if playing else "▶ Play")
        self.btn_play.blockSignals(False)

    # ---- masks + sam3 ---- #

    def _on_masks_toggled(self) -> None:
        on = self.btn_masks.isChecked()
        self.btn_masks.setText("Masks: ON" if on else "Masks: OFF")
        self.toggle_masks_clicked.emit()

    def set_sam3_status(self, text: str) -> None:
        self.sam3_status.setText(text)

    def set_source(self, path: str) -> None:
        """Show the current frame-source path (elided; full path in tooltip)."""
        disp = path or "—"
        self.source_label.setToolTip(path or "No source loaded")
        metrics = self.source_label.fontMetrics()
        self.source_label.setText(metrics.elidedText(
            f"Source: {disp}", Qt.TextElideMode.ElideMiddle, 320))

    def set_sam3_running(self, running: bool) -> None:
        """Enable Cancel while SAM3 is busy. The per-frame run buttons stay
        enabled — presses while busy are queued by the window. Only the
        all-frames batch refuses to start while something is running."""
        self.btn_run_sam3.setEnabled(True)
        self.btn_reseg.setEnabled(True)
        self.btn_autolabel_frame.setEnabled(True)
        self.btn_sam3_all_frames.setEnabled(not running)
        self.btn_autolabel_all.setEnabled(not running)
        self.btn_cancel_sam3.setEnabled(running)

    def set_interp_status(self, text: str) -> None:
        self.interp_status.setText(text)

    def set_interp_running(self, running: bool) -> None:
        """Enable Stop and disable the trigger buttons while busy."""
        self.btn_interpolate.setEnabled(not running)
        self.btn_keyframe.setEnabled(not running)
        self.btn_cancel_interp.setEnabled(running)

    def set_annotated_progress(self, annotated: int, total: int) -> None:
        self.annot_progress.setMaximum(max(total, 1))
        self.annot_progress.setValue(min(annotated, max(total, 1)))

    def _on_recat_entered(self) -> None:
        txt = self.recat_edit.text().strip()
        if txt.isdigit():
            self.recat_selected.emit(int(txt))
            self.recat_edit.clear()
        else:
            self.recat_edit.selectAll()

    def _on_track_entered(self) -> None:
        txt = self.track_edit.text().strip()
        if txt == "":
            self.track_id_selected.emit(None)
            self.track_edit.clear()
        elif txt.isdigit():
            self.track_id_selected.emit(int(txt))
            self.track_edit.clear()
        else:
            self.track_edit.selectAll()

    def prefill_track(self, track_id: Optional[int]) -> None:
        """Show a box's track id (or empty) in the track field."""
        self.track_edit.setText("" if track_id is None else str(track_id))

    def set_mask_opacity(self, pct: int) -> None:
        """Sync the opacity slider without re-emitting (e.g. on restore)."""
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(pct)
        self.opacity_value_label.setText(f"{pct}%")
        self.opacity_slider.blockSignals(False)

    # ---- categories ---- #

    def _on_add_cat_entered(self) -> None:
        name = self.add_cat_edit.text().strip()
        if name:
            self.add_cat_clicked.emit(name)
            self.add_cat_edit.clear()

    def _selected_cat_id(self) -> Optional[int]:
        item = self.cat_list.currentItem()
        if item is None:
            return None
        cat_id = item.data(Qt.UserRole)
        return int(cat_id) if cat_id is not None else None

    def _on_rename_cat(self) -> None:
        cat_id = self._selected_cat_id()
        if cat_id is not None:
            self.rename_cat_clicked.emit(cat_id)

    def _on_del_cat(self) -> None:
        cat_id = self._selected_cat_id()
        if cat_id is not None:
            self.del_cat_clicked.emit(cat_id)

    def _on_cat_clicked(self, item: QListWidgetItem) -> None:
        cat_id = item.data(Qt.UserRole)
        if cat_id is not None:
            cat_id = int(cat_id)
            self._preselected_cat_id = cat_id
            # Only an explicit Ctrl/Shift click marks the highlight as an
            # autolabel restriction; a plain click just preselects.
            self._restrict_selection = bool(
                QtWidgets.QApplication.keyboardModifiers()
                & (Qt.KeyboardModifier.ControlModifier
                   | Qt.KeyboardModifier.ShiftModifier))
            # Visual hint: bold the clicked row.
            f = item.font()
            f.setBold(True)
            item.setFont(f)
            # Un-bold other rows.
            for i in range(self.cat_list.count()):
                if i is not self.cat_list.row(item):
                    other = self.cat_list.item(i)
                    of = other.font()
                    if of.bold():
                        of.setBold(False)
                        other.setFont(of)
            self.preselect_cat.emit(cat_id)
            self.cat_clicked.emit(cat_id)

    def get_preselected_cat_id(self) -> Optional[int]:
        return self._preselected_cat_id

    def set_autolabel_detector(self, detector: str) -> None:
        """Retitle the autolabel section for the active backend."""
        label = DETECTOR_LABELS.get(detector, detector)
        if detector == "owlv2_exemplar":
            how = ("1-shot exemplar detection — the box selected on the "
                   "canvas is the visual query")
        else:
            how = "text-prompt detection"
        masks = (" with masks" if detector in DETECTORS_WITH_MASKS else "")
        self.autolabel_header.setText(f"{label} Autolabel:")
        self.btn_autolabel_frame.setToolTip(
            f"{label} {how} on this frame; detections are added as editable "
            f"boxes{masks} (NMS-deduplicated). Prompts with the categories "
            "highlighted in the list above (multi-select), or ALL categories "
            "when none are highlighted.")
        self.btn_autolabel_all.setToolTip(
            f"Background {label} autolabel on every frame, for the "
            "categories highlighted in the list above (or ALL categories "
            "when none are highlighted). Cancel anytime.")

    def get_selected_cat_ids(self) -> List[int]:
        """All highlighted categories, sorted.

        Empty unless the highlight is an explicit restriction: a Ctrl/Shift
        click, or ANY multi-selection (2+ rows). A plain single-click just
        preselects a category for drawing and must not silently restrict an
        autolabel run to it."""
        items = self.cat_list.selectedItems()
        if not self._restrict_selection and len(items) < 2:
            return []
        ids = []
        for item in items:
            cid = item.data(Qt.UserRole)
            if cid is not None:
                ids.append(int(cid))
        return sorted(set(ids))

    def _rebuild_cat_list(self) -> None:
        self.cat_list.clear()
        for cat in sorted(self.coco.categories, key=lambda c: c["id"]):
            txt = f"{cat['id']} — {cat['name']}"
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, cat["id"])
            self.cat_list.addItem(it)

    # Widget groups that can be hidden via the config's "ui": {"hide": [...]}.
    # Values are attribute names on this panel; an attribute may be a single
    # widget or a list of widgets.
    _HIDEABLE = {
        "keyframe": ["btn_keyframe"],
        "interpolate": ["btn_interpolate", "btn_cancel_interp",
                        "interp_status"],
        "jump": ["jump_buttons"],
        "sam3_run": ["sam3_header", "btn_run_sam3", "btn_reseg",
                     "btn_cancel_sam3", "btn_propagate"],
        "sam3_all_frames": ["btn_sam3_all_frames"],
        "autolabel": ["autolabel_header", "btn_autolabel_frame",
                      "btn_autolabel_all"],
        "masks": ["btn_masks", "opacity_slider", "opacity_value_label"],
        "play": ["btn_play", "combo_speed"],
    }

    def set_hidden_groups(self, groups: List[str]) -> None:
        """Hide the widget groups named in `groups` (see _HIDEABLE) and
        re-show any known group not listed, so a runtime config reload can
        both hide and un-hide."""
        want = set(groups)
        for g, attrs in self._HIDEABLE.items():
            for attr in attrs:
                w = getattr(self, attr, None)
                if w is None:
                    continue
                for widget in (w if isinstance(w, list) else [w]):
                    widget.setVisible(g not in want)
        for g in sorted(want - set(self._HIDEABLE)):
            print(f"⚠️ Unknown ui.hide group: {g!r} "
                  f"(known: {sorted(self._HIDEABLE)})")

    def set_track_ids_visible(self, visible: bool) -> None:
        """Show/hide everything track-id related: the "Track of selected"
        row and the " T<id>" suffix in the box list (the canvas labels are
        gated separately via CanvasWidget.show_track_ids)."""
        self.show_track_ids = visible
        self.track_row_label.setVisible(visible)
        self.track_edit.setVisible(visible)

    def set_slider_max(self, n: int) -> None:
        self.frame_slider.setMaximum(max(0, n - 1))

    def set_slider(self, idx: int) -> None:
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)

    def set_info(self, idx: int, total: int, ts_ns: int,
                 boxes: int, ts_real: bool = True) -> None:
        # Timestamp-named image series show the real timestamp; otherwise
        # the synthetic ts is meaningless, so show the plain image index.
        pos = f"ts_ns: {ts_ns}" if ts_real else f"index: {idx}"
        self.info_label.setText(
            f"Frame: {idx + 1}/{total}<br>"
            f"{pos}<br>"
            f"Boxes on frame: {boxes}"
        )



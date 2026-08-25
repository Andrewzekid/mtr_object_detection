"""Config dialog (runtime settings editor)."""

import json
from typing import Any, Dict

from ..qt_compat import QtWidgets  # first: enum shims
from ..qt_compat import (  # noqa: F401
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout,
)

from .side_panel import SidePanel  # noqa: F401  (hide-group metadata)


# ---------------------------------------------------------------------------
# Config dialog (runtime settings editor)
# ---------------------------------------------------------------------------

class ConfigDialog(QtWidgets.QDialog):
    """Settings editor opened from the ⚙ Config button (File → Config…).

    Contains an "Advanced settings" checkbox (off by default) that gates the
    interpolation/tracking sections and the keyframe/interpolate hide
    checkboxes, plus checkboxes for hiding UI groups and SAM3 /
    mask-opacity fields. Buttons:
      - Load from file… — read a JSON config, fill the widgets, apply it.
      - Apply — apply the current widget state without saving.
      - Save… — apply, then write the widget state to a JSON file.
      - Close — dismiss the dialog.
    The dialog is pre-filled from the live window state when opened, and
    collects the same config schema as scripts/config/label_review.example.json.
    """

    _HIDE_LABELS = {
        "keyframe": "Keyframe controls",
        "interpolate": "Interpolate controls",
        "jump": "Jump buttons (−10/−5/+5/+10)",
        "sam3_run": "SAM3 run buttons",
        "sam3_all_frames": "SAM3 all-frames button",
        "autolabel": "Autolabel buttons",
        "masks": "Mask toggle + opacity slider",
        "play": "Play controls",
    }

    def __init__(self, win: "ReviewWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Config settings")
        self.setModal(True)
        self.resize(480, 640)
        root = QVBoxLayout(self)

        # Scrollable content (the dialog has grown past one screen).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget()
        body = QVBoxLayout(content)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # --- UI visibility -------------------------------------------------
        ui_box = QtWidgets.QGroupBox("Hide UI elements")
        ui_form = QVBoxLayout(ui_box)
        ui_form.addWidget(QLabel("Checked = hidden"))
        self.hide_checks: Dict[str, QCheckBox] = {}
        for key, label in self._HIDE_LABELS.items():
            cb = QCheckBox(label)
            self.hide_checks[key] = cb
            ui_form.addWidget(cb)
        body.addWidget(ui_box)

        # --- Interpolation -------------------------------------------------
        self.interp_box = QtWidgets.QGroupBox("Interpolation")
        interp_form = QtWidgets.QFormLayout(self.interp_box)
        self.combo_flow = QtWidgets.QComboBox()
        self.combo_flow.addItems(["dis", "klt", "farneback"])
        interp_form.addRow("Flow method", self.combo_flow)
        self.combo_cam = QtWidgets.QComboBox()
        self.combo_cam.addItems(["none", "global"])
        interp_form.addRow("Camera model", self.combo_cam)
        self.spin_match_frac = QtWidgets.QDoubleSpinBox()
        self.spin_match_frac.setRange(0.01, 1.0)
        self.spin_match_frac.setSingleStep(0.05)
        self.spin_match_frac.setDecimals(2)
        interp_form.addRow("Match max dist frac", self.spin_match_frac)
        self.check_confirm_mismatch = QCheckBox("Confirm on mismatch")
        interp_form.addRow(self.check_confirm_mismatch)
        self.check_show_track_ids = QCheckBox(
            "Show track ids (T<id> labels + track edit field)")
        self.check_show_track_ids.setToolTip(
            "Unchecked (default): track ids are not displayed — box labels\n"
            "and the box list show only the category and annotation id, and\n"
            "the \"Track of selected\" row is hidden. Checked: show them.")
        interp_form.addRow(self.check_show_track_ids)
        body.addWidget(self.interp_box)

        # --- SAM3 ----------------------------------------------------------
        sam3_box = QtWidgets.QGroupBox("SAM3")
        sam3_form = QtWidgets.QFormLayout(sam3_box)
        self.combo_device = QtWidgets.QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])
        sam3_form.addRow("Device", self.combo_device)
        self.edit_sam3_model = QtWidgets.QLineEdit()
        self.edit_sam3_model.setPlaceholderText(
            "default: core/sam3/models/sam3-model/sam3.pt")
        self.edit_sam3_model.setToolTip(
            "Path to the SAM3 weights (.pt). Leave empty for the default\n"
            "location. Takes effect on the next SAM3 run.")
        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.clicked.connect(self._browse_sam3_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.edit_sam3_model, 1)
        model_row.addWidget(btn_browse_model)
        sam3_form.addRow("Model path", model_row)
        self.spin_sam3_conf = QtWidgets.QDoubleSpinBox()
        self.spin_sam3_conf.setRange(0.0, 1.0)
        self.spin_sam3_conf.setSingleStep(0.05)
        self.spin_sam3_conf.setDecimals(2)
        sam3_form.addRow("Confidence", self.spin_sam3_conf)
        self.check_auto_segment = QCheckBox("Auto-segment on box add")
        sam3_form.addRow(self.check_auto_segment)
        self.spin_min_poly_area = QtWidgets.QSpinBox()
        self.spin_min_poly_area.setRange(0, 1000000)
        self.spin_min_poly_area.setSuffix(" px²")
        self.spin_min_poly_area.setToolTip(
            "Mask contours smaller than this area are dropped when saving\n"
            "polygons — filters the scattered specks SAM3 occasionally\n"
            "produces. The largest contour is always kept;\n"
            "0 keeps every contour.")
        sam3_form.addRow("Min polygon area", self.spin_min_poly_area)
        self.spin_nms_iou = QtWidgets.QDoubleSpinBox()
        self.spin_nms_iou.setRange(0.0, 1.0)
        self.spin_nms_iou.setSingleStep(0.05)
        self.spin_nms_iou.setDecimals(2)
        self.spin_nms_iou.setToolTip(
            "Autolabel NMS: after a text-prompt run, overlapping detections\n"
            "of the SAME category with IoU above this are deduplicated\n"
            "(highest confidence kept). SAM3 often returns several masks\n"
            "for one object. 1.0 disables dedup.")
        sam3_form.addRow("Autolabel NMS IoU", self.spin_nms_iou)
        self.combo_propagate_method = QtWidgets.QComboBox()
        self.combo_propagate_method.addItem(
            "Memory bank (SAM3 video session)", "memory")
        self.combo_propagate_method.addItem(
            "Frame-by-frame chain (IoU)", "chain")
        self.combo_propagate_method.setToolTip(
            "How 'Propagate →' tracks objects forward:\n"
            "- Memory bank: one SAM3 video session per side; all selected\n"
            "  boxes tracked together with SAM3's internal memory. A lost\n"
            "  object may recover on later frames. Builds a temp clip of\n"
            "  the remaining range first.\n"
            "- Frame-by-frame chain: re-detects each object on every frame\n"
            "  (previous box as prompt, IoU-gated). A track stops\n"
            "  permanently at the first frame with no detection.")
        sam3_form.addRow("Propagate method", self.combo_propagate_method)
        self.spin_propagate_min_iou = QtWidgets.QDoubleSpinBox()
        self.spin_propagate_min_iou.setRange(0.0, 1.0)
        self.spin_propagate_min_iou.setSingleStep(0.05)
        self.spin_propagate_min_iou.setDecimals(2)
        self.spin_propagate_min_iou.setToolTip(
            "Chain mode only: a detection with IoU to the previous frame's\n"
            "box below this is rejected (track reported lost) — stops the\n"
            "chain latching onto an unrelated similar object.")
        sam3_form.addRow("Propagate min IoU", self.spin_propagate_min_iou)
        self.spin_propagate_seed_iou = QtWidgets.QDoubleSpinBox()
        self.spin_propagate_seed_iou.setRange(0.0, 1.0)
        self.spin_propagate_seed_iou.setSingleStep(0.05)
        self.spin_propagate_seed_iou.setDecimals(2)
        self.spin_propagate_seed_iou.setToolTip(
            "Chain mode only: a detection must also overlap the SEED box\n"
            "by at least this much — anchors the chain against drift under\n"
            "large camera motion.")
        sam3_form.addRow("Propagate min seed IoU",
                         self.spin_propagate_seed_iou)
        body.addWidget(sam3_box)

        # --- Autolabel detector ------------------------------------------
        al_box = QtWidgets.QGroupBox("Autolabel detector")
        al_form = QtWidgets.QFormLayout(al_box)
        self.combo_autolabel_detector = QtWidgets.QComboBox()
        self.combo_autolabel_detector.addItem(
            "SAM3 (text-prompt boxes + masks)", "sam3")
        self.combo_autolabel_detector.addItem(
            "OWLv2 (zero-shot boxes)", "owlv2")
        self.combo_autolabel_detector.addItem(
            "OWLv2 exemplar (1-shot, uses selected box)", "owlv2_exemplar")
        self.combo_autolabel_detector.addItem(
            "Grounding DINO (zero-shot boxes)", "grounding_dino")
        self.combo_autolabel_detector.addItem(
            "Florence-2 (phrase-grounding boxes)", "florence2")
        self.combo_autolabel_detector.addItem(
            "Falcon Perception (boxes + masks)", "falcon")
        self.combo_autolabel_detector.setToolTip(
            "Which model the 'Autolabel frame' / 'Autolabel ALL frames'\n"
            "buttons use.\n"
            "- SAM3: text-prompt detection with segmentation masks.\n"
            "- OWLv2: zero-shot text-prompt box detection (faster, no\n"
            "  masks). Default checkpoint: google/owlv2-large-patch14-\n"
            "  ensemble.\n"
            "- OWLv2 exemplar: 1-shot image-guided detection — the\n"
            "  currently selected box is cropped out and used as the\n"
            "  visual query; matching objects get its category.\n"
            "- Grounding DINO: zero-shot text-prompt boxes (no masks).\n"
            "  Default checkpoint: IDEA-Research/grounding-dino-base.\n"
            "- Florence-2: phrase-grounding boxes, one prompt per\n"
            "  category (no masks, no confidence scores).\n"
            "  Default checkpoint: microsoft/Florence-2-large.\n"
            "- Falcon Perception: open-vocabulary grounding with real\n"
            "  instance masks (no confidence scores).\n"
            "  Default checkpoint: tiiuae/Falcon-Perception.")
        al_form.addRow("Detector", self.combo_autolabel_detector)
        self.edit_owlv2_model = QtWidgets.QLineEdit()
        self.edit_owlv2_model.setPlaceholderText(
            "default: google/owlv2-large-patch14-ensemble")
        self.edit_owlv2_model.setToolTip(
            "OWLv2 checkpoint (HuggingFace model id or local path).\n"
            "Takes effect on the next autolabel run.")
        al_form.addRow("OWLv2 model", self.edit_owlv2_model)
        self.spin_owlv2_conf = QtWidgets.QDoubleSpinBox()
        self.spin_owlv2_conf.setRange(0.0, 1.0)
        self.spin_owlv2_conf.setSingleStep(0.05)
        self.spin_owlv2_conf.setDecimals(2)
        self.spin_owlv2_conf.setToolTip(
            "OWLv2 detection confidence threshold.")
        al_form.addRow("OWLv2 confidence", self.spin_owlv2_conf)
        self.edit_gdino_model = QtWidgets.QLineEdit()
        self.edit_gdino_model.setPlaceholderText(
            "default: IDEA-Research/grounding-dino-base")
        self.edit_gdino_model.setToolTip(
            "Grounding DINO checkpoint (HuggingFace model id or local\n"
            "path). Takes effect on the next autolabel run.")
        al_form.addRow("Grounding DINO model", self.edit_gdino_model)
        self.spin_gdino_conf = QtWidgets.QDoubleSpinBox()
        self.spin_gdino_conf.setRange(0.0, 1.0)
        self.spin_gdino_conf.setSingleStep(0.05)
        self.spin_gdino_conf.setDecimals(2)
        self.spin_gdino_conf.setToolTip(
            "Grounding DINO box threshold (detection confidence).")
        al_form.addRow("Grounding DINO confidence", self.spin_gdino_conf)
        self.edit_florence2_model = QtWidgets.QLineEdit()
        self.edit_florence2_model.setPlaceholderText(
            "default: microsoft/Florence-2-large")
        self.edit_florence2_model.setToolTip(
            "Florence-2 checkpoint (HuggingFace model id or local path).\n"
            "Takes effect on the next autolabel run.")
        al_form.addRow("Florence-2 model", self.edit_florence2_model)
        self.edit_falcon_model = QtWidgets.QLineEdit()
        self.edit_falcon_model.setPlaceholderText(
            "default: tiiuae/Falcon-Perception")
        self.edit_falcon_model.setToolTip(
            "Falcon Perception checkpoint (HuggingFace model id or local\n"
            "path). Takes effect on the next autolabel run.")
        al_form.addRow("Falcon model", self.edit_falcon_model)
        body.addWidget(al_box)

        # --- Mask opacity --------------------------------------------------
        mask_box = QtWidgets.QGroupBox("Masks")
        mask_form = QtWidgets.QFormLayout(mask_box)
        self.spin_opacity = QtWidgets.QSpinBox()
        self.spin_opacity.setRange(0, 100)
        self.spin_opacity.setSuffix(" %")
        mask_form.addRow("Mask opacity", self.spin_opacity)
        body.addWidget(mask_box)

        # --- Display -------------------------------------------------------
        display_box = QtWidgets.QGroupBox("Display")
        display_form = QtWidgets.QFormLayout(display_box)
        self.spin_max_image_dim = QtWidgets.QSpinBox()
        self.spin_max_image_dim.setRange(0, 16384)
        self.spin_max_image_dim.setSingleStep(128)
        self.spin_max_image_dim.setSuffix(" px")
        self.spin_max_image_dim.setSpecialValueText("Original")
        self.spin_max_image_dim.setToolTip(
            "Maximum displayed image dimension (width or height) in\n"
            "pixels. Larger frames are downscaled for display only —\n"
            "box/mask coordinates stay in original image pixels.\n"
            "0 (\"Original\") disables downscaling.")
        display_form.addRow("Max image size", self.spin_max_image_dim)
        body.addWidget(display_box)

        # --- Tracking ------------------------------------------------------
        self.track_box = QtWidgets.QGroupBox("Tracking")
        track_form = QtWidgets.QFormLayout(self.track_box)
        self.check_sticky_ids = QCheckBox(
            "Sticky track ids (inherit from previous frame)")
        self.check_sticky_ids.setToolTip(
            "Unchecked (default): every new box gets a fresh auto-increment "
            "track id.\nChecked: the k-th box drawn on a frame inherits the "
            "k-th track id\nfrom the nearest earlier annotated frame (what "
            "interpolation expects).")
        track_form.addRow(self.check_sticky_ids)
        body.addWidget(self.track_box)

        # --- Advanced toggle (at the very bottom of the settings) ----------
        self.check_advanced = QCheckBox(
            "Advanced settings (interpolation, tracking, keyframe/interpolate "
            "controls)")
        self.check_advanced.setToolTip(
            "Unchecked (default): the interpolation and tracking sections\n"
            "above are hidden, and the keyframe/interpolate buttons stay\n"
            "hidden in the side panel.")
        body.addWidget(self.check_advanced)

        # --- Buttons -------------------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load from file…")
        self.btn_apply = QPushButton("Apply")
        self.btn_save = QPushButton("Save…")
        self.btn_close = QPushButton("Close")
        # QPushButton auto-defaults inside a QDialog: without this, pressing
        # Enter in any field (e.g. the SAM3 confidence spinbox) fires the
        # first button — opening the Load file dialog unexpectedly.
        for b in (self.btn_load, self.btn_apply, self.btn_save,
                  self.btn_close):
            b.setAutoDefault(False)
            b.setDefault(False)
        btn_row.addWidget(self.btn_load)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_close.clicked.connect(self.accept)
        self.check_advanced.toggled.connect(self._update_advanced_visibility)

        self._prefill_from_window()
        self._update_advanced_visibility()

    def _update_advanced_visibility(self) -> None:
        """Show the interpolation/tracking sections and the keyframe +
        interpolate hide checkboxes only in advanced mode."""
        adv = self.check_advanced.isChecked()
        self.interp_box.setVisible(adv)
        self.track_box.setVisible(adv)
        self.hide_checks["interpolate"].setVisible(adv)
        self.hide_checks["keyframe"].setVisible(adv)
        self.adjustSize()

    # -- prefill / collect ---------------------------------------------------

    def _is_group_hidden(self, key: str) -> bool:
        attrs = SidePanel._HIDEABLE.get(key, [])
        if not attrs:
            return False
        w = getattr(self.win.side, attrs[0], None)
        if isinstance(w, list):
            w = w[0] if w else None
        return bool(w is not None and w.isHidden())

    def _prefill_from_window(self) -> None:
        """Fill the widgets from the live window state."""
        win = self.win
        self.check_advanced.setChecked(win.advanced_ui)
        for key, cb in self.hide_checks.items():
            cb.setChecked(self._is_group_hidden(key))
        self.combo_flow.setCurrentText(win.interp_flow_method)
        self.combo_cam.setCurrentText(win.interp_camera_model)
        self.spin_match_frac.setValue(win.interp_match_frac)
        self.check_confirm_mismatch.setChecked(win.interp_confirm_mismatch)
        self.check_show_track_ids.setChecked(win.show_track_ids)
        self.combo_device.setCurrentText(win.sam3_device)
        self.edit_sam3_model.setText(win.sam3_model or "")
        self.spin_sam3_conf.setValue(win.sam3_conf)
        self.check_auto_segment.setChecked(win.auto_segment)
        self.spin_min_poly_area.setValue(int(win.coco.min_polygon_area))
        self.spin_nms_iou.setValue(win.sam3_nms_iou)
        idx = self.combo_autolabel_detector.findData(
            getattr(win, "autolabel_detector", "sam3"))
        self.combo_autolabel_detector.setCurrentIndex(max(0, idx))
        self.edit_owlv2_model.setText(
            getattr(win, "owlv2_model", "") or "")
        self.spin_owlv2_conf.setValue(
            getattr(win, "owlv2_conf", 0.3))
        self.edit_gdino_model.setText(
            getattr(win, "gdino_model", "") or "")
        self.spin_gdino_conf.setValue(
            getattr(win, "gdino_conf", 0.35))
        self.edit_florence2_model.setText(
            getattr(win, "florence2_model", "") or "")
        self.edit_falcon_model.setText(
            getattr(win, "falcon_model", "") or "")
        idx = self.combo_propagate_method.findData(
            getattr(win, "propagate_method", "memory"))
        self.combo_propagate_method.setCurrentIndex(max(0, idx))
        self.spin_propagate_min_iou.setValue(
            getattr(win, "propagate_min_iou", 0.3))
        self.spin_propagate_seed_iou.setValue(
            getattr(win, "propagate_min_seed_iou", 0.2))
        self.spin_opacity.setValue(win.side.opacity_slider.value())
        self.spin_max_image_dim.setValue(win.display_max_dim)
        self.check_sticky_ids.setChecked(win.coco.sticky_track_ids)

    def _prefill_from_config(self, cfg: Dict[str, Any]) -> None:
        """Fill the widgets from a config dict (same schema as _collect)."""
        interp = cfg.get("interpolation", {})
        if interp.get("flow_method") in ("dis", "klt", "farneback"):
            self.combo_flow.setCurrentText(interp["flow_method"])
        if interp.get("camera_model") in ("none", "global"):
            self.combo_cam.setCurrentText(interp["camera_model"])
        if "match_max_dist_frac" in interp:
            self.spin_match_frac.setValue(float(interp["match_max_dist_frac"]))
        if "confirm_mismatch" in interp:
            self.check_confirm_mismatch.setChecked(
                bool(interp["confirm_mismatch"]))
        sam3 = cfg.get("sam3", {})
        if sam3.get("device") in ("auto", "cuda", "cpu"):
            self.combo_device.setCurrentText(sam3["device"])
        if sam3.get("model"):
            self.edit_sam3_model.setText(str(sam3["model"]))
        if "conf" in sam3:
            self.spin_sam3_conf.setValue(float(sam3["conf"]))
        if "auto_segment" in sam3:
            self.check_auto_segment.setChecked(bool(sam3["auto_segment"]))
        if "min_polygon_area" in sam3:
            self.spin_min_poly_area.setValue(
                max(0, int(sam3["min_polygon_area"])))
        if "autolabel_nms_iou" in sam3:
            self.spin_nms_iou.setValue(
                max(0.0, min(1.0, float(sam3["autolabel_nms_iou"]))))
        if sam3.get("propagate_method") in ("memory", "chain"):
            self.combo_propagate_method.setCurrentIndex(
                self.combo_propagate_method.findData(
                    sam3["propagate_method"]))
        if "propagate_min_iou" in sam3:
            self.spin_propagate_min_iou.setValue(
                max(0.0, min(1.0, float(sam3["propagate_min_iou"]))))
        if "propagate_min_seed_iou" in sam3:
            self.spin_propagate_seed_iou.setValue(
                max(0.0, min(1.0, float(sam3["propagate_min_seed_iou"]))))
        autolabel = cfg.get("autolabel", {})
        if autolabel.get("detector") in ("sam3", "owlv2", "owlv2_exemplar",
                                         "grounding_dino", "florence2",
                                         "falcon"):
            self.combo_autolabel_detector.setCurrentIndex(
                self.combo_autolabel_detector.findData(
                    autolabel["detector"]))
        if autolabel.get("owlv2_model"):
            self.edit_owlv2_model.setText(str(autolabel["owlv2_model"]))
        if "owlv2_conf" in autolabel:
            self.spin_owlv2_conf.setValue(
                max(0.0, min(1.0, float(autolabel["owlv2_conf"]))))
        if autolabel.get("gdino_model"):
            self.edit_gdino_model.setText(str(autolabel["gdino_model"]))
        if "gdino_conf" in autolabel:
            self.spin_gdino_conf.setValue(
                max(0.0, min(1.0, float(autolabel["gdino_conf"]))))
        if autolabel.get("florence2_model"):
            self.edit_florence2_model.setText(
                str(autolabel["florence2_model"]))
        if autolabel.get("falcon_model"):
            self.edit_falcon_model.setText(str(autolabel["falcon_model"]))
        ui = cfg.get("ui", {})
        if "advanced" in ui:
            self.check_advanced.setChecked(bool(ui["advanced"]))
            self._update_advanced_visibility()
        if "hide" in ui:
            hidden = set(ui["hide"] or [])
            for key, cb in self.hide_checks.items():
                cb.setChecked(key in hidden)
        if "mask_opacity" in ui:
            self.spin_opacity.setValue(
                max(0, min(100, int(ui["mask_opacity"]))))
        display = cfg.get("display", {})
        if "max_image_dim" in display:
            self.spin_max_image_dim.setValue(
                max(0, int(display["max_image_dim"])))
        tracking = cfg.get("tracking", {})
        if "sticky_ids" in tracking:
            self.check_sticky_ids.setChecked(bool(tracking["sticky_ids"]))
        if "show_ids" in tracking:
            self.check_show_track_ids.setChecked(bool(tracking["show_ids"]))

    def _collect(self) -> Dict[str, Any]:
        """Widget state as a config dict (same schema as the example JSON)."""
        advanced = self.check_advanced.isChecked()
        hidden = [k for k, cb in self.hide_checks.items() if cb.isChecked()]
        if not advanced:
            # Keyframe/interpolate checkboxes are hidden in basic mode; the
            # groups stay hidden in the side panel regardless.
            hidden = sorted(set(hidden) | {"interpolate", "keyframe"})
        return {
            "interpolation": {
                "flow_method": self.combo_flow.currentText(),
                "camera_model": self.combo_cam.currentText(),
                "match_max_dist_frac": self.spin_match_frac.value(),
                "confirm_mismatch":
                    self.check_confirm_mismatch.isChecked(),
            },
            "sam3": {
                "device": self.combo_device.currentText(),
                "model": self.edit_sam3_model.text().strip() or None,
                "conf": self.spin_sam3_conf.value(),
                "auto_segment": self.check_auto_segment.isChecked(),
                "min_polygon_area": self.spin_min_poly_area.value(),
                "autolabel_nms_iou": self.spin_nms_iou.value(),
                "propagate_method":
                    self.combo_propagate_method.currentData(),
                "propagate_min_iou": self.spin_propagate_min_iou.value(),
                "propagate_min_seed_iou":
                    self.spin_propagate_seed_iou.value(),
            },
            "autolabel": {
                "detector": self.combo_autolabel_detector.currentData(),
                "owlv2_model": self.edit_owlv2_model.text().strip() or None,
                "owlv2_conf": self.spin_owlv2_conf.value(),
                "gdino_model": self.edit_gdino_model.text().strip() or None,
                "gdino_conf": self.spin_gdino_conf.value(),
                "florence2_model":
                    self.edit_florence2_model.text().strip() or None,
                "falcon_model": self.edit_falcon_model.text().strip() or None,
            },
            "ui": {
                "advanced": advanced,
                "hide": hidden,
                "mask_opacity": self.spin_opacity.value(),
            },
            "display": {
                "max_image_dim": self.spin_max_image_dim.value(),
            },
            "tracking": {
                "sticky_ids": self.check_sticky_ids.isChecked(),
                "show_ids": self.check_show_track_ids.isChecked(),
            },
        }

    # -- button handlers -----------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", "scripts/config", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load config", str(e))
            return
        self._prefill_from_config(cfg)
        self.win._apply_runtime_config(self._collect())
        self.win.statusBar().showMessage(f"Config loaded: {path}", 4000)

    def _browse_sam3_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "SAM3 weights", "", "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.edit_sam3_model.setText(path)

    def _on_apply(self) -> None:
        self.win._apply_runtime_config(self._collect())
        self.win.statusBar().showMessage("Config applied", 3000)

    def _on_save(self) -> None:
        cfg = self._collect()
        self.win._apply_runtime_config(cfg)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "scripts/config/label_review.json",
            "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        except Exception as e:
            QMessageBox.warning(self, "Save config", str(e))
            return
        self.win.statusBar().showMessage(f"Config saved: {path}", 4000)


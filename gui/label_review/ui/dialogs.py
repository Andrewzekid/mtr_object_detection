"""Config dialog (runtime settings editor)."""

import json
from typing import Any, Dict

from ..qt_compat import QtWidgets  # first: enum shims
from ..qt_compat import (  # noqa: F401
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout,
)

from .side_panel import SidePanel  # noqa: F401  (hide-group metadata)
from . import theme as ui_theme


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
        "viewpoint": "Viewpoint selection buttons",
        "rerun": "Rerun viewer / map buttons",
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

        # --- Appearance ------------------------------------------------------
        appear_box = QtWidgets.QGroupBox("Appearance")
        appear_form = QtWidgets.QFormLayout(appear_box)
        self.combo_theme = QtWidgets.QComboBox()
        for name in ui_theme.THEMES:
            self.combo_theme.addItem(name.capitalize(), name)
        self.combo_theme.setToolTip(
            "UI theme (same choices as the View menu). Applied immediately\n"
            "when you press Apply / Load / Save, and persisted.")
        appear_form.addRow("Theme", self.combo_theme)
        body.addWidget(appear_box)

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
        self.spin_sam3_imgsz = QtWidgets.QSpinBox()
        self.spin_sam3_imgsz.setRange(0, 4096)
        self.spin_sam3_imgsz.setSingleStep(64)
        self.spin_sam3_imgsz.setToolTip(
            "SAM3 inference size in pixels (square). 0 = library default.\n"
            "Lower values (e.g. 768/512) segment faster; boxes/masks are\n"
            "always rescaled back to the original image size. Takes effect\n"
            "on the next SAM3 run (re-segment, point segment, autolabel,\n"
            "SAM3 ALL, propagate).")
        sam3_form.addRow("Inference imgsz", self.spin_sam3_imgsz)
        self.combo_sam3_quantize = QtWidgets.QComboBox()
        self.combo_sam3_quantize.addItem("FP32 (full precision)", 32)
        self.combo_sam3_quantize.addItem("FP16 (recommended on GPU)", 16)
        self.combo_sam3_quantize.addItem("FP8 (fastest, may lose quality)", 8)
        self.combo_sam3_quantize.setToolTip(
            "SAM3 precision. FP16 roughly halves memory and speeds up\n"
            "inference on GPU with no visible quality loss; FP8 is\n"
            "experimental. FP32 matches the checkpoint exactly.")
        sam3_form.addRow("Precision", self.combo_sam3_quantize)
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
        # Optional separate weights for "Propagate →". SAM3.1 multiplex
        # (core/sam3/models/sam3.1-model/sam3.1_multiplex.pt) generally
        # tracks better than the base SAM3 checkpoint; leaving this empty
        # uses the same Model path as everything else.
        self.edit_propagate_model = QtWidgets.QLineEdit()
        self.edit_propagate_model.setPlaceholderText(
            "default: same as Model path (e.g. "
            "core/sam3/models/sam3.1-model/sam3.1_multiplex.pt)")
        self.edit_propagate_model.setToolTip(
            "Optional path to SAM3 weights used ONLY by 'Propagate →'\n"
            "(the memory-bank video session). Set this to the SAM3.1\n"
            "multiplex checkpoint for better tracking. Leave empty to use\n"
            "the same Model path as the rest of SAM3. Takes effect on the\n"
            "next propagate run.")
        btn_browse_prop_model = QPushButton("Browse…")
        btn_browse_prop_model.clicked.connect(self._browse_propagate_model)
        prop_model_row = QHBoxLayout()
        prop_model_row.addWidget(self.edit_propagate_model, 1)
        prop_model_row.addWidget(btn_browse_prop_model)
        sam3_form.addRow("Propagate model", prop_model_row)
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
            "  Default checkpoint: IDEA-Research/grounding-dino-base.")
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
            "OWLv2 detection confidence threshold (text-prompted).")
        al_form.addRow("OWLv2 confidence", self.spin_owlv2_conf)
        self.spin_owlv2_exemplar_conf = QtWidgets.QDoubleSpinBox()
        self.spin_owlv2_exemplar_conf.setRange(0.0, 1.0)
        self.spin_owlv2_exemplar_conf.setSingleStep(0.05)
        self.spin_owlv2_exemplar_conf.setDecimals(2)
        self.spin_owlv2_exemplar_conf.setToolTip(
            "OWLv2 exemplar (1-shot image-guided) confidence threshold.\n"
            "Image-guided scores run much hotter than text-prompt scores,\n"
            "so this needs a higher value than the text threshold (~0.6).")
        al_form.addRow("OWLv2 exemplar confidence",
                       self.spin_owlv2_exemplar_conf)
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

        # --- Rerun map / pose DB --------------------------------------------
        map_box = QtWidgets.QGroupBox("Rerun map / pose DB")
        map_form = QtWidgets.QFormLayout(map_box)
        self.combo_pose_db_match = QtWidgets.QComboBox()
        for label, data in (("Auto (filename, then id, then timestamp)",
                             "auto"),
                            ("Image filename = DB filename column",
                             "filename"),
                            ("Image filename stem = DB id column",
                             "filename_id"),
                            ("Image filename stem = timestamp (ns)",
                             "timestamp")):
            self.combo_pose_db_match.addItem(label, data)
        self.combo_pose_db_match.setToolTip(
            "How 'Show annotated in Rerun' matches each frame to a row in\n"
            "the pose database. Use 'id column' when your image files are\n"
            "named by the images-table id (e.g. 1042.jpg), 'timestamp' when\n"
            "they are named by nanosecond timestamps. Auto tries filename,\n"
            "then id, then timestamp. Applies to the currently open DB.")
        map_form.addRow("Pose DB match", self.combo_pose_db_match)
        body.addWidget(map_box)

        # --- Tracking (always visible — not advanced-gated) ------------------
        self.track_box = QtWidgets.QGroupBox("Tracking")
        track_form = QtWidgets.QFormLayout(self.track_box)
        self.check_sticky_ids = QCheckBox(
            "Sticky track ids (inherit from previous frame)")
        self.check_sticky_ids.setToolTip(
            "Checked (default): the k-th box drawn on a frame inherits the\n"
            "k-th track id from the nearest earlier annotated frame (what\n"
            "interpolation expects). Unchecked: every new box gets a fresh\n"
            "auto-increment track id.")
        track_form.addRow(self.check_sticky_ids)
        self.check_show_track_ids = QCheckBox(
            "Show track ids (T<id> labels + track edit field)")
        self.check_show_track_ids.setToolTip(
            "Checked (default): boxes are labelled T<id> on the canvas and\n"
            "in the box list, and the \"Track of selected\" edit row shows.")
        track_form.addRow(self.check_show_track_ids)
        body.addWidget(self.track_box)

        # --- Advanced toggle (at the very bottom of the settings) ----------
        self.check_advanced = QCheckBox(
            "Advanced settings (interpolation parameters, hide checkboxes)")
        self.check_advanced.setToolTip(
            "Unchecked (default): the interpolation parameter section and\n"
            "the 'hide UI elements' checkboxes for interpolation/keyframe\n"
            "are hidden. The keyframe/interpolate buttons themselves are\n"
            "always visible in the side panel.")
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
        """Show the interpolation-parameter section and the interpolate /
        keyframe hide checkboxes only in advanced mode."""
        adv = self.check_advanced.isChecked()
        self.interp_box.setVisible(adv)
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
        idx = self.combo_theme.findData(ui_theme.current_theme())
        self.combo_theme.setCurrentIndex(max(0, idx))
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
        self.spin_sam3_imgsz.setValue(win.sam3_imgsz or 0)
        qidx = self.combo_sam3_quantize.findData(
            win.sam3_quantize if win.sam3_quantize in (8, 16, 32) else 32)
        self.combo_sam3_quantize.setCurrentIndex(max(0, qidx))
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
        self.spin_owlv2_exemplar_conf.setValue(
            getattr(win, "owlv2_exemplar_conf", 0.6))
        self.edit_gdino_model.setText(
            getattr(win, "gdino_model", "") or "")
        self.spin_gdino_conf.setValue(
            getattr(win, "gdino_conf", 0.35))
        idx = self.combo_propagate_method.findData(
            getattr(win, "propagate_method", "memory"))
        self.combo_propagate_method.setCurrentIndex(max(0, idx))
        self.edit_propagate_model.setText(
            getattr(win, "propagate_model", "") or "")
        self.spin_propagate_min_iou.setValue(
            getattr(win, "propagate_min_iou", 0.3))
        self.spin_propagate_seed_iou.setValue(
            getattr(win, "propagate_min_seed_iou", 0.2))
        self.spin_opacity.setValue(win.side.opacity_slider.value())
        self.spin_max_image_dim.setValue(win.display_max_dim)
        idx = self.combo_pose_db_match.findData(
            getattr(win, "pose_db_match", "auto"))
        self.combo_pose_db_match.setCurrentIndex(max(0, idx))
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
        if "imgsz" in sam3:
            try:
                self.spin_sam3_imgsz.setValue(
                    max(0, int(sam3["imgsz"] or 0)))
            except (TypeError, ValueError):
                pass
        if sam3.get("quantize") in (8, 16, 32):
            qidx = self.combo_sam3_quantize.findData(int(sam3["quantize"]))
            self.combo_sam3_quantize.setCurrentIndex(max(0, qidx))
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
        if sam3.get("propagate_model"):
            self.edit_propagate_model.setText(str(sam3["propagate_model"]))
        if "propagate_min_iou" in sam3:
            self.spin_propagate_min_iou.setValue(
                max(0.0, min(1.0, float(sam3["propagate_min_iou"]))))
        if "propagate_min_seed_iou" in sam3:
            self.spin_propagate_seed_iou.setValue(
                max(0.0, min(1.0, float(sam3["propagate_min_seed_iou"]))))
        autolabel = cfg.get("autolabel", {})
        if autolabel.get("detector") in ("sam3", "owlv2", "owlv2_exemplar",
                                         "grounding_dino"):
            self.combo_autolabel_detector.setCurrentIndex(
                self.combo_autolabel_detector.findData(
                    autolabel["detector"]))
        if autolabel.get("owlv2_model"):
            self.edit_owlv2_model.setText(str(autolabel["owlv2_model"]))
        if "owlv2_conf" in autolabel:
            self.spin_owlv2_conf.setValue(
                max(0.0, min(1.0, float(autolabel["owlv2_conf"]))))
        if "owlv2_exemplar_conf" in autolabel:
            self.spin_owlv2_exemplar_conf.setValue(
                max(0.0, min(1.0, float(autolabel["owlv2_exemplar_conf"]))))
        if autolabel.get("gdino_model"):
            self.edit_gdino_model.setText(str(autolabel["gdino_model"]))
        if "gdino_conf" in autolabel:
            self.spin_gdino_conf.setValue(
                max(0.0, min(1.0, float(autolabel["gdino_conf"]))))
        ui = cfg.get("ui", {})
        if "advanced" in ui:
            self.check_advanced.setChecked(bool(ui["advanced"]))
            self._update_advanced_visibility()
        if ui.get("theme") in ui_theme.THEMES:
            self.combo_theme.setCurrentIndex(
                self.combo_theme.findData(ui["theme"]))
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
        pose_db = cfg.get("pose_db", {})
        if pose_db.get("match") in ("auto", "filename", "filename_id",
                                    "timestamp"):
            self.combo_pose_db_match.setCurrentIndex(
                self.combo_pose_db_match.findData(pose_db["match"]))
        tracking = cfg.get("tracking", {})
        if "sticky_ids" in tracking:
            self.check_sticky_ids.setChecked(bool(tracking["sticky_ids"]))
        if "show_ids" in tracking:
            self.check_show_track_ids.setChecked(bool(tracking["show_ids"]))

    def _collect(self) -> Dict[str, Any]:
        """Widget state as a config dict (same schema as the example JSON)."""
        hidden = [k for k, cb in self.hide_checks.items() if cb.isChecked()]
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
                "imgsz": self.spin_sam3_imgsz.value() or None,
                "quantize": self.combo_sam3_quantize.currentData(),
                "auto_segment": self.check_auto_segment.isChecked(),
                "min_polygon_area": self.spin_min_poly_area.value(),
                "autolabel_nms_iou": self.spin_nms_iou.value(),
                "propagate_method":
                    self.combo_propagate_method.currentData(),
                "propagate_model":
                    self.edit_propagate_model.text().strip() or None,
                "propagate_min_iou": self.spin_propagate_min_iou.value(),
                "propagate_min_seed_iou":
                    self.spin_propagate_seed_iou.value(),
            },
            "autolabel": {
                "detector": self.combo_autolabel_detector.currentData(),
                "owlv2_model": self.edit_owlv2_model.text().strip() or None,
                "owlv2_conf": self.spin_owlv2_conf.value(),
                "owlv2_exemplar_conf": self.spin_owlv2_exemplar_conf.value(),
                "gdino_model": self.edit_gdino_model.text().strip() or None,
                "gdino_conf": self.spin_gdino_conf.value(),
            },
            "ui": {
                "advanced": self.check_advanced.isChecked(),
                "hide": hidden,
                "mask_opacity": self.spin_opacity.value(),
                "theme": self.combo_theme.currentData(),
            },
            "display": {
                "max_image_dim": self.spin_max_image_dim.value(),
            },
            "pose_db": {
                "match": self.combo_pose_db_match.currentData(),
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

    def _browse_propagate_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Propagate weights (SAM3 / SAM3.1)", "",
            "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.edit_propagate_model.setText(path)

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


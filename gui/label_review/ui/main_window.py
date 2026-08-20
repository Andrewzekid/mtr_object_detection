"""Main window — the review tool's top-level QMainWindow."""

import json
import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from PyQt6.QtCore import QRunnable, QThreadPool

from ..qt_compat import Qt, QtCore, QtGui, QtWidgets, QEvent, QTimer, _QT_HORZ  # first: enum shims  # noqa: E501
from ..qt_compat import (  # noqa: F401
    QAction, QApplication, QCheckBox, QFileDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QShortcut, QSplitter,
)

from ..config import _resolve_device
from ..state.coco_state import CocoState
from ..state.index import ImageFolderIndex, StereoIndex
from ..workers.label_review_workers import (  # noqa: F401
    SAM3Worker, SAM3BatchWorker, InterpBatchWorker,
    SAM3AutolabelWorker, SAM3AutolabelBatchWorker, SAM3PropagateWorker,
    _get_interp13, _SAM3_AVAILABLE, run_sam3, _iou_xyxy,
)

from .canvas import CanvasWidget
from .side_panel import SidePanel
from .dialogs import ConfigDialog


# ---------------------------------------------------------------------------
# Prefetch runnable: decode one frame in the background for playback.
# ---------------------------------------------------------------------------

class _PrefetchRunnable(QRunnable):
    """Decode a single frame into the index's LRU cache.

    The cache is thread-safe, so multiple runnables can run in parallel.
    If the frame is already cached, decode_image returns immediately.
    """

    def __init__(self, frame_index, idx: int, side: Optional[str]):
        super().__init__()
        self.frame_index = frame_index
        self.idx = idx
        self.side = side

    def run(self) -> None:
        try:
            if self.side is None:
                self.frame_index.decode_image(self.idx)
            else:
                self.frame_index.decode_image(self.idx, side=self.side)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ReviewWindow(QMainWindow):

    def __init__(self, frame_index: "ImageFolderIndex | EmptyIndex",
                 coco: CocoState,
                 sam3_model: Optional[str] = None,
                 sam3_device: str = "cuda",
                  sam3_conf: float = 0.25,
                  propagate_method: str = "memory",
                  propagate_min_iou: float = 0.3,
                  propagate_min_seed_iou: float = 0.2,
                  auto_segment: bool = False,
                  interp_flow_method: str = "dis",
                  interp_camera_model: str = "none",
                  interp_match_frac: float = 0.2,
                  interp_confirm_mismatch: bool = True,
                  ui_hide: Optional[List[str]] = None,
                  mask_opacity: Optional[int] = None,
                   advanced_ui: bool = False,
                   show_track_ids: bool = True,
                   display_max_dim: int = 0,
                   parent=None):
        super().__init__(parent)
        self.frame_index = frame_index
        self.coco = coco
        self.sam3_model = sam3_model
        self.sam3_device = sam3_device
        self.sam3_conf = sam3_conf
        # Propagate method: "memory" (SAM3 video memory bank, one session
        # per side) or "chain" (frame-by-frame re-detection, IoU-chained,
        # permanent stop on first miss). Config: sam3.propagate_method.
        self.propagate_method: str = (
            propagate_method if propagate_method in ("memory", "chain")
            else "memory")
        # Chain-mode IoU gates (config: sam3.propagate_min_iou /
        # sam3.propagate_min_seed_iou).
        self.propagate_min_iou: float = float(propagate_min_iou)
        self.propagate_min_seed_iou: float = float(propagate_min_seed_iou)
        # IoU threshold for class-aware NMS on autolabel detections
        # (config: sam3.autolabel_nms_iou; 1.0 disables dedup).
        self.sam3_nms_iou: float = 0.7
        self.auto_segment = auto_segment
        self.interp_flow_method = interp_flow_method
        self.interp_camera_model = interp_camera_model
        # Fraction of min(frame w, h) used as the max pairing distance when
        # matching anchor boxes for interpolation.
        self.interp_match_frac = interp_match_frac
        # Whether the advanced controls (interpolation + tracking settings,
        # keyframe/interpolate buttons) are exposed. Off by default; toggled
        # from the Config dialog's "Advanced settings" checkbox.
        self.advanced_ui: bool = advanced_ui
        # Whether track ids are shown (canvas "T<id>" labels, box-list
        # suffix, "Track of selected" row). On by default; config key
        # tracking.show_ids, editable under the dialog's advanced section.
        self.show_track_ids: bool = show_track_ids
        # Whether to show the track-id mismatch confirmation dialog before
        # interpolating.
        self.interp_confirm_mismatch = interp_confirm_mismatch
        # Display-only downscale cap for frame pixmaps (0 = original
        # resolution; config: display.max_image_dim). Box/mask coordinates
        # are unaffected — the canvases keep logical image coords.
        self.display_max_dim: int = max(0, int(display_max_dim))

        self.setWindowTitle("Computer Vision Label Review Tool")
        self.resize(1600, 900)

        self._current_idx = coco.current_idx
        self._current_image_id: Optional[int] = None
        self._pending_cat_id: Optional[int] = None
        # Sticky category: the last category assigned to a drawn box. New
        # draws auto-assign it (unless the user preselected another cat).
        self._last_cat_id: Optional[int] = None
        self._sam3_worker: Optional[SAM3Worker] = None
        self._sam3_batch_worker: Optional[SAM3BatchWorker] = None
        self._sam3_autolabel_worker: Optional[SAM3AutolabelWorker] = None
        self._sam3_autolabel_batch_worker: Optional[SAM3AutolabelBatchWorker] = None
        self._sam3_propagate_worker: Optional[SAM3PropagateWorker] = None
        # Bumped on every source switch. Workers capture the value at start
        # (`_lr_session`); result handlers drop signals from stale sessions
        # so a job started on the old source can't write masks/boxes into
        # the new CocoState (whose ids are renumbered from 1).
        self._session_seq: int = 0
        # FIFO of pending single-frame SAM3 jobs (img_path, bboxes_xyxy,
        # concepts, ann_ids). Filled when a run is requested while another
        # is in flight; drained when the current worker finishes/cancels.
        self._sam3_queue: List[Dict[str, Any]] = []
        # Last SAM3 status body (without the "(N in queue)" prefix), so the
        # queue count can be re-rendered without losing the message.
        self._sam3_status_body: str = "idle"
        self._batch_frames_done: int = 0
        # SAM3-all chain state (stereo: left batch, then right batch).
        self._sam3_all_pending: List[str] = []
        self._sam3_all_jobs_by_side: Dict[str, List[Dict[str, Any]]] = {}
        self._sam3_all_ok: int = 0
        self._sam3_all_fail: int = 0
        self._interp_worker: Optional[InterpBatchWorker] = None
        # Tmp file path for the current frame's image (run_sam3 needs a path).
        self._tmp_image_path: Optional[str] = None

        # Persisted UI state (window geometry, sidebar width, etc.).
        self._ui_state_path = Path.home() / ".config" / "cv_label_review" / "state.json"

        # Frame playback timer.
        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_interval_ms: int = 33  # 1x = 30 fps (matches combo_speed)
        self._playing: bool = False
        # Prefetch timer: during playback, decode the upcoming frames in
        # the background so playback is not capped by decode latency.
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.timeout.connect(self._prefetch_tick)
        self._prefetch_lookahead = 8
        # Per-tick counter used to throttle the keyframe/annotated/progress
        # UI syncs during playback (refresh every ~8 ticks instead of every
        # tick so the sync cost doesn't cap playback speed).
        self._play_tick_count: int = 0
        # Session-only opt-out from the X discard-all confirmation dialog.
        self._skip_discard_confirm: bool = False
        # Set once quit has been confirmed, so closeEvent doesn't ask twice.
        self._quit_confirmed: bool = False

        # ---------- layout ----------
        self._stereo: bool = bool(getattr(frame_index, "stereo", False))
        self._splitter = QSplitter(_QT_HORZ)
        self.side = SidePanel(coco)
        # One canvas per side; `self.canvas` stays the left/mono canvas
        # (compat attr — mono sessions have exactly this one).
        self.canvases: Dict[str, CanvasWidget] = {}
        self.canvas = self._make_canvas("left")
        self.canvases["left"] = self.canvas
        self._splitter.addWidget(self.canvas)
        if self._stereo:
            canvas_right = self._make_canvas("right")
            self.canvases["right"] = canvas_right
            self._splitter.addWidget(canvas_right)
        # The canvas that window-level edit shortcuts (D/A/X/R/digits…) and
        # per-frame ops act on — the last one clicked (left in mono).
        self._active_canvas: CanvasWidget = self.canvas
        # Side each running batch op (SAM3 ALL / autolabel ALL / propagate /
        # interpolate) was started on; "left" defaults keep direct handler
        # calls in mono tests valid.
        self._batch_side: str = "left"
        self._autolabel_batch_side: str = "left"
        self._interp_side: str = "left"
        self._splitter.addWidget(self.side)
        self._splitter.setSizes([1200, 360])

        # Track-id visibility (side-panel row/list; canvases get it in
        # _make_canvas).
        self.side.set_track_ids_visible(self.show_track_ids)

        self.setCentralWidget(self._splitter)
        self._build_menu()

        # Config-driven UI tweaks: hide button groups, preset mask opacity.
        # Without advanced mode the keyframe/interpolate groups are always
        # hidden, regardless of the ui_hide list.
        hide_groups = list(ui_hide or [])
        if not self.advanced_ui:
            hide_groups = sorted(set(hide_groups) | {"interpolate", "keyframe"})
        if hide_groups:
            self.side.set_hidden_groups(hide_groups)
        if mask_opacity is not None:
            pct = max(0, min(100, int(mask_opacity)))
            self.side.opacity_slider.setValue(pct)
            # Signals aren't connected yet at this point, so set the canvas
            # overlay alpha directly as well.
            for c in self.canvases.values():
                c.set_mask_alpha(round(pct * 255 / 100))

        # ---------- status bar (save indicator + transient messages) ----------
        self._save_indicator = QLabel("✓ Saved")
        self._save_indicator.setStyleSheet("padding: 0 6px;")
        self.statusBar().addPermanentWidget(self._save_indicator)
        self._reviewed_label = QLabel("Reviewed: 0/0")
        self._reviewed_label.setStyleSheet("padding: 0 6px;")
        self.statusBar().addPermanentWidget(self._reviewed_label)
        self.statusBar().showMessage("Ready", 3000)
        # Refresh the indicator whenever the dirty flag may have changed.
        # We piggyback on a 250ms timer instead of patching every mutation site.
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_save_indicator)
        self._status_timer.start(250)

        # ---------- signals ----------
        # (Canvas signals are wired in _make_canvas, one canvas per side.)
        self.side.cat_clicked.connect(self._on_side_cat_clicked)
        self.side.slider_moved.connect(self._on_slider_moved)
        self.side.slider_released.connect(self._on_slider_released)
        self.side.nav_delta.connect(self._on_frame_nav)
        self.side.run_sam3_clicked.connect(self._on_run_sam3_all)
        self.side.toggle_masks_clicked.connect(self._on_toggle_masks)
        self.side.resegment_clicked.connect(self._on_resegment_selected)
        self.side.play_pause_clicked.connect(self._on_play_pause)
        self.side.play_speed_changed.connect(self._on_play_speed_changed)
        self.side.box_selected.connect(self._on_box_list_selected)
        self.side.preselect_cat.connect(self._on_preselect_cat)
        self.side.mask_opacity_changed.connect(self._on_mask_opacity_changed)
        self.side.cancel_sam3_clicked.connect(self._on_cancel_sam3)
        self.side.recat_selected.connect(self._on_recat_selected)
        self.side.sam3_all_frames_clicked.connect(self._on_sam3_all_frames)
        self.side.autolabel_frame_clicked.connect(self._on_autolabel_frame)
        self.side.autolabel_all_clicked.connect(self._on_autolabel_all)
        self.side.propagate_clicked.connect(self._on_propagate_track)
        self.side.toggle_keyframe_clicked.connect(self._on_toggle_keyframe)
        self.side.toggle_annotated_clicked.connect(self._on_toggle_annotated)
        self.side.interpolate_clicked.connect(self._on_interpolate)
        self.side.cancel_interp_clicked.connect(self._on_cancel_interp)
        self.side.track_id_selected.connect(self._on_track_selected)
        self.side.add_cat_clicked.connect(self._on_add_category)
        self.side.rename_cat_clicked.connect(self._on_rename_category)
        self.side.del_cat_clicked.connect(self._on_delete_category)

        # Frame slider config
        self.side.set_slider_max(len(self.frame_index))

        # Shortcuts that work even when canvas doesn't have focus.
        # These mirror the canvas keyPressEvent so the user doesn't need
        # to click the canvas first to give it focus. Edit keys (F/D/A/X)
        # forward to the ACTIVE canvas (the last one clicked — always the
        # left/only one in mono).
        for key, slot in [
            (Qt.Key_N, lambda: self._on_frame_nav(+1)),
            (Qt.Key_B, lambda: self._on_frame_nav(-1)),
            (Qt.Key_S, self._on_save_quit),
            (Qt.Key_Q, self._on_quit),
            (Qt.Key_Space, self._on_play_pause),
            (Qt.Key_F, lambda: self._active_canvas._fit_to_view()),
            (Qt.Key_D, lambda: self._active_canvas.keyPressEvent(
                QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_D, Qt.NoModifier))),
            (Qt.Key_A, lambda: self._active_canvas.keyPressEvent(
                QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_A, Qt.NoModifier))),
            (Qt.Key_X, lambda: self._active_canvas.keyPressEvent(
                QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_X, Qt.NoModifier))),
            (Qt.Key_M, self._on_toggle_masks),
            (Qt.Key_R, self._on_resegment_selected),
            (Qt.Key_Z, self._on_zoom_to_selected),
            (Qt.Key_U, self._on_next_unlabeled),
            (Qt.Key_C, self._focus_recat_edit),
            (Qt.Key_T, self._focus_track_edit),
            (Qt.Key_K, self._on_toggle_keyframe),
            (Qt.Key_I, self._on_interpolate),
        ]:
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(slot)
        # Ctrl+Z = undo, Ctrl+Shift+Z / Ctrl+Y = redo. These need the
        # Control modifier, so they can't share the simple-key loop above.
        sc_undo = QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        sc_undo.activated.connect(self._on_undo)
        sc_redo = QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self)
        sc_redo.activated.connect(self._on_redo)
        sc_redo_y = QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        sc_redo_y.activated.connect(self._on_redo)
        # Ctrl+A = select all boxes on the current frame (active side).
        sc_sel_all = QShortcut(QtGui.QKeySequence("Ctrl+A"), self)
        sc_sel_all.activated.connect(lambda: self._active_canvas.select_all())

        # Load first frame, then restore persisted UI state.
        QTimer.singleShot(50, self._load_current)
        QTimer.singleShot(150, self._load_ui_state)

        # Clicking anywhere outside a focused line edit (e.g. the "new
        # category name" box) defocuses it, so hotkeys work again without
        # having to press Enter/Escape first.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._update_source_label()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            # Clicking a stereo canvas makes it the active side (the target
            # of window-level edit shortcuts and per-frame ops).
            if isinstance(obj, CanvasWidget) \
                    and any(obj is c for c in self.canvases.values()) \
                    and obj is not self._active_canvas:
                self._set_active_canvas(obj)
            fw = QApplication.focusWidget()
            if isinstance(fw, QLineEdit) and fw is not obj:
                fw.clearFocus()
        return super().eventFilter(obj, event)

    # ----------------------- canvases (mono / stereo) ------------------- #

    def _make_canvas(self, side: str) -> CanvasWidget:
        """Create and wire one canvas ("left" / "right")."""
        c = CanvasWidget(side=side)
        c.parent_window = self  # type: ignore[attr-defined]
        c.show_track_ids = self.show_track_ids
        c.display_max_dim = self.display_max_dim
        c.box_added.connect(self._on_box_added)
        c.box_deleted.connect(self._on_box_deleted)
        c.box_moved.connect(self._on_box_moved)
        c.box_resized.connect(self._on_box_resized)
        c.frame_nav.connect(self._on_frame_nav)
        c.save_quit.connect(self._on_save_quit)
        c.quit_request.connect(self._on_quit)
        c.discard_all.connect(self._on_discard_all)
        c.cat_pick_requested.connect(self._on_cat_pick_requested)
        c.toggle_masks.connect(self._on_toggle_masks)
        c.resegment_selected.connect(self._on_resegment_selected)
        c.play_pause.connect(self._on_play_pause)
        c.fit_view.connect(c._fit_to_view)
        c.selection_changed.connect(
            lambda i, canvas=c: self._on_canvas_selection_changed(canvas, i))
        c.zoom_to_selected.connect(self._on_zoom_to_selected)
        c.next_unlabeled.connect(self._on_next_unlabeled)
        return c

    def _rebuild_canvases(self, stereo: bool) -> None:
        """Switch between the mono (1 canvas) and stereo (2 canvases)
        layouts — called when a source switch changes stereo-ness."""
        for c in self.canvases.values():
            c.setParent(None)
            c.deleteLater()
        self.canvases = {}
        self.canvas = self._make_canvas("left")
        self.canvases["left"] = self.canvas
        self._splitter.insertWidget(0, self.canvas)
        if stereo:
            right = self._make_canvas("right")
            self.canvases["right"] = right
            self._splitter.insertWidget(1, right)
        self._active_canvas = self.canvas
        self._stereo = stereo
        # Keep the current mask overlay settings on the fresh canvases.
        alpha = round(self.side.opacity_slider.value() * 255 / 100)
        for c in self.canvases.values():
            c.set_mask_alpha(alpha)

    def _src_canvas(self) -> CanvasWidget:
        """The canvas that emitted the signal being handled; falls back to
        the active canvas for direct calls (keyboard shortcuts, tests)."""
        s = self.sender()
        return s if isinstance(s, CanvasWidget) else self._active_canvas

    def _set_active_canvas(self, canvas: CanvasWidget) -> None:
        """Make `canvas` the active side: window-level edit shortcuts and
        per-frame ops target it, and the side panel mirrors its box list."""
        self._active_canvas = canvas
        self._current_image_id = canvas._image_id
        self.side.set_boxes(canvas._boxes)
        self.side.highlight_box_row(canvas._selected_idx)
        self._update_propagate_button()

    def _on_canvas_selection_changed(self, canvas: CanvasWidget,
                                     box_idx: int) -> None:
        # Selection on the inactive side doesn't drive the side panel.
        if canvas is not self._active_canvas:
            return
        self.side.highlight_box_row(box_idx)
        self._prefill_recat()
        self._update_propagate_button()

    def _frame_at_side(self, idx: int, side: str) -> Dict[str, Any]:
        """frame_at with a stereo side; plain frame_at in mono (mono
        indices don't take a side kwarg)."""
        if self._stereo:
            return self.frame_index.frame_at(idx, side=side)
        return self.frame_index.frame_at(idx)

    def _decode_side(self, idx: int, side: str) -> np.ndarray:
        if self._stereo:
            return self.frame_index.decode_image(idx, side=side)
        return self.frame_index.decode_image(idx)

    def _side_worker_index(self, side: str):
        """The frame index a batch worker runs on — the single-side view in
        stereo, the plain index in mono."""
        if self._stereo:
            return self.frame_index.side_index(side)
        return self.frame_index

    def _update_source_label(self) -> None:
        """Show the current frame source in the side panel: the image
        folder for file sources (both folders in stereo mode)."""
        idx = self.frame_index
        if getattr(idx, "stereo", False):
            ldir = (os.path.dirname(os.path.abspath(idx.files_left[0]))
                    if idx.files_left else "")
            rdir = (os.path.dirname(os.path.abspath(idx.files_right[0]))
                    if idx.files_right else "")
            self.side.set_source(f"L: {ldir}  |  R: {rdir}")
            return
        files = getattr(idx, "files", None)
        if files:
            dirs = sorted({os.path.dirname(os.path.abspath(f))
                           for f in files})
            path = (dirs[0] if len(dirs) == 1
                    else f"{len(files)} files from {len(dirs)} folders")
        else:
            path = ""
        self.side.set_source(path)

    # ----------------------- status bar + save indicator ----------------- #

    def _refresh_save_indicator(self) -> None:
        """Update the permanent '✓ Saved' / '● Unsaved' label in the status bar."""
        if self.coco.dirty:
            self._save_indicator.setText("● Unsaved")
            self._save_indicator.setStyleSheet(
                "padding: 0 6px; color: #ff8040; font-weight: bold;"
            )
        else:
            self._save_indicator.setText("✓ Saved")
            self._save_indicator.setStyleSheet("padding: 0 6px; color: #40c060;")
        self._reviewed_label.setText(
            f"Reviewed: {len(self.coco.reviewed)}/{len(self.frame_index)}"
        )

    def _load_ui_state(self) -> None:
        """Apply persisted UI state (mask opacity)."""
        try:
            if not self._ui_state_path.exists():
                return
            with open(self._ui_state_path, "r") as f:
                state = json.load(f)
            opacity = state.get("mask_opacity")
            if isinstance(opacity, int) and 0 <= opacity <= 100:
                for c in self.canvases.values():
                    c.set_mask_alpha(round(opacity * 255 / 100))
                self.side.set_mask_opacity(opacity)
        except Exception:
            pass

    def _write_ui_state(self, **updates) -> None:
        """Merge `updates` into the persisted UI state file."""
        try:
            state = {}
            if self._ui_state_path.exists():
                with open(self._ui_state_path, "r") as f:
                    state = json.load(f)
            state.update(updates)
            self._ui_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._ui_state_path, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _on_mask_opacity_changed(self, pct: int) -> None:
        for c in self.canvases.values():
            c.set_mask_alpha(round(pct * 255 / 100))
        self._write_ui_state(mask_opacity=int(pct))

    # ----------------------- open images / folder ----------------------- #

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        act_imgs = QAction("Open image file(s)…", self)
        act_imgs.setShortcut("Ctrl+O")
        act_imgs.triggered.connect(self._open_image_files)
        m.addAction(act_imgs)
        act_dir = QAction("Open folder…", self)
        act_dir.setShortcut("Ctrl+Shift+O")
        act_dir.triggered.connect(self._open_image_folder)
        m.addAction(act_dir)
        act_stereo = QAction("Open stereo folders…", self)
        act_stereo.setToolTip(
            "Open a left and a right image folder. Frames are paired "
            "positionally (matching filenames expected) and navigated "
            "together; each side gets its own canvas and annotations.")
        act_stereo.triggered.connect(self._open_stereo_folders)
        m.addAction(act_stereo)
        act_anns = QAction("Load annotations file…", self)
        act_anns.setShortcut("Ctrl+I")
        act_anns.setToolTip(
            "Import boxes/masks from a COCO JSON (e.g. from a previous "
            "session or a model run), matching images by file name. "
            "Categories are merged by name.")
        act_anns.triggered.connect(self._load_annotations_dialog)
        m.addAction(act_anns)
        m.addSeparator()
        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.setToolTip(
            "Save the COCO JSON to the current output path (no quit).")
        act_save.triggered.connect(self._on_save)
        m.addAction(act_save)
        act_save_as = QAction("Save as…", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.setToolTip(
            "Save the COCO JSON to a chosen location; the session keeps "
            "saving there afterwards.")
        act_save_as.triggered.connect(self._on_save_as)
        m.addAction(act_save_as)
        m.addSeparator()
        act_cfg = QAction("Config settings…", self)
        act_cfg.setShortcut("Ctrl+G")
        act_cfg.setToolTip(
            "Open the settings dialog (hide UI elements / interpolation / "
            "SAM3 / mask opacity; load/save a JSON config).")
        act_cfg.triggered.connect(self._load_config_dialog)
        m.addAction(act_cfg)
        # A visible Config button in the menu bar's top-right corner.
        cfg_btn = QPushButton("⚙ Config")
        cfg_btn.setToolTip(
            "Open the settings dialog (hide UI elements / interpolation / "
            "SAM3 / mask opacity; load/save a JSON config).")
        cfg_btn.clicked.connect(self._load_config_dialog)
        self.menuBar().setCornerWidget(cfg_btn)

    def _switch_source(self, new_index, out_json: str, label: str) -> None:
        """Shared source-switch: save the current session, swap the frame
        index, and start a fresh COCO session at `out_json`, keeping the
        current category list."""
        # Invalidate the old session's background work: queued jobs carry
        # the old CocoState's ann/image ids, which the new state renumbers
        # from 1 — letting them drain would write masks onto the wrong
        # boxes. Cancel what's cancellable; anything still running finishes
        # into the void because the session guard drops its results.
        self._session_seq += 1
        self._sam3_queue.clear()
        for w in (self._sam3_worker, self._sam3_batch_worker,
                  self._sam3_autolabel_batch_worker,
                  self._sam3_propagate_worker, self._interp_worker):
            if w is not None and w.isRunning():
                w.cancel()
        # Reset the SAM3 + interpolation UI. A cancelled worker's terminal
        # signals are dropped as stale (see _stale_sender), so they can't
        # re-enable the batch buttons / disable Cancel / re-enable the canvas
        # on their own — do it here, or a switch out of a running job leaves
        # Cancel stuck on and the canvas locked.
        self.side.set_sam3_running(False)
        self._set_sam3_status("idle")
        for c in self.canvases.values():
            c.setEnabled(True)
        self.side.set_interp_running(False)
        self.side.set_interp_status("Interpolation: idle")
        # Save the current session before switching away from it.
        try:
            self.coco.save(is_final=False)
        except Exception:
            pass
        self._stop_playback()
        self.side.set_playing(False)
        self.frame_index = new_index
        # Settings survive the source switch.
        sticky = self.coco.sticky_track_ids
        min_poly = self.coco.min_polygon_area
        self.coco = CocoState(out_json, self.coco.categories)
        self.coco.sticky_track_ids = sticky
        self.coco.min_polygon_area = min_poly
        self.coco.load_existing()
        self._update_source_label()
        self.side.coco = self.coco  # side panel keeps its own reference
        # load_existing may have merged categories saved in the project
        # file — refresh the visible list (it still shows the previous
        # session's categories, e.g. empty after an idle start).
        self.side._rebuild_cat_list()
        self._current_idx = self.coco.load_progress(len(self.frame_index))
        self._current_image_id = None
        self._last_cat_id = None
        self._pending_cat_id = None
        for c in self.canvases.values():
            c._image_id = None
            c.reset_state()
        self._active_canvas = self.canvas
        self.side.set_slider_max(len(self.frame_index))
        self.setWindowTitle(f"Computer Vision Label Review Tool — {label}")
        self._load_current()
        self.statusBar().showMessage(
            f"Loaded {len(self.frame_index)} frame(s) — saving to {out_json}",
            5000)

    def _load_annotations_dialog(self) -> None:
        """File → Load annotations file…: import boxes/masks from a COCO
        JSON into the current session, matching images by file name."""
        if len(self.frame_index) == 0:
            QMessageBox.information(
                self, "Load annotations",
                "Open an image file/folder first, then load annotations.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load annotations file", "", "COCO JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load annotations", str(e))
            return
        if getattr(self.frame_index, "stereo", False):
            # Stereo: match imported images by (basename, side).
            file_to_frame = {}
            for side, files in (("left", self.frame_index.files_left),
                                ("right", self.frame_index.files_right)):
                for i, fp in enumerate(files):
                    file_to_frame[(os.path.basename(fp), side)] = i
        else:
            file_to_frame = {
                os.path.basename(fp): i
                for i, fp in enumerate(getattr(self.frame_index, "files", []))
            }
        n_frames, n_ok, n_skip = self.coco.import_coco(
            data, file_to_frame, self.frame_index)
        self.side._rebuild_cat_list()  # import may have added categories
        self._refresh_boxes()
        self._update_progress()
        self.coco.save(is_final=False)
        self.statusBar().showMessage(
            f"Imported {n_ok} annotation(s) on {n_frames} frame(s) from "
            f"{os.path.basename(path)} ({n_skip} skipped)", 6000)

    def _load_config_dialog(self) -> None:
        """Open the settings dialog (⚙ Config button / File → Config…).

        The dialog edits live settings and can load/save a JSON config from
        inside itself."""
        dlg = ConfigDialog(self)
        dlg.exec()

    def _apply_runtime_config(self, cfg: Dict[str, Any]) -> None:
        """Apply a config dict to the running window. Interpolation/SAM3
        values affect subsequent operations; ui.hide both hides and
        un-hides groups; mask_opacity applies immediately."""
        interp_cfg = cfg.get("interpolation", {})
        fm = interp_cfg.get("flow_method")
        if fm in ("dis", "klt", "farneback"):
            self.interp_flow_method = fm
        cm = interp_cfg.get("camera_model")
        if cm in ("none", "global"):
            self.interp_camera_model = cm
        if "match_max_dist_frac" in interp_cfg:
            self.interp_match_frac = float(interp_cfg["match_max_dist_frac"])
        if "confirm_mismatch" in interp_cfg:
            self.interp_confirm_mismatch = bool(
                interp_cfg["confirm_mismatch"])
        sam3_cfg = cfg.get("sam3", {})
        dev = sam3_cfg.get("device")
        if dev in ("auto", "cuda", "cpu"):
            # 'auto' must resolve to a concrete device here — workers hand
            # sam3_device straight to run_sam3, which won't interpret 'auto'.
            self.sam3_device = _resolve_device(dev)
        if sam3_cfg.get("model"):
            self.sam3_model = str(sam3_cfg["model"])
        if "conf" in sam3_cfg:
            self.sam3_conf = float(sam3_cfg["conf"])
        if "auto_segment" in sam3_cfg:
            self.auto_segment = bool(sam3_cfg["auto_segment"])
        if "min_polygon_area" in sam3_cfg:
            self.coco.min_polygon_area = max(
                0.0, float(sam3_cfg["min_polygon_area"]))
        if "autolabel_nms_iou" in sam3_cfg:
            self.sam3_nms_iou = max(
                0.0, min(1.0, float(sam3_cfg["autolabel_nms_iou"])))
        pm = sam3_cfg.get("propagate_method")
        if pm in ("memory", "chain"):
            self.propagate_method = pm
        if "propagate_min_iou" in sam3_cfg:
            self.propagate_min_iou = max(
                0.0, min(1.0, float(sam3_cfg["propagate_min_iou"])))
        if "propagate_min_seed_iou" in sam3_cfg:
            self.propagate_min_seed_iou = max(
                0.0, min(1.0, float(sam3_cfg["propagate_min_seed_iou"])))
        ui_cfg = cfg.get("ui", {})
        if "advanced" in ui_cfg:
            self.advanced_ui = bool(ui_cfg["advanced"])
        if "hide" in ui_cfg or "advanced" in ui_cfg:
            groups = list(ui_cfg.get("hide") or [])
            if not self.advanced_ui:
                groups = sorted(set(groups) | {"interpolate", "keyframe"})
            self.side.set_hidden_groups(groups)
        if "mask_opacity" in ui_cfg:
            pct = max(0, min(100, int(ui_cfg["mask_opacity"])))
            # Signals are connected by now, so this also updates the canvas.
            self.side.opacity_slider.setValue(pct)
        display_cfg = cfg.get("display", {})
        if "max_image_dim" in display_cfg:
            self.display_max_dim = max(0, int(display_cfg["max_image_dim"]))
            for c in self.canvases.values():
                c.display_max_dim = self.display_max_dim
            # Re-render the current frame at the new display resolution.
            self._load_current()
        tracking_cfg = cfg.get("tracking", {})
        if "sticky_ids" in tracking_cfg:
            self.coco.sticky_track_ids = bool(tracking_cfg["sticky_ids"])
        if "show_ids" in tracking_cfg:
            self.show_track_ids = bool(tracking_cfg["show_ids"])
            for c in self.canvases.values():
                c.show_track_ids = self.show_track_ids
            self.side.set_track_ids_visible(self.show_track_ids)
            self._refresh_boxes()  # re-render labels without/with the T-ids


    def _open_image_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open image file(s)", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if files:
            self._switch_to_images(files)

    def _open_image_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open image folder", "")
        if d:
            self._switch_to_images([d])

    def _open_stereo_folders(self) -> None:
        """File → Open stereo folders…: pick the left, then the right
        folder; frames pair positionally and navigate together."""
        d_left = QFileDialog.getExistingDirectory(
            self, "Open LEFT image folder", "")
        if not d_left:
            return
        d_right = QFileDialog.getExistingDirectory(
            self, "Open RIGHT image folder", d_left)
        if not d_right:
            return
        self._switch_to_images([d_left], right_paths=[d_right])

    def _switch_to_images(self, paths: List[str],
                          right_paths: Optional[List[str]] = None) -> None:
        """Replace the frame source with plain image files/folders — or a
        stereo pair of folders when `right_paths` is given.

        Starts a fresh COCO session next to the (left) source
        (``labels_coco.json`` in the folder, or beside the first file),
        keeping the current category list.
        """
        try:
            new_index = (StereoIndex(paths, right_paths) if right_paths
                         else ImageFolderIndex(paths))
        except (FileNotFoundError, RuntimeError) as e:
            QMessageBox.warning(self, "Open images", str(e))
            return
        stereo = bool(getattr(new_index, "stereo", False))
        if stereo != self._stereo:
            self._rebuild_canvases(stereo)
        out_dir = (paths[0] if os.path.isdir(paths[0])
                   else os.path.dirname(os.path.abspath(paths[0])))
        out_json = os.path.join(out_dir, "labels_coco.json")
        label = f"{out_dir} (stereo)" if stereo else out_dir
        self._switch_source(new_index, out_json, label)
        if stereo and getattr(new_index, "pairing_warning", None):
            print(f"⚠️ {new_index.pairing_warning}")
            self.statusBar().showMessage(new_index.pairing_warning, 8000)

    # ----------------------- frame playback ----------------------------- #

    def _on_play_pause(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()
        self.side.set_playing(self._playing)

    def _on_play_speed_changed(self, ms: int) -> None:
        self._play_interval_ms = ms
        if self._playing:
            # Restart the timer with the new interval.
            self._play_timer.stop()
            self._play_timer.start(self._play_interval_ms)

    def _prefetch_tick(self) -> None:
        """Decode upcoming frames in the background to keep the LRU cache
        warm during playback."""
        if not self._playing:
            return
        frame_index = self.frame_index
        n = len(frame_index)
        if n == 0:
            return
        # Determine which caches we can warm. Fake test indexes may not
        # expose the cache, in which case we just skip prefetching.
        if self._stereo and hasattr(frame_index, "left"):
            sides = ["left", "right"]
            caches = {
                "left": getattr(getattr(frame_index, "left", None),
                                "_decode_cache", None),
                "right": getattr(getattr(frame_index, "right", None),
                                 "_decode_cache", None),
            }
        elif hasattr(frame_index, "_decode_cache"):
            sides = [None]
            caches = {None: frame_index._decode_cache}
        else:
            return
        for side in sides:
            cache = caches.get(side)
            if cache is None:
                continue
            for offset in range(1, self._prefetch_lookahead + 1):
                idx = self._current_idx + offset
                if idx >= n:
                    break
                frame = (frame_index.frame_at(idx) if side is None
                         else frame_index.frame_at(idx, side=side))
                fp = frame.get("file_path")
                if fp and cache.contains(fp):
                    continue
                QThreadPool.globalInstance().start(
                    _PrefetchRunnable(frame_index, idx, side))

    def _start_playback(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._play_timer.start(self._play_interval_ms)
        self._prefetch_tick()  # warm the first lookahead batch immediately
        self._prefetch_timer.start(100)
        self.statusBar().showMessage(
            f"Playing at ~{1000 / self._play_interval_ms:.0f} fps "
            f"({self._play_interval_ms} ms/frame)", 2000
        )

    def _stop_playback(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._play_timer.stop()
        self._prefetch_timer.stop()
        # Refresh the throttled UI syncs immediately on pause so the
        # keyframe/annotated buttons, progress bar, and box list are
        # accurate for the current frame.
        self._refresh_boxes()
        self._sync_keyframe_button()
        self._sync_annotated_button()
        self._update_progress()
        # Persist progress once per play run instead of per tick.
        if 0 <= self._current_idx < len(self.frame_index):
            self.coco.save(is_final=False)
        self.statusBar().showMessage("Paused", 1500)

    def _on_play_tick(self) -> None:
        """Advance one frame per timer tick; stop at the end."""
        if self._current_idx + 1 >= len(self.frame_index):
            self._stop_playback()
            self.side.set_playing(False)
            self.statusBar().showMessage("Reached last frame", 3000)
            return
        self._on_frame_nav(+1, save=False)

    # ----------------------- frame loading ----------------------------- #

    def _boxes_for_image(self, image_id: Optional[int]) -> List[Dict[str, Any]]:
        """Build the canvas/side-panel box dicts for one COCO image id."""
        boxes = []
        if image_id is None:
            return boxes
        for ann in self.coco.anns_for_image(image_id):
            x, y, bw, bh = ann["bbox"]
            boxes.append({
                "id": ann["id"],
                "bbox": [x, y, bw, bh],
                "cat_id": ann["category_id"],
                "cat_name": self.coco.cat_map.get(ann["category_id"], "?"),
                "mask": ann.get("_mask"),
                "track_id": ann.get("track_id"),
                "interp": ann.get("interp", False),
            })
        return boxes

    def _load_current(self) -> None:
        idx = self._current_idx
        if idx < 0 or idx >= len(self.frame_index):
            return
        ts_real = getattr(self.frame_index, "timestamps_real", True)
        # Decode + fill one canvas per side (just "left" in mono). Frame
        # navigation is index-based and moves all canvases together.
        # Uses load_frame (batched: 1 repaint/canvas instead of 3) — the
        # per-tick repaint cost was capping playback below the speed setting.
        for side, canvas in self.canvases.items():
            frame = self._frame_at_side(idx, side)
            arr = self._decode_side(idx, side)
            h, w = arr.shape[:2]
            image_id = self.coco.ensure_image(frame, w, h, side=side)
            boxes = self._boxes_for_image(image_id)
            pos = (f"ts={frame['timestamp_ns']}" if ts_real else f"index={idx}")
            tag = f"{side.upper()}  |  " if self._stereo else ""
            info = (f"{tag}Frame {idx + 1}/{len(self.frame_index)}  |  {pos}")
            canvas.load_frame(arr, boxes, info, image_id)
        # The compat attribute tracks the ACTIVE side's image id (identical
        # to the only canvas's in mono).
        self._current_image_id = self._active_canvas._image_id

        # Side panel mirrors the active side's box list. During playback,
        # the box list rebuild (QListWidget clear+re-add) is skipped — it's
        # O(boxes) per tick and the user is watching the canvas, not the
        # list. The info label + slider stay live (cheap, always relevant).
        # The full box list + buttons refresh on pause / explicit nav.
        boxes = self._active_canvas._boxes
        if not self._playing:
            self.side.set_boxes(boxes)
            self.side.highlight_box_row(-1)
        self.side.set_slider(idx)
        self.side.set_info(idx, len(self.frame_index),
                           self._frame_at_side(idx, "left")["timestamp_ns"],
                           len(boxes),
                           ts_real=ts_real)
        # Update current index for save paths. We do NOT autosave on every
        # frame navigation (it would write the JSON every tick during
        # playback). Progress is saved:
        #   - on explicit N/B/X keystrokes (see _on_frame_nav / _on_discard_all)
        #   - on slider release (see _on_slider_released)
        #   - on quit (closeEvent, _on_save_quit, _on_quit)
        self.coco.current_idx = idx
        # Throttle the per-frame sync calls during playback: keyframe /
        # annotated button sync + progress bar rebuild are cheap
        # individually but add up at high speeds. During playback we
        # refresh them on a coarse interval via _status_timer (250 ms)
        # instead of every tick; on pause / explicit nav they refresh
        # immediately for accurate UI.
        if self._playing:
            self._play_tick_count += 1
            # Refresh progress + buttons every ~8 ticks during playback
            # (≈4×/s at 30 fps) — smooth enough to feel live, cheap enough
            # not to cap playback speed.
            if self._play_tick_count & 7 == 0:
                self._sync_keyframe_button()
                self._sync_annotated_button()
                self._update_progress()
        else:
            self._sync_keyframe_button()
            self._sync_annotated_button()
            self._update_progress()

    # ----------------------- event handlers ---------------------------- #

    def _on_frame_nav(self, delta: int, save: bool = True) -> None:
        # Forward nav (N) means "this frame is reviewed" — record it.
        if delta > 0:
            self.coco.mark_reviewed(self._current_idx)
        new_idx = self._current_idx + delta
        if 0 <= new_idx < len(self.frame_index):
            self._current_idx = new_idx
            self._load_current()
            if save:
                # Save progress (tmp) on explicit nav — not on autoplay
                # ticks (a full-json save per tick capped playback at a
                # few fps regardless of the speed setting).
                self.coco.save(is_final=False)

    def _on_slider_moved(self, idx: int) -> None:
        if 0 <= idx < len(self.frame_index) and idx != self._current_idx:
            self._current_idx = idx
            self._load_current()
            # No save here: valueChanged fires on every drag tick and
            # save() rewrites the whole COCO json + re-polygonizes every
            # mask — that per-tick cost is what made scrubbing laggy.
            # Progress is saved once per drag on sliderReleased instead.

    def _on_slider_released(self) -> None:
        """Save progress once at the end of a slider drag."""
        if 0 <= self._current_idx < len(self.frame_index):
            self.coco.save(is_final=False)

    def _on_box_deleted(self, ann_id: int) -> None:
        self.coco.remove_box(ann_id)
        self._refresh_boxes()
        self.statusBar().showMessage(f"Deleted box #{ann_id}", 2500)

    def _on_box_moved(self, ann_id: int, x: float, y: float,
                     w: float, h: float) -> None:
        """Commit a drag-move of an existing box. Mask stays attached;
        the user can press R to re-segment if they want a fresh mask."""
        self.coco.move_box(ann_id, x, y, w, h)
        # Don't call _refresh_boxes() — the canvas already shows the new
        # position from the live drag. Just update the side panel info.
        self.side.set_boxes(self._src_canvas()._boxes)
        self.statusBar().showMessage(
            f"Moved box #{ann_id} to ({int(x)},{int(y)})", 2500
        )

    def _on_box_resized(self, ann_id: int, x: float, y: float,
                        w: float, h: float) -> None:
        self.coco.resize_box(ann_id, x, y, w, h)
        self.side.set_boxes(self._src_canvas()._boxes)
        self.statusBar().showMessage(
            f"Resized box #{ann_id} to {int(w)}x{int(h)}", 2500
        )

    def _on_zoom_to_selected(self) -> None:
        """Zoom the canvas so the selected box fills ~80% of the view."""
        canvas = self._src_canvas()
        sel = canvas._selected_idx
        if sel < 0 or sel >= len(canvas._boxes):
            self.statusBar().showMessage("No box selected to zoom to", 2000)
            return
        x, y, w, h = canvas._boxes[sel]["bbox"]
        iw, ih = canvas._image_size
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            return
        vw, vh = canvas.width(), canvas.height()
        # Scale so the box fills 80% of the smaller view dimension.
        scale = min(vw / (w * 1.25), vh / (h * 1.25))
        scale = max(0.05, min(40.0, scale))
        canvas._scale = scale
        # Center the box in the view.
        cx_img = x + w / 2.0
        cy_img = y + h / 2.0
        canvas._offset = QtCore.QPointF(
            vw / 2.0 - cx_img * scale, vh / 2.0 - cy_img * scale
        )
        canvas._user_zoomed = True  # resize must not undo this zoom
        canvas.update()
        self.statusBar().showMessage(
            f"Zoomed to box #{canvas._boxes[sel]['id']}", 2000
        )

    # ----------------------- interpolation ----------------------------- #

    def _sync_keyframe_button(self) -> None:
        """Reflect whether the current frame is a marked keyframe."""
        self.side.btn_keyframe.blockSignals(True)
        self.side.btn_keyframe.setChecked(
            self._current_idx in self.coco.keyframes)
        self.side.btn_keyframe.blockSignals(False)

    def _on_toggle_keyframe(self) -> None:
        """K key / ★ Keyframe button: mark or unmark the current frame."""
        idx = self._current_idx
        if idx in self.coco.keyframes:
            self.coco.keyframes.discard(idx)
            msg = f"Keyframe unmarked (frame {idx + 1})"
        else:
            self.coco.keyframes.add(idx)
            msg = f"Keyframe marked (frame {idx + 1})"
        self._sync_keyframe_button()
        self.coco.save(is_final=False)
        self.statusBar().showMessage(msg, 2500)

    def _sync_annotated_button(self) -> None:
        """Reflect whether the current frame is marked as annotated."""
        self.side.btn_mark_annotated.blockSignals(True)
        self.side.btn_mark_annotated.setChecked(
            self._current_idx in self.coco.annotated_marks)
        self.side.btn_mark_annotated.blockSignals(False)

    def _on_toggle_annotated(self) -> None:
        """'✔ Mark as annotated' button: count this frame as annotated
        (it lands in the output JSON's annotated_image_idxs) even though
        it has no boxes."""
        idx = self._current_idx
        if idx in self.coco.annotated_marks:
            self.coco.annotated_marks.discard(idx)
            msg = f"Annotation mark removed (frame {idx + 1})"
        else:
            self.coco.annotated_marks.add(idx)
            msg = f"Marked as annotated (frame {idx + 1})"
        # The mark changes the saved JSON's annotated_image_idxs, so force
        # a full write on the next save (not just the progress sidecar).
        self.coco.dirty = True
        self._sync_annotated_button()
        self.coco.save(is_final=False)
        self.statusBar().showMessage(msg, 2500)

    @staticmethod
    def _ann_to_interp_dict(ann: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a COCO annotation to a 13-style box dict."""
        x, y, w, h = (float(v) for v in ann["bbox"])
        return {
            "ann_id": ann["id"],
            "category_id": int(ann["category_id"]),
            "track_id": ann.get("track_id"),
            "xywh": [x, y, w, h],
            "xyxy": [x, y, x + w, y + h],
            "center": np.array([x + w / 2.0, y + h / 2.0]),
        }

    def _pair_boxes(self, boxes_a: List[Dict[str, Any]],
                    boxes_b: List[Dict[str, Any]],
                    max_dist: float) -> Tuple[List, List[str]]:
        """Match anchor boxes for interpolation.

        Exact (category_id, track_id) pairs first; the remainder falls back
        to 13's greedy nearest-center match within max_dist (same category).
        Returns (pairs, warnings): pairs is a list of (box_a, box_b).
        """
        mod = _get_interp13()
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        warnings: List[str] = []
        unmatched_a: List[int] = []
        matched_b: set = set()
        for i, ba in enumerate(boxes_a):
            hit = None
            tid = ba.get("track_id")
            if tid is not None:
                for j, bb in enumerate(boxes_b):
                    if j in matched_b:
                        continue
                    if (bb.get("track_id") == tid
                            and bb["category_id"] == ba["category_id"]):
                        hit = j
                        break
            if hit is not None:
                matched_b.add(hit)
                pairs.append((ba, boxes_b[hit]))
            else:
                unmatched_a.append(i)
        unmatched_b = [j for j in range(len(boxes_b))
                       if j not in matched_b]
        # Track ids present on only one anchor side → mismatch warnings.
        tids_a = {ba.get("track_id") for ba in boxes_a}
        tids_b = {bb.get("track_id") for bb in boxes_b}
        for tid in sorted((tids_a - tids_b) - {None}):
            warnings.append(
                f"track {tid}: box on the start frame but no box with the "
                f"same track id on the end frame")
        for tid in sorted((tids_b - tids_a) - {None}):
            warnings.append(
                f"track {tid}: box on the end frame but no box with the "
                f"same track id on the start frame")
        # Nearest-center fallback for the leftovers.
        if unmatched_a and unmatched_b and max_dist > 0:
            subs_a = [boxes_a[i] for i in unmatched_a]
            subs_b = [boxes_b[j] for j in unmatched_b]
            matches, _ua, _ub = mod.match_pairs(subs_a, subs_b, max_dist)
            for i2, j2, _d in matches:
                i, j = unmatched_a[i2], unmatched_b[j2]
                pairs.append((boxes_a[i], boxes_b[j]))
                name = self.coco.cat_map.get(boxes_a[i]["category_id"], "?")
                warnings.append(
                    f"{name}: no shared track id — paired by nearest center "
                    f"(box #{boxes_a[i]['ann_id']} ↔ box #{boxes_b[j]['ann_id']})")
        return pairs, warnings

    def _ensure_image_id(self, frame_idx: int,
                         side: Optional[str] = None) -> Optional[int]:
        """Ensure a COCO image record exists for (frame_idx, side) — no
        full decode.

        Borrows the size from any already-visited image record; only decodes
        the frame itself if no dimensions are known at all. `side` defaults
        to the active canvas's side.
        """
        if side is None:
            side = self._active_canvas.side
        known = self.coco._img_id_by_idx.get((frame_idx, side))
        if known is not None:
            return known
        frame = self._frame_at_side(frame_idx, side)
        w = h = 0
        for img in self.coco.images:
            if img.get("width") and img.get("height"):
                w, h = img["width"], img["height"]
                break
        if not w or not h:
            arr = self._decode_side(frame_idx, side)
            h, w = arr.shape[:2]
        return self.coco.ensure_image(frame, w, h, side=side)

    def _on_interpolate(self) -> None:
        """I key / Interpolate button: flow-interpolate the gap around the
        current frame between the nearest labeled anchor frames.

        If the current frame has boxes it is the END anchor and the gap is
        filled backward; otherwise anchors are needed on both sides. Interior
        frames that already have boxes are skipped. In stereo mode the whole
        operation runs on the ACTIVE side only (anchors and the output boxes
        are per-side).
        """
        if (self._interp_worker is not None
                and self._interp_worker.isRunning()):
            self.statusBar().showMessage("Interpolation already running", 2500)
            return
        cur = self._current_idx
        total = len(self.frame_index)
        if cur < 0 or cur >= total:
            return
        side = self._active_canvas.side
        anchors = self.coco.anchor_candidates(side)
        if not anchors:
            self.statusBar().showMessage(
                "Interpolate: label at least one frame first", 3000)
            return
        before = [f for f in anchors if f < cur]
        after = [f for f in anchors if f > cur]
        if self.coco.frame_has_boxes(cur, side):
            if not before:
                self.statusBar().showMessage(
                    "Interpolate: no earlier labeled frame to start from",
                    4000)
                return
            a, b = max(before), cur
        else:
            if not before or not after:
                self.statusBar().showMessage(
                    "Interpolate: need a labeled frame before AND after the "
                    "current frame", 4500)
                return
            a, b = max(before), min(after)
        if b - a < 2:
            self.statusBar().showMessage(
                f"Nothing to interpolate between frames {a + 1} and {b + 1}",
                3000)
            return
        img_a = self.coco._img_id_by_idx.get((a, side))
        img_b = self.coco._img_id_by_idx.get((b, side))
        anns_a = self.coco.anns_for_image(img_a) if img_a else []
        anns_b = self.coco.anns_for_image(img_b) if img_b else []
        if not anns_a or not anns_b:
            self.statusBar().showMessage(
                "Interpolate: anchor frames must have boxes", 3000)
            return
        boxes_a = [self._ann_to_interp_dict(ann) for ann in anns_a]
        boxes_b = [self._ann_to_interp_dict(ann) for ann in anns_b]
        # Match threshold: fraction of the smaller frame dimension (config:
        # interpolation.match_max_dist_frac), from the anchor image records.
        max_dist = 0.0
        for img_id in (img_a, img_b):
            for img in self.coco.images:
                if img["id"] == img_id and img.get("width") \
                        and img.get("height"):
                    max_dist = max(max_dist,
                                   self.interp_match_frac
                                   * min(img["width"], img["height"]))
                    break
        pairs, warnings = self._pair_boxes(boxes_a, boxes_b, max_dist)
        if not pairs:
            self.statusBar().showMessage(
                "Interpolate: no box pairs to interpolate", 3000)
            return
        if warnings and self.interp_confirm_mismatch:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Track id mismatch")
            msg.setText(
                f"Interpolating frames {a + 1} → {b + 1}:\n\n"
                + "\n".join(f"• {w_}" for w_ in warnings)
                + "\n\nProceed with this pairing?")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Ok
                | QMessageBox.StandardButton.Cancel)
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            if msg.exec() != QMessageBox.StandardButton.Ok:
                return
        elif warnings:
            # Confirmation disabled via config — log the mismatches instead.
            for w_ in warnings:
                print(f"⚠️ Interpolate pairing: {w_}")
        jobs = [{"a": a, "b": b, "box_a": ba, "box_b": bb}
                for ba, bb in pairs]
        tmp_base = str(Path(self.coco.output_json).parent
                       / "_tmp_interp_imgs")
        self.side.set_interp_running(True)
        self.side.set_interp_status(
            f"Interpolating {len(jobs)} pair(s), frames {a + 1}–{b + 1}…")
        for c in self.canvases.values():
            c.setEnabled(False)
        self._interp_side = side
        self._interp_worker = InterpBatchWorker(
            self._side_worker_index(side), jobs, tmp_base,
            flow_method=self.interp_flow_method,
            camera_model=self.interp_camera_model,
            parent=self)
        self._interp_worker._lr_session = self._session_seq
        self._interp_worker.progress_signal.connect(self._on_interp_progress)
        self._interp_worker.finished_signal.connect(self._on_interp_finished)
        self._interp_worker.failed_signal.connect(self._on_interp_failed)
        self._interp_worker.cancelled_signal.connect(
            self._on_interp_cancelled)
        self._interp_worker.start()

    def _on_cancel_interp(self) -> None:
        if self._interp_worker is not None:
            self._interp_worker.cancel()

    def _on_interp_progress(self, done: int, total: int) -> None:
        self.side.set_interp_status(
            f"Interpolating {done}/{total} pair(s)…")

    def _on_interp_finished(self, results: list) -> None:
        if self._stale_sender():
            return
        for c in self.canvases.values():
            c.setEnabled(True)
        self.side.set_interp_running(False)
        side = self._interp_side
        added = 0
        skipped = 0
        with self.coco.undo_stack.group("interpolate boxes"):
            for job, pairs in results:
                for p in range(job["a"] + 1, job["b"]):
                    res = pairs.get(p)
                    if res is None:
                        continue
                    if self.coco.frame_has_boxes(p, side):
                        skipped += 1
                        continue
                    img_id = self._ensure_image_id(p, side)
                    if img_id is None:
                        continue
                    x1, y1, x2, y2 = res["xyxy"]
                    self.coco.add_interp_box(
                        img_id, x1, y1, x2 - x1, y2 - y1,
                        job["box_a"]["category_id"],
                        job["box_a"].get("track_id"),
                        res.get("source", "flow"),
                        res.get("conf", 0.5))
                    added += 1
        self.coco.save(is_final=False)
        self.side.set_interp_status(
            f"Interpolated {added} box(es)"
            + (f", skipped {skipped} labeled frame(s)" if skipped else ""))
        self._refresh_boxes()
        self.statusBar().showMessage(
            f"Interpolation done: {added} box(es) added (Ctrl+Z undoes all)",
            4000)

    def _on_interp_failed(self, err: str) -> None:
        if self._stale_sender():
            print(f"⚠️ Dropping stale interpolation failure from previous "
                  f"source: {err}")
            return
        for c in self.canvases.values():
            c.setEnabled(True)
        self.side.set_interp_running(False)
        self.side.set_interp_status("Interpolation failed")
        QMessageBox.critical(self, "Interpolation failed", err)

    def _on_interp_cancelled(self) -> None:
        if self._stale_sender():
            return
        for c in self.canvases.values():
            c.setEnabled(True)
        self.side.set_interp_running(False)
        self.side.set_interp_status("Interpolation cancelled")
        self._refresh_boxes()
        self.statusBar().showMessage("Interpolation cancelled", 3000)

    # ----------------------- undo / redo -------------------------------- #

    def _on_undo(self) -> None:
        entry = self.coco.undo_stack.pop_undo()
        if entry is None:
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        desc, undo, _redo = entry
        try:
            undo()
        except Exception as e:
            self.statusBar().showMessage(f"Undo failed: {e}", 3000)
            return
        self._refresh_boxes()
        self.statusBar().showMessage(f"Undid: {desc}", 2500)

    def _on_redo(self) -> None:
        entry = self.coco.undo_stack.pop_redo()
        if entry is None:
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        desc, _undo, redo = entry
        try:
            redo()
        except Exception as e:
            self.statusBar().showMessage(f"Redo failed: {e}", 3000)
            return
        self._refresh_boxes()
        self.statusBar().showMessage(f"Redid: {desc}", 2500)

    def _on_box_added(self, image_id: int, x: float, y: float,
                      w: float, h: float, cat_id: int) -> None:
        canvas = self._src_canvas()
        new_ann_id = self.coco.add_box(image_id, x, y, w, h, cat_id)
        self._last_cat_id = cat_id
        self._refresh_boxes()
        # Auto-select the new box so R / Re-seg sel / C / T act on it
        # immediately after drawing (refresh resets the selection).
        for i, b in enumerate(canvas._boxes):
            if b["id"] == new_ann_id:
                canvas._selected_idx = i
                canvas._multi_selected = {i}
                self.side.highlight_box_row(i)
                canvas.update()
                break
        ann = self.coco.get_box(new_ann_id)
        tid = ann.get("track_id") if ann else None
        tid_txt = f" T{tid}" if tid is not None else ""
        self.statusBar().showMessage(
            f"Added box cat={self.coco.cat_map.get(cat_id, '?')}{tid_txt} "
            f"(ann_id={new_ann_id})", 3000
        )
        if self.auto_segment and _SAM3_AVAILABLE:
            # Auto-run SAM3 on the freshly added box.
            img_path = self._write_tmp_image(canvas.side)
            if img_path is not None:
                self._start_sam3_worker(
                    img_path,
                    bboxes_xyxy=[[x, y, x + w, y + h]],
                    concepts=[self.coco.cat_map[cat_id]],
                    ann_ids=[new_ann_id],
                )

    def _assign_pending_cat(self, cat_id: int) -> None:
        """Number-key category pick for the pending rectangle (on the
        active side)."""
        if cat_id not in self.coco.cat_map:
            self.statusBar().showMessage(f"⚠️ Category {cat_id} not found", 3000)
            return
        canvas = self._active_canvas
        rect = canvas.get_pending_rect()
        if rect is None or self._current_image_id is None:
            return
        x, y, w, h = rect
        canvas.reset_state()
        # Reuse the shared add-box path.
        self._on_box_added(self._current_image_id, x, y, w, h, cat_id)

    def _on_discard_all(self) -> None:
        # X discards all boxes on the ACTIVE side of the current frame.
        if self._current_image_id is None:
            return
        anns = list(self.coco.anns_for_image(self._current_image_id))
        if not anns:
            self.coco.mark_reviewed(self._current_idx)
            self._on_frame_nav(+1)
            return
        if not self._confirm_discard(len(anns)):
            return
        self.coco.mark_reviewed(self._current_idx)
        idx = self._current_idx

        def _jump_back(i: int = idx) -> None:
            if 0 <= i < len(self.frame_index) and i != self._current_idx:
                self._current_idx = i
                self._load_current()

        # One undo entry for the whole discard: Ctrl+Z restores every box
        # and jumps back to this frame (the view advances after discard,
        # so without the jump the undo would look like a no-op).
        with self.coco.undo_stack.group(
                f"discard {len(anns)} box(es) on frame {idx + 1}"):
            self.coco.undo_stack.push(
                "jump back", undo=_jump_back, redo=lambda: None)
            for ann in anns:
                self.coco.remove_box(ann["id"])
        self._active_canvas.set_boxes([])
        self._on_frame_nav(+1)

    def _confirm_discard(self, n_boxes: int) -> bool:
        """Confirm the X discard-all. Returns True to proceed."""
        if self._skip_discard_confirm:
            return True
        msg = QMessageBox(self)
        msg.setWindowTitle("Discard all boxes?")
        msg.setText(f"Discard all {n_boxes} box(es) on frame "
                    f"{self._current_idx + 1}?\n(Ctrl+Z restores them.)")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        cb = QCheckBox("Don't ask again this session")
        msg.setCheckBox(cb)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return False
        if cb.isChecked():
            self._skip_discard_confirm = True
        return True

    def _on_next_unlabeled(self) -> None:
        """Jump to the next frame that is neither reviewed nor has boxes.

        Scans forward from the current frame, wrapping around once. A frame
        counts as unlabeled when it has no live annotations AND is not in
        the reviewed set (so discarded frames are skipped too).
        """
        total = len(self.frame_index)
        for step in range(1, total + 1):
            idx = (self._current_idx + step) % total
            if idx in self.coco.reviewed:
                continue
            # A frame counts as labeled when EITHER side has boxes
            # (progress/UX semantics; frame_has_boxes with side=None).
            if self.coco.frame_has_boxes(idx):
                continue
            self._current_idx = idx
            self._load_current()
            self.statusBar().showMessage(
                f"Next unlabeled: frame {idx + 1}/{total}", 2500)
            return
        self.statusBar().showMessage("No unlabeled frames left 🎉", 3000)

    def _on_save(self) -> None:
        """File → Save (Ctrl+S): write the final COCO JSON to the current
        output path without quitting."""
        self.coco.save(is_final=True)
        self.statusBar().showMessage(
            f"Saved → {self.coco.output_json}", 4000)

    def _on_save_as(self) -> None:
        """File → Save as… (Ctrl+Shift+S): write the COCO JSON to a chosen
        location and make it this session's output path (later saves,
        including progress/tmp writes, go there too)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save COCO JSON as", self.coco.output_json,
            "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.coco.output_json = path
        self.coco.progress_file = path.replace(".json", ".progress")
        self.coco.save(is_final=True)
        self.statusBar().showMessage(f"Saved → {path}", 4000)

    def _on_save_quit(self) -> None:
        ret = QMessageBox.question(
            self, "Save final output and quit?",
            f"Write the final COCO JSON to\n{self.coco.output_json}\nand quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.coco.save(is_final=True)
        self._quit_confirmed = True
        self._shutdown_workers()
        QtWidgets.QApplication.quit()

    def _on_quit(self) -> None:
        res = self._confirm_quit()
        if res is None:
            return
        if res:
            self.coco.save(is_final=False)
        self._quit_confirmed = True
        self._shutdown_workers()
        QtWidgets.QApplication.quit()

    def _confirm_quit(self) -> Optional[bool]:
        """Ask before quitting with unsaved changes.

        Returns True = save progress and quit, False = quit without saving,
        None = cancel (stay). Returns True immediately when nothing is dirty.
        """
        if not self.coco.dirty:
            return True
        ret = QMessageBox.question(
            self, "Quit label review?",
            "You have unsaved changes since the last save.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if ret == QMessageBox.StandardButton.Save:
            return True
        if ret == QMessageBox.StandardButton.Discard:
            return False
        return None

    # -------------- category assignment for pending rect -------------- #

    def _on_cat_pick_requested(self, x: float, y: float, w: float, h: float) -> None:
        print(f"📐 Drawn rect x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f} — pick category")
        print("  Categories:")
        for cid, name in sorted(self.coco.cat_map.items()):
            print(f"    {cid} -> {name}")

    def _on_side_cat_clicked(self, cat_id: int) -> None:
        # If we have a pending rect (on the active side), assign; else
        # preselect for the next draw.
        if self._active_canvas.get_pending_rect() is not None:
            self._assign_pending_cat(cat_id)
        else:
            self._on_preselect_cat(cat_id)

    def _refresh_boxes(self) -> None:
        """Rebuild every canvas's box list from the COCO state; the side
        panel mirrors the active side."""
        for canvas in self.canvases.values():
            canvas.set_boxes(self._boxes_for_image(canvas._image_id))
        if self._current_image_id is None:
            return
        canvas = self._active_canvas
        self.side.set_boxes(canvas._boxes)
        self.side.highlight_box_row(canvas._selected_idx)
        self._update_propagate_button()
        self.side.set_info(self._current_idx, len(self.frame_index),
                           self._frame_at_side(self._current_idx, "left")["timestamp_ns"],
                           len(canvas._boxes))
        self._update_progress()

    # ---- box list panel + preselected category ---- #

    def _on_box_list_selected(self, box_idx: int) -> None:
        """Clicking a box-list row selects that box on the active canvas
        (the side panel mirrors the active side)."""
        canvas = self._active_canvas
        if 0 <= box_idx < len(canvas._boxes):
            canvas._selected_idx = box_idx
            canvas._multi_selected = {box_idx}
            canvas.update()
            self.side.highlight_box_row(box_idx)
            self._prefill_recat()

    def _on_add_category(self, name: str) -> None:
        """Add a new category from the side panel's name field."""
        name = name.strip()
        if not name:
            return
        if name in self.coco.cat_name_to_id:
            # Make the no-op visible: select the existing row so it's
            # clear why nothing was added.
            existing = self.coco.cat_name_to_id[name]
            for i in range(self.side.cat_list.count()):
                if self.side.cat_list.item(i).data(Qt.UserRole) == existing:
                    self.side.cat_list.setCurrentRow(i)
                    break
            self.statusBar().showMessage(
                f"⚠️ Category {name!r} already exists (id {existing}) — "
                "selected it in the list", 5000)
            return
        new_id = max((c["id"] for c in self.coco.categories), default=-1) + 1
        self.coco.categories.append({"id": new_id, "name": name})
        self.coco.cat_map[new_id] = name
        self.coco.cat_name_to_id[name] = new_id
        self.coco.dirty = True
        self.side._rebuild_cat_list()
        # Preselect it so the next draw uses it without a keypress.
        self._on_preselect_cat(new_id)
        self.statusBar().showMessage(
            f"Added category {new_id} — {name} (preselected)", 4000)

    def _on_rename_category(self, cat_id: int) -> None:
        """Rename a category from the side panel's Rename button.

        Annotations reference category_id, so a rename touches only the
        category name — all boxes keep their assignment."""
        old = self.coco.cat_map.get(cat_id)
        if old is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename category", f"New name for category {cat_id}:",
            text=old)
        name = name.strip()
        if not ok or not name or name == old:
            return
        if name in self.coco.cat_name_to_id:
            self.statusBar().showMessage(
                f"⚠️ Category {name!r} already exists", 3000)
            return
        for c in self.coco.categories:
            if c["id"] == cat_id:
                c["name"] = name
                break
        self.coco.cat_map[cat_id] = name
        del self.coco.cat_name_to_id[old]
        self.coco.cat_name_to_id[name] = cat_id
        self.coco.dirty = True
        self.side._rebuild_cat_list()
        self._refresh_boxes()  # canvas labels / box list show the name
        self.statusBar().showMessage(
            f"Renamed category {cat_id}: {old!r} → {name!r}", 4000)

    def _on_delete_category(self, cat_id: int) -> None:
        """Delete a category from the side panel's Delete button.

        Boxes using the category are deleted too (soft delete via
        removed_ids, so the undo stack's per-box undo can still restore
        them if the category were re-added — in practice treat as final)."""
        name = self.coco.cat_map.get(cat_id)
        if name is None:
            return
        affected = [a["id"] for a in self.coco.annotations
                    if a["category_id"] == cat_id
                    and a["id"] not in self.coco.removed_ids]
        msg = f"Delete category {cat_id} — {name!r}?"
        if affected:
            msg += (f"\n\n{len(affected)} box(es) use this category and "
                    "will be deleted too.")
        ret = QMessageBox.question(
            self, "Delete category", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.coco.removed_ids.update(affected)
        self.coco.categories = [
            c for c in self.coco.categories if c["id"] != cat_id]
        del self.coco.cat_map[cat_id]
        del self.coco.cat_name_to_id[name]
        self.coco.dirty = True
        # Direct removed_ids mutation bypasses remove_box(), so the ann
        # cache must be invalidated explicitly (otherwise _refresh_boxes
        # would still show the deleted boxes via the stale cache).
        self.coco._invalidate_ann_caches()
        # Clear any pending preselection pointing at the deleted category.
        if self._pending_cat_id == cat_id:
            self._pending_cat_id = None
            for c in self.canvases.values():
                c._drawing = False
        if self.side._preselected_cat_id == cat_id:
            self.side._preselected_cat_id = None
        self.side._rebuild_cat_list()
        self._refresh_boxes()
        self.statusBar().showMessage(
            f"Deleted category {cat_id} — {name!r}"
            + (f" ({len(affected)} box(es) removed)" if affected else ""),
            4000)

    def _on_preselect_cat(self, cat_id: int) -> None:
        """A category was clicked in the side panel — remember it for the
        next draw so the user doesn't need to press a number key."""
        self._pending_cat_id = cat_id
        name = self.coco.cat_map.get(cat_id, "?")
        # Put the canvas in draw mode immediately so the next click-drag draws.
        self._active_canvas._drawing = True
        self._active_canvas.update()
        self.statusBar().showMessage(f"Drawing: {name} — drag on the image", 4000)

    def _on_recat_selected(self, cat_id: int) -> None:
        """Reassign the selected box's category (typed id + Enter)."""
        canvas = self._active_canvas
        sel = canvas._selected_idx
        if not (0 <= sel < len(canvas._boxes)):
            self.statusBar().showMessage("Select a box first", 2500)
            return
        if cat_id not in self.coco.cat_map:
            self.statusBar().showMessage(f"⚠️ Category {cat_id} not found", 3000)
            return
        ann_id = canvas._boxes[sel]["id"]
        if self.coco.set_cat(ann_id, cat_id):
            # Recat updates the sticky category too: the next drawn box
            # auto-assigns the category just chosen here.
            self._last_cat_id = cat_id
            self._refresh_boxes()
            canvas._selected_idx = sel
            canvas._multi_selected = {sel}
            canvas.update()
            name = self.coco.cat_map.get(cat_id, "?")
            self.statusBar().showMessage(
                f"Box #{ann_id} → {name} (cat {cat_id})", 2500)
            canvas.setFocus()  # keep hotkeys working after Enter

    def _prefill_recat(self) -> None:
        """Show the selected box's cat id + track id in their fields."""
        canvas = self._active_canvas
        sel = canvas._selected_idx
        if 0 <= sel < len(canvas._boxes):
            box = canvas._boxes[sel]
            self.side.recat_edit.setText(str(box.get("cat_id", "")))
            self.side.prefill_track(box.get("track_id"))

    def _focus_recat_edit(self) -> None:
        """C key: prefill with the current cat and focus the recat field."""
        self._prefill_recat()
        self.side.recat_edit.setFocus()
        self.side.recat_edit.selectAll()

    def _focus_track_edit(self) -> None:
        """T key: prefill with the current track id and focus the field."""
        if not self.show_track_ids:
            self.statusBar().showMessage(
                "Track ids are hidden — enable them in Config → Advanced "
                "settings → Interpolation", 3500)
            return
        self._prefill_recat()
        self.side.track_edit.setFocus()
        self.side.track_edit.selectAll()

    def _on_track_selected(self, value) -> None:
        """Set the selected box's track id (typed id + Enter; None clears)."""
        canvas = self._active_canvas
        sel = canvas._selected_idx
        if not (0 <= sel < len(canvas._boxes)):
            self.statusBar().showMessage("Select a box first", 2500)
            return
        ann_id = canvas._boxes[sel]["id"]
        if self.coco.set_track_id(ann_id, value):
            self._refresh_boxes()
            canvas._selected_idx = sel
            canvas._multi_selected = {sel}
            canvas.update()
            self.side.prefill_track(value)
            self.statusBar().showMessage(
                f"Box #{ann_id} track id → "
                f"{value if value is not None else '(cleared)'}", 2500)
            canvas.setFocus()  # keep hotkeys working after Enter

    def _update_progress(self) -> None:
        """Refresh the annotation-coverage progress bar in the side panel.

        Counts frames annotated on EITHER side (union) — in stereo an
        annotated frame has up to two image records but is one frame."""
        annotated = len(self.coco.labeled_frame_idxs())
        self.side.set_annotated_progress(annotated, len(self.frame_index))

    # ----------------------- SAM3 segmentation ------------------------- #

    def _set_sam3_status(self, body: str) -> None:
        """SAM3 status line with the pending-queue count, e.g.
        'SAM3 (1 in queue): done — 4 mask(s), 0 failed'."""
        self._sam3_status_body = body
        n = len(self._sam3_queue)
        prefix = f"SAM3 ({n} in queue)" if n else "SAM3"
        self.side.set_sam3_status(f"{prefix}: {body}")

    def _refresh_sam3_status(self) -> None:
        """Re-render the status line (queue count changed, body didn't)."""
        self._set_sam3_status(self._sam3_status_body)

    def _start_next_queued_sam3(self) -> None:
        """Pop the next queued single-frame SAM3 job, if nothing is running."""
        if not self._sam3_queue:
            return
        if self._sam3_busy():
            # Previous worker may still be tearing down — retry shortly.
            QTimer.singleShot(200, self._start_next_queued_sam3)
            return
        job = self._sam3_queue.pop(0)
        if job.get("kind") == "autolabel":
            self._start_autolabel_worker(job["img_path"], job["concepts"],
                                         job["cat_ids"], job["image_id"])
        elif job.get("kind") == "propagate":
            self._start_propagate_worker(
                job["start_frame_idx"], job["seeds"], side=job.get("side"))
        else:
            self._start_sam3_worker(job["img_path"], job["bboxes_xyxy"],
                                    job["concepts"], job["ann_ids"])

    def _write_tmp_image(self, side: Optional[str] = None) -> Optional[str]:
        """Write the current frame's image to a tmp file (for run_sam3).

        File-backed sources are used directly — no copy. Falls back to
        decode_image → PNG when a frame has no backing file. `side` selects
        the stereo side (default: the active canvas's; irrelevant in mono).
        """
        if self._current_idx < 0 or self._current_idx >= len(self.frame_index):
            return None
        if side is None:
            side = self._active_canvas.side
        frame = self._frame_at_side(self._current_idx, side)
        fp = frame.get("file_path")
        if fp and os.path.exists(fp):
            return fp
        tmp_dir = Path(self.coco.output_json).parent / "_tmp_sam3_imgs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        blob = frame.get("image_blob")
        if not blob:
            # Last resort: decode in memory and write a PNG.
            try:
                arr = self._decode_side(self._current_idx, side)
            except Exception as e:
                print(f"⚠️ Failed to decode frame for SAM3: {e}")
                return None
            # The side suffix keeps left/right tmp files from clobbering.
            path = tmp_dir / f"frame_{self._current_idx}_{side}.png"
            try:
                Image.fromarray(arr).save(path)
                self._tmp_image_path = str(path)
                return self._tmp_image_path
            except Exception as e:
                print(f"⚠️ Failed to write tmp image for SAM3: {e}")
                return None
        # Pick suffix from media_type if available, else .jpg
        mt = frame.get("media_type") or "image/jpeg"
        ext = ".jpg"
        if "png" in mt:
            ext = ".png"
        elif "webp" in mt:
            ext = ".webp"
        elif "bmp" in mt:
            ext = ".bmp"
        path = tmp_dir / f"frame_{self._current_idx}_{side}{ext}"
        try:
            with open(path, "wb") as f:
                f.write(blob)
            self._tmp_image_path = str(path)
            return self._tmp_image_path
        except Exception as e:
            print(f"⚠️ Failed to write tmp image for SAM3: {e}")
            return None

    def _on_run_sam3_all(self) -> None:
        """Run SAM3 on every bbox on the current frame. In stereo this
        covers BOTH sides: the first side's job starts immediately, the
        second waits in the SAM3 queue (masks apply by global ann_id)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable. "
                                 "Install ultralytics + segment-anything and "
                                 "place weights at "
                                 "core/sam3/models/sam3-model/sam3.pt")
            return
        ran = False
        for canvas in self.canvases.values():
            img_id = canvas._image_id
            if img_id is None:
                continue
            anns = self.coco.anns_for_image(img_id)
            if not anns:
                continue
            img_path = self._write_tmp_image(canvas.side)
            if img_path is None:
                self._set_sam3_status(
                    f"failed — no image for {canvas.side} frame")
                self.statusBar().showMessage(
                    f"⚠️ Could not prepare the {canvas.side} frame image "
                    f"for SAM3", 4000)
                continue
            bboxes_xyxy = []
            concepts = []
            ann_ids = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                bboxes_xyxy.append([x, y, x + w, y + h])
                concepts.append(
                    self.coco.cat_map.get(ann["category_id"], "object"))
                ann_ids.append(ann["id"])
            self._start_sam3_worker(img_path, bboxes_xyxy, concepts, ann_ids)
            ran = True
        if not ran:
            print("ℹ️ No bboxes on this frame — nothing to segment.")

    def _sam3_all_jobs(self, side: str) -> Tuple[List[Dict[str, Any]], int]:
        """(jobs, total_boxes) for a SAM3-all run on one side: every
        visited frame of that side with boxes that have no mask yet."""
        jobs = []
        total_boxes = 0
        for idx in range(len(self.frame_index)):
            img_id = self.coco._img_id_by_idx.get((idx, side))
            if img_id is None:
                continue  # never visited → no annotations possible
            anns = [a for a in self.coco.anns_for_image(img_id)
                    if a.get("_mask") is None]
            if not anns:
                continue
            bboxes, concepts, ann_ids = [], [], []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                bboxes.append([x, y, x + w, y + h])
                concepts.append(
                    self.coco.cat_map.get(ann["category_id"], "object"))
                ann_ids.append(ann["id"])
            jobs.append({"frame_idx": idx, "bboxes_xyxy": bboxes,
                         "concepts": concepts, "ann_ids": ann_ids})
            total_boxes += len(anns)
        return jobs, total_boxes

    def _on_sam3_all_frames(self) -> None:
        """Background SAM3 over every frame that has boxes without masks.
        In stereo this covers BOTH sides: the left batch runs first, the
        right batch is chained on finish (masks apply by global ann_id, so
        the result handlers are side-agnostic)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable.")
            return
        if self._sam3_busy():
            self.statusBar().showMessage("SAM3 already running", 2500)
            return
        sides = ["left", "right"] if self._stereo else ["left"]
        jobs_by_side = {}
        total_boxes = 0
        for s in sides:
            jobs, n = self._sam3_all_jobs(s)
            if jobs:
                jobs_by_side[s] = jobs
                total_boxes += n
        if not jobs_by_side:
            self.statusBar().showMessage(
                "Nothing to do — every annotated box already has a mask", 4000)
            return
        n_frames = sum(len(j) for j in jobs_by_side.values())
        scope = (f"{len(jobs_by_side)} sides" if self._stereo else
                 f"{n_frames} frame(s)")
        ret = QMessageBox.question(
            self, "Auto-annotate all with SAM3?",
            f"Run SAM3 on {total_boxes} box(es) without masks across "
            f"{scope} ({n_frames} frame(s) total)?\n"
            f"Runs in the background (device: {self.sam3_device}, CPU "
            f"fallback on CUDA OOM). You can cancel anytime.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if ret != QMessageBox.StandardButton.Yes:
            return
        self._sam3_all_jobs_by_side = jobs_by_side
        self._sam3_all_pending = [s for s in sides if s in jobs_by_side]
        self._sam3_all_ok = 0
        self._sam3_all_fail = 0
        self._start_sam3_all_next_side()

    def _start_sam3_all_next_side(self) -> None:
        """Start the SAM3-all batch for the next pending side."""
        side = self._sam3_all_pending.pop(0)
        jobs = self._sam3_all_jobs_by_side[side]
        tmp_dir = str(Path(self.coco.output_json).parent / "_tmp_sam3_imgs")
        self._batch_side = side
        self._batch_frames_done = 0
        tag = f"all[{side}]" if self._stereo else "all"
        self._set_sam3_status(f"{tag}: 0/{len(jobs)} frames…")
        self.side.set_sam3_running(True)
        # Canvas stays enabled: masks apply by ann_id, so the user can
        # keep navigating/drawing on other frames while SAM3 runs.
        self._sam3_batch_worker = SAM3BatchWorker(
            self._side_worker_index(side), jobs, tmp_dir,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            parent=self,
        )
        self._sam3_batch_worker.frame_done_signal.connect(
            self._on_batch_frame_done)
        self._sam3_batch_worker.progress_signal.connect(
            self._on_batch_progress)
        self._sam3_batch_worker.finished_signal.connect(
            self._on_batch_finished)
        self._sam3_batch_worker.failed_signal.connect(self._on_sam3_failed)
        self._sam3_batch_worker.cancelled_signal.connect(
            self._on_batch_cancelled)
        self._sam3_batch_worker._lr_session = self._session_seq
        self._sam3_batch_worker.start()

    def _on_batch_frame_done(self, frame_idx: int, results: list) -> None:
        """Apply one frame's masks (grouped as a single undo entry)."""
        if self._stale_sender():
            return
        with self.coco.undo_stack.group(f"SAM3 masks on frame {frame_idx + 1}"):
            for r in results:
                if r.get("ann_id") is not None and r.get("mask") is not None:
                    self.coco.set_mask(r["ann_id"], r["mask"])
        if frame_idx == self._current_idx:
            self._refresh_boxes()
        self._batch_frames_done += 1
        # Checkpoint every 10 frames so masks survive a crash.
        if self._batch_frames_done % 10 == 0:
            self.coco.save(is_final=False)

    def _on_batch_progress(self, done: int, total: int) -> None:
        tag = (f"all[{self._batch_side}]" if self._stereo else "all")
        self._set_sam3_status(f"{tag}: {done}/{total} frames…")

    def _on_batch_finished(self, n_ok: int, n_fail: int) -> None:
        if self._stale_sender():
            return
        self._sam3_all_ok += n_ok
        self._sam3_all_fail += n_fail
        if self._sam3_all_pending:
            # Stereo run: chain the other side's batch.
            self.coco.save(is_final=False)
            self._start_sam3_all_next_side()
            return
        self.side.set_sam3_running(False)
        self._set_sam3_status(
            f"all: done — {self._sam3_all_ok} mask(s), "
            f"{self._sam3_all_fail} failed")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage(
            f"Auto-annotate finished: {self._sam3_all_ok} masks", 4000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_batch_cancelled(self) -> None:
        if self._stale_sender():
            return
        self._sam3_all_pending = []  # don't chain the other side
        self.side.set_sam3_running(False)
        self._set_sam3_status("all: cancelled")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage("Auto-annotate cancelled", 3000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    # ------------------- SAM3 autolabel (text prompts) ------------------- #

    def _stale_sender(self) -> bool:
        """True when the signal's sender worker was started before the last
        source switch (its ids belong to the old CocoState — drop them).
        Direct (non-signal) calls have no sender and are never stale, which
        keeps the offscreen tests calling handlers directly valid."""
        s = self.sender()
        return (s is not None
                and getattr(s, "_lr_session", self._session_seq)
                != self._session_seq)

    def _sam3_busy(self) -> bool:
        """Any SAM3 worker (box segmentation, autolabel, propagate)
        running?"""
        for w in (self._sam3_worker, self._sam3_batch_worker,
                  self._sam3_autolabel_worker,
                  self._sam3_autolabel_batch_worker,
                  self._sam3_propagate_worker):
            if w is not None and w.isRunning():
                return True
        return False

    def _autolabel_concepts(self):
        """(concepts, cat_ids) for an autolabel run; None (+ a status
        message) when there is nothing to prompt with.

        Categories highlighted in the side list (Ctrl/Shift multi-select)
        restrict the run to those; with no highlight every category is
        prompted."""
        selected = [cid for cid in self.side.get_selected_cat_ids()
                    if cid in self.coco.cat_map]
        if selected:
            return ([self.coco.cat_map[cid] for cid in selected],
                    list(selected))
        if not self.coco.categories:
            self.statusBar().showMessage(
                "No categories yet — add one in the side panel first", 3000)
            return None
        cats = sorted(self.coco.categories, key=lambda c: c["id"])
        return [c["name"] for c in cats], [c["id"] for c in cats]

    def _on_autolabel_frame(self) -> None:
        """Autolabel the current frame (highlighted categories, or all)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                "SAM3 (ultralytics) is not importable.")
            return
        if self._current_image_id is None:
            return
        pair = self._autolabel_concepts()
        if pair is None:
            return
        concepts, cat_ids = pair
        img_path = self._write_tmp_image()
        if img_path is None:
            self._set_sam3_status("autolabel failed — no image for frame")
            return
        self._start_autolabel_worker(img_path, concepts, cat_ids,
                                     self._current_image_id)

    def _start_autolabel_worker(self, img_path: str, concepts: list,
                                cat_ids: list, image_id: int) -> None:
        if self._sam3_busy():
            # Busy — queue the job; it starts when the current run ends.
            self._sam3_queue.append({
                "kind": "autolabel",
                "img_path": img_path,
                "concepts": concepts,
                "cat_ids": cat_ids,
                "image_id": image_id,
            })
            self._refresh_sam3_status()
            self.statusBar().showMessage(
                f"SAM3 busy — autolabel queued "
                f"({len(self._sam3_queue)} in queue)", 2500)
            return
        self._set_sam3_status(
            f"autolabel: detecting {len(concepts)} categor"
            f"{'y' if len(concepts) == 1 else 'ies'}…")
        self.side.set_sam3_running(True)
        # Canvas stays enabled: results apply by image_id, so the user can
        # keep working on other frames while SAM3 runs.
        self._sam3_autolabel_worker = SAM3AutolabelWorker(
            image_path=img_path,
            concepts=concepts,
            cat_ids=cat_ids,
            image_id=image_id,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            parent=self,
        )
        self._sam3_autolabel_worker.finished_signal.connect(
            self._on_autolabel_finished)
        self._sam3_autolabel_worker.failed_signal.connect(self._on_sam3_failed)
        self._sam3_autolabel_worker._lr_session = self._session_seq
        self._sam3_autolabel_worker.start()

    def _apply_autolabel_dets(self, image_id: int,
                              dets: list) -> Tuple[int, int]:
        """Turn detections into annotations (boxes + masks, one undo group).

        Runs class-aware NMS first (SAM3 often returns several overlapping
        masks for one object — highest confidence wins within a category,
        IoU > sam3.autolabel_nms_iou). Detections duplicating an existing
        same-category box (IoU > 0.7) are skipped. Returns (added, skipped).
        """
        # Class-aware NMS over the incoming detections.
        ordered = sorted(
            (d for d in dets if d.get("cat_id") is not None),
            key=lambda d: -d.get("confidence", 0.0))
        kept: list = []
        for d in ordered:
            if any(d["cat_id"] == k["cat_id"]
                   and _iou_xyxy(d["bbox_xyxy"], k["bbox_xyxy"])
                   > self.sam3_nms_iou
                   for k in kept):
                continue
            kept.append(d)
        nms_dropped = len(ordered) - len(kept)
        n_unknown = len(dets) - len(ordered)  # no cat_id — undetectable
        existing = list(self.coco.anns_for_image(image_id))
        added = 0
        skipped = nms_dropped + n_unknown
        with self.coco.undo_stack.group("autolabel"):
            for d in kept:
                cat_id = d["cat_id"]
                x1, y1, x2, y2 = d["bbox_xyxy"]
                w, h = x2 - x1, y2 - y1
                if w < 2 or h < 2:
                    skipped += 1
                    continue
                dup = any(
                    a["category_id"] == cat_id and _iou_xyxy(
                        [x1, y1, x2, y2],
                        [a["bbox"][0], a["bbox"][1],
                         a["bbox"][0] + a["bbox"][2],
                         a["bbox"][1] + a["bbox"][3]]) > 0.7
                    for a in existing)
                if dup:
                    skipped += 1
                    continue
                ann_id = self.coco.add_box(image_id, x1, y1, w, h, cat_id)
                if d.get("mask") is not None:
                    self.coco.set_mask(ann_id, d["mask"])
                existing.append(self.coco.get_box(ann_id))
                added += 1
        return added, skipped

    def _on_autolabel_finished(self, image_id: int, dets: list) -> None:
        if self._stale_sender():
            return
        self.side.set_sam3_running(False)
        added, skipped = self._apply_autolabel_dets(image_id, dets)
        self._set_sam3_status(
            f"autolabel: +{added} box(es)"
            + (f", {skipped} skipped" if skipped else ""))
        if image_id == self._current_image_id:
            self._refresh_boxes()
        self.coco.save(is_final=False)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_autolabel_all(self) -> None:
        """Background text-prompt autolabel over every frame — on the
        ACTIVE side only (in stereo)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                "SAM3 (ultralytics) is not importable.")
            return
        n = len(self.frame_index)
        if n == 0:
            return
        if self._sam3_busy():
            self.statusBar().showMessage("SAM3 already running", 2500)
            return
        side = self._active_canvas.side
        pair = self._autolabel_concepts()
        if pair is None:
            return
        concepts, cat_ids = pair
        selected_note = (f" — restricted to the {len(concepts)} highlighted "
                         f"categor{'y' if len(concepts) == 1 else 'ies'}"
                         if self.side.get_selected_cat_ids() else "")
        ret = QMessageBox.question(
            self, "Autolabel all frames with SAM3?",
            f"Run text-prompt detection on all {n} frame(s) for "
            f"{len(concepts)} categor{'y' if len(concepts) == 1 else 'ies'} "
            f"({', '.join(concepts[:5])}{'…' if len(concepts) > 5 else ''})"
            f"{selected_note}?\n"
            f"Runs in the background (device: {self.sam3_device}, CPU "
            f"fallback on CUDA OOM). Detections become editable boxes with "
            f"masks; duplicates of existing boxes are skipped. "
            f"You can cancel anytime.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if ret != QMessageBox.StandardButton.Yes:
            return
        tmp_dir = str(Path(self.coco.output_json).parent / "_tmp_sam3_imgs")
        self._autolabel_batch_side = side
        self._batch_frames_done = 0
        self._batch_added = 0
        self._set_sam3_status(f"autolabel all: 0/{n} frames…")
        self.side.set_sam3_running(True)
        self._sam3_autolabel_batch_worker = SAM3AutolabelBatchWorker(
            self._side_worker_index(side), list(range(n)), concepts, cat_ids,
            tmp_dir,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            parent=self,
        )
        self._sam3_autolabel_batch_worker.frame_done_signal.connect(
            self._on_autolabel_batch_frame_done)
        self._sam3_autolabel_batch_worker.progress_signal.connect(
            lambda d, t: self._set_sam3_status(
                f"autolabel all: {d}/{t} frames…"))
        self._sam3_autolabel_batch_worker.finished_signal.connect(
            self._on_autolabel_batch_finished)
        self._sam3_autolabel_batch_worker.failed_signal.connect(
            self._on_sam3_failed)
        self._sam3_autolabel_batch_worker.cancelled_signal.connect(
            self._on_autolabel_batch_cancelled)
        self._sam3_autolabel_batch_worker._lr_session = self._session_seq
        self._sam3_autolabel_batch_worker.start()

    def _on_autolabel_batch_cancelled(self) -> None:
        if self._stale_sender():
            return
        self.side.set_sam3_running(False)
        self._set_sam3_status("autolabel all: cancelled")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage("Autolabel-all cancelled", 3000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_autolabel_batch_frame_done(self, frame_idx: int,
                                       dets: list) -> None:
        """Add one frame's autolabel detections as boxes+masks (on the
        side the batch was started on)."""
        if self._stale_sender():
            return
        if dets:
            side = self._autolabel_batch_side
            frame = self._frame_at_side(frame_idx, side)
            mask0 = next((d["mask"] for d in dets
                          if d.get("mask") is not None), None)
            if mask0 is not None:
                h, w = mask0.shape
            else:
                arr = self._decode_side(frame_idx, side)
                h, w = arr.shape[:2]
            image_id = self.coco.ensure_image(frame, w, h, side=side)
            added, _skipped = self._apply_autolabel_dets(image_id, dets)
            self._batch_added += added
            if frame_idx == self._current_idx:
                self._refresh_boxes()
        self._batch_frames_done += 1
        # Checkpoint every 10 frames so boxes survive a crash.
        if self._batch_frames_done % 10 == 0:
            self.coco.save(is_final=False)

    def _on_autolabel_batch_finished(self, total_dets: int) -> None:
        if self._stale_sender():
            return
        self.side.set_sam3_running(False)
        # total_dets is the raw detection count; NMS + existing-box dedup
        # drop many, so report the boxes actually added.
        added = self._batch_added
        self._set_sam3_status(f"autolabel all: done — {added} box(es) added")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage(
            f"Autolabel finished: {added} box(es) added "
            f"({total_dets} detected)", 4000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    # ------------------- SAM3 track propagation ------------------------- #

    def _selected_boxes(self) -> List[Dict[str, Any]]:
        """All selected box dicts on the ACTIVE canvas (multi-select
        aware)."""
        canvas = self._active_canvas
        sel = sorted(i for i in canvas._multi_selected
                     if 0 <= i < len(canvas._boxes))
        if not sel and 0 <= canvas._selected_idx < len(canvas._boxes):
            sel = [canvas._selected_idx]
        return [canvas._boxes[i] for i in sel]

    def _selected_boxes_all_sides(self) -> List[Tuple[str, Dict[str, Any]]]:
        """(side, box) for every selected box across ALL canvases — for
        ops that should honor a cross-side multi-select (Propagate)."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        for side, canvas in self.canvases.items():
            sel = sorted(i for i in canvas._multi_selected
                         if 0 <= i < len(canvas._boxes))
            if not sel and 0 <= canvas._selected_idx < len(canvas._boxes):
                sel = [canvas._selected_idx]
            out.extend((side, canvas._boxes[i]) for i in sel)
        return out

    def _selected_single_box(self) -> Optional[Dict[str, Any]]:
        """The canvas box dict when exactly one box is selected, else None."""
        boxes = self._selected_boxes()
        return boxes[0] if len(boxes) == 1 else None

    def _update_propagate_button(self) -> None:
        """Propagate needs at least one selected box on a non-last frame."""
        ok = (bool(self._selected_boxes_all_sides())
              and self._current_idx < len(self.frame_index) - 1)
        self.side.btn_propagate.setEnabled(ok)

    def _on_propagate_track(self) -> None:
        """Seed SAM3 propagation from the selected boxes (Propagate →).

        With several boxes selected, each distinct track is propagated —
        one run per side covering ALL of that side's seeds (in stereo the
        two sides run back-to-back through the SAM3 queue). Boxes sharing
        a track id on the SAME side collapse into a single seed; boxes
        without a track id each seed a fresh track.
        """
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                "SAM3 (ultralytics) is not importable.")
            return
        boxes = self._selected_boxes_all_sides()
        if not boxes:
            self.statusBar().showMessage(
                "Select one or more boxes first, then Propagate →", 3000)
            return
        n = len(self.frame_index)
        if self._current_idx >= n - 1:
            self.statusBar().showMessage(
                "Seed is on the last frame — nothing to propagate to", 3000)
            return
        # One seed per (side, track id); track-less boxes are separate
        # seeds. Sides are kept apart: a left and a right box sharing a
        # track id are the same object in two views — both propagate.
        seeds: List[Tuple[str, Dict[str, Any]]] = []
        seen_tids = set()
        for side, box in boxes:
            tid = box.get("track_id")
            if tid is not None:
                key = (side, tid)
                if key in seen_tids:
                    continue
                seen_tids.add(key)
            seeds.append((side, box))
        # Preview the labels without consuming fresh track ids (see the
        # single-track comment below): only commit them after confirmation.
        labels = []
        next_fresh = self.coco.peek_fresh_track_id()
        for side, box in seeds:
            tid = box.get("track_id")
            cat_id = box.get("cat_id", 0)
            concept = self.coco.cat_map.get(cat_id, "object")
            if tid is None:
                tid_shown = next_fresh
                next_fresh += 1
            else:
                tid_shown = tid
            label = f"T{tid_shown} ({concept})"
            if self._stereo:
                label += f" [{side}]"
            labels.append(label)
        if self.propagate_method == "chain":
            method_blurb = (
                "Frame-by-frame chain: each object is re-detected with the\n"
                "previous frame's box as prompt, IoU-gated. A track stops\n"
                "permanently at the first frame with no detection.")
        else:
            method_blurb = (
                "SAM3 video memory bank: one session per side covering all\n"
                "of that side's selected boxes. A track reported lost may\n"
                "recover on later frames. The remaining range is built into\n"
                "a temporary clip first.")
        ret = QMessageBox.question(
            self, "Propagate track(s) forward?",
            f"Propagate {', '.join(labels)} from frame "
            f"{self._current_idx + 1} to the end ({n - self._current_idx - 1} "
            f"frame(s)) with method '{self.propagate_method}'?\n\n"
            f"{method_blurb}\n\n"
            f"Frames that already have a box with that track id are skipped. "
            f"Each side runs as one background job "
            f"(device: {self.sam3_device}). Cancel anytime. "
            f"Each side's run is one Ctrl+Z step.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if ret != QMessageBox.StandardButton.Yes:
            return
        by_side: Dict[str, List[Dict[str, Any]]] = {}
        # Track-id changes made for seed-less boxes are recorded per side
        # so each side's composite undo entry reverses only its own
        # track-id assignments (sharing one list across sides would make
        # both sides' undo entries undo the same changes, double-applying
        # _undo_set_track and permanently losing track ids).
        seed_tid_changes_by_side: Dict[str, List[Tuple[int, Optional[int], int]]] = {}
        with self.coco.undo_stack.mute():
            for side, box in seeds:
                tid = box.get("track_id")
                if tid is None:
                    old_tid = box.get("track_id")
                    tid = self.coco._fresh_track_id()
                    self.coco.set_track_id(box["id"], tid)
                    seed_tid_changes_by_side.setdefault(side, []).append(
                        (box["id"], old_tid, tid))
                    box["track_id"] = tid
                cat_id = box.get("cat_id", 0)
                x, y, w, h = box["bbox"]
                by_side.setdefault(side, []).append({
                    "track_id": tid,
                    "cat_id": cat_id,
                    "concept": self.coco.cat_map.get(cat_id, ""),
                    "bbox_xyxy": [x, y, x + w, y + h],
                })
        # One run per side covers ALL of that side's seeds at once; the
        # second side queues behind the first.
        for side, side_seeds in by_side.items():
            self._start_propagate_worker(
                self._current_idx, side_seeds, side,
                seed_tid_changes=seed_tid_changes_by_side.get(side, []))

    def _propagate_label(self, seeds: List[Dict[str, Any]],
                         side: str) -> str:
        """Status/undo label for a propagate run."""
        if len(seeds) == 1:
            return f"T{seeds[0]['track_id']}"
        tag = f" [{side}]" if self._stereo else ""
        return f"{len(seeds)} tracks{tag}"

    def _start_propagate_worker(self, start_frame_idx: int,
                                seeds: List[Dict[str, Any]],
                                side: Optional[str] = None,
                                seed_tid_changes:
                                Optional[List[Tuple[int, Optional[int],
                                                    int]]] = None
                                ) -> None:
        if side is None:
            side = self._active_canvas.side
        if self._sam3_busy():
            self._sam3_queue.append({
                "kind": "propagate",
                "start_frame_idx": start_frame_idx,
                "seeds": seeds,
                "side": side,
            })
            self._refresh_sam3_status()
            self.statusBar().showMessage(
                f"SAM3 busy — propagate queued "
                f"({len(self._sam3_queue)} in queue)", 2500)
            return
        label = self._propagate_label(seeds, side)
        total = len(self.frame_index) - start_frame_idx - 1
        self._set_sam3_status(f"propagate {label}: 0/{total} frames…")
        self.side.set_sam3_running(True)
        tmp_dir = str(Path(self.coco.output_json).parent / "_tmp_sam3_imgs")
        self._sam3_propagate_worker = SAM3PropagateWorker(
            self._side_worker_index(side), start_frame_idx, seeds, tmp_dir,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            method=self.propagate_method,
            min_iou=self.propagate_min_iou,
            min_seed_iou=self.propagate_min_seed_iou,
            parent=self,
        )
        self._sam3_propagate_worker._meta = {
            "seeds": seeds, "side": side,
            "added": 0, "ann_ids": [], "anns": [],
            "seed_tid_changes": seed_tid_changes or []}
        self._last_propagate_meta = self._sam3_propagate_worker._meta
        self._sam3_propagate_worker._lr_session = self._session_seq
        self._sam3_propagate_worker.frame_done_signal.connect(
            self._on_propagate_frame_done)
        self._sam3_propagate_worker.stage_signal.connect(
            lambda s: self._set_sam3_status(f"propagate {label}: {s}"))
        self._sam3_propagate_worker.progress_signal.connect(
            lambda d, t: self._set_sam3_status(
                f"propagate {label}: {d}/{t} frames…"))
        self._sam3_propagate_worker.finished_signal.connect(
            self._on_propagate_finished)
        self._sam3_propagate_worker.failed_signal.connect(
            self._on_propagate_failed)
        self._sam3_propagate_worker.cancelled_signal.connect(
            self._on_propagate_cancelled)
        self._sam3_propagate_worker.start()

    def _propagate_meta(self) -> Dict[str, Any]:
        """Return the meta dict attached to the propagate worker that sent
        the signal, or the legacy fallback. Using the worker's own meta
        makes concurrent/interleaved runs safe."""
        sender = self.sender()
        if sender is not None and isinstance(sender, SAM3PropagateWorker):
            return getattr(sender, "_meta", {})
        return getattr(self, "_last_propagate_meta", {})

    def _on_propagate_frame_done(self, frame_idx: int, dets) -> None:
        """Add propagated boxes (one per seed, keeping each seed's track
        id), or skip. `dets` is aligned with meta["seeds"]."""
        if self._stale_sender():
            return
        meta = self._propagate_meta()
        if not meta:
            return
        side = meta.get("side", "left")
        for seed, det in zip(meta.get("seeds", []), dets or []):
            if det is None:
                continue
            frame = self._frame_at_side(frame_idx, side)
            mask = det.get("mask")
            if mask is not None:
                h, w = mask.shape
            else:
                arr = self._decode_side(frame_idx, side)
                h, w = arr.shape[:2]
            image_id = self.coco.ensure_image(frame, w, h, side=side)
            # Never overwrite a frame that already has this track id.
            have = any(a.get("track_id") == seed["track_id"]
                       for a in self.coco.anns_for_image(image_id))
            if have:
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            # Muted: per-frame pushes are dropped; _end_propagate
            # re-pushes the whole run as ONE undo entry, so user
            # edits made while the run is in flight stay separate.
            with self.coco.undo_stack.mute():
                # Pass the track id in — a separate add_box (which
                # consumes a fresh id) + set_track_id would leak the
                # global counter every frame.
                ann_id = self.coco.add_box(image_id, x1, y1,
                                           x2 - x1, y2 - y1,
                                           seed["cat_id"],
                                           track_id=seed["track_id"])
                ann = self.coco.get_box(ann_id)
                if ann is not None:
                    ann["propagated"] = True
                    ann["confidence"] = float(det.get("confidence", 1.0))
                if mask is not None:
                    self.coco.set_mask(ann_id, mask)
            meta["ann_ids"].append(ann_id)
            # Deep copy so later edits to the annotation don't corrupt the
            # redo snapshot stored in the composite undo entry.
            ann_copy = copy.deepcopy(self.coco.get_box(ann_id))
            meta["anns"].append(ann_copy)
            meta["added"] += 1
        if frame_idx == self._current_idx:
            self._refresh_boxes()
        # Checkpoint every 10 frames so propagated boxes survive a crash.
        if (frame_idx + 1) % 10 == 0:
            self.coco.save(is_final=False)

    def _end_propagate(self, status: str, message: Optional[str]) -> None:
        """Shared teardown for finished / failed / cancelled."""
        meta = self._propagate_meta()
        ids = list(meta.get("ann_ids", []))
        anns = list(meta.get("anns", []))
        seed_tid_changes = list(meta.get("seed_tid_changes", []))
        if ids or seed_tid_changes:
            # One composite undo entry for the whole run (the per-frame
            # pushes were muted). It also reverses any fresh track ids
            # assigned to previously track-less seed boxes.
            coco = self.coco
            seeds = meta.get("seeds", [])
            what = (f"track T{seeds[0]['track_id']}" if len(seeds) == 1
                    else f"{len(seeds)} tracks")
            def _undo():
                for i in reversed(ids):
                    coco._undo_remove(i)
                for ann_id, old_tid, _ in seed_tid_changes:
                    coco._undo_set_track(ann_id, old_tid)
            def _redo():
                for a in anns:
                    coco._redo_add(a)
                for ann_id, _, new_tid in seed_tid_changes:
                    coco._undo_set_track(ann_id, new_tid)
            coco.undo_stack.push(
                f"propagate {what} ({len(ids)} box(es))",
                undo=_undo, redo=_redo)
        self.side.set_sam3_running(False)
        self._set_sam3_status(status)
        self.coco.save(is_final=False)
        self._refresh_boxes()
        if message:
            self.statusBar().showMessage(message, 4000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_propagate_finished(self, n_found: int, lost_map) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta()
        seeds = meta.get("seeds", [])
        label = self._propagate_label(seeds, meta.get("side", "left"))
        # n_found counts detections, but frames that already carried the
        # track id (or a sub-2px box) were skipped — report boxes actually
        # added.
        added = meta.get("added", 0)
        status = f"propagate {label}: done — {added} box(es) added"
        notes = [f"T{seeds[i]['track_id']} lost at frame {f + 1}"
                 for i, f in sorted((lost_map or {}).items())
                 if i < len(seeds)]
        if notes:
            status += f" ({', '.join(notes)})"
        self._end_propagate(status, "P" + status[1:])

    def _on_propagate_failed(self, err: str) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta()
        label = self._propagate_label(meta.get("seeds", []),
                                      meta.get("side", "left"))
        print(f"❌ SAM3 propagate failed: {err}")
        self._end_propagate(f"propagate {label}: failed — {err}",
                            f"SAM3 propagate failed: {err}")

    def _on_propagate_cancelled(self) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta()
        label = self._propagate_label(meta.get("seeds", []),
                                      meta.get("side", "left"))
        added = meta.get("added", 0)
        self._end_propagate(
            f"propagate {label}: cancelled ({added} box(es) kept)",
            "Propagate cancelled — boxes already added were kept")

    def _on_resegment_selected(self) -> None:
        """Re-run SAM3 on the selected bbox(es) (R key / button) — on the
        active side.

        With a shift-click multi-selection, every selected box is
        re-segmented (single worker job, queued as one when busy)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable.")
            return
        canvas = self._active_canvas
        sel = [i for i in sorted(canvas._multi_selected)
               if 0 <= i < len(canvas._boxes)]
        if not sel and 0 <= canvas._selected_idx < len(canvas._boxes):
            sel = [canvas._selected_idx]
        if not sel:
            self.statusBar().showMessage(
                "No box selected — click a box first, then re-seg", 3000)
            print("ℹ️ No box selected — press R after clicking a box.")
            return
        img_path = self._write_tmp_image(canvas.side)
        if img_path is None:
            return
        bboxes_xyxy = []
        concepts = []
        ann_ids = []
        for i in sel:
            box = canvas._boxes[i]
            x, y, w, h = box["bbox"]
            bboxes_xyxy.append([x, y, x + w, y + h])
            concepts.append(
                self.coco.cat_map.get(box.get("cat_id", 0), "object"))
            ann_ids.append(box["id"])
        print(f"🔬 Re-segmenting ann_ids={ann_ids}")
        self._start_sam3_worker(img_path, bboxes_xyxy, concepts, ann_ids)

    def _start_sam3_worker(self, img_path: str, bboxes_xyxy: list,
                           concepts: list, ann_ids: list) -> None:
        if self._sam3_busy():
            # Busy — queue the job; it starts when the current run ends.
            self._sam3_queue.append({
                "img_path": img_path,
                "bboxes_xyxy": bboxes_xyxy,
                "concepts": concepts,
                "ann_ids": ann_ids,
            })
            self._refresh_sam3_status()
            self.statusBar().showMessage(
                f"SAM3 busy — job queued "
                f"({len(self._sam3_queue)} in queue)", 2500)
            return
        self._set_sam3_status(f"running on {len(bboxes_xyxy)} box(es)…")
        self.side.set_sam3_running(True)
        # Canvas stays enabled: masks apply by ann_id, so the user can
        # keep navigating/drawing on other frames while SAM3 runs.
        self._sam3_worker = SAM3Worker(
            image_path=img_path,
            bboxes_xyxy=bboxes_xyxy,
            concepts=concepts,
            ann_ids=ann_ids,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            parent=self,
        )
        self._sam3_worker.finished_signal.connect(self._on_sam3_finished)
        self._sam3_worker.failed_signal.connect(self._on_sam3_failed)
        self._sam3_worker.progress_signal.connect(self._on_sam3_progress)
        self._sam3_worker.cancelled_signal.connect(self._on_sam3_cancelled)
        self._sam3_worker._lr_session = self._session_seq
        self._sam3_worker.start()

    def _on_sam3_progress(self, done: int, total: int, concept: str) -> None:
        self._set_sam3_status(
            f"{done}/{total} concept(s) done — last: {concept}"
        )

    def _on_cancel_sam3(self) -> None:
        cancelled_any = False
        cleared_queue = 0
        # Capture "was running" BEFORE cancel() — a worker may leave
        # isRunning() quickly, and the status text depends on it.
        running_cancelled = False
        if self._sam3_queue:
            cleared_queue = len(self._sam3_queue)
            self._sam3_queue.clear()  # cancel drops queued jobs too
            cancelled_any = True
        for w in (self._sam3_worker, self._sam3_batch_worker,
                  self._sam3_autolabel_batch_worker,
                  self._sam3_propagate_worker):
            if w is not None and w.isRunning():
                w.cancel()
                cancelled_any = True
                running_cancelled = True
        # The single-frame autolabel worker is one predict call with no
        # cooperative cancel — say so instead of pretending to stop it.
        autolabel_running = (
            self._sam3_autolabel_worker is not None
            and self._sam3_autolabel_worker.isRunning())
        if cancelled_any:
            if running_cancelled:
                self._set_sam3_status("cancelling…")
            else:
                # Only queued jobs were dropped — nothing was running.
                self._set_sam3_status(
                    f"cancelled — {cleared_queue} queued job(s) dropped")
                if not autolabel_running:
                    self.side.set_sam3_running(False)
            if autolabel_running:
                self.statusBar().showMessage(
                    "Single-frame autolabel can't be cancelled — it "
                    "finishes on its own", 3500)
        elif autolabel_running:
            self.statusBar().showMessage(
                "Single-frame autolabel can't be cancelled — it finishes "
                "on its own", 3500)

    def _on_sam3_cancelled(self) -> None:
        if self._stale_sender():
            return
        self.side.set_sam3_running(False)
        self._set_sam3_status("cancelled — no masks applied")
        self.statusBar().showMessage("SAM3 cancelled", 2500)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_sam3_finished(self, results: list) -> None:
        if self._stale_sender():
            return
        self.side.set_sam3_running(False)
        n_ok = sum(1 for r in results if r.get("success"))
        n_fail = len(results) - n_ok
        # Surface the first failure reason in the status label — the per-box
        # errors also go to the terminal, but that's easy to miss.
        err_txt = ""
        if n_fail:
            first_err = next(
                (r.get("error") for r in results
                 if not r.get("success") and r.get("error")), None)
            if first_err:
                err_txt = f" — {first_err}"
        self._set_sam3_status(
            f"done — {n_ok} mask(s), {n_fail} failed{err_txt}"
        )
        with self.coco.undo_stack.group("SAM3 masks"):
            for r in results:
                ann_id = r.get("ann_id")
                mask = r.get("mask")
                if ann_id is None:
                    continue
                # Never clear an existing mask on failure — a failed concept
                # returns mask=None and the box may already have a good one.
                if mask is not None:
                    self.coco.set_mask(ann_id, mask)
                if not r.get("success"):
                    print(f"  ⚠️ ann_id={ann_id}: {r.get('error', 'no mask')}")
        self._refresh_boxes()
        # Save progress so masks survive a crash.
        self.coco.save(is_final=False)
        print(f"✅ SAM3 finished — {n_ok}/{len(results)} masks assigned")
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_sam3_failed(self, msg: str) -> None:
        if self._stale_sender():
            print(f"⚠️ Dropping stale SAM3 failure from previous "
                  f"source: {msg}")
            return
        self.side.set_sam3_running(False)
        self._sam3_queue.clear()  # hard failure — don't keep retrying queued jobs
        self._sam3_all_pending = []  # and don't chain the other stereo side
        self._set_sam3_status(f"failed — {msg}")
        QMessageBox.warning(self, "SAM3 failed", msg)

    def _on_toggle_masks(self) -> None:
        new_vis = not self._active_canvas.masks_visible()
        for c in self.canvases.values():
            c.set_masks_visible(new_vis)
        # Keep the side-panel button in sync.
        self.side.btn_masks.blockSignals(True)
        self.side.btn_masks.setChecked(new_vis)
        self.side.btn_masks.setText("Masks: ON" if new_vis else "Masks: OFF")
        self.side.btn_masks.blockSignals(False)

    # ----------------------- shutdown ---------------------------------- #

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # _on_quit / _on_save_quit already confirmed + saved; don't ask twice.
        if not self._quit_confirmed:
            res = self._confirm_quit()
            if res is None:
                ev.ignore()
                return
            if res:
                self.coco.save(is_final=False)
        self._shutdown_workers()
        super().closeEvent(ev)

    def _shutdown_workers(self) -> None:
        """Cancel and reap the background workers (SAM3 single/all-frames/
        autolabel/propagate, interpolation) so no QThread is destroyed
        mid-run at exit — Qt aborts the process when that happens."""
        self._sam3_queue.clear()
        # Stop playback + prefetch so no new prefetch runnables are queued.
        self._playing = False
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._prefetch_timer.isActive():
            self._prefetch_timer.stop()
        # Clear pending prefetch runnables from the global thread pool and
        # wait briefly for any in-flight ones to finish — they only call
        # decode_image (no Qt object access), so a short wait is enough.
        QThreadPool.globalInstance().clear()
        QThreadPool.globalInstance().waitForDone(500)
        for w in (self._sam3_worker, self._sam3_batch_worker,
                  self._sam3_autolabel_worker,
                  self._sam3_autolabel_batch_worker,
                  self._sam3_propagate_worker,
                  self._interp_worker):
            if w is None or not w.isRunning():
                continue
            # The single-frame autolabel worker is one predict call and has
            # no cooperative cancel — it just gets the wait below.
            if hasattr(w, "cancel"):
                w.cancel()
            if not w.wait(2000):
                # A run_sam3 / interpolate call is still in flight —
                # last resort before exit.
                w.terminate()
                w.wait(1000)


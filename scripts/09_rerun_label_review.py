#!/usr/bin/env python3
"""
Interactive 2D bbox reviewer for a Rerun .rrd recording OR a folder of images.

Code layout: this file holds the UI (canvas, side panel, main window) and
the COCO/undo state. The background worker threads (SAM3 segmentation,
optical-flow interpolation) live in ``label_review_workers.py`` next to
this file and are imported back at the top of this module.

Run it
------
    # Rerun mode (recording + embedded Rerun web viewer):
    python scripts/09_rerun_label_review.py \
        --rrd complete3/output.rrd \
        --output_json output/complete3/coco_fresh.json \
        --no-seed --sam3-device cpu

    # Image mode (plain files, no Rerun viewer):
    python scripts/09_rerun_label_review.py \
        --images /path/to/folder_or_image.jpg \
        --output_json output/my_labels/coco.json

    # Idle mode (default when no source is given) — pick a source from the
    # File menu once the window is up:
    python scripts/09_rerun_label_review.py

What this does
---------------
* Opens an existing Rerun ``.rrd`` recording — or, with ``--images``, a set
  of plain image files (one or more files/folders; folders are scanned for
  jpg/jpeg/png/bmp/webp/tif, sorted by name). If every file stem is a bare
  integer (e.g. ``1712345678901234567.jpg``), the filenames are treated as
  nanosecond timestamps: frames sort by timestamp and the slider/info show
  them; otherwise the UI shows the plain image index. Without ``--rrd`` or
  ``--images`` the app starts idle (no Rerun viewer) until you load a
  source from the File menu.
* New categories can be added from the side panel at any time (name field +
  Add button under the category list); the new category gets the next free
  id and is preselected for the next draw. The Rename / Delete buttons next
  to it act on the category selected in the list: rename only changes the
  name (boxes keep their category), delete asks for confirmation and removes
  the boxes using that category too.
* The File menu switches the frame source at runtime (the current session
  is saved first, categories are kept): ``Ctrl+O`` open image files,
  ``Ctrl+Shift+O`` open folder, ``Ctrl+R`` load a Rerun .rrd (indexes it,
  spawns the web viewer, and embeds it on the spot). ``Ctrl+G`` opens the
  Config settings dialog. ``Ctrl+S`` saves the COCO JSON to the current
  output path without quitting; ``Ctrl+Shift+S`` (Save as…) picks a new
  location and keeps saving there. Image sessions write
  ``labels_coco.json`` next to the chosen images; rrd sessions write
  ``<name>_labels.json`` beside the recording unless ``--output_json``
  is given.
* Embeds the official Rerun *web* viewer inside a PyQt6 window using
  ``QWebEngineView`` pointed at a locally hosted ``rerun.serve_web_viewer()``
  instance. The viewer shows the full 3D world + camera images, and you can
  scrub the timeline there exactly like in the native viewer.
* Alongside the Rerun viewer, renders a 2D canvas of the *current frame's*
  camera image, with the existing 2D bboxes loaded from the recording, plus
  any new bboxes you draw. Number keys pick the category (same UX as
  ``08_click_review_coco.py``).
* Persists edits to a COCO json + ``.progress`` file in the same on-disk
  format that ``08_click_review_coco.py`` produces, so downstream scripts
  (08b, 13_interpolate_tracks.py, ...) keep working unchanged.
* The current frame is *driven by the Rerun viewer's timeline*: when you drag
  the timeline in the embedded Rerun viewer, a JS→Python event propagates the
  new time, the script looks up the matching frame, and the 2D canvas updates
  automatically. You can also navigate with N/B keys or a slider.

SAM3 segmentation
-----------------
* ``M`` toggles mask overlay visibility.
* ``Run SAM3 (all)`` button (or ``Shift+M``) runs Ultralytics SAM3 on every
  bbox on the current frame and overlays the resulting masks. Masks are
  stored per-annotation in the COCO output as polygon ``segmentation``
  (same shape as ``scripts/results.json``; the legacy base64-PNG ``mask``
  field is still read on load) so they round-trip with the json file.
  Requesting another run while one is in flight queues it (FIFO); the
  status line shows the pending count, e.g. ``SAM3 (1 in queue): done —
  4 mask(s), 0 failed``. Cancel drops the running job and the queue.
* ``R`` re-segments the *selected* bbox only — useful after you've redrawn a
  bbox to fix a bad SAM3 mask: delete the old box, draw a new one, then press
  R while it's selected to regenerate the mask.
* ``--auto-segment`` runs SAM3 automatically right after each new bbox is
  drawn, so you don't need to press anything.
  While SAM3 runs, the side panel shows per-concept progress and a Cancel
  button (cooperative — the in-flight concept finishes first). The mask
  overlay opacity is adjustable via the slider next to the mask toggle.
  "SAM3 ALL frames" runs SAM3 in the background over every box without a
  mask on every frame; the progress bar above the frame slider shows how
  many frames have at least one box.

The .rrd must contain at least one ``EncodedImage`` entity (auto-detected).
The script auto-detects:
  * the image entity path (first entity with archetype EncodedImage),
  * the matching ``Boxes2D`` sibling entity (``<image>/bboxes2d`` by default),
  * the timeline to use (prefers ``ros_time`` if present, else ``log_time``),
  * categories start empty; they are created from the side panel's Add
    field, come back with a resumed labels file, or are created on the fly
    from the ``Boxes2D:labels`` column of an .rrd (``--json`` can still
    seed an explicit category list).

Timestamps are stored in the COCO output as the ``timestamp_ns`` field on
each image, and a side-table ``timestamp_ns → image_id`` is written into the
JSON so the result can be joined back to the inspection SQLite database
(``complete3/inspection_v2.db`` ``images.timestamp_ns`` column).

USAGE
-----
    python scripts/09_rerun_label_review.py \
        --rrd complete3/output.rrd \
        --output_json output/complete3/coco_reviewed.json \
        [--json <seed coco>] \
        [--image-entity /world/leveled/camera_init/body/camera_right/image] \
        [--timeline ros_time] \
        [--output-yolo-dir ...] [--data-yaml ...] \
        [--grpc-port 9876] [--web-port 9090]

    # or, without an .rrd (mutually exclusive with --rrd):
    python scripts/09_rerun_label_review.py \
        --images <folder | image file> [more paths ...] \
        --output_json output/my_labels/coco.json

    # or with no source at all (idle; pick one later from the File menu):
    python scripts/09_rerun_label_review.py

Key bindings (in the 2D canvas, when it has focus — click it once):
    D / del / backspace : delete selected box(es)
    shift+click : toggle a box in/out of the multi-selection (delete
               applies to all selected; the last clicked box stays the
               primary for move/resize/recat/track)
    A        : toggle draw mode (click-drag a new box)
    N / →    : next frame
    B / ←    : previous frame
    X        : discard ALL boxes on this frame and advance
    S        : save final result and quit
    Q / ESC  : quit (progress saved in tmp file)
    0..9     : when drawing, assign category id to the pending rectangle
               (use the buttons in the side panel for ids > 9). After the
               first box, new draws auto-reuse the previous box's category
               (preselect another cat in the side panel to change it)
    M        : toggle mask overlay visibility
    R        : re-segment the selected bbox with SAM3 (replaces its mask)
    + / =    : zoom in   |  -  : zoom out  |  0 : reset zoom
    arrows   : pan (when zoomed)
    Ctrl+Z   : undo last edit (add/delete/move/resize/mask/discard-all)
    Ctrl+Shift+Z / Ctrl+Y : redo
    U        : jump to the next unlabeled frame (no boxes, not yet reviewed)
    C        : focus the "Cat of selected" field — type any category id
               (e.g. 13) and press Enter to reassign the selected box
    T        : focus the "Track of selected" field — type a track id and
               press Enter to set it (clear the field to unset). Track ids
               are auto-assigned by inheritance: the k-th box drawn on a
               frame gets the k-th track id from the nearest earlier
               annotated frame (box 1 on frame 2 → track 1, etc.), falling
               back to a global counter (1, 2, 3, ...) when there is no
               earlier frame or it has fewer boxes
    K        : toggle keyframe on the current frame (anchors for
               interpolation; keyframes with boxes take priority over
               other labeled frames)
    I        : interpolate — fill the gap between the nearest labeled
               frames with optical-flow boxes (frames that already have
               boxes are skipped; Ctrl+Z undoes the whole fill)

Reviewed frames (marked on N forward-nav and X discard) are tracked in the
.progress sidecar and shown in the status bar; X, S and quitting with
unsaved changes ask for confirmation first.

Config file (--config)
----------------------
A JSON file can adjust interpolation, SAM3, and UI settings. Values in the
config override the corresponding CLI flags. Example:
``scripts/config/label_review.example.json``.

    {
      "interpolation": {
        "flow_method": "dis",         // dis | klt | farneback
        "camera_model": "none",       // none | global
        "match_max_dist_frac": 0.2,   // anchor box pairing distance,
                                      // fraction of min(frame w, h)
        "confirm_mismatch": true      // false = skip the track-id mismatch
                                      // dialog (mismatches are logged)
      },
      "sam3": {
        "device": "cpu",              // cpu | cuda
        "conf": 0.25,
        "auto_segment": false
      },
      "ui": {
        "hide": ["sam3_all_frames"],  // button groups to hide; known groups:
                                      // keyframe, interpolate, jump,
                                      // sam3_run, sam3_all_frames, masks,
                                      // play, rerun_toggle.
                                      // Default (no config): ["interpolate",
                                      // "keyframe"]; set [] to show all.
        "mask_opacity": 47            // initial mask overlay opacity (0-100)
      },
      "tracking": {
        "sticky_ids": false           // false (default): every new box gets
                                      // a fresh auto-increment track id;
                                      // true: the k-th box on a frame
                                      // inherits the k-th track id from the
                                      // nearest earlier annotated frame
                                      // (what interpolation pairing expects)
      }
    }

The same settings can be edited at runtime from the ⚙ Config button in the
menu bar (or File → Config settings…, Ctrl+G): a dialog with checkboxes for
the UI groups plus interpolation / SAM3 / opacity fields, with buttons to
Load from file…, Apply, Save… (writes the JSON above), and Close.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

# Rerun
import rerun as rr
import rerun.blueprint as rrb

# Qt
import PyQt6.QtCore as QtCore
import PyQt6.QtGui as QtGui
import PyQt6.QtWidgets as QtWidgets
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QObject, QEvent, pyqtSlot
# PyQt6 uses scoped enums. Provide short aliases for PyQt5-style names used below.
_QT_HORZ = Qt.Orientation.Horizontal
_QT_VERT = Qt.Orientation.Vertical
Qt.StrongFocus = Qt.FocusPolicy.StrongFocus  # type: ignore[attr-defined]
Qt.NoFocus = Qt.FocusPolicy.NoFocus  # type: ignore[attr-defined]
Qt.AlignCenter = Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
Qt.AlignLeft = Qt.AlignmentFlag.AlignLeft  # type: ignore[attr-defined]
Qt.DashLine = Qt.PenStyle.DashLine  # type: ignore[attr-defined]
Qt.LeftButton = Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
Qt.MiddleButton = Qt.MouseButton.MiddleButton  # type: ignore[attr-defined]
Qt.RightButton = Qt.MouseButton.RightButton  # type: ignore[attr-defined]
Qt.UserRole = Qt.ItemDataRole.UserRole  # type: ignore[attr-defined]
Qt.Key_Escape = Qt.Key.Key_Escape  # type: ignore[attr-defined]
Qt.Key_D = Qt.Key.Key_D  # type: ignore[attr-defined]curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
Qt.Key_Delete = Qt.Key.Key_Delete  # type: ignore[attr-defined]
Qt.Key_A = Qt.Key.Key_A  # type: ignore[attr-defined]
Qt.Key_N = Qt.Key.Key_N  # type: ignore[attr-defined]
Qt.Key_Right = Qt.Key.Key_Right  # type: ignore[attr-defined]
Qt.Key_B = Qt.Key.Key_B  # type: ignore[attr-defined]
Qt.Key_Left = Qt.Key.Key_Left  # type: ignore[attr-defined]
Qt.Key_X = Qt.Key.Key_X  # type: ignore[attr-defined]
Qt.Key_S = Qt.Key.Key_S  # type: ignore[attr-defined]
Qt.Key_Q = Qt.Key.Key_Q  # type: ignore[attr-defined]
Qt.Key_Plus = Qt.Key.Key_Plus  # type: ignore[attr-defined]
Qt.Key_Equal = Qt.Key.Key_Equal  # type: ignore[attr-defined]
Qt.Key_Minus = Qt.Key.Key_Minus  # type: ignore[attr-defined]
Qt.Key_0 = Qt.Key.Key_0  # type: ignore[attr-defined]
Qt.Key_M = Qt.Key.Key_M  # type: ignore[attr-defined]
Qt.Key_R = Qt.Key.Key_R  # type: ignore[attr-defined]
Qt.Key_Shift = Qt.Key.Key_Shift  # type: ignore[attr-defined]
Qt.Key_Space = Qt.Key.Key_Space  # type: ignore[attr-defined]
Qt.Key_F = Qt.Key.Key_F  # type: ignore[attr-defined]
Qt.Key_Z = Qt.Key.Key_Z  # type: ignore[attr-defined]
Qt.Key_Control = Qt.Key.Key_Control  # type: ignore[attr-defined]
Qt.Key_Y = Qt.Key.Key_Y  # type: ignore[attr-defined]
Qt.Key_U = Qt.Key.Key_U  # type: ignore[attr-defined]
Qt.Key_C = Qt.Key.Key_C  # type: ignore[attr-defined]
# Generic fallback: alias every remaining Qt.Key.Key_* member as Qt.Key_*,
# so newly added shortcuts (T, K, I, ...) don't each need a line above.
for _qn in dir(Qt.Key):
    if _qn.startswith("Key_") and not hasattr(Qt, _qn):
        setattr(Qt, _qn, getattr(Qt.Key, _qn))
# Cursor shapes (PyQt6 scoped enums)
Qt.SizeFDiagCursor = Qt.CursorShape.SizeFDiagCursor  # type: ignore[attr-defined]
Qt.SizeBDiagCursor = Qt.CursorShape.SizeBDiagCursor  # type: ignore[attr-defined]
Qt.SizeAllCursor = Qt.CursorShape.SizeAllCursor  # type: ignore[attr-defined]
Qt.CrossCursor = Qt.CursorShape.CrossCursor  # type: ignore[attr-defined]
Qt.NoModifier = Qt.KeyboardModifier.NoModifier  # type: ignore[attr-defined]
Qt.ControlModifier = Qt.KeyboardModifier.ControlModifier  # type: ignore[attr-defined]
Qt.ShiftModifier = Qt.KeyboardModifier.ShiftModifier  # type: ignore[attr-defined]
from PyQt6.QtGui import QPen, QColor, QPainter, QPixmap, QFont, QTransform
# PyQt6 scoped-enum shims for QtGui
QPainter.Antialiasing = QPainter.RenderHint.Antialiasing  # type: ignore[attr-defined]
QPainter.SmoothPixmapTransform = QPainter.RenderHint.SmoothPixmapTransform  # type: ignore[attr-defined]
QtGui.QImage.Format_RGB888 = QtGui.QImage.Format.Format_RGB888  # type: ignore[attr-defined]
QtGui.QImage.Format_ARGB32 = QtGui.QImage.Format.Format_ARGB32  # type: ignore[attr-defined]
# QFont doesn't accept "Sans" string in PyQt6; we use QFont() with family name below.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSlider,
    QSplitter, QFrame, QSizePolicy, QMessageBox, QCheckBox,
    QLineEdit, QProgressBar, QFileDialog,
)
# QSizePolicy scoped enum alias
QSizePolicy.Expanding = QSizePolicy.Policy.Expanding  # type: ignore[attr-defined]
QSizePolicy.Fixed = QSizePolicy.Policy.Fixed  # type: ignore[attr-defined]
QSizePolicy.Preferred = QSizePolicy.Policy.Preferred  # type: ignore[attr-defined]
from PyQt6.QtGui import QShortcut, QAction
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebChannel import QWebChannel
    _HAS_WEBENGINE = True
except Exception:
    _HAS_WEBENGINE = False
    QWebEngineView = None  # type: ignore[assignment]
    QWebChannel = None  # type: ignore[assignment]

# Rerun experimental chunk reader for parsing the .rrd
try:
    import rerun.experimental as exp
    _HAS_EXP = True
except Exception:
    _HAS_EXP = False

import pyarrow as pa

# Worker threads (SAM3 / interpolation) live in a sibling module to keep
# this file UI-focused. The script dir goes on sys.path so the import also
# works when this file is exec'd by path (tests) instead of run as a script.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from label_review_workers import (  # noqa: E402
    SAM3Worker, SAM3BatchWorker, InterpBatchWorker,
    _get_interp13, _SAM3_AVAILABLE, run_sam3,
)



# ---------------------------------------------------------------------------
# Mask encoding / decoding helpers (PNG in-memory)
# ---------------------------------------------------------------------------

def _encode_mask_png(mask: np.ndarray) -> Optional[bytes]:
    """Encode a boolean HxW mask as PNG bytes (single-channel, 0/255)."""
    if mask is None or mask.size == 0:
        return None
    arr = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decode_mask_png(blob: bytes) -> Optional[np.ndarray]:
    """Decode PNG bytes back to a boolean HxW mask. Returns None on failure."""
    if not blob:
        return None
    try:
        with Image.open(io.BytesIO(blob)) as im:
            arr = np.array(im.convert("L"))
        return arr > 0
    except Exception:
        return None


def _mask_to_polygons(mask: np.ndarray) -> List[List[int]]:
    """Convert a boolean HxW mask to COCO polygon segmentation format —
    a list of flat [x1, y1, x2, y2, ...] outer contours in int pixels
    (same shape as the ``segmentation`` field in scripts/results.json).
    Returns [] when the mask is empty or no contour has ≥3 points."""
    import cv2  # lazy: heavy import, only needed when masks are saved
    if mask is None or mask.size == 0 or not mask.any():
        return []
    arr = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        flat = c.reshape(-1).tolist()
        if len(flat) >= 6:  # a polygon needs at least 3 points
            polys.append([int(v) for v in flat])
    return polys


def _polygons_to_mask(polys: List[List[float]], height: int,
                      width: int) -> Optional[np.ndarray]:
    """Rasterize COCO polygon segmentation back to a boolean HxW mask."""
    import cv2  # lazy
    if not polys or height <= 0 or width <= 0:
        return None
    pts = [np.array(p, np.int32).reshape(-1, 2) for p in polys
           if len(p) >= 6]
    if not pts:
        return None
    arr = np.zeros((height, width), np.uint8)
    cv2.fillPoly(arr, pts, 1)
    return arr.astype(bool)



# ---------------------------------------------------------------------------
# RrdFrameIndex: scan the .rrd, build (frame_idx -> timestamp_ns, image bytes)
# ---------------------------------------------------------------------------

class RrdFrameIndex:
    """Scan a .rrd once and build an index of frames for one image entity.

    A "frame" is a unique value of the chosen timeline at which an
    ``EncodedImage`` was logged for ``image_entity``. Each frame stores:
      * ``frame_idx`` (0-based, ordered by timeline value)
      * ``timestamp_ns`` (int nanos since epoch, taken from the chosen timeline)
      * ``log_time_ns`` (int nanos since epoch from ``log_time``)
      * ``image_bytes`` (the EncodedImage blob, JPEG/PNG bytes)
      * ``existing_boxes`` (list of (cx, cy, hw, hh, label) from Boxes2D)
    """

    def __init__(
        self,
        rrd_path: str,
        image_entity: Optional[str] = None,
        bboxes_entity: Optional[str] = None,
        timeline: Optional[str] = None,
        progress_cb=None,
    ):
        if not _HAS_EXP:
            raise RuntimeError(
                "rerun.experimental is required to read .rrd files. "
                "Install rerun-sdk >= 0.30."
            )
        self.rrd_path = str(rrd_path)
        self.progress_cb = progress_cb
        reader = exp.RrdReader(self.rrd_path)
        recs = reader.recordings()
        if not recs:
            raise RuntimeError(f"No recordings found in {self.rrd_path}")
        store = reader.store(store=recs[0])

        # ---- Auto-detect image entity, bboxes entity, timeline ----
        schema_cols = list(store.schema())
        image_entities = sorted({
            str(c.entity_path)
            for c in schema_cols
            if getattr(c, "archetype", None) == "rerun.archetypes.EncodedImage"
        })
        if not image_entities:
            raise RuntimeError(
                "No EncodedImage entity found in this .rrd. "
                "Cannot label images without an image stream."
            )
        self.image_entity = image_entity or image_entities[0]
        if image_entity and image_entity not in image_entities:
            raise RuntimeError(
                f"--image-entity {image_entity} not found. "
                f"Available: {image_entities}"
            )
        self.image_entity = image_entity or image_entities[0]

        # Bboxes sibling: <image_entity>/bboxes2d (auto) unless overridden.
        candidate_bbox = bboxes_entity or f"{self.image_entity}/bboxes2d"
        bbox_entities = sorted({
            str(c.entity_path)
            for c in schema_cols
            if getattr(c, "archetype", None) == "rerun.archetypes.Boxes2D"
        })
        self.bboxes_entity = (
            candidate_bbox if candidate_bbox in bbox_entities
            else (bbox_entities[0] if bbox_entities else None)
        )

        # Timeline: prefer ros_time, then log_time.
        if timeline is None:
            timeline = "ros_time"
        # Validate the timeline exists. Timeline index columns appear as
        # top-level columns named just `log_time`, `ros_time`, etc., without
        # an entity_path attribute (they apply to the whole recording).
        available_timelines = set()
        for c in schema_cols:
            if hasattr(c, "entity_path"):
                continue
            name = getattr(c, "name", "")
            if name and not name.startswith("/") and not name.startswith("Index("):
                available_timelines.add(name)
        if timeline not in available_timelines:
            if "log_time" in available_timelines:
                timeline = "log_time"
            elif available_timelines:
                timeline = next(iter(sorted(available_timelines)))
            else:
                raise RuntimeError("Image entity has no timeline column.")
        self.timeline = timeline

        # ---- Stream chunks and collect per-frame data ----
        self.frames: List[Dict[str, Any]] = []
        self._scan(store)

    # ------------------------------------------------------------------ #

    def _scan(self, store) -> None:
        """Walk the chunk stream, collect image + bboxes for each timeline value."""
        # First pass: collect image chunks. We accumulate per-timestamp rows.
        # Each chunk's rows share a timeline column; we de-duplicate by
        # (ros_time) to handle multi-row chunks (rare but possible).
        img_rows: Dict[int, Dict[str, Any]] = {}
        bbox_rows: Dict[int, List[Tuple[float, float, float, float, str]]] = {}

        stream = store.stream()
        n_chunks = 0
        for ch in stream:
            n_chunks += 1
            ep = str(ch.entity_path)
            if ep != self.image_entity and ep != self.bboxes_entity:
                continue
            rb = ch.to_record_batch()
            schema_names = rb.schema.names

            # Determine which timeline column to use.
            tl_name = None
            for nm in schema_names:
                if nm == self.timeline:
                    tl_name = nm
                    break
            if tl_name is None:
                continue

            tl_col = rb.column(tl_name)
            # Convert timestamp[ns] → int64
            try:
                tl_int = tl_col.cast(pa.int64())
            except Exception:
                # already int64
                tl_int = tl_col

            if ep == self.image_entity:
                blob_col = rb.column("EncodedImage:blob")
                mt_col = (
                    rb.column("EncodedImage:media_type")
                    if "EncodedImage:media_type" in schema_names
                    else None
                )
                log_col = (
                    rb.column("log_time").cast(pa.int64())
                    if "log_time" in schema_names else None
                )
                n = len(tl_int)
                for i in range(n):
                    if tl_col[i].is_valid is False if hasattr(tl_col[i], "is_valid") else False:
                        continue
                    ts_ns = int(tl_int[i].as_py())
                    blob_val = blob_col[i].as_py()
                    # blob is list[list[uint8]] per rerun schema — flatten.
                    blob = b"".join(bytes(x) for x in blob_val) if blob_val else b""
                    mt = None
                    if mt_col is not None:
                        mt_list = mt_col[i].as_py()
                        if mt_list:
                            mt = mt_list[0] if isinstance(mt_list, list) else mt_list
                    log_ns = int(log_col[i].as_py()) if log_col is not None else ts_ns
                    # If multiple rows share the same timestamp, prefer the
                    # first non-empty one (they should be identical anyway).
                    if ts_ns not in img_rows or not img_rows[ts_ns].get("blob"):
                        img_rows[ts_ns] = {
                            "blob": blob,
                            "media_type": mt,
                            "log_time_ns": log_ns,
                        }
            elif ep == self.bboxes_entity:
                centers_col = rb.column("Boxes2D:centers")
                hs_col = rb.column("Boxes2D:half_sizes")
                labels_col = (
                    rb.column("Boxes2D:labels")
                    if "Boxes2D:labels" in schema_names
                    else None
                )
                n = len(tl_int)
                for i in range(n):
                    ts_ns = int(tl_int[i].as_py())
                    centers = centers_col[i].as_py() or []
                    hsizes = hs_col[i].as_py() or []
                    labels = labels_col[i].as_py() if labels_col is not None else []
                    boxes = []
                    for j in range(min(len(centers), len(hsizes))):
                        cx, cy = centers[j]
                        hw, hh = hsizes[j]
                        lbl = labels[j] if j < len(labels) else ""
                        boxes.append((float(cx), float(cy),
                                      float(hw), float(hh), str(lbl)))
                    bbox_rows.setdefault(ts_ns, []).extend(boxes)

            if self.progress_cb and n_chunks % 50 == 0:
                self.progress_cb(n_chunks, len(img_rows))

        # Build ordered frame list.
        ts_sorted = sorted(img_rows.keys())
        for idx, ts in enumerate(ts_sorted):
            info = img_rows[ts]
            self.frames.append({
                "frame_idx": idx,
                "timestamp_ns": ts,
                "log_time_ns": info["log_time_ns"],
                "image_blob": info["blob"],
                "media_type": info["media_type"],
                "existing_boxes": bbox_rows.get(ts, []),
            })

    # ------------------------- accessors ------------------------------ #

    def __len__(self) -> int:
        return len(self.frames)

    def frame_at(self, idx: int) -> Dict[str, Any]:
        return self.frames[idx]

    def decode_image(self, idx: int) -> np.ndarray:
        blob = self.frames[idx]["image_blob"]
        if not blob:
            # Return a small placeholder so the canvas doesn't crash.
            return np.zeros((4, 4, 3), dtype=np.uint8)
        with Image.open(io.BytesIO(blob)) as im:
            return np.array(im.convert("RGB"))

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        """Binary search the sorted timeline. Returns -1 if not found."""
        lo, hi = 0, len(self.frames) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self.frames[mid]["timestamp_ns"]
            if v == ts_ns:
                return mid
            elif v < ts_ns:
                lo = mid + 1
            else:
                hi = mid - 1
        # Snap to nearest.
        if lo >= len(self.frames):
            return len(self.frames) - 1
        if hi < 0:
            return 0
        return lo if abs(self.frames[lo]["timestamp_ns"] - ts_ns) < \
                    abs(self.frames[hi]["timestamp_ns"] - ts_ns) else hi


# ---------------------------------------------------------------------------
# Image-folder frame index — same interface as RrdFrameIndex, backed by
# plain image files instead of an .rrd recording (used with --images).
# ---------------------------------------------------------------------------

class ImageFolderIndex:
    """Frame index over plain image files.

    Accepts a mix of image files and folders; folders are scanned
    (non-recursively) for common image extensions and sorted by name.
    If every file stem is a bare integer (e.g. ``1712345678901234567.jpg``),
    the filenames are treated as nanosecond timestamps: frames are sorted
    by timestamp and the slider/info show the real timestamps. Otherwise
    timestamps are synthetic (1 ms per frame) and the UI shows the image
    index instead.
    Exposes the same interface as RrdFrameIndex: ``__len__``, ``frame_at``,
    ``decode_image``, ``find_idx_by_timestamp``, and a ``.frames`` list with
    timestamp_ns / log_time_ns / frame_idx / existing_boxes / image_blob keys.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, paths: List[str], allow_empty: bool = False):
        files: List[str] = []
        for p in paths:
            if os.path.isdir(p):
                for name in sorted(os.listdir(p)):
                    if os.path.splitext(name)[1].lower() in self.IMG_EXTS:
                        files.append(os.path.join(p, name))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"--images path not found: {p}")
        if not files and not allow_empty:
            raise RuntimeError(f"No image files found under: {paths}")
        # Timestamp-named series? Only when EVERY stem is a bare integer.
        stems = [os.path.splitext(os.path.basename(fp))[0] for fp in files]
        self.timestamps_real = bool(stems) and all(s.isdigit() for s in stems)
        if self.timestamps_real:
            files = sorted(files,
                           key=lambda fp: (int(os.path.splitext(
                               os.path.basename(fp))[0]), fp))
        self.files = files
        self.frames: List[Dict[str, Any]] = []
        for idx, fp in enumerate(files):
            if self.timestamps_real:
                ts = int(os.path.splitext(os.path.basename(fp))[0])
            else:
                ts = idx * 1_000_000  # synthetic: 1 ms per frame
            self.frames.append({
                "frame_idx": idx,
                "timestamp_ns": ts,
                "log_time_ns": ts,
                "image_blob": None,
                "media_type": None,
                "existing_boxes": [],
                "file_path": fp,
                "file_name": os.path.basename(fp),
            })

    def __len__(self) -> int:
        return len(self.frames)

    def frame_at(self, idx: int) -> Dict[str, Any]:
        return self.frames[idx]

    def decode_image(self, idx: int) -> np.ndarray:
        with Image.open(self.frames[idx]["file_path"]) as im:
            return np.array(im.convert("RGB"))

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        if not self.frames:
            return -1
        if not self.timestamps_real:
            # Synthetic timestamps are idx * 1 ms; snap to the nearest frame.
            return min(max(round(ts_ns / 1_000_000), 0),
                       len(self.frames) - 1)
        # Binary search the sorted real timestamps; snap to nearest.
        lo, hi = 0, len(self.frames) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self.frames[mid]["timestamp_ns"]
            if v == ts_ns:
                return mid
            elif v < ts_ns:
                lo = mid + 1
            else:
                hi = mid - 1
        if lo >= len(self.frames):
            return len(self.frames) - 1
        if hi < 0:
            return 0
        return lo if abs(self.frames[lo]["timestamp_ns"] - ts_ns) < \
                    abs(self.frames[hi]["timestamp_ns"] - ts_ns) else hi


class EmptyIndex:
    """Zero-frame index used when the app starts with no source (idle mode;
    the user picks a source via the File menu)."""

    timestamps_real = False
    frames: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return 0

    def frame_at(self, idx: int) -> Dict[str, Any]:
        raise IndexError(idx)

    def decode_image(self, idx: int) -> np.ndarray:
        raise IndexError(idx)

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        return -1


# ---------------------------------------------------------------------------
# Undo stack — keeps a bounded history of inverse operations.
# ---------------------------------------------------------------------------

class UndoStack:
    """A small stack of (description, undo_callable, redo_callable) tuples.

    Each mutation in CocoState pushes an entry here. Ctrl+Z pops and runs
    the undo callable; Ctrl+Shift+Z / Ctrl+Y pops and runs the redo callable
    (the redo stack is cleared on any new mutation).

    Use ``with stack.group("...")`` around a multi-step mutation (e.g.
    discard-all) so it undoes/redoes as a single entry.

    Callables are zero-arg. They run on the main thread synchronously.
    """

    MAX_DEPTH = 100

    def __init__(self) -> None:
        self._undo: List[Tuple[str, Any, Any]] = []
        self._redo: List[Tuple[str, Any, Any]] = []
        # When not None, push() collects entries here instead of pushing
        # them directly; group() flushes the batch as one composite entry.
        self._batch: Optional[List[Tuple[str, Any, Any]]] = None

    def push(self, description: str, undo: Any, redo: Any) -> None:
        """Push an (undo, redo) pair. Clears the redo stack."""
        if self._batch is not None:
            self._batch.append((description, undo, redo))
            self._redo.clear()
            return
        self._undo.append((description, undo, redo))
        if len(self._undo) > self.MAX_DEPTH:
            self._undo.pop(0)
        self._redo.clear()

    @contextmanager
    def group(self, description: str):
        """Coalesce every push inside the block into one undo/redo entry.

        Nested groups merge into the outermost one. Undo runs the collected
        undo callables in reverse order; redo runs them in original order.
        """
        outer = self._batch
        self._batch = []
        try:
            yield
        finally:
            entries = self._batch
            self._batch = outer
        if not entries:
            return
        if outer is not None:
            outer.extend(entries)
            return

        def undo_all(entries=tuple(entries)) -> None:
            for _desc, undo, _redo in reversed(entries):
                undo()

        def redo_all(entries=tuple(entries)) -> None:
            for _desc, _undo, redo in entries:
                redo()

        self.push(description, undo_all, redo_all)

    def pop_undo(self) -> Optional[Tuple[str, Any, Any]]:
        if not self._undo:
            return None
        entry = self._undo.pop()
        # Move to redo stack so Ctrl+Shift+Z can re-apply.
        self._redo.append(entry)
        return entry

    def pop_redo(self) -> Optional[Tuple[str, Any, Any]]:
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry)
        return entry

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


# ---------------------------------------------------------------------------
# COCO state
# ---------------------------------------------------------------------------

class CocoState:
    """In-memory COCO dataset, mirroring 08_click_review_coco.py's schema.

    Stores per-image: image_id (1-based), timestamp_ns, file_name, width,
    height. Stores per-annotation: id, image_id, category_id, bbox (xywh),
    area, iscrowd. Tracks removed ids so the seed boxes can be 'deleted'.
    """

    def __init__(self, output_json: str, categories: List[Dict[str, Any]],
                 undo_stack: Optional[UndoStack] = None):
        self.output_json = output_json
        self.progress_file = output_json.replace(".json", ".progress")
        self.categories = categories
        self.cat_map = {c["id"]: c["name"] for c in categories}
        self.cat_name_to_id = {c["name"]: c["id"] for c in categories}
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.removed_ids: set = set()
        self.current_idx = 0
        # Frame indices the user has explicitly reviewed (N forward-nav or
        # X discard). Persisted in the .progress sidecar.
        self.reviewed: set = set()
        # Frames the user marked as interpolation keyframes (K key / button).
        # Persisted in the .progress sidecar like `reviewed`.
        self.keyframes: set = set()
        # Track id assignment for newly drawn boxes. Default (False): every
        # new box gets a fresh id from the global auto-increment counter.
        # When True (config "tracking": {"sticky_ids": true}), the k-th box
        # drawn on a frame inherits the k-th track id from the nearest
        # earlier annotated frame (see _next_track_id) — what interpolation
        # pairing expects. Deleted ids are never recycled either way.
        self.sticky_track_ids: bool = False
        self._track_next: int = 1
        self._img_id_by_ts: Dict[int, int] = {}
        self._img_id_by_idx: Dict[int, int] = {}
        self._ann_id_next = 1
        # Dirty flag — True when there are unsaved mutations since the last
        # successful save(). Cleared by save(). UI shows a "●" indicator.
        self.dirty: bool = False
        self.undo_stack: UndoStack = undo_stack or UndoStack()

    # ------------------------- persistence ---------------------------- #

    def load_existing(self) -> None:
        # Prefer the newest of the final JSON and the _tmp progress JSON —
        # quitting with Q saves to _tmp only, so without this the previous
        # session's edits would be silently ignored on relaunch.
        tmp_json = self.output_json.replace(".json", "_tmp.json")
        candidates = [p for p in (self.output_json, tmp_json)
                      if os.path.exists(p)]
        if candidates:
            path = max(candidates, key=os.path.getmtime)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.images = data.get("images", [])
            self.annotations = data.get("annotations", [])
            # Merge categories instead of replacing: the saved file may
            # predate categories added later via the side panel or --json
            # seed, so union by id and name rather than clobbering.
            saved_cats = data.get("categories", [])
            # Only inherit categories from a session that actually has
            # content. A totally empty previous session (0 imgs, 0 anns —
            # e.g. an idle window opened and closed) has nothing worth
            # preserving, and merging would resurrect stale defaults.
            if saved_cats and (self.images or self.annotations):
                have_ids = {c["id"] for c in saved_cats}
                have_names = {c["name"] for c in saved_cats}
                merged = list(saved_cats)
                for c in self.categories:
                    if c["id"] not in have_ids and c["name"] not in have_names:
                        merged.append(c)
                self.categories = merged
            self.cat_map = {c["id"]: c["name"] for c in self.categories}
            self.cat_name_to_id = {c["name"]: c["id"] for c in self.categories}
            img_dims = {i["id"]: (i.get("height", 0), i.get("width", 0))
                        for i in self.images}
            for ann in self.annotations:
                self._ann_id_next = max(self._ann_id_next, ann["id"] + 1)
                # Rebuild the global track-id counter so new boxes continue
                # the existing numbering without reusing ids.
                tid = ann.get("track_id")
                if isinstance(tid, int) and not isinstance(tid, bool):
                    self._track_next = max(self._track_next, tid + 1)
                # Restore the in-memory mask: COCO polygon "segmentation"
                # (current format) or legacy base64 PNG ("mask").
                mask_b64 = ann.get("mask")
                if isinstance(mask_b64, str) and mask_b64:
                    try:
                        import base64
                        # None on decode failure; no `or None` — ndarray
                        # truthiness is ambiguous and raises.
                        ann["_mask"] = _decode_mask_png(
                            base64.b64decode(mask_b64))
                    except Exception:
                        ann["_mask"] = None
                elif ann.get("segmentation"):
                    h, w = img_dims.get(ann["image_id"], (0, 0))
                    ann["_mask"] = _polygons_to_mask(ann["segmentation"], h, w)
                else:
                    ann["_mask"] = ann.get("_mask")  # may be None or ndarray
            for img in self.images:
                ts = img.get("timestamp_ns")
                if ts is not None:
                    self._img_id_by_ts[ts] = img["id"]
                self._img_id_by_idx[img.get("frame_idx", 0)] = img["id"]
            print(f"📂 Loaded existing COCO: {path} "
                  f"({len(self.images)} imgs, {len(self.annotations)} anns)")

    def load_progress(self, total_frames: int) -> int:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                idx = data.get("last_index", 0)
                self.reviewed = set(data.get("reviewed", []))
                self.keyframes = set(data.get("keyframes", []))
                if 0 <= idx < total_frames:
                    print(f"⏳ Resuming from frame {idx + 1}/{total_frames}")
                    return idx
            except Exception:
                pass
        return 0

    def save(self, is_final: bool) -> None:
        final_anns = []
        for ann in self.annotations:
            if ann["id"] in self.removed_ids:
                continue
            out = {k: v for k, v in ann.items() if not k.startswith("_")}
            # Legacy base64-PNG "mask" field is superseded by COCO polygon
            # "segmentation" (same shape as scripts/results.json).
            out.pop("mask", None)
            mask = ann.get("_mask")
            if mask is not None and isinstance(mask, np.ndarray) and mask.size:
                polys = _mask_to_polygons(mask)
                if polys:
                    out["segmentation"] = polys
            final_anns.append(out)
        data = {
            "images": self.images,
            "annotations": final_anns,
            "categories": self.categories,
        }
        path = self.output_json if is_final else self.output_json.replace(
            ".json", "_tmp.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(self.progress_file, "w") as f:
            json.dump({"last_index": self.current_idx + 1,
                       "reviewed": sorted(self.reviewed),
                       "keyframes": sorted(self.keyframes)}, f)
        self.dirty = False
        print(f"✅ Saved {'final' if is_final else 'progress'} → {path} "
              f"(idx {self.current_idx + 1})")

    # ------------------------- mutation -------------------------------- #

    def mark_reviewed(self, frame_idx: int) -> None:
        """Record that the user has reviewed frame `frame_idx` (0-based).

        Called on explicit forward navigation (N) and discard-all (X).
        Not a COCO mutation, so it does not set dirty or push undo — but it
        is persisted in the .progress sidecar on the next save().
        """
        self.reviewed.add(frame_idx)

    def ensure_image(self, frame: Dict[str, Any], width: int, height: int) -> int:
        ts = frame["timestamp_ns"]
        if ts in self._img_id_by_ts:
            img_id = self._img_id_by_ts[ts]
            # update size if needed
            for img in self.images:
                if img["id"] == img_id:
                    img["width"] = width
                    img["height"] = height
                    break
            return img_id
        img_id = len(self.images) + 1
        img_rec = {
            "id": img_id,
            "file_name": frame.get("file_name") or f"{frame['frame_idx']:06d}.jpg",
            "width": width,
            "height": height,
            "timestamp_ns": ts,
            "log_time_ns": frame.get("log_time_ns", ts),
            "frame_idx": frame["frame_idx"],
        }
        self.images.append(img_rec)
        self._img_id_by_ts[ts] = img_id
        self._img_id_by_idx[frame["frame_idx"]] = img_id
        return img_id

    def seed_box(self, image_id: int, cx: float, cy: float,
                 hw: float, hh: float, label: str) -> None:
        cat_id = self._resolve_cat_id(label)
        x = cx - hw
        y = cy - hh
        ann = {
            "id": self._ann_id_next,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(hw * 2), float(hh * 2)],
            "area": float(hw * 2 * hh * 2),
            "iscrowd": 0,
            "seed": True,
            "track_id": self._fresh_track_id(),
        }
        self.annotations.append(ann)
        self._ann_id_next += 1
        self.dirty = True
        # No undo entry: seeding happens automatically on first frame visit,
        # so undoable seeds would (a) evict real user history from the
        # bounded stack and (b) let Ctrl+Z mutate frames the user isn't
        # looking at. Deleting a seeded box is itself undoable.

    def add_box(self, image_id: int, x: float, y: float,
                w: float, h: float, cat_id: int) -> int:
        ann = {
            "id": self._ann_id_next,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
            "track_id": (self._next_track_id(image_id, cat_id)
                         if self.sticky_track_ids
                         else self._fresh_track_id()),
        }
        self.annotations.append(ann)
        self._ann_id_next += 1
        self.dirty = True
        new_id = ann["id"]
        # Undo: remove the added box.
        self.undo_stack.push(
            f"add box #{new_id}",
            undo=lambda: self._undo_remove(new_id),
            redo=lambda: self._redo_add(ann),
        )
        return ann["id"]

    def remove_box(self, ann_id: int) -> None:
        # Snapshot the annotation so undo can restore it (including mask).
        prev = None
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev = dict(ann)
                if "_mask" in prev and isinstance(prev["_mask"], np.ndarray):
                    prev["_mask"] = prev["_mask"].copy()
                break
        self.removed_ids.add(ann_id)
        self.dirty = True
        # Undo: re-add the box (un-remove).
        self.undo_stack.push(
            f"delete box #{ann_id}",
            undo=lambda: self._undo_restore(ann_id, prev),
            redo=lambda: self._redo_remove(ann_id),
        )

    def set_mask(self, ann_id: int, mask: Optional[np.ndarray]) -> None:
        """Attach (or clear) a SAM3 mask to an annotation, in-memory only."""
        prev_mask = None
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev_mask = ann.get("_mask")
                if isinstance(prev_mask, np.ndarray):
                    prev_mask = prev_mask.copy()
                if mask is None:
                    ann.pop("_mask", None)
                else:
                    ann["_mask"] = mask
                self.dirty = True
                # Undo: restore the previous mask (or None).
                mask_copy = mask.copy() if isinstance(mask, np.ndarray) else None
                self.undo_stack.push(
                    f"set mask #{ann_id}",
                    undo=lambda: self._undo_set_mask(ann_id, prev_mask),
                    redo=lambda: self._redo_set_mask(ann_id, mask_copy),
                )
                return

    def move_box(self, ann_id: int, new_x: float, new_y: float,
                 w: float, h: float) -> None:
        """Update an annotation's bbox position (size unchanged)."""
        prev_bbox = None
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev_bbox = list(ann["bbox"])
                ann["bbox"] = [float(new_x), float(new_y), float(w), float(h)]
                ann["area"] = float(w * h)
                self.dirty = True
                self.undo_stack.push(
                    f"move box #{ann_id}",
                    undo=lambda: self._undo_set_bbox(ann_id, prev_bbox),
                    redo=lambda: self._redo_set_bbox(ann_id, [new_x, new_y, w, h]),
                )
                return

    def resize_box(self, ann_id: int, new_x: float, new_y: float,
                   new_w: float, new_h: float) -> None:
        """Update an annotation's bbox size and position (corner drag)."""
        prev_bbox = None
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev_bbox = list(ann["bbox"])
                ann["bbox"] = [float(new_x), float(new_y),
                               float(new_w), float(new_h)]
                ann["area"] = float(new_w * new_h)
                self.dirty = True
                self.undo_stack.push(
                    f"resize box #{ann_id}",
                    undo=lambda: self._undo_set_bbox(ann_id, prev_bbox),
                    redo=lambda: self._redo_set_bbox(
                        ann_id, [new_x, new_y, new_w, new_h]
                    ),
                )
                return

    def set_cat(self, ann_id: int, cat_id: int) -> bool:
        """Change an annotation's category. Returns False if not found."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev = ann["category_id"]
                if prev == cat_id:
                    return True
                ann["category_id"] = cat_id
                self.dirty = True
                self.undo_stack.push(
                    f"recat box #{ann_id} → {cat_id}",
                    undo=lambda: self._undo_set_cat(ann_id, prev),
                    redo=lambda: self._undo_set_cat(ann_id, cat_id),
                )
                return True
        return False

    def _fresh_track_id(self) -> int:
        """Next id from the global creation-order counter (1-based).

        Deleted boxes do not recycle their ids (the counter only moves
        forward)."""
        n = self._track_next
        self._track_next += 1
        return n

    def _reference_track_ids(self, image_id: int) -> List[Optional[int]]:
        """Track ids of the nearest earlier annotated frame, in that
        frame's creation order — the frame new boxes should inherit
        track ids from (this is what interpolation pairing expects)."""
        idx = None
        for img in self.images:
            if img["id"] == image_id:
                idx = img.get("frame_idx")
                break
        if not idx:  # None or frame 0 → no earlier frame possible
            return []
        for fi in range(idx - 1, -1, -1):
            ref_img_id = self._img_id_by_idx.get(fi)
            if ref_img_id is None:
                continue
            tids = [a.get("track_id")
                    for a in sorted(self.annotations,
                                    key=lambda a: a["id"])
                    if a["image_id"] == ref_img_id
                    and a["id"] not in self.removed_ids]
            if any(t is not None for t in tids):
                return tids
        return []

    def _next_track_id(self, image_id: int, cat_id: int) -> int:
        """Track id for a newly drawn box.

        The k-th box drawn on this frame (in creation order) inherits the
        k-th track id from the nearest earlier annotated frame — so the
        first box drawn on frame 2 continues track 1 from frame 1, the
        second continues track 2, etc. When there is no such reference, or
        this frame already has more boxes than the reference, or the id is
        already taken on this frame, a fresh global id is used. ``cat_id``
        is accepted for call-site compatibility and ignored."""
        ref = self._reference_track_ids(image_id)
        if ref:
            live = [a for a in self.annotations
                    if a["image_id"] == image_id
                    and a["id"] not in self.removed_ids]
            k = len(live)
            used = {a.get("track_id") for a in live}
            if k < len(ref):
                tid = ref[k]
                if tid is not None and tid not in used:
                    return tid
        return self._fresh_track_id()

    def set_track_id(self, ann_id: int, value: Optional[int]) -> bool:
        """Undoable set of the annotation's track id (None clears it)."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                prev = ann.get("track_id")
                if prev == value:
                    return True
                if value is None:
                    ann.pop("track_id", None)
                else:
                    ann["track_id"] = int(value)
                self.dirty = True
                shown = value if value is not None else "(none)"
                self.undo_stack.push(
                    f"set track id #{ann_id} → {shown}",
                    undo=lambda: self._undo_set_track(ann_id, prev),
                    redo=lambda: self._undo_set_track(ann_id, value),
                )
                return True
        return False

    def add_interp_box(self, image_id: int, x: float, y: float,
                       w: float, h: float, cat_id: int,
                       track_id: Optional[int], source: str,
                       confidence: float) -> int:
        """Add a flow-interpolated box (undoable, like add_box).

        Carries provenance: ``interp=True``, ``source`` (flow/linear/...),
        ``confidence`` in [0,1], and the track id inherited from the start
        anchor box.
        """
        ann = {
            "id": self._ann_id_next,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
            "interp": True,
            "source": source,
            "confidence": float(confidence),
        }
        if track_id is not None:
            ann["track_id"] = int(track_id)
        self.annotations.append(ann)
        self._ann_id_next += 1
        self.dirty = True
        new_id = ann["id"]
        self.undo_stack.push(
            f"interp box #{new_id}",
            undo=lambda: self._undo_remove(new_id),
            redo=lambda: self._redo_add(ann),
        )
        return new_id

    # ------------------- interpolation helpers ------------------------- #

    def labeled_frame_idxs(self) -> List[int]:
        """Sorted frame_idxs that have at least one live annotation."""
        boxed_img_ids = {
            ann["image_id"] for ann in self.annotations
            if ann["id"] not in self.removed_ids
        }
        out = {
            img.get("frame_idx", 0)
            for img in self.images if img["id"] in boxed_img_ids
        }
        return sorted(out)

    def frame_has_boxes(self, frame_idx: int) -> bool:
        img_id = self._img_id_by_idx.get(frame_idx)
        if img_id is None:
            return False
        return any(
            ann["image_id"] == img_id and ann["id"] not in self.removed_ids
            for ann in self.annotations
        )

    def anchor_candidates(self) -> List[int]:
        """Sorted frame_idxs usable as interpolation anchors (have boxes).

        Keyframes (K) take priority: if any keyframe has boxes, only those
        are candidates — so the user can pin exactly which frames bound an
        interpolation span even when other frames also have boxes.
        """
        boxed = self.labeled_frame_idxs()
        keyed = [f for f in boxed if f in self.keyframes]
        return sorted(keyed if keyed else boxed)

    # ---- undo/redo primitives (called via lambdas on the stack) ---- #

    def _undo_set_cat(self, ann_id: int, cat_id: int) -> None:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                ann["category_id"] = cat_id
                self.dirty = True
                return

    def _undo_set_track(self, ann_id: int, value: Optional[int]) -> None:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                if value is None:
                    ann.pop("track_id", None)
                else:
                    ann["track_id"] = value
                self.dirty = True
                return

    def _undo_remove(self, ann_id: int) -> None:
        """Inverse of add_box / seed_box — remove the box."""
        self.removed_ids.add(ann_id)
        self.dirty = True

    def _redo_add(self, ann_snapshot: Dict[str, Any]) -> None:
        """Re-apply an add_box — re-insert the snapshot and un-remove."""
        ann_id = ann_snapshot["id"]
        # If the annotation still exists (just was removed), un-remove it.
        if ann_id in self.removed_ids:
            self.removed_ids.discard(ann_id)
        else:
            # Append a fresh copy (mask included if present).
            new = dict(ann_snapshot)
            if "_mask" in new and isinstance(new["_mask"], np.ndarray):
                new["_mask"] = new["_mask"].copy()
            self.annotations.append(new)
        self.dirty = True

    def _undo_restore(self, ann_id: int, prev: Optional[Dict[str, Any]]) -> None:
        """Inverse of remove_box — restore the box."""
        if prev is None:
            return
        # Un-remove if currently removed.
        if ann_id in self.removed_ids:
            self.removed_ids.discard(ann_id)
        else:
            # Re-append a copy.
            new = dict(prev)
            if "_mask" in new and isinstance(new["_mask"], np.ndarray):
                new["_mask"] = new["_mask"].copy()
            self.annotations.append(new)
        self.dirty = True

    def _redo_remove(self, ann_id: int) -> None:
        self.removed_ids.add(ann_id)
        self.dirty = True

    def _undo_set_bbox(self, ann_id: int, prev_bbox: Optional[List[float]]) -> None:
        if prev_bbox is None:
            return
        for ann in self.annotations:
            if ann["id"] == ann_id:
                ann["bbox"] = list(prev_bbox)
                ann["area"] = float(prev_bbox[2] * prev_bbox[3])
                self.dirty = True
                return

    def _redo_set_bbox(self, ann_id: int, new_bbox: List[float]) -> None:
        self._undo_set_bbox(ann_id, new_bbox)

    def _undo_set_mask(self, ann_id: int,
                       prev_mask: Optional[np.ndarray]) -> None:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                if prev_mask is None:
                    ann.pop("_mask", None)
                else:
                    ann["_mask"] = prev_mask.copy() if isinstance(prev_mask, np.ndarray) else prev_mask
                self.dirty = True
                return

    def _redo_set_mask(self, ann_id: int,
                       new_mask: Optional[np.ndarray]) -> None:
        self._undo_set_mask(ann_id, new_mask)

    def get_box(self, ann_id: int) -> Optional[Dict[str, Any]]:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                return ann
        return None

    def get_mask(self, ann_id: int) -> Optional[np.ndarray]:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                return ann.get("_mask")
        return None

    def anns_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        return [
            ann for ann in self.annotations
            if ann["image_id"] == image_id and ann["id"] not in self.removed_ids
        ]

    def _resolve_cat_id(self, label: str) -> int:
        # If the label parses as an int matching an existing cat id, use it.
        try:
            v = int(label)
            if v in self.cat_map:
                return v
        except (TypeError, ValueError):
            pass
        # Else look up by name.
        if label in self.cat_name_to_id:
            return self.cat_name_to_id[label]
        # Else create a new category.
        new_id = max(self.cat_map.keys(), default=-1) + 1
        self.categories.append({"id": new_id, "name": label})
        self.cat_map[new_id] = label
        self.cat_name_to_id[label] = new_id
        return new_id


# ---------------------------------------------------------------------------
# Canvas widget: shows the image + bboxes, supports draw/select/delete
# ---------------------------------------------------------------------------

class CanvasWidget(QWidget):
    """2D image canvas with bbox + mask overlay. Emits signals on edits."""

    box_added = pyqtSignal(int, float, float, float, float, int)  # img_id,x,y,w,h,cat_id
    box_deleted = pyqtSignal(int)  # ann_id
    box_moved = pyqtSignal(int, float, float, float, float)   # ann_id, new_x, new_y, w, h
    box_resized = pyqtSignal(int, float, float, float, float)  # ann_id, x, y, w, h
    frame_nav = pyqtSignal(int)    # delta (-1 / +1)
    save_quit = pyqtSignal()
    quit_request = pyqtSignal()
    discard_all = pyqtSignal()
    cat_pick_requested = pyqtSignal(float, float, float, float)  # pending rect
    resegment_selected = pyqtSignal()  # R key on selected box
    toggle_masks = pyqtSignal()       # M key
    play_pause = pyqtSignal()          # Space key
    fit_view = pyqtSignal()            # F key
    selection_changed = pyqtSignal(int)  # selected box index (or -1)
    zoom_to_selected = pyqtSignal()    # Z key
    next_unlabeled = pyqtSignal()      # U key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pixmap: Optional[QPixmap] = None
        self._image_size: Tuple[int, int] = (0, 0)  # (w, h)
        self._boxes: List[Dict[str, Any]] = []  # see set_boxes
        self._masks_visible: bool = True
        self._mask_alpha: int = 120  # 0-255 overlay alpha for mask fill
        self._selected_idx: int = -1
        # All currently selected box indices (includes _selected_idx, which
        # is the "primary" selection used for move/resize/recat/track).
        # Shift+click toggles membership; a plain click resets to {hit}.
        self._multi_selected: set = set()
        self._drawing: bool = False
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_current: Optional[Tuple[float, float]] = None
        self._waiting_cat: bool = False
        self._pending_rect: Optional[Tuple[float, float, float, float]] = None

        # Box editing state. _edit_mode is one of:
        #   "idle" | "draw" | "move" | "resize_tl" | "resize_tr" |
        #   "resize_bl" | "resize_br"
        # where tl/tr/bl/br = top-left/top-right/bottom-left/bottom-right corner.
        self._edit_mode: str = "idle"
        # Starting bbox + cursor position when a move/resize began, in
        # image coords. Used to compute the new bbox on each mousemove.
        self._edit_start_box: Optional[Tuple[float, float, float, float]] = None
        self._edit_start_cursor: Optional[Tuple[float, float]] = None
        # Radius (in widget px) within which a corner handle is "grabbed".
        self._handle_radius_px: int = 8

        # Pan / zoom transforms. We display image in 'fit' mode by default.
        self._scale = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._panning = False
        self._pan_start: Optional[QtCore.QPointF] = None

        self._info_text = ""

    # ----------------------- public api -------------------------------- #

    def set_image(self, arr: np.ndarray) -> None:
        h, w, _ = arr.shape
        self._image_size = (w, h)
        self._pixmap = QPixmap.fromImage(
            QtGui.QImage(arr.data, w, h, 3 * w,
                         QtGui.QImage.Format_RGB888).copy()
        )
        self._fit_to_view()
        self.update()

    def set_boxes(self, boxes: List[Dict[str, Any]]) -> None:
        """boxes: list of dicts with keys id, bbox=[x,y,w,h], cat_name, cat_id,
        optional mask (HxW bool array)."""
        self._boxes = list(boxes)
        self._selected_idx = -1
        self._multi_selected = set()
        self._waiting_cat = False
        self._pending_rect = None
        self.update()

    def set_masks_visible(self, visible: bool) -> None:
        self._masks_visible = visible
        self.update()

    def masks_visible(self) -> bool:
        return self._masks_visible

    def set_mask_alpha(self, alpha: int) -> None:
        """Set mask overlay alpha (0-255)."""
        self._mask_alpha = max(0, min(255, int(alpha)))
        self.update()

    def mask_alpha(self) -> int:
        return self._mask_alpha

    def set_info(self, text: str) -> None:
        self._info_text = text
        self.update()

    def get_pending_rect(self) -> Optional[Tuple[float, float, float, float]]:
        return self._pending_rect

    def reset_state(self) -> None:
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._waiting_cat = False
        self._pending_rect = None
        self._selected_idx = -1
        self._multi_selected = set()
        self.update()

    # ----------------------- coordinate maps --------------------------- #

    def _fit_to_view(self) -> None:
        if not self._pixmap or self._image_size == (0, 0):
            return
        iw, ih = self._image_size
        vw, vh = self.width(), self.height()
        if vw <= 0 or vh <= 0:
            return
        scale = min(vw / iw, vh / ih) * 0.95
        self._scale = scale
        # Center.
        self._offset = QtCore.QPointF(
            (vw - iw * scale) / 2.0, (vh - ih * scale) / 2.0
        )
        self.update()

    def _img_to_widget(self, x: float, y: float) -> QtCore.QPointF:
        return QtCore.QPointF(
            x * self._scale + self._offset.x(),
            y * self._scale + self._offset.y(),
        )

    def _widget_to_img(self, px: float, py: float) -> Tuple[float, float]:
        return (
            (px - self._offset.x()) / self._scale,
            (py - self._offset.y()) / self._scale,
        )

    def _clamp_img_pt(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp an image-coord point to the image bounds."""
        iw, ih = self._image_size
        return (min(max(x, 0.0), float(iw)), min(max(y, 0.0), float(ih)))

    # ----------------------- painting ---------------------------------- #

    # Per-class color palette (RGB) for mask overlays.
    _MASK_COLORS = [
        (255, 0, 128), (0, 200, 255), (120, 220, 60), (255, 160, 0),
        (160, 0, 255), (0, 255, 200), (220, 40, 40), (40, 220, 220),
        (255, 220, 40), (180, 220, 255),
    ]

    def _color_for_cat(self, cat_id: int) -> Tuple[int, int, int]:
        return self._MASK_COLORS[cat_id % len(self._MASK_COLORS)]

    def _corner_handles(self, box_idx: int) -> List[Tuple[float, float]]:
        """Return widget-coord centers of the 4 corner handles for box_idx,
        in order: top-left, top-right, bottom-left, bottom-right."""
        x, y, w, h = self._boxes[box_idx]["bbox"]
        tl = self._img_to_widget(x, y)
        tr = self._img_to_widget(x + w, y)
        bl = self._img_to_widget(x, y + h)
        br = self._img_to_widget(x + w, y + h)
        return [(tl.x(), tl.y()), (tr.x(), tr.y()),
                (bl.x(), bl.y()), (br.x(), br.y())]

    def _hit_corner(self, box_idx: int, px: float, py: float) -> Optional[str]:
        """Return 'tl'/'tr'/'bl'/'br' if (px,py) is on a corner handle, else None."""
        handles = self._corner_handles(box_idx)
        names = ["tl", "tr", "bl", "br"]
        r = self._handle_radius_px + 2  # small tolerance
        for (hx, hy), name in zip(handles, names):
            if (px - hx) ** 2 + (py - hy) ** 2 <= r * r:
                return name
        return None

    def _paint_masks(self, p: QPainter) -> None:
        """Paint semi-transparent masks for every box that has one."""
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0:
            return
        for i, box in enumerate(self._boxes):
            mask = box.get("mask")
            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                continue
            # Build an RGBA8888 QImage from the mask tinted with the class color.
            h, w = mask.shape[:2]
            if (w, h) != (iw, ih):
                # Resize the mask to the current image size (nearest neighbor).
                from PIL import Image as _PILImage
                m_pil = _PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L")
                m_pil = m_pil.resize((iw, ih), _PILImage.NEAREST)
                mask = np.array(m_pil) > 0
            cat_id = box.get("cat_id", 0)
            r, g, b = self._color_for_cat(cat_id)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = r
            rgba[..., 1] = g
            rgba[..., 2] = b
            rgba[..., 3] = (mask.astype(np.uint8) * self._mask_alpha)
            qimg = QtGui.QImage(rgba.data, w, h, 4 * w,
                               QtGui.QImage.Format.Format_RGBA8888)
            mask_pixmap = QPixmap.fromImage(qimg.copy())
            tl = self._img_to_widget(0, 0)
            br = self._img_to_widget(iw, ih)
            p.drawPixmap(QtCore.QRectF(tl, br), mask_pixmap,
                         QtCore.QRectF(0, 0, iw, ih))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(20, 20, 25))

        if self._pixmap is None:
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        # Draw image.
        target = QtCore.QRectF(self._offset.x(), self._offset.y(),
                               self._image_size[0] * self._scale,
                               self._image_size[1] * self._scale)
        p.drawPixmap(target, self._pixmap,
                     QtCore.QRectF(0, 0, self._pixmap.width(),
                                   self._pixmap.height()))

        # Draw masks (semi-transparent overlay, per-class color).
        if self._masks_visible and self._boxes:
            self._paint_masks(p)

        # Draw boxes.
        font = QFont("Sans", max(8, int(10 * self._scale + 4)))
        p.setFont(font)
        for i, box in enumerate(self._boxes):
            x, y, w, h = box["bbox"]
            if i == self._selected_idx:
                color, width = QColor(255, 80, 80), 2      # primary selection
            elif i in self._multi_selected:
                color, width = QColor(255, 150, 40), 2     # shift-selected
            else:
                color, width = QColor(255, 220, 30), 1.2
            pen = QPen(color, width)
            p.setPen(pen)
            tl = self._img_to_widget(x, y)
            br = self._img_to_widget(x + w, y + h)
            p.drawRect(QtCore.QRectF(tl, br))
            tid = box.get("track_id")
            ttxt = f"T{tid} " if tid is not None else ""
            itxt = "~" if box.get("interp") else ""
            label = f"{itxt}{ttxt}{box.get('cat_name','?')} (id:{box['id']})"
            p.fillRect(
                int(tl.x()), max(0, int(tl.y() - 16)),
                8 * len(label) + 6, 16, QColor(0, 0, 0, 160)
            )
            p.setPen(color)
            p.drawText(int(tl.x() + 3), max(12, int(tl.y() - 4)), label)

        # Draw corner handles on the selected box.
        if 0 <= self._selected_idx < len(self._boxes):
            handles = self._corner_handles(self._selected_idx)
            p.setPen(QPen(QColor(255, 80, 80), 1.5))
            p.setBrush(QColor(255, 255, 255, 200))
            r = self._handle_radius_px
            for hx, hy in handles:
                p.drawRect(int(hx - r), int(hy - r), 2 * r, 2 * r)

        # Draw pending rectangle while dragging.
        if self._drawing and self._draw_start and self._draw_current:
            x0, y0 = self._draw_start
            x1, y1 = self._draw_current
            tl_img = (min(x0, x1), min(y0, y1))
            br_img = (max(x0, x1), max(y0, y1))
            tl = self._img_to_widget(*tl_img)
            br = self._img_to_widget(*br_img)
            pen = QPen(QColor(80, 180, 255), 2, Qt.DashLine)
            p.setPen(pen)
            p.drawRect(QtCore.QRectF(tl, br))

        # Info HUD.
        p.setPen(QColor(230, 230, 230))
        p.setFont(QFont("Sans", 10))
        p.drawText(8, 16, self._info_text)

        if self._waiting_cat:
            p.fillRect(8, self.height() - 28, 380, 22, QColor(0, 0, 0, 180))
            p.setPen(QColor(255, 220, 30))
            p.setFont(QFont("Sans", 10))
            p.drawText(12, self.height() - 12,
                       "Pick category: 0-9 / click button   |  ESC to cancel")

    # ----------------------- mouse ------------------------------------- #

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            if self._waiting_cat:
                return
            px, py = ev.position().x(), ev.position().y()
            if self._drawing:
                ix, iy = self._clamp_img_pt(*self._widget_to_img(px, py))
                self._draw_start = (ix, iy)
                self._draw_current = (ix, iy)
                self._edit_mode = "draw"
                return
            # 1. Hit-test corner handles on the currently selected box.
            if 0 <= self._selected_idx < len(self._boxes):
                corner = self._hit_corner(self._selected_idx, px, py)
                if corner is not None:
                    # Begin resize.
                    bx, by, bw, bh = self._boxes[self._selected_idx]["bbox"]
                    self._edit_mode = f"resize_{corner}"
                    self._edit_start_box = (bx, by, bw, bh)
                    ix, iy = self._widget_to_img(px, py)
                    self._edit_start_cursor = (ix, iy)
                    return
            # 2. Hit-test box bodies (topmost first).
            ix, iy = self._widget_to_img(px, py)
            hit = -1
            for i in range(len(self._boxes) - 1, -1, -1):
                x, y, w, h = self._boxes[i]["bbox"]
                if x <= ix <= x + w and y <= iy <= y + h:
                    hit = i
                    break
            if hit >= 0:
                if ev.modifiers() & Qt.ShiftModifier:
                    # Shift+click: toggle membership in the multi-selection;
                    # no drag. The clicked box becomes the primary.
                    if hit in self._multi_selected:
                        self._multi_selected.discard(hit)
                        if self._selected_idx == hit:
                            self._selected_idx = (
                                next(iter(self._multi_selected))
                                if self._multi_selected else -1)
                    else:
                        self._multi_selected.add(hit)
                        self._selected_idx = hit
                    self.selection_changed.emit(self._selected_idx)
                    self.update()
                    return
                # Select the box and begin move.
                self._selected_idx = hit
                self._multi_selected = {hit}
                self.selection_changed.emit(hit)
                bx, by, bw, bh = self._boxes[hit]["bbox"]
                self._edit_mode = "move"
                self._edit_start_box = (bx, by, bw, bh)
                self._edit_start_cursor = (ix, iy)
                self.update()
            else:
                # Click in empty space — deselect.
                self._selected_idx = -1
                self._multi_selected = set()
                self.selection_changed.emit(-1)
                self.update()
        elif ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.position()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        px, py = ev.position().x(), ev.position().y()
        # Cursor feedback.
        if self._edit_mode == "idle" and not self._drawing and not self._panning:
            self._update_cursor(px, py)

        if self._edit_mode == "draw" and self._draw_start is not None:
            self._draw_current = self._clamp_img_pt(
                *self._widget_to_img(px, py))
            self.update()
        elif self._edit_mode == "move":
            ix, iy = self._widget_to_img(px, py)
            sx, sy = self._edit_start_cursor
            dx, dy = ix - sx, iy - sy
            bx, by, bw, bh = self._edit_start_box
            # Keep the whole box inside the image.
            iw, ih = self._image_size
            new_x = min(max(bx + dx, 0.0), max(iw - bw, 0.0))
            new_y = min(max(by + dy, 0.0), max(ih - bh, 0.0))
            # Update the box in-place so the canvas shows it moving.
            self._boxes[self._selected_idx]["bbox"] = [new_x, new_y, bw, bh]
            self.update()
        elif self._edit_mode.startswith("resize_"):
            ix, iy = self._widget_to_img(px, py)
            sx, sy = self._edit_start_cursor
            dx, dy = ix - sx, iy - sy
            bx, by, bw, bh = self._edit_start_box
            corner = self._edit_mode.split("_", 1)[1]
            new_x, new_y, new_w, new_h = bx, by, bw, bh
            if corner == "tl":
                new_x = bx + dx; new_y = by + dy
                new_w = bw - dx; new_h = bh - dy
            elif corner == "tr":
                new_y = by + dy
                new_w = bw + dx; new_h = bh - dy
            elif corner == "bl":
                new_x = bx + dx
                new_w = bw - dx; new_h = bh + dy
            elif corner == "br":
                new_w = bw + dx; new_h = bh + dy
            # Clamp: keep the box inside the image bounds.
            iw, ih = self._image_size
            new_x = max(new_x, 0.0)
            new_y = max(new_y, 0.0)
            new_w = min(new_w, iw - new_x)
            new_h = min(new_h, ih - new_y)
            # Clamp: don't allow the box to flip (negative w/h).
            if new_w > 2 and new_h > 2:
                self._boxes[self._selected_idx]["bbox"] = [
                    new_x, new_y, new_w, new_h
                ]
                self.update()
        elif self._panning and self._pan_start is not None:
            delta = ev.position() - self._pan_start
            self._offset += delta
            self._pan_start = ev.position()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            if self._edit_mode == "draw" and self._draw_start and self._draw_current:
                x0, y0 = self._draw_start
                x1, y1 = self._draw_current
                x = min(x0, x1); y = min(y0, y1)
                w = abs(x1 - x0); h = abs(y1 - y0)
                self._drawing = False
                self._edit_mode = "idle"
                self._draw_start = None
                self._draw_current = None
                if w > 2 and h > 2:
                    # If the user preselected a category (clicked a cat in
                    # the side panel before drawing), assign it now without
                    # waiting for a number key. Otherwise reuse the last
                    # assigned category (sticky), so consecutive boxes of the
                    # same class need no keypress at all.
                    pre = getattr(self.parent_window, "_pending_cat_id", None)
                    if pre is None:
                        pre = getattr(self.parent_window, "_last_cat_id", None)
                    if pre not in self.parent_window.coco.cat_map:
                        # Stale (e.g. category deleted since last use).
                        pre = None
                    if pre is not None:
                        # Reset pending cat so the next draw asks again.
                        self.parent_window._pending_cat_id = None
                        # Don't clear the preselection visual in the side panel;
                        # the user can click again to re-preselect.
                        self.box_added.emit(
                            self.parent_window._current_image_id or 0,
                            x, y, w, h, pre,
                        )
                        self.update()
                    else:
                        self._pending_rect = (x, y, w, h)
                        self._waiting_cat = True
                        self.update()
                        self.cat_pick_requested.emit(x, y, w, h)
                else:
                    self.update()
            elif self._edit_mode == "move":
                # Commit the new position.
                if 0 <= self._selected_idx < len(self._boxes):
                    box = self._boxes[self._selected_idx]
                    x, y, w, h = box["bbox"]
                    self.box_moved.emit(box["id"], x, y, w, h)
                self._edit_mode = "idle"
                self._edit_start_box = None
                self._edit_start_cursor = None
            elif self._edit_mode.startswith("resize_"):
                if 0 <= self._selected_idx < len(self._boxes):
                    box = self._boxes[self._selected_idx]
                    x, y, w, h = box["bbox"]
                    self.box_resized.emit(box["id"], x, y, w, h)
                self._edit_mode = "idle"
                self._edit_start_box = None
                self._edit_start_cursor = None
        elif ev.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None

    def _update_cursor(self, px: float, py: float) -> None:
        """Set the cursor shape based on what's under it (handle / box / empty)."""
        if 0 <= self._selected_idx < len(self._boxes):
            corner = self._hit_corner(self._selected_idx, px, py)
            if corner in ("tl", "br"):
                self.setCursor(Qt.SizeFDiagCursor)
                return
            if corner in ("tr", "bl"):
                self.setCursor(Qt.SizeBDiagCursor)
                return
        # Hit-test box body.
        ix, iy = self._widget_to_img(px, py)
        for i in range(len(self._boxes) - 1, -1, -1):
            x, y, w, h = self._boxes[i]["bbox"]
            if x <= ix <= x + w and y <= iy <= y + h:
                self.setCursor(Qt.SizeAllCursor)
                return
        if self._drawing:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        if self._pixmap is not None and self._scale < 1e-6:
            self._fit_to_view()
        super().resizeEvent(ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        # Zoom around cursor.
        degrees = ev.angleDelta().y() / 120.0
        factor = 1.1 ** degrees
        new_scale = max(0.05, min(40.0, self._scale * factor))
        # Keep cursor anchored.
        cx, cy = ev.position().x(), ev.position().y()
        ix, iy = self._widget_to_img(cx, cy)
        self._scale = new_scale
        # Re-offset so that (ix, iy) maps to (cx, cy) under new scale.
        self._offset = QtCore.QPointF(
            cx - ix * self._scale, cy - iy * self._scale
        )
        self.update()

    # ----------------------- keyboard ---------------------------------- #

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        k = ev.key()
        if self._waiting_cat:
            if k == Qt.Key_Escape:
                self._waiting_cat = False
                self._pending_rect = None
                self.update()
                return
            # number keys
            txt = ev.text()
            if txt.isdigit():
                self.parent_window._assign_pending_cat(int(txt))
            return

        if k in (Qt.Key_D, Qt.Key_Delete, Qt.Key_Backspace):
            # Delete every selected box (shift-click multi-select), or just
            # the primary selection when there is no multi-selection.
            # Capture ids up front: each emitted delete triggers a refresh
            # that rewrites _boxes, so indexing lazily would skip boxes.
            sel = sorted(self._multi_selected)
            if not sel and 0 <= self._selected_idx < len(self._boxes):
                sel = [self._selected_idx]
            ids = [self._boxes[i]["id"] for i in sel
                   if 0 <= i < len(self._boxes)]
            for ann_id in ids:
                self.box_deleted.emit(ann_id)
        elif k in (Qt.Key_A,):
            self._drawing = not self._drawing
            self._draw_start = None
            self._draw_current = None
            self.update()
        elif k in (Qt.Key_N, Qt.Key_Right):
            self.frame_nav.emit(+1)
        elif k in (Qt.Key_B, Qt.Key_Left):
            self.frame_nav.emit(-1)
        elif k in (Qt.Key_X,):
            self.discard_all.emit()
        elif k in (Qt.Key_S,):
            self.save_quit.emit()
        elif k in (Qt.Key_Q, Qt.Key_Escape):
            self.quit_request.emit()
        elif k == Qt.Key_M:
            self.toggle_masks.emit()
        elif k == Qt.Key_R:
            self.resegment_selected.emit()
        elif k == Qt.Key_Space:
            self.play_pause.emit()
        elif k == Qt.Key_F:
            self._fit_to_view()
        elif k == Qt.Key_Z:
            self.zoom_to_selected.emit()
        elif k == Qt.Key_U:
            self.next_unlabeled.emit()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self._scale = min(40.0, self._scale * 1.2); self.update()
        elif k == Qt.Key_Minus:
            self._scale = max(0.05, self._scale / 1.2); self.update()
        elif k == Qt.Key_0:
            self._fit_to_view()
        else:
            super().keyPressEvent(ev)


# ---------------------------------------------------------------------------
# Side panel: category list + buttons + frame slider
# ---------------------------------------------------------------------------

class SidePanel(QWidget):

    cat_clicked = pyqtSignal(int)  # cat_id
    slider_moved = pyqtSignal(int)  # frame_idx
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
    toggle_keyframe_clicked = pyqtSignal()   # "★ Keyframe" button (K)
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
            "Current frame source (image folder / rrd recording).")
        layout.addWidget(self.source_label)

        self.cat_label = QLabel("Categories (click to preselect for next draw, or press 0-9 when drawing):")
        layout.addWidget(self.cat_label)

        self.cat_list = QListWidget()
        layout.addWidget(self.cat_list, 1)
        self.cat_list.itemClicked.connect(self._on_cat_clicked)
        self._rebuild_cat_list()
        self._preselected_cat_id: Optional[int] = None

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
        # fallback); edit it when a track continues.
        track_row = QHBoxLayout()
        track_row.addWidget(QLabel("Track of selected:"))
        self.track_edit = QLineEdit()
        self.track_edit.setPlaceholderText("id, e.g. 2 (empty clears)")
        self.track_edit.setToolTip(
            "Select a box, type a track id, press Enter to set it. "
            "Clear the field and press Enter to unset. Press T to focus.")
        self.track_edit.returnPressed.connect(self._on_track_entered)
        track_row.addWidget(self.track_edit, 1)
        layout.addLayout(track_row)

        # Frame playback controls: play/pause + speed.
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Play / pause the frame timeline (Space).")
        self.btn_play.clicked.connect(self._on_play_clicked)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(QLabel("speed:"))
        self.combo_speed = QtWidgets.QComboBox()
        # Speeds are ms-per-frame; 1x = 100 ms/frame (what used to be 10x).
        for label, ms in [("0.25x", 400), ("0.5x", 200), ("1x", 100),
                          ("2x", 50), ("5x", 20), ("10x", 10)]:
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

        self.btn_sam3_all_frames = QPushButton("SAM3 ALL frames")
        self.btn_sam3_all_frames.setToolTip(
            "Background auto-annotate: run SAM3 on every box that has no "
            "mask yet, across ALL frames. Cancel anytime.")
        self.btn_sam3_all_frames.clicked.connect(self.sam3_all_frames_clicked.emit)
        layout.addWidget(self.btn_sam3_all_frames)

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
            ttxt = f" T{tid}" if tid is not None else ""
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
        self.btn_sam3_all_frames.setEnabled(not running)
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
        "sam3_run": ["btn_run_sam3", "btn_reseg", "btn_cancel_sam3"],
        "sam3_all_frames": ["btn_sam3_all_frames"],
        "masks": ["btn_masks", "opacity_slider", "opacity_value_label"],
        "play": ["btn_play", "combo_speed"],
    }

    def set_hidden_groups(self, groups: List[str]) -> None:
        """Hide the widget groups named in `groups` (see _HIDEABLE) and
        re-show any known group not listed, so a runtime config reload can
        both hide and un-hide. ``rerun_toggle`` is valid but handled by the
        main window, not this panel."""
        want = set(groups)
        for g, attrs in self._HIDEABLE.items():
            for attr in attrs:
                w = getattr(self, attr, None)
                if w is None:
                    continue
                for widget in (w if isinstance(w, list) else [w]):
                    widget.setVisible(g not in want)
        for g in sorted(want - set(self._HIDEABLE) - {"rerun_toggle"}):
            print(f"⚠️ Unknown ui.hide group: {g!r} "
                  f"(known: {sorted(self._HIDEABLE)}, rerun_toggle)")

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


# ---------------------------------------------------------------------------
# Time-bridge: receive time_update events from the embedded web viewer.
# ---------------------------------------------------------------------------

# Minimal JS injected into the web viewer page that hooks the rerun
# notebook widget's `on_viewer_event` and posts JSON back to PyQt via
# `QWebChannel`.
_BRIDGE_JS = r"""
(function(){
  function tryHook(){
    if (window.rrWidgetBridge) return;
    if (!window.qWebChannel) return;
    new QWebChannel(qt.webChannelTransport, function(channel){
      window.rrWidgetBridge = channel.objects.bridge;
      // Hook the viewer's event dispatch.
      var origSend = null;
      function patchWidget(){
        var w = window.rerunWidget;
        if (!w || !w._on_raw_event) { setTimeout(patchWidget, 100); return; }
        // Wrap the model.send path so we get a copy of every raw event.
        if (!w.__patched){
          w.__patched = true;
          var origDispatch = w._dispatch_raw_event
            ? w._dispatch_raw_event.bind(w)
            : null;
          if (origDispatch){
            w._dispatch_raw_event = function(json){
              try {
                if (window.rrWidgetBridge){
                  window.rrWidgetBridge.onViewerEvent(json);
                }
              } catch(e){}
              origDispatch(json);
            };
          }
        }
      }
      patchWidget();
    });
  }
  tryHook();
})();
"""

class TimeBridge(QObject):
    """Receives raw viewer events (JSON strings) from the web widget.

    Parses the `time_update` event and re-emits as a Qt signal with the
    integer nanosecond timestamp. Other event types are ignored here but
    could be extended (e.g. selection_change).
    """

    time_changed = pyqtSignal(int)  # timestamp in nanos since epoch

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(str)
    def onViewerEvent(self, event_json: str) -> None:
        try:
            ev = json.loads(event_json)
        except Exception:
            return
        if not isinstance(ev, dict):
            return
        etype = ev.get("type")
        if etype == "time_update":
            t = ev.get("time")
            if t is None:
                return
            # `time` is in seconds (float) per rerun notebook widget convention.
            ns = int(round(float(t) * 1e9))
            self.time_changed.emit(ns)
        elif etype == "timeline_change":
            # We could remember the active timeline here if the user
            # switches timelines in the viewer. For now we ignore.
            pass


# ---------------------------------------------------------------------------
# Config dialog (runtime settings editor)
# ---------------------------------------------------------------------------

class ConfigDialog(QtWidgets.QDialog):
    """Settings editor opened from the ⚙ Config button (File → Config…).

    Contains checkboxes for hiding UI groups plus interpolation / SAM3 /
    mask-opacity / tracking fields. Buttons:
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
        "masks": "Mask toggle + opacity slider",
        "play": "Play controls",
        "rerun_toggle": "Rerun viewer toggle",
    }

    def __init__(self, win: "ReviewWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Config settings")
        self.setModal(True)
        root = QVBoxLayout(self)

        # --- UI visibility -------------------------------------------------
        ui_box = QtWidgets.QGroupBox("Hide UI elements")
        ui_form = QVBoxLayout(ui_box)
        ui_form.addWidget(QLabel("Checked = hidden"))
        self.hide_checks: Dict[str, QCheckBox] = {}
        for key, label in self._HIDE_LABELS.items():
            cb = QCheckBox(label)
            self.hide_checks[key] = cb
            ui_form.addWidget(cb)
        root.addWidget(ui_box)

        # --- Interpolation -------------------------------------------------
        interp_box = QtWidgets.QGroupBox("Interpolation")
        interp_form = QtWidgets.QFormLayout(interp_box)
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
        root.addWidget(interp_box)

        # --- SAM3 ----------------------------------------------------------
        sam3_box = QtWidgets.QGroupBox("SAM3")
        sam3_form = QtWidgets.QFormLayout(sam3_box)
        self.combo_device = QtWidgets.QComboBox()
        self.combo_device.addItems(["cuda", "cpu"])
        sam3_form.addRow("Device", self.combo_device)
        self.spin_sam3_conf = QtWidgets.QDoubleSpinBox()
        self.spin_sam3_conf.setRange(0.0, 1.0)
        self.spin_sam3_conf.setSingleStep(0.05)
        self.spin_sam3_conf.setDecimals(2)
        sam3_form.addRow("Confidence", self.spin_sam3_conf)
        self.check_auto_segment = QCheckBox("Auto-segment on box add")
        sam3_form.addRow(self.check_auto_segment)
        root.addWidget(sam3_box)

        # --- Mask opacity --------------------------------------------------
        mask_box = QtWidgets.QGroupBox("Masks")
        mask_form = QtWidgets.QFormLayout(mask_box)
        self.spin_opacity = QtWidgets.QSpinBox()
        self.spin_opacity.setRange(0, 100)
        self.spin_opacity.setSuffix(" %")
        mask_form.addRow("Mask opacity", self.spin_opacity)
        root.addWidget(mask_box)

        # --- Tracking ------------------------------------------------------
        track_box = QtWidgets.QGroupBox("Tracking")
        track_form = QtWidgets.QFormLayout(track_box)
        self.check_sticky_ids = QCheckBox(
            "Sticky track ids (inherit from previous frame)")
        self.check_sticky_ids.setToolTip(
            "Unchecked (default): every new box gets a fresh auto-increment "
            "track id.\nChecked: the k-th box drawn on a frame inherits the "
            "k-th track id\nfrom the nearest earlier annotated frame (what "
            "interpolation expects).")
        track_form.addRow(self.check_sticky_ids)
        root.addWidget(track_box)

        # --- Buttons -------------------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load from file…")
        self.btn_apply = QPushButton("Apply")
        self.btn_save = QPushButton("Save…")
        self.btn_close = QPushButton("Close")
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

        self._prefill_from_window()

    # -- prefill / collect ---------------------------------------------------

    def _is_group_hidden(self, key: str) -> bool:
        if key == "rerun_toggle":
            return self.win.btn_toggle_rerun.isHidden()
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
        for key, cb in self.hide_checks.items():
            cb.setChecked(self._is_group_hidden(key))
        self.combo_flow.setCurrentText(win.interp_flow_method)
        self.combo_cam.setCurrentText(win.interp_camera_model)
        self.spin_match_frac.setValue(win.interp_match_frac)
        self.check_confirm_mismatch.setChecked(win.interp_confirm_mismatch)
        self.combo_device.setCurrentText(win.sam3_device)
        self.spin_sam3_conf.setValue(win.sam3_conf)
        self.check_auto_segment.setChecked(win.auto_segment)
        self.spin_opacity.setValue(win.side.opacity_slider.value())
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
        if sam3.get("device") in ("cuda", "cpu"):
            self.combo_device.setCurrentText(sam3["device"])
        if "conf" in sam3:
            self.spin_sam3_conf.setValue(float(sam3["conf"]))
        if "auto_segment" in sam3:
            self.check_auto_segment.setChecked(bool(sam3["auto_segment"]))
        ui = cfg.get("ui", {})
        if "hide" in ui:
            hidden = set(ui["hide"] or [])
            for key, cb in self.hide_checks.items():
                cb.setChecked(key in hidden)
        if "mask_opacity" in ui:
            self.spin_opacity.setValue(
                max(0, min(100, int(ui["mask_opacity"]))))
        tracking = cfg.get("tracking", {})
        if "sticky_ids" in tracking:
            self.check_sticky_ids.setChecked(bool(tracking["sticky_ids"]))

    def _collect(self) -> Dict[str, Any]:
        """Widget state as a config dict (same schema as the example JSON)."""
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
                "conf": self.spin_sam3_conf.value(),
                "auto_segment": self.check_auto_segment.isChecked(),
            },
            "ui": {
                "hide": [k for k, cb in self.hide_checks.items()
                         if cb.isChecked()],
                "mask_opacity": self.spin_opacity.value(),
            },
            "tracking": {
                "sticky_ids": self.check_sticky_ids.isChecked(),
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


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ReviewWindow(QMainWindow):

    def __init__(self, rrd_index: "RrdFrameIndex | ImageFolderIndex",
                 coco: CocoState,
                 grpc_uri: str, web_port: int,
                 sam3_model: Optional[str] = None,
                 sam3_device: str = "cuda",
                  sam3_conf: float = 0.25,
                  auto_segment: bool = False,
                  seed_from_rrd: bool = True,
                  interp_flow_method: str = "dis",
                  interp_camera_model: str = "none",
                  interp_match_frac: float = 0.2,
                  interp_confirm_mismatch: bool = True,
                  has_viewer: bool = True,
                  ui_hide: Optional[List[str]] = None,
                  mask_opacity: Optional[int] = None,
                  grpc_port: int = 9876,
                  parent=None):
        super().__init__(parent)
        self.rrd_index = rrd_index
        self.coco = coco
        self.grpc_uri = grpc_uri
        self.web_port = web_port
        self.seed_from_rrd = seed_from_rrd
        self.sam3_model = sam3_model
        self.sam3_device = sam3_device
        self.sam3_conf = sam3_conf
        self.auto_segment = auto_segment
        self.interp_flow_method = interp_flow_method
        self.interp_camera_model = interp_camera_model
        # Fraction of min(frame w, h) used as the max pairing distance when
        # matching anchor boxes for interpolation.
        self.interp_match_frac = interp_match_frac
        # Whether to show the track-id mismatch confirmation dialog before
        # interpolating.
        self.interp_confirm_mismatch = interp_confirm_mismatch
        self.grpc_port = grpc_port
        # Rerun viewer subprocess spawned at runtime (File → Load from rrd).
        self._rerun_proc = None

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
        # FIFO of pending single-frame SAM3 jobs (img_path, bboxes_xyxy,
        # concepts, ann_ids). Filled when a run is requested while another
        # is in flight; drained when the current worker finishes/cancels.
        self._sam3_queue: List[Dict[str, Any]] = []
        # Last SAM3 status body (without the "(N in queue)" prefix), so the
        # queue count can be re-rendered without losing the message.
        self._sam3_status_body: str = "idle"
        self._batch_frames_done: int = 0
        self._interp_worker: Optional[InterpBatchWorker] = None
        # Tmp file path for the current frame's image (run_sam3 needs a path).
        self._tmp_image_path: Optional[str] = None

        # Persisted UI state (collapsed Rerun panel etc.).
        self._ui_state_path = Path.home() / ".config" / "rerun_label_review" / "state.json"

        # Frame playback timer.
        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_interval_ms: int = 100  # 1x default (matches combo_speed)
        self._playing: bool = False
        # Session-only opt-out from the X discard-all confirmation dialog.
        self._skip_discard_confirm: bool = False
        # Set once quit has been confirmed, so closeEvent doesn't ask twice.
        self._quit_confirmed: bool = False

        # ---------- layout ----------
        splitter = QSplitter(_QT_HORZ)
        self.canvas = CanvasWidget()
        self.canvas.parent_window = self  # type: ignore[attr-defined]
        splitter.addWidget(self.canvas)

        self.side = SidePanel(coco)
        splitter.addWidget(self.side)
        splitter.setSizes([1200, 360])

        # Right column: web viewer (in a collapsible container) below the canvas
        right_split = QSplitter(_QT_VERT)
        right_split.addWidget(splitter)

        # Rerun viewer wrapped in a container with a toggle button.
        self.rerun_container = QWidget()
        rerun_layout = QVBoxLayout(self.rerun_container)
        self._rerun_layout = rerun_layout  # kept for runtime rrd loading
        rerun_layout.setContentsMargins(0, 0, 0, 0)
        rerun_layout.setSpacing(0)
        # Toggle toolbar.
        self.rerun_toolbar = QWidget()
        tb_layout = QHBoxLayout(self.rerun_toolbar)
        tb_layout.setContentsMargins(6, 2, 6, 2)
        self.btn_toggle_rerun = QPushButton("Hide Rerun ▾")
        self.btn_toggle_rerun.setCheckable(True)
        self.btn_toggle_rerun.clicked.connect(self._on_toggle_rerun)
        tb_layout.addWidget(self.btn_toggle_rerun)
        tb_layout.addStretch(1)
        rerun_layout.addWidget(self.rerun_toolbar)

        self.web_view: Optional[QWebEngineView] = None
        self.bridge: Optional[TimeBridge] = None
        self.channel: Optional[QWebChannel] = None
        if _HAS_WEBENGINE and has_viewer:
            self.web_view = QWebEngineView()
            rerun_layout.addWidget(self.web_view, 1)
            right_split.addWidget(self.rerun_container)
            right_split.setSizes([700, 600])
            self._setup_web_bridge()
        else:
            if has_viewer:
                placeholder_text = (
                    "PyQt6-WebEngine not installed.\n"
                    "Install with: pip install PyQt6-WebEngine\n"
                    "Run `rerun` separately and scrub its timeline."
                )
            else:
                placeholder_text = (
                    "No Rerun viewer (idle / image mode).\n"
                    "File → Open image file(s) / Open folder /\n"
                    "Load from rrd to pick a source."
                )
            placeholder = QLabel(placeholder_text)
            placeholder.setAlignment(Qt.AlignCenter)
            rerun_layout.addWidget(placeholder, 1)
            right_split.addWidget(self.rerun_container)
            self.btn_toggle_rerun.setEnabled(False)

        self.setCentralWidget(right_split)
        self._build_menu()

        # Config-driven UI tweaks: hide button groups, preset mask opacity.
        if ui_hide:
            self.side.set_hidden_groups(ui_hide)
            if "rerun_toggle" in ui_hide:
                self.btn_toggle_rerun.hide()
        if mask_opacity is not None:
            pct = max(0, min(100, int(mask_opacity)))
            self.side.opacity_slider.setValue(pct)
            # Signals aren't connected yet at this point, so set the canvas
            # overlay alpha directly as well.
            self.canvas.set_mask_alpha(round(pct * 255 / 100))

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
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.box_deleted.connect(self._on_box_deleted)
        self.canvas.box_moved.connect(self._on_box_moved)
        self.canvas.box_resized.connect(self._on_box_resized)
        self.canvas.frame_nav.connect(self._on_frame_nav)
        self.canvas.save_quit.connect(self._on_save_quit)
        self.canvas.quit_request.connect(self._on_quit)
        self.canvas.discard_all.connect(self._on_discard_all)
        self.canvas.cat_pick_requested.connect(self._on_cat_pick_requested)
        self.canvas.toggle_masks.connect(self._on_toggle_masks)
        self.canvas.resegment_selected.connect(self._on_resegment_selected)
        self.canvas.play_pause.connect(self._on_play_pause)
        self.canvas.fit_view.connect(lambda: self.canvas._fit_to_view())
        self.canvas.selection_changed.connect(self.side.highlight_box_row)
        self.canvas.selection_changed.connect(lambda _i: self._prefill_recat())
        self.canvas.zoom_to_selected.connect(self._on_zoom_to_selected)
        self.canvas.next_unlabeled.connect(self._on_next_unlabeled)
        self.side.cat_clicked.connect(self._on_side_cat_clicked)
        self.side.slider_moved.connect(self._on_slider_moved)
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
        self.side.toggle_keyframe_clicked.connect(self._on_toggle_keyframe)
        self.side.interpolate_clicked.connect(self._on_interpolate)
        self.side.cancel_interp_clicked.connect(self._on_cancel_interp)
        self.side.track_id_selected.connect(self._on_track_selected)
        self.side.add_cat_clicked.connect(self._on_add_category)
        self.side.rename_cat_clicked.connect(self._on_rename_category)
        self.side.del_cat_clicked.connect(self._on_delete_category)

        if self.bridge is not None:
            self.bridge.time_changed.connect(self._on_viewer_time_changed)

        # Frame slider config
        self.side.set_slider_max(len(self.rrd_index))

        # Shortcuts that work even when canvas doesn't have focus.
        # These mirror the canvas keyPressEvent so the user doesn't need
        # to click the canvas first to give it focus.
        for key, slot in [
            (Qt.Key_N, lambda: self._on_frame_nav(+1)),
            (Qt.Key_B, lambda: self._on_frame_nav(-1)),
            (Qt.Key_S, self._on_save_quit),
            (Qt.Key_Q, self._on_quit),
            (Qt.Key_Space, self._on_play_pause),
            (Qt.Key_F, lambda: self.canvas._fit_to_view()),
            (Qt.Key_D, lambda: self.canvas.keyPressEvent(
                QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_D, Qt.NoModifier))),
            (Qt.Key_A, lambda: self.canvas.keyPressEvent(
                QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_A, Qt.NoModifier))),
            (Qt.Key_X, lambda: self.canvas.keyPressEvent(
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
            fw = QApplication.focusWidget()
            if isinstance(fw, QLineEdit) and fw is not obj:
                fw.clearFocus()
        return super().eventFilter(obj, event)

    def _update_source_label(self) -> None:
        """Show the current frame source in the side panel: the image
        folder for file sources, the .rrd path for recordings."""
        idx = self.rrd_index
        files = getattr(idx, "files", None)
        if files:
            dirs = sorted({os.path.dirname(os.path.abspath(f))
                           for f in files})
            path = (dirs[0] if len(dirs) == 1
                    else f"{len(files)} files from {len(dirs)} folders")
        else:
            path = getattr(idx, "rrd_path", "") or ""
        self.side.set_source(path)

    # ----------------------- web bridge setup -------------------------- #

    def _setup_web_bridge(self) -> None:
        if self.web_view is None:
            return
        self.bridge = TimeBridge(self)
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        # Point the embedded web view at the locally-hosted rerun web
        # viewer (started by `rerun --serve-web --web-viewer-port <port>`).
        url = f"http://127.0.0.1:{self.web_port}/"
        self.web_view.setUrl(QUrl(url))
        # Inject the bridge JS once the page finishes loading.
        self.web_view.loadFinished.connect(self._inject_bridge_js)

    def _inject_bridge_js(self, ok: bool) -> None:
        if not ok or self.web_view is None:
            return
        # Allow a few retries — the wasm viewer takes a moment to expose
        # `window.rerunWidget`.
        js = _BRIDGE_JS + "\n;(function(){tryHook();})();"
        for attempt in range(8):
            QTimer.singleShot(500 * (attempt + 1),
                              lambda: self.web_view and
                              self.web_view.page().runJavaScript(js))
        # After the widget is hooked, pause its timeline so it doesn't
        # autoplay through every frame on load (which would flood the
        # canvas with time_update events). We retry a few times because
        # `rerunWidget` is exposed asynchronously by the wasm viewer.
        pause_js = (
            "(function pause(){"
            "  var w = window.rerunWidget;"
            "  if (w && w.set_time_ctrl){"
            "    try { w.set_time_ctrl(null, null, false); } catch(e){}"
            "  } else { setTimeout(pause, 250); }"
            "})();"
        )
        for attempt in range(20):
            QTimer.singleShot(750 + 250 * attempt,
                              lambda: self.web_view and
                              self.web_view.page().runJavaScript(pause_js))

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
            f"Reviewed: {len(self.coco.reviewed)}/{len(self.rrd_index)}"
        )

    def _on_toggle_rerun(self) -> None:
        """Collapse or expand the embedded Rerun web viewer."""
        collapsed = self.btn_toggle_rerun.isChecked()
        # Hide/show the web view itself; keep the toggle button visible.
        if self.web_view is not None:
            self.web_view.setVisible(not collapsed)
        # Hide/show any non-button widgets in the rerun container layout
        # (placeholder etc.). The toolbar (with the button) stays.
        for i in range(self.rerun_container.layout().count()):
            w = self.rerun_container.layout().itemAt(i).widget()
            if w is self.rerun_toolbar:
                continue
            if w is not None:
                w.setVisible(not collapsed)
        self.btn_toggle_rerun.setText("Show Rerun ▸" if collapsed else "Hide Rerun ▾")
        # Save preference.
        self._write_ui_state(web_collapsed=collapsed)
        # Force the splitter to give the freed space to the canvas.
        rs = self.centralWidget()
        if isinstance(rs, QSplitter):
            sizes = rs.sizes()
            if collapsed:
                rs.setSizes([sizes[0] + sizes[1], 0])
            else:
                rs.setSizes([700, 600])

    def _load_ui_state(self) -> None:
        """Apply persisted UI state (web collapsed etc.)."""
        try:
            if not self._ui_state_path.exists():
                return
            with open(self._ui_state_path, "r") as f:
                state = json.load(f)
            if state.get("web_collapsed") and self.btn_toggle_rerun.isEnabled():
                self.btn_toggle_rerun.setChecked(True)
                self._on_toggle_rerun()
            opacity = state.get("mask_opacity")
            if isinstance(opacity, int) and 0 <= opacity <= 100:
                self.canvas.set_mask_alpha(round(opacity * 255 / 100))
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
        self.canvas.set_mask_alpha(round(pct * 255 / 100))
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
        act_rrd = QAction("Load from rrd…", self)
        act_rrd.setShortcut("Ctrl+R")
        act_rrd.setToolTip("Open a Rerun .rrd recording and embed its viewer.")
        act_rrd.triggered.connect(self._load_rrd_dialog)
        m.addAction(act_rrd)
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
        # Save the current session before switching away from it.
        try:
            self.coco.save(is_final=False)
        except Exception:
            pass
        self._stop_playback()
        self.side.set_playing(False)
        self.rrd_index = new_index
        sticky = self.coco.sticky_track_ids  # setting survives the switch
        self.coco = CocoState(out_json, self.coco.categories)
        self.coco.sticky_track_ids = sticky
        self.coco.load_existing()
        self._update_source_label()
        self.side.coco = self.coco  # side panel keeps its own reference
        self._current_idx = self.coco.load_progress(len(self.rrd_index))
        self._current_image_id = None
        self._last_cat_id = None
        self._pending_cat_id = None
        self.canvas.reset_state()
        self.side.set_slider_max(len(self.rrd_index))
        self.setWindowTitle(f"Computer Vision Label Review Tool — {label}")
        self._load_current()
        self.statusBar().showMessage(
            f"Loaded {len(self.rrd_index)} frame(s) — saving to {out_json}",
            5000)

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
        if dev in ("cuda", "cpu"):
            self.sam3_device = dev
        if "conf" in sam3_cfg:
            self.sam3_conf = float(sam3_cfg["conf"])
        if "auto_segment" in sam3_cfg:
            self.auto_segment = bool(sam3_cfg["auto_segment"])
        ui_cfg = cfg.get("ui", {})
        if "hide" in ui_cfg:
            groups = ui_cfg["hide"] or []
            self.side.set_hidden_groups(groups)
            self.btn_toggle_rerun.setVisible("rerun_toggle" not in groups)
        if "mask_opacity" in ui_cfg:
            pct = max(0, min(100, int(ui_cfg["mask_opacity"])))
            # Signals are connected by now, so this also updates the canvas.
            self.side.opacity_slider.setValue(pct)
        tracking_cfg = cfg.get("tracking", {})
        if "sticky_ids" in tracking_cfg:
            self.coco.sticky_track_ids = bool(tracking_cfg["sticky_ids"])

    def _load_rrd_dialog(self) -> None:
        """File → Load from rrd: index a recording, spawn its web viewer,
        and switch the session to it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load RRD recording", "", "Rerun recordings (*.rrd)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(
            f"Indexing {path} — can take a minute for big recordings…")
        QApplication.processEvents()
        try:
            new_index = RrdFrameIndex(
                path,
                progress_cb=lambda n_chunks, n_imgs: print(
                    f"  scanned {n_chunks} chunks, {n_imgs} unique image "
                    "timestamps", end="\r"),
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Load rrd", str(e))
            return
        QApplication.restoreOverrideCursor()
        if len(new_index) == 0:
            QMessageBox.warning(self, "Load rrd", "No image frames found.")
            return
        # Replace any previously spawned viewer and start a new one.
        if self._rerun_proc is not None:
            try:
                self._rerun_proc.terminate()
            except Exception:
                pass
        self._rerun_proc = _spawn_rerun_web_viewer(
            path, self.web_port, self.grpc_port)
        QApplication.instance().aboutToQuit.connect(
            lambda: self._rerun_proc is not None
            and self._rerun_proc.poll() is None
            and self._rerun_proc.terminate())
        # Create the embedded web view now if we started without one.
        if _HAS_WEBENGINE and self.web_view is None:
            while self._rerun_layout.count() > 1:
                item = self._rerun_layout.takeAt(1)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self.web_view = QWebEngineView()
            self._rerun_layout.addWidget(self.web_view, 1)
            self._setup_web_bridge()
            if self.bridge is not None:
                self.bridge.time_changed.connect(
                    self._on_viewer_time_changed)
        self.btn_toggle_rerun.setEnabled(True)
        # If no categories yet (fresh empty start), seed them from the
        # recording's labels.
        cats = self.coco.categories
        if not cats:
            seeded = _seed_categories(new_index, None)
            if seeded != cats:
                self.coco.categories = seeded
                self.coco.cat_map = {c["id"]: c["name"] for c in seeded}
                self.coco.cat_name_to_id = {c["name"]: c["id"]
                                            for c in seeded}
                self.side._rebuild_cat_list()
        out_json = os.path.splitext(path)[0] + "_labels.json"
        self._switch_source(new_index, out_json, os.path.dirname(path))

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

    def _switch_to_images(self, paths: List[str]) -> None:
        """Replace the frame source with plain image files/folders.

        Starts a fresh COCO session next to the source (``labels_coco.json``
        in the folder, or beside the first file), keeping the current
        category list.
        """
        try:
            new_index = ImageFolderIndex(paths)
        except (FileNotFoundError, RuntimeError) as e:
            QMessageBox.warning(self, "Open images", str(e))
            return
        out_dir = (paths[0] if os.path.isdir(paths[0])
                   else os.path.dirname(os.path.abspath(paths[0])))
        out_json = os.path.join(out_dir, "labels_coco.json")
        self._switch_source(new_index, out_json, out_dir)

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

    def _start_playback(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._play_timer.start(self._play_interval_ms)
        # Pause the embedded Rerun viewer so its timeline doesn't fight
        # with our playback timer (both would emit time_update events
        # and double-advance the canvas).
        self._pause_rerun_viewer()
        self.statusBar().showMessage(
            f"Playing at {self._play_interval_ms} ms/frame", 2000
        )

    def _stop_playback(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._play_timer.stop()
        self.statusBar().showMessage("Paused", 1500)

    def _on_play_tick(self) -> None:
        """Advance one frame per timer tick; stop at the end."""
        if self._current_idx + 1 >= len(self.rrd_index):
            self._stop_playback()
            self.side.set_playing(False)
            self.statusBar().showMessage("Reached last frame", 3000)
            return
        self._on_frame_nav(+1)

    def _pause_rerun_viewer(self) -> None:
        """Send a JS command to pause the embedded Rerun viewer."""
        if self.web_view is None:
            return
        js = (
            "(function(){"
            "  var w = window.rerunWidget;"
            "  if (w && w.set_time_ctrl){"
            "    try { w.set_time_ctrl(null, null, false); } catch(e){}"
            "  }"
            "})();"
        )
        try:
            self.web_view.page().runJavaScript(js)
        except Exception:
            pass

    # ----------------------- frame loading ----------------------------- #

    def _load_current(self) -> None:
        idx = self._current_idx
        if idx < 0 or idx >= len(self.rrd_index):
            return
        frame = self.rrd_index.frame_at(idx)
        arr = self.rrd_index.decode_image(idx)
        h, w = arr.shape[:2]
        image_id = self.coco.ensure_image(frame, w, h)
        self._current_image_id = image_id

        # Seed existing boxes from the .rrd on first visit (unless --no-seed).
        existing = frame.get("existing_boxes", [])
        already = self.coco.anns_for_image(image_id)
        if self.seed_from_rrd and not already and existing:
            for (cx, cy, hw, hh, label) in existing:
                self.coco.seed_box(image_id, cx, cy, hw, hh, label)

        # Build box list for canvas + side panel.
        boxes = []
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
        self.canvas.set_image(arr)
        self.canvas.set_boxes(boxes)
        self.side.set_boxes(boxes)
        self.side.highlight_box_row(-1)
        ts_real = getattr(self.rrd_index, "timestamps_real", True)
        pos = (f"ts={frame['timestamp_ns']}" if ts_real else f"index={idx}")
        self.canvas.set_info(
            f"Frame {idx + 1}/{len(self.rrd_index)}  |  {pos}"
        )
        self.side.set_slider(idx)
        self.side.set_info(idx, len(self.rrd_index),
                           frame["timestamp_ns"], len(boxes),
                           ts_real=ts_real)
        # Update current index for save paths. We do NOT autosave on every
        # frame navigation (it would write the JSON every tick when the
        # embedded Rerun viewer autoplays the timeline). Progress is saved:
        #   - on explicit N/B/X keystrokes (see _on_frame_nav / _on_discard_all)
        #   - on quit (closeEvent, _on_save_quit, _on_quit)
        self.coco.current_idx = idx
        self._sync_keyframe_button()
        self._update_progress()

    # ----------------------- event handlers ---------------------------- #

    def _on_frame_nav(self, delta: int) -> None:
        # Forward nav (N) means "this frame is reviewed" — record it.
        if delta > 0:
            self.coco.mark_reviewed(self._current_idx)
        new_idx = self._current_idx + delta
        if 0 <= new_idx < len(self.rrd_index):
            self._current_idx = new_idx
            self._load_current()
            # Save progress (tmp) on explicit nav — not on autoplay ticks.
            self.coco.save(is_final=False)

    def _on_slider_moved(self, idx: int) -> None:
        if 0 <= idx < len(self.rrd_index) and idx != self._current_idx:
            self._current_idx = idx
            self._load_current()
            self.coco.save(is_final=False)

    def _on_viewer_time_changed(self, ts_ns: int) -> None:
        # Only an RrdFrameIndex is backed by the embedded viewer. After
        # switching to images (or in idle mode) the viewer still shows the
        # old recording, whose timestamps are meaningless for the new index.
        if not isinstance(self.rrd_index, RrdFrameIndex):
            return
        # If the user manually scrubs the Rerun viewer while our play timer
        # is running, stop our playback so the two timelines don't fight.
        if self._playing:
            self._stop_playback()
            self.side.set_playing(False)
        idx = self.rrd_index.find_idx_by_timestamp(ts_ns)
        if idx != self._current_idx and 0 <= idx < len(self.rrd_index):
            self._current_idx = idx
            self._load_current()

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
        self.side.set_boxes(self.canvas._boxes)
        self.statusBar().showMessage(
            f"Moved box #{ann_id} to ({int(x)},{int(y)})", 2500
        )

    def _on_box_resized(self, ann_id: int, x: float, y: float,
                        w: float, h: float) -> None:
        self.coco.resize_box(ann_id, x, y, w, h)
        self.side.set_boxes(self.canvas._boxes)
        self.statusBar().showMessage(
            f"Resized box #{ann_id} to {int(w)}x{int(h)}", 2500
        )

    def _on_zoom_to_selected(self) -> None:
        """Zoom the canvas so the selected box fills ~80% of the view."""
        sel = self.canvas._selected_idx
        if sel < 0 or sel >= len(self.canvas._boxes):
            self.statusBar().showMessage("No box selected to zoom to", 2000)
            return
        x, y, w, h = self.canvas._boxes[sel]["bbox"]
        iw, ih = self.canvas._image_size
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            return
        vw, vh = self.canvas.width(), self.canvas.height()
        # Scale so the box fills 80% of the smaller view dimension.
        scale = min(vw / (w * 1.25), vh / (h * 1.25))
        scale = max(0.05, min(40.0, scale))
        self.canvas._scale = scale
        # Center the box in the view.
        cx_img = x + w / 2.0
        cy_img = y + h / 2.0
        self.canvas._offset = QtCore.QPointF(
            vw / 2.0 - cx_img * scale, vh / 2.0 - cy_img * scale
        )
        self.canvas.update()
        self.statusBar().showMessage(
            f"Zoomed to box #{self.canvas._boxes[sel]['id']}", 2000
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

    def _ensure_image_id(self, frame_idx: int) -> Optional[int]:
        """Ensure a COCO image record exists for frame_idx (no full decode).

        Borrows the size from any already-visited image record; only decodes
        the frame itself if no dimensions are known at all.
        """
        known = self.coco._img_id_by_idx.get(frame_idx)
        if known is not None:
            return known
        frame = self.rrd_index.frame_at(frame_idx)
        w = h = 0
        for img in self.coco.images:
            if img.get("width") and img.get("height"):
                w, h = img["width"], img["height"]
                break
        if not w or not h:
            arr = self.rrd_index.decode_image(frame_idx)
            h, w = arr.shape[:2]
        return self.coco.ensure_image(frame, w, h)

    def _on_interpolate(self) -> None:
        """I key / Interpolate button: flow-interpolate the gap around the
        current frame between the nearest labeled anchor frames.

        If the current frame has boxes it is the END anchor and the gap is
        filled backward; otherwise anchors are needed on both sides. Interior
        frames that already have boxes are skipped.
        """
        if (self._interp_worker is not None
                and self._interp_worker.isRunning()):
            self.statusBar().showMessage("Interpolation already running", 2500)
            return
        cur = self._current_idx
        total = len(self.rrd_index)
        if cur < 0 or cur >= total:
            return
        anchors = self.coco.anchor_candidates()
        if not anchors:
            self.statusBar().showMessage(
                "Interpolate: label at least one frame first", 3000)
            return
        before = [f for f in anchors if f < cur]
        after = [f for f in anchors if f > cur]
        if self.coco.frame_has_boxes(cur):
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
        img_a = self.coco._img_id_by_idx.get(a)
        img_b = self.coco._img_id_by_idx.get(b)
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
        self.canvas.setEnabled(False)
        self._interp_worker = InterpBatchWorker(
            self.rrd_index, jobs, tmp_base,
            flow_method=self.interp_flow_method,
            camera_model=self.interp_camera_model,
            parent=self)
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
        self.canvas.setEnabled(True)
        self.side.set_interp_running(False)
        added = 0
        skipped = 0
        with self.coco.undo_stack.group("interpolate boxes"):
            for job, pairs in results:
                for p in range(job["a"] + 1, job["b"]):
                    res = pairs.get(p)
                    if res is None:
                        continue
                    if self.coco.frame_has_boxes(p):
                        skipped += 1
                        continue
                    img_id = self._ensure_image_id(p)
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
        self.canvas.setEnabled(True)
        self.side.set_interp_running(False)
        self.side.set_interp_status("Interpolation failed")
        QMessageBox.critical(self, "Interpolation failed", err)

    def _on_interp_cancelled(self) -> None:
        self.canvas.setEnabled(True)
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
        new_ann_id = self.coco.add_box(image_id, x, y, w, h, cat_id)
        self._last_cat_id = cat_id
        self._refresh_boxes()
        ann = self.coco.get_box(new_ann_id)
        tid = ann.get("track_id") if ann else None
        tid_txt = f" T{tid}" if tid is not None else ""
        self.statusBar().showMessage(
            f"Added box cat={self.coco.cat_map.get(cat_id, '?')}{tid_txt} "
            f"(ann_id={new_ann_id})", 3000
        )
        if self.auto_segment and _SAM3_AVAILABLE:
            # Auto-run SAM3 on the freshly added box.
            img_path = self._write_tmp_image()
            if img_path is not None:
                self._start_sam3_worker(
                    img_path,
                    bboxes_xyxy=[[x, y, x + w, y + h]],
                    concepts=[self.coco.cat_map[cat_id]],
                    ann_ids=[new_ann_id],
                )

    def _assign_pending_cat(self, cat_id: int) -> None:
        """Number-key category pick for the pending rectangle."""
        if cat_id not in self.coco.cat_map:
            self.statusBar().showMessage(f"⚠️ Category {cat_id} not found", 3000)
            return
        rect = self.canvas.get_pending_rect()
        if rect is None or self._current_image_id is None:
            return
        x, y, w, h = rect
        self.canvas.reset_state()
        # Reuse the shared add-box path.
        self._on_box_added(self._current_image_id, x, y, w, h, cat_id)

    def _on_discard_all(self) -> None:
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
            if 0 <= i < len(self.rrd_index) and i != self._current_idx:
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
        self.canvas.set_boxes([])
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
        total = len(self.rrd_index)
        for step in range(1, total + 1):
            idx = (self._current_idx + step) % total
            if idx in self.coco.reviewed:
                continue
            img_id = self.coco._img_id_by_idx.get(idx)
            if img_id is not None and self.coco.anns_for_image(img_id):
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
        # If we have a pending rect, assign; else preselect for the next draw.
        if self.canvas.get_pending_rect() is not None:
            self._assign_pending_cat(cat_id)
        else:
            self._on_preselect_cat(cat_id)

    def _refresh_boxes(self) -> None:
        if self._current_image_id is None:
            return
        boxes = []
        for ann in self.coco.anns_for_image(self._current_image_id):
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
        self.canvas.set_boxes(boxes)
        self.side.set_boxes(boxes)
        self.side.highlight_box_row(self.canvas._selected_idx)
        self.side.set_info(self._current_idx, len(self.rrd_index),
                           self.rrd_index.frame_at(self._current_idx)["timestamp_ns"],
                           len(boxes))
        self._update_progress()

    # ---- box list panel + preselected category ---- #

    def _on_box_list_selected(self, box_idx: int) -> None:
        """Clicking a box-list row selects that box on the canvas."""
        if 0 <= box_idx < len(self.canvas._boxes):
            self.canvas._selected_idx = box_idx
            self.canvas._multi_selected = {box_idx}
            self.canvas.update()
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
        # Clear any pending preselection pointing at the deleted category.
        if self._pending_cat_id == cat_id:
            self._pending_cat_id = None
            self.canvas._drawing = False
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
        self.canvas._drawing = True
        self.canvas.update()
        self.statusBar().showMessage(f"Drawing: {name} — drag on the image", 4000)

    def _on_recat_selected(self, cat_id: int) -> None:
        """Reassign the selected box's category (typed id + Enter)."""
        sel = self.canvas._selected_idx
        if not (0 <= sel < len(self.canvas._boxes)):
            self.statusBar().showMessage("Select a box first", 2500)
            return
        if cat_id not in self.coco.cat_map:
            self.statusBar().showMessage(f"⚠️ Category {cat_id} not found", 3000)
            return
        ann_id = self.canvas._boxes[sel]["id"]
        if self.coco.set_cat(ann_id, cat_id):
            self._refresh_boxes()
            self.canvas._selected_idx = sel
            self.canvas._multi_selected = {sel}
            self.canvas.update()
            name = self.coco.cat_map.get(cat_id, "?")
            self.statusBar().showMessage(
                f"Box #{ann_id} → {name} (cat {cat_id})", 2500)
            self.canvas.setFocus()  # keep hotkeys working after Enter

    def _prefill_recat(self) -> None:
        """Show the selected box's cat id + track id in their fields."""
        sel = self.canvas._selected_idx
        if 0 <= sel < len(self.canvas._boxes):
            box = self.canvas._boxes[sel]
            self.side.recat_edit.setText(str(box.get("cat_id", "")))
            self.side.prefill_track(box.get("track_id"))

    def _focus_recat_edit(self) -> None:
        """C key: prefill with the current cat and focus the recat field."""
        self._prefill_recat()
        self.side.recat_edit.setFocus()
        self.side.recat_edit.selectAll()

    def _focus_track_edit(self) -> None:
        """T key: prefill with the current track id and focus the field."""
        self._prefill_recat()
        self.side.track_edit.setFocus()
        self.side.track_edit.selectAll()

    def _on_track_selected(self, value) -> None:
        """Set the selected box's track id (typed id + Enter; None clears)."""
        sel = self.canvas._selected_idx
        if not (0 <= sel < len(self.canvas._boxes)):
            self.statusBar().showMessage("Select a box first", 2500)
            return
        ann_id = self.canvas._boxes[sel]["id"]
        if self.coco.set_track_id(ann_id, value):
            self._refresh_boxes()
            self.canvas._selected_idx = sel
            self.canvas._multi_selected = {sel}
            self.canvas.update()
            self.side.prefill_track(value)
            self.statusBar().showMessage(
                f"Box #{ann_id} track id → "
                f"{value if value is not None else '(cleared)'}", 2500)
            self.canvas.setFocus()  # keep hotkeys working after Enter

    def _update_progress(self) -> None:
        """Refresh the annotation-coverage progress bar in the side panel."""
        annotated = len({a["image_id"] for a in self.coco.annotations
                         if a["id"] not in self.coco.removed_ids})
        self.side.set_annotated_progress(annotated, len(self.rrd_index))

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
        if (self._sam3_worker is not None
                and self._sam3_worker.isRunning()) or \
           (self._sam3_batch_worker is not None
                and self._sam3_batch_worker.isRunning()):
            return
        job = self._sam3_queue.pop(0)
        self._start_sam3_worker(job["img_path"], job["bboxes_xyxy"],
                                job["concepts"], job["ann_ids"])

    def _write_tmp_image(self) -> Optional[str]:
        """Write the current frame's image to a tmp file (for run_sam3).

        File-backed sources (image-folder mode) are used directly — no
        copy. rrd frames write their blob. Falls back to decode_image →
        PNG when a frame has neither blob nor file."""
        if self._current_idx < 0 or self._current_idx >= len(self.rrd_index):
            return None
        frame = self.rrd_index.frame_at(self._current_idx)
        fp = frame.get("file_path")
        if fp and os.path.exists(fp):
            return fp
        tmp_dir = Path(self.coco.output_json).parent / "_tmp_sam3_imgs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        blob = frame.get("image_blob")
        if not blob:
            # Last resort: decode in memory and write a PNG.
            try:
                arr = self.rrd_index.decode_image(self._current_idx)
            except Exception as e:
                print(f"⚠️ Failed to decode frame for SAM3: {e}")
                return None
            path = tmp_dir / f"frame_{self._current_idx}.png"
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
        path = tmp_dir / f"frame_{self._current_idx}{ext}"
        try:
            with open(path, "wb") as f:
                f.write(blob)
            self._tmp_image_path = str(path)
            return self._tmp_image_path
        except Exception as e:
            print(f"⚠️ Failed to write tmp image for SAM3: {e}")
            return None

    def _on_run_sam3_all(self) -> None:
        """Run SAM3 on every bbox on the current frame."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable. "
                                 "Install ultralytics + segment-anything and "
                                 "place weights at "
                                 "core/sam3/models/sam3-model/sam3.pt")
            return
        if self._current_image_id is None:
            return
        anns = self.coco.anns_for_image(self._current_image_id)
        if not anns:
            print("ℹ️ No bboxes on this frame — nothing to segment.")
            return
        img_path = self._write_tmp_image()
        if img_path is None:
            self._set_sam3_status("failed — no image for this frame")
            self.statusBar().showMessage(
                "⚠️ Could not prepare the frame image for SAM3", 4000)
            return
        bboxes_xyxy = []
        concepts = []
        ann_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes_xyxy.append([x, y, x + w, y + h])
            concepts.append(self.coco.cat_map.get(ann["category_id"], "object"))
            ann_ids.append(ann["id"])
        self._start_sam3_worker(img_path, bboxes_xyxy, concepts, ann_ids)

    def _on_sam3_all_frames(self) -> None:
        """Background SAM3 over every frame that has boxes without masks."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable.")
            return
        if (self._sam3_worker is not None and self._sam3_worker.isRunning()) or \
           (self._sam3_batch_worker is not None
                and self._sam3_batch_worker.isRunning()):
            self.statusBar().showMessage("SAM3 already running", 2500)
            return
        jobs = []
        total_boxes = 0
        for idx in range(len(self.rrd_index)):
            img_id = self.coco._img_id_by_idx.get(idx)
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
        if not jobs:
            self.statusBar().showMessage(
                "Nothing to do — every annotated box already has a mask", 4000)
            return
        ret = QMessageBox.question(
            self, "Auto-annotate all with SAM3?",
            f"Run SAM3 on {total_boxes} box(es) without masks across "
            f"{len(jobs)} frame(s)?\n"
            f"Runs in the background (device: {self.sam3_device}, CPU "
            f"fallback on CUDA OOM). You can cancel anytime.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes)
        if ret != QMessageBox.StandardButton.Yes:
            return
        tmp_dir = str(Path(self.coco.output_json).parent / "_tmp_sam3_imgs")
        self._batch_frames_done = 0
        self._set_sam3_status(f"all: 0/{len(jobs)} frames…")
        self.side.set_sam3_running(True)
        self.canvas.setEnabled(False)
        self._sam3_batch_worker = SAM3BatchWorker(
            self.rrd_index, jobs, tmp_dir,
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
        self._sam3_batch_worker.start()

    def _on_batch_frame_done(self, frame_idx: int, results: list) -> None:
        """Apply one frame's masks (grouped as a single undo entry)."""
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
        self._set_sam3_status(f"all: {done}/{total} frames…")

    def _on_batch_finished(self, n_ok: int, n_fail: int) -> None:
        self.canvas.setEnabled(True)
        self.side.set_sam3_running(False)
        self._set_sam3_status(
            f"all: done — {n_ok} mask(s), {n_fail} failed")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage(
            f"Auto-annotate finished: {n_ok} masks", 4000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_batch_cancelled(self) -> None:
        self.canvas.setEnabled(True)
        self.side.set_sam3_running(False)
        self._set_sam3_status("all: cancelled")
        self.coco.save(is_final=False)
        self._refresh_boxes()
        self.statusBar().showMessage("Auto-annotate cancelled", 3000)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_resegment_selected(self) -> None:
        """Re-run SAM3 on just the selected bbox (R key / button)."""
        if not _SAM3_AVAILABLE:
            QMessageBox.warning(self, "SAM3 unavailable",
                                 "core.models_inference.run_sam3 not importable.")
            return
        sel = self.canvas._selected_idx
        if sel < 0 or sel >= len(self.canvas._boxes):
            print("ℹ️ No box selected — press R after clicking a box.")
            return
        box = self.canvas._boxes[sel]
        ann_id = box["id"]
        x, y, w, h = box["bbox"]
        img_path = self._write_tmp_image()
        if img_path is None:
            return
        bboxes_xyxy = [[x, y, x + w, y + h]]
        concepts = [self.coco.cat_map.get(box.get("cat_id", 0), "object")]
        ann_ids = [ann_id]
        print(f"🔬 Re-segmenting ann_id={ann_id} "
              f"bbox={bboxes_xyxy[0]} cat={concepts[0]}")
        self._start_sam3_worker(img_path, bboxes_xyxy, concepts, ann_ids)

    def _start_sam3_worker(self, img_path: str, bboxes_xyxy: list,
                           concepts: list, ann_ids: list) -> None:
        if (self._sam3_worker is not None and self._sam3_worker.isRunning()) or \
           (self._sam3_batch_worker is not None
                and self._sam3_batch_worker.isRunning()):
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
        self.canvas.setEnabled(False)
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
        self._sam3_worker.start()

    def _on_sam3_progress(self, done: int, total: int, concept: str) -> None:
        self._set_sam3_status(
            f"{done}/{total} concept(s) done — last: {concept}"
        )

    def _on_cancel_sam3(self) -> None:
        cancelled_any = False
        if self._sam3_queue:
            self._sam3_queue.clear()  # cancel drops queued jobs too
            cancelled_any = True
        if self._sam3_worker is not None and self._sam3_worker.isRunning():
            self._sam3_worker.cancel()
            cancelled_any = True
        if (self._sam3_batch_worker is not None
                and self._sam3_batch_worker.isRunning()):
            self._sam3_batch_worker.cancel()
            cancelled_any = True
        if cancelled_any:
            self._set_sam3_status("cancelling…")
            self.side.btn_cancel_sam3.setEnabled(False)

    def _on_sam3_cancelled(self) -> None:
        self.canvas.setEnabled(True)
        self.side.set_sam3_running(False)
        self._set_sam3_status("cancelled — no masks applied")
        self.statusBar().showMessage("SAM3 cancelled", 2500)
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_sam3_finished(self, results: list) -> None:
        self.canvas.setEnabled(True)
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
        for r in results:
            ann_id = r.get("ann_id")
            mask = r.get("mask")
            if ann_id is None:
                continue
            self.coco.set_mask(ann_id, mask)
            if not r.get("success"):
                print(f"  ⚠️ ann_id={ann_id}: {r.get('error', 'no mask')}")
        self._refresh_boxes()
        # Save progress so masks survive a crash.
        self.coco.save(is_final=False)
        print(f"✅ SAM3 finished — {n_ok}/{len(results)} masks assigned")
        QTimer.singleShot(0, self._start_next_queued_sam3)

    def _on_sam3_failed(self, msg: str) -> None:
        self.canvas.setEnabled(True)
        self.side.set_sam3_running(False)
        self._sam3_queue.clear()  # hard failure — don't keep retrying queued jobs
        self._set_sam3_status(f"failed — {msg}")
        QMessageBox.warning(self, "SAM3 failed", msg)

    def _on_toggle_masks(self) -> None:
        new_vis = not self.canvas.masks_visible()
        self.canvas.set_masks_visible(new_vis)
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
        """Cancel and reap the background workers (SAM3 single/all-frames,
        interpolation) so no QThread is destroyed mid-run at exit — Qt
        aborts the process when that happens."""
        self._sam3_queue.clear()
        for w in (self._sam3_worker, self._sam3_batch_worker,
                  self._interp_worker):
            if w is None or not w.isRunning():
                continue
            w.cancel()
            if not w.wait(2000):
                # A run_sam3 / interpolate call is still in flight —
                # last resort before exit.
                w.terminate()
                w.wait(1000)


# ---------------------------------------------------------------------------
# Web viewer host: spawn `rerun --serve-web --web-viewer-port <port>` so the
# embedded QWebEngineView has something to point at. We don't use
# `rr.serve_web_viewer()` from the SDK here because that needs an active
# RecordingStream; we want to point at the *existing* .rrd file.
# ---------------------------------------------------------------------------

def _spawn_rerun_web_viewer(rrd_path: str, web_port: int,
                            grpc_port: int) -> "subprocess.Popen":
    import subprocess
    cmd = [
        "rerun",
        "--port", str(grpc_port),
        "--web-viewer-port", str(web_port),
        "--web-viewer",
        "--serve-web",
        "--bind", "127.0.0.1",
        str(rrd_path),
    ]
    print(f"Starting rerun web viewer: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    # Wait briefly for the HTTP server to come up.
    import urllib.request
    url = f"http://127.0.0.1:{web_port}/"
    for _ in range(40):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            print(f"Rerun web viewer ready at {url}")
            return proc
        except Exception:
            time.sleep(0.25)
    print(f"⚠️ Rerun web viewer not responding at {url} after 10s — "
          f"the embedded view may stay blank. You can still label via "
          f"the slider/canvas.")
    return proc


# ---------------------------------------------------------------------------
# Category bootstrapping
# ---------------------------------------------------------------------------

def _seed_categories(rrd_index: RrdFrameIndex,
                     seed_json: Optional[str]) -> List[Dict[str, Any]]:
    """Build the category list. Priority:

    1. If --json is given, use its categories (and merge any labels found
       in the .rrd as new categories).
    2. Otherwise start empty — categories are created as needed: from the
       side panel's Add field, from a resumed labels file, or on the fly
       from .rrd box labels (see CocoState._resolve_cat_id).

    For .rrd sources the recording's Boxes2D:labels are still merged in so
    seeded boxes get proper category names instead of bare ids.
    """
    cats: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}

    if seed_json and os.path.exists(seed_json):
        with open(seed_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("categories", []):
            cats.append({"id": c["id"], "name": c["name"]})
            seen[c["name"]] = c["id"]

    # Scan labels in the .rrd.
    label_set: set = set()
    for frame in rrd_index.frames:
        for (_, _, _, _, lbl) in frame["existing_boxes"]:
            if lbl:
                label_set.add(lbl)
    # Decide whether labels are category names or instance ids.
    name_labels = [l for l in label_set if not l.isdigit()]
    if name_labels and not cats:
        for i, n in enumerate(sorted(name_labels)):
            cats.append({"id": i, "name": n})
            seen[n] = i
    elif name_labels:
        # Merge new names into the seed categories.
        next_id = max((c["id"] for c in cats), default=-1) + 1
        for n in sorted(name_labels):
            if n not in seen:
                cats.append({"id": next_id, "name": n})
                seen[n] = next_id
                next_id += 1

    # No seed file and no .rrd labels → start with an empty category list;
    # the user adds categories from the side panel (or a resumed labels
    # file brings its own).
    return cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 2D bbox reviewer for a .rrd recording "
                    "or a folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rrd", help="Path to the .rrd recording "
                        "(mutually exclusive with --images).")
    parser.add_argument("--images", nargs="+", metavar="PATH",
                        help="Image files and/or folders to review instead of "
                             "an .rrd (folders are scanned for "
                             "jpg/jpeg/png/bmp/webp/tif, sorted by name).")
    parser.add_argument("--output_json",
                        help="Output COCO JSON path (progress file is "
                             "auto-saved). Default: <source>/labels_coco.json "
                             "for --images, <rrd>_labels.json for --rrd, "
                             "./untitled_labels_coco.json in idle mode.")
    parser.add_argument("--json", help="Seed COCO JSON for categories / existing labels.")
    parser.add_argument("--db", help="Deprecated/unused: categories are no longer read "
                        "from a SQLite DB. Kept so old commands still parse.")
    parser.add_argument("--image-entity", help="Override the image entity path (auto-detect by default).")
    parser.add_argument("--bboxes-entity", help="Override the Boxes2D entity path (auto-detect).")
    parser.add_argument("--timeline", help="Timeline to use (auto-detect; prefers ros_time).")
    parser.add_argument("--output-yolo-dir", help="Also export YOLO dataset on exit.")
    parser.add_argument("--data-yaml", help="Reference data.yaml for YOLO class order.")
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--web-port", type=int, default=9090)
    # SAM3 options
    parser.add_argument("--sam3-model", default=None,
                        help="Path to SAM3 weights (default: "
                             "core/sam3/models/sam3-model/sam3.pt).")
    parser.add_argument("--sam3-device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for SAM3 inference (default: cuda).")
    parser.add_argument("--sam3-conf", type=float, default=0.25,
                        help="SAM3 confidence threshold (default: 0.25).")
    parser.add_argument("--auto-segment", action="store_true",
                        help="Automatically run SAM3 after each new bbox is drawn.")
    parser.add_argument("--no-seed", action="store_true",
                        help="Do not seed boxes from the .rrd's existing "
                             "Boxes2D track (start with a blank canvas).")
    # Interpolation options (I key / Interpolate button)
    parser.add_argument("--interp-flow-method",
                        choices=["dis", "klt", "farneback"], default="dis",
                        help="Optical flow method for gap interpolation "
                             "(default: dis — dense inverse search, best for "
                             "handheld camera shake).")
    parser.add_argument("--interp-camera-model",
                        choices=["none", "global"], default="none",
                        help="Camera motion model for gap interpolation "
                             "(default: none; 'global' fits a per-frame "
                             "RANSAC similarity transform).")
    parser.add_argument("--config",
                        help="JSON config file (interpolation/sam3/ui "
                             "settings; values override the corresponding "
                             "CLI flags). See scripts/config/"
                             "label_review.example.json.")
    args = parser.parse_args()

    if args.rrd and args.images:
        parser.error("--rrd and --images are mutually exclusive")
    # Neither source given → start idle; the user picks a source from the
    # File menu (Open image file(s) / Open folder / Load from rrd).

    # Optional JSON config; its values override the corresponding CLI flags.
    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"⚙️  Loaded config: {args.config}")
        except Exception as e:
            print(f"❌ Could not read config {args.config}: {e}")
            sys.exit(1)
    interp_cfg = cfg.get("interpolation", {})
    sam3_cfg = cfg.get("sam3", {})
    ui_cfg = cfg.get("ui", {})
    tracking_cfg = cfg.get("tracking", {})

    flow_method = interp_cfg.get("flow_method", args.interp_flow_method)
    if flow_method not in ("dis", "klt", "farneback"):
        print(f"⚠️ config interpolation.flow_method {flow_method!r} "
              "invalid; using 'dis'")
        flow_method = "dis"
    camera_model = interp_cfg.get("camera_model", args.interp_camera_model)
    if camera_model not in ("none", "global"):
        print(f"⚠️ config interpolation.camera_model {camera_model!r} "
              "invalid; using 'none'")
        camera_model = "none"
    sam3_device = sam3_cfg.get("device", args.sam3_device)
    if sam3_device not in ("cuda", "cpu"):
        print(f"⚠️ config sam3.device {sam3_device!r} invalid; "
              f"using '{args.sam3_device}'")
        sam3_device = args.sam3_device

    # ---------- 1. Index the frame source ----------
    rrd_path = None
    if args.images:
        print(f"🖼️  Loading images from: {args.images}")
        rrd_index = ImageFolderIndex(args.images)
        print(f"✅ Indexed {len(rrd_index)} images")
        if rrd_index.timestamps_real:
            print("🕒 Filenames look like timestamps — "
                  "slider/info will show them.")
        if not args.output_json:
            base = (args.images[0] if os.path.isdir(args.images[0])
                    else os.path.dirname(os.path.abspath(args.images[0])))
            args.output_json = os.path.join(base, "labels_coco.json")
    elif args.rrd:
        rrd_path = os.path.abspath(args.rrd)
        if not os.path.exists(rrd_path):
            print(f"❌ .rrd not found: {rrd_path}")
            sys.exit(1)
        print(f"🔍 Scanning {rrd_path} (this can take a minute for big recordings)…")
        rrd_index = RrdFrameIndex(
            rrd_path,
            image_entity=args.image_entity,
            bboxes_entity=args.bboxes_entity,
            timeline=args.timeline,
            progress_cb=lambda n_chunks, n_imgs: print(
                f"  scanned {n_chunks} chunks, {n_imgs} unique image timestamps",
                end="\r"),
        )
        print(f"\n✅ Indexed {len(rrd_index)} frames "
              f"(image entity: {rrd_index.image_entity}, "
              f"bboxes entity: {rrd_index.bboxes_entity}, "
              f"timeline: {rrd_index.timeline})")
        if not args.output_json:
            args.output_json = os.path.splitext(rrd_path)[0] + "_labels.json"
    else:
        # Idle start — no source; pick one from the File menu.
        rrd_index = EmptyIndex()
        if not args.output_json:
            args.output_json = os.path.abspath("untitled_labels_coco.json")
        print("💤 No source given — idle mode. Use File → Open image "
              "file(s) / Open folder / Load from rrd to begin.")
    if len(rrd_index) == 0 and (args.rrd or args.images):
        print("❌ No image frames found.")
        sys.exit(1)

    # ---------- 2. Categories / COCO state ----------
    categories = _seed_categories(rrd_index, args.json)
    print(f"🏷️  Categories: {[c['name'] for c in categories]}")
    coco = CocoState(args.output_json, categories)
    # Default: fresh auto-increment track ids; sticky inheritance is opt-in.
    coco.sticky_track_ids = bool(tracking_cfg.get("sticky_ids", False))
    coco.load_existing()
    coco.current_idx = coco.load_progress(len(rrd_index))

    # ---------- 3. Spawn the Rerun web viewer (rrd mode only) ----------
    proc = None
    if args.rrd:
        proc = _spawn_rerun_web_viewer(rrd_path, args.web_port, args.grpc_port)

    # ---------- 4. Qt app ----------
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Computer Vision Label Review Tool")

    # Clean up the spawned rerun process on exit.
    def _shutdown():
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    app.aboutToQuit.connect(_shutdown)
    signal.signal(signal.SIGINT, lambda *a: app.quit())

    win = ReviewWindow(rrd_index, coco,
                       grpc_uri=(f"rerun+http://127.0.0.1:{args.grpc_port}/proxy"
                                 if args.rrd else ""),
                       web_port=args.web_port,
                       sam3_model=args.sam3_model,
                       sam3_device=sam3_device,
                       sam3_conf=float(sam3_cfg.get("conf", args.sam3_conf)),
                       auto_segment=bool(sam3_cfg.get("auto_segment",
                                                      args.auto_segment)),
                        seed_from_rrd=not args.no_seed and bool(args.rrd),
                        has_viewer=bool(args.rrd),
                        interp_flow_method=flow_method,
                        interp_camera_model=camera_model,
                        interp_match_frac=float(interp_cfg.get(
                            "match_max_dist_frac", 0.2)),
                        interp_confirm_mismatch=bool(interp_cfg.get(
                            "confirm_mismatch", True)),
                        # Interpolation + keyframe controls are hidden by
                        # default; an explicit "ui": {"hide": [...]} config
                        # overrides this (set [] to show everything), and
                        # the Config dialog can toggle them at runtime.
                        ui_hide=ui_cfg.get("hide",
                                           ["interpolate", "keyframe"]),
                        mask_opacity=ui_cfg.get("mask_opacity"),
                        grpc_port=args.grpc_port)
    win.show()
    exit_code = app.exec()

    # ---------- 5. Optional YOLO export ----------
    if args.output_yolo_dir:
        print(f"\nExporting YOLO dataset to {args.output_yolo_dir}")
        # Reuse helpers from 08_click_review_coco.py if importable.
        try:
            from importlib import util
            here = Path(__file__).resolve().parent
            mod_path = here / "08_click_review_coco.py"
            if mod_path.exists():
                spec = util.spec_from_file_location("click_review_coco", mod_path)
                mod = util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                final_coco = {
                    "images": coco.images,
                    "annotations": [
                        ann for ann in coco.annotations
                        if ann["id"] not in coco.removed_ids
                    ],
                    "categories": coco.categories,
                }
                # Write images to a temp dir: from the .rrd blobs, or copy
                # the source files directly in --images mode.
                tmp_img_dir = Path(args.output_yolo_dir) / "_src_images"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                for img in coco.images:
                    frame = rrd_index.frame_at(img["frame_idx"])
                    blob = frame["image_blob"]
                    if blob:
                        with open(tmp_img_dir / img["file_name"], "wb") as f:
                            f.write(blob)
                    elif frame.get("file_path"):
                        with open(frame["file_path"], "rb") as src, \
                                open(tmp_img_dir / img["file_name"], "wb") as dst:
                            dst.write(src.read())
                mod.export_yolo_from_coco(
                    final_coco, img_dir=tmp_img_dir,
                    output_dir=args.output_yolo_dir,
                    data_yaml=args.data_yaml,
                    copy_images=True,
                )
            else:
                print("⚠️ 08_click_review_coco.py not found; skipping YOLO export.")
        except Exception as e:
            print(f"⚠️ YOLO export failed: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
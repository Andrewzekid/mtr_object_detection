#!/usr/bin/env python3
"""
Rerun-based interactive 2D bbox reviewer for an existing .rrd recording.

What this does
---------------
* Opens an existing Rerun ``.rrd`` recording.
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
  stored per-annotation in the COCO output (PNG-encoded in the ``mask`` field)
  so they round-trip with the json file.
* ``R`` re-segments the *selected* bbox only — useful after you've redrawn a
  bbox to fix a bad SAM3 mask: delete the old box, draw a new one, then press
  R while it's selected to regenerate the mask.
* ``--auto-segment`` runs SAM3 automatically right after each new bbox is
  drawn, so you don't need to press anything.

The .rrd must contain at least one ``EncodedImage`` entity (auto-detected).
The script auto-detects:
  * the image entity path (first entity with archetype EncodedImage),
  * the matching ``Boxes2D`` sibling entity (``<image>/bboxes2d`` by default),
  * the timeline to use (prefers ``ros_time`` if present, else ``log_time``),
  * categories from the ``Boxes2D:labels`` column if it contains category
    names; otherwise uses the categories from a seeded ``--json`` COCO file
    or falls back to integer ids.

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

Key bindings (in the 2D canvas, when it has focus — click it once):
    D / del  : delete selected box
    A        : toggle draw mode (click-drag a new box)
    N / →    : next frame
    B / ←    : previous frame
    X        : discard ALL boxes on this frame and advance
    S        : save final result and quit
    Q / ESC  : quit (progress saved in tmp file)
    0..9     : when drawing, assign category id to the pending rectangle
               (use the buttons in the side panel for ids > 9)
    M        : toggle mask overlay visibility
    R        : re-segment the selected bbox with SAM3 (replaces its mask)
    + / =    : zoom in   |  -  : zoom out  |  0 : reset zoom
    arrows   : pan (when zoomed)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import sqlite3
import sys
import threading
import time
import warnings
from collections import defaultdict
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
Qt.Key_D = Qt.Key.Key_D  # type: ignore[attr-defined]
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
    QSplitter, QFrame, QSizePolicy, QMessageBox,
)
# QSizePolicy scoped enum alias
QSizePolicy.Expanding = QSizePolicy.Policy.Expanding  # type: ignore[attr-defined]
QSizePolicy.Fixed = QSizePolicy.Policy.Fixed  # type: ignore[attr-defined]
QSizePolicy.Preferred = QSizePolicy.Policy.Preferred  # type: ignore[attr-defined]
from PyQt6.QtGui import QShortcut
from PyQt6.QtCore import QThread
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

# SAM3 (optional — core.models_inference.run_sam3)
_SAM3_AVAILABLE = False
try:
    import sys as _sys
    _PROJ_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJ_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJ_ROOT)
    from core.models_inference import run_sam3  # type: ignore[import-not-found]
    _SAM3_AVAILABLE = True
except Exception as _sam3_import_err:
    run_sam3 = None  # type: ignore[assignment]
    print(f"⚠️ SAM3 not available ({_sam3_import_err}). "
          f"Segmentation features will be disabled.")


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


# ---------------------------------------------------------------------------
# SAM3Worker: runs run_sam3 on a worker thread so the UI doesn't freeze.
# ---------------------------------------------------------------------------

class SAM3Worker(QThread):
    """Asynchronous SAM3 inference thread.

    Inputs:
      image_path : str — path to a temp image file on disk (run_sam3 needs a
                   path; we dump the .rrd blob there).
      bboxes_xyxy: list of [x1,y1,x2,y2] pixel coords (one per region).
      concepts   : list of class names (one per bbox, used for labelling).
      model_path, device, conf : forwarded to run_sam3.

    Emits:
      finished_signal(list_of_dicts) where each dict is
        {ann_id: int|None, bbox_xyxy: [...], mask: HxW bool array|None,
         label: str, area: float, success: bool, error: str|None}
    """

    finished_signal = pyqtSignal(list)
    failed_signal = pyqtSignal(str)

    def __init__(self, image_path: str, bboxes_xyxy: list,
                 concepts: list, ann_ids: list,
                 model_path: Optional[str], device: str, conf: float,
                 parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.bboxes_xyxy = bboxes_xyxy
        self.concepts = concepts
        self.ann_ids = ann_ids
        self.model_path = model_path
        self.device = device
        self.conf = conf

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit(
                "SAM3 is not installed. Install ultralytics + segment-anything "
                "and place model weights under core/sam3/models/sam3-model/sam3.pt"
            )
            return
        # Group bboxes by concept so SAM3 sees exemplars of the same class
        # together. Each unique concept gets one run_sam3 call with all
        # bboxes for that concept as exemplars.
        per_concept: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(self.concepts):
            per_concept[c].append(i)

        results: List[Dict[str, Any]] = []
        for concept, idxs in per_concept.items():
            bxs = [self.bboxes_xyxy[i] for i in idxs]
            try:
                res = run_sam3(
                    image_path=self.image_path,
                    bboxes=bxs,
                    concepts=[concept],
                    model_path=self.model_path,
                    device=self.device,
                    conf=self.conf,
                )
            except Exception as e:
                for i in idxs:
                    results.append({
                        "ann_id": self.ann_ids[i],
                        "bbox_xyxy": self.bboxes_xyxy[i],
                        "mask": None,
                        "label": concept,
                        "area": 0.0,
                        "success": False,
                        "error": str(e),
                    })
                continue

            if not res.get("success"):
                for i in idxs:
                    results.append({
                        "ann_id": self.ann_ids[i],
                        "bbox_xyxy": self.bboxes_xyxy[i],
                        "mask": None,
                        "label": concept,
                        "area": 0.0,
                        "success": False,
                        "error": res.get("error", "SAM3 failed"),
                    })
                continue

            masks = res.get("masks", []) or []
            dets = res.get("detections", []) or []
            # Pair each input bbox with the closest detection's mask.
            # dets[i].bbox is xyxy. We match by IoU.
            for k, i in enumerate(idxs):
                bx = self.bboxes_xyxy[i]
                best_mask = None
                best_iou = -1.0
                for d_idx, d in enumerate(dets):
                    db = d.get("bbox", [0, 0, 0, 0])
                    iou = _iou_xyxy(bx, db)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = masks[d_idx] if d_idx < len(masks) else None
                # If no detection matched by IoU, fall back to the k-th mask.
                if best_mask is None and k < len(masks):
                    best_mask = masks[k]
                area = float(best_mask.sum()) if best_mask is not None else 0.0
                results.append({
                    "ann_id": self.ann_ids[i],
                    "bbox_xyxy": bx,
                    "mask": best_mask,
                    "label": concept,
                    "area": area,
                    "success": best_mask is not None,
                    "error": None if best_mask is not None else "no matching mask",
                })

        self.finished_signal.emit(results)


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    """IoU between two xyxy boxes. Returns 0 if either has zero area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


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
# COCO state
# ---------------------------------------------------------------------------

class CocoState:
    """In-memory COCO dataset, mirroring 08_click_review_coco.py's schema.

    Stores per-image: image_id (1-based), timestamp_ns, file_name, width,
    height. Stores per-annotation: id, image_id, category_id, bbox (xywh),
    area, iscrowd. Tracks removed ids so the seed boxes can be 'deleted'.
    """

    def __init__(self, output_json: str, categories: List[Dict[str, Any]]):
        self.output_json = output_json
        self.progress_file = output_json.replace(".json", ".progress")
        self.categories = categories
        self.cat_map = {c["id"]: c["name"] for c in categories}
        self.cat_name_to_id = {c["name"]: c["id"] for c in categories}
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.removed_ids: set = set()
        self.current_idx = 0
        self._img_id_by_ts: Dict[int, int] = {}
        self._img_id_by_idx: Dict[int, int] = {}
        self._ann_id_next = 1

    # ------------------------- persistence ---------------------------- #

    def load_existing(self) -> None:
        if os.path.exists(self.output_json):
            with open(self.output_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.images = data.get("images", [])
            self.annotations = data.get("annotations", [])
            self.categories = data.get("categories", self.categories)
            self.cat_map = {c["id"]: c["name"] for c in self.categories}
            self.cat_name_to_id = {c["name"]: c["id"] for c in self.categories}
            for ann in self.annotations:
                self._ann_id_next = max(self._ann_id_next, ann["id"] + 1)
                # Decode any persisted mask (base64-encoded PNG).
                mask_b64 = ann.get("mask")
                if isinstance(mask_b64, str) and mask_b64:
                    try:
                        import base64
                        ann["_mask"] = _decode_mask_png(
                            base64.b64decode(mask_b64)
                        ) or None
                    except Exception:
                        ann["_mask"] = None
                else:
                    ann["_mask"] = ann.get("_mask")  # may be None or ndarray
            for img in self.images:
                ts = img.get("timestamp_ns")
                if ts is not None:
                    self._img_id_by_ts[ts] = img["id"]
                self._img_id_by_idx[img.get("frame_idx", 0)] = img["id"]
            print(f"📂 Loaded existing COCO: {self.output_json} "
                  f"({len(self.images)} imgs, {len(self.annotations)} anns)")

    def load_progress(self, total_frames: int) -> int:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                idx = data.get("last_index", 0)
                if 0 <= idx < total_frames:
                    print(f"⏳ Resuming from frame {idx + 1}/{total_frames}")
                    return idx
            except Exception:
                pass
        return 0

    def save(self, is_final: bool) -> None:
        import base64
        final_anns = []
        for ann in self.annotations:
            if ann["id"] in self.removed_ids:
                continue
            out = {k: v for k, v in ann.items() if not k.startswith("_")}
            # Persist mask as base64-encoded PNG so it survives json round-trip.
            mask = ann.get("_mask")
            if mask is not None and isinstance(mask, np.ndarray) and mask.size:
                png = _encode_mask_png(mask)
                if png:
                    out["mask"] = base64.b64encode(png).decode("ascii")
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
            json.dump({"last_index": self.current_idx + 1}, f)
        print(f"✅ Saved {'final' if is_final else 'progress'} → {path} "
              f"(idx {self.current_idx + 1})")

    # ------------------------- mutation -------------------------------- #

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
            "file_name": f"{frame['frame_idx']:06d}.jpg",
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
        }
        self.annotations.append(ann)
        self._ann_id_next += 1

    def add_box(self, image_id: int, x: float, y: float,
                w: float, h: float, cat_id: int) -> int:
        ann = {
            "id": self._ann_id_next,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
        }
        self.annotations.append(ann)
        self._ann_id_next += 1
        return ann["id"]

    def remove_box(self, ann_id: int) -> None:
        self.removed_ids.add(ann_id)

    def set_mask(self, ann_id: int, mask: Optional[np.ndarray]) -> None:
        """Attach (or clear) a SAM3 mask to an annotation, in-memory only."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                if mask is None:
                    ann.pop("_mask", None)
                else:
                    ann["_mask"] = mask
                return

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
    frame_nav = pyqtSignal(int)    # delta (-1 / +1)
    save_quit = pyqtSignal()
    quit_request = pyqtSignal()
    discard_all = pyqtSignal()
    cat_pick_requested = pyqtSignal(float, float, float, float)  # pending rect
    resegment_selected = pyqtSignal()  # R key on selected box
    toggle_masks = pyqtSignal()       # M key

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
        self._selected_idx: int = -1
        self._drawing: bool = False
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_current: Optional[Tuple[float, float]] = None
        self._waiting_cat: bool = False
        self._pending_rect: Optional[Tuple[float, float, float, float]] = None

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
        self._waiting_cat = False
        self._pending_rect = None
        self.update()

    def set_masks_visible(self, visible: bool) -> None:
        self._masks_visible = visible
        self.update()

    def masks_visible(self) -> bool:
        return self._masks_visible

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

    # ----------------------- painting ---------------------------------- #

    # Per-class color palette (RGB) for mask overlays.
    _MASK_COLORS = [
        (255, 0, 128), (0, 200, 255), (120, 220, 60), (255, 160, 0),
        (160, 0, 255), (0, 255, 200), (220, 40, 40), (40, 220, 220),
        (255, 220, 40), (180, 220, 255),
    ]

    def _color_for_cat(self, cat_id: int) -> Tuple[int, int, int]:
        return self._MASK_COLORS[cat_id % len(self._MASK_COLORS)]

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
            rgba[..., 3] = (mask.astype(np.uint8) * 120)  # 47% alpha
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
            color = QColor(255, 80, 80) if i == self._selected_idx else QColor(255, 220, 30)
            pen = QPen(color, 2 if i == self._selected_idx else 1.2)
            p.setPen(pen)
            tl = self._img_to_widget(x, y)
            br = self._img_to_widget(x + w, y + h)
            p.drawRect(QtCore.QRectF(tl, br))
            label = f"{box.get('cat_name','?')} (id:{box['id']})"
            p.fillRect(
                int(tl.x()), max(0, int(tl.y() - 16)),
                8 * len(label) + 6, 16, QColor(0, 0, 0, 160)
            )
            p.setPen(color)
            p.drawText(int(tl.x() + 3), max(12, int(tl.y() - 4)), label)

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
                ix, iy = self._widget_to_img(px, py)
                self._draw_start = (ix, iy)
                self._draw_current = (ix, iy)
            else:
                # Hit-test: topmost box first.
                ix, iy = self._widget_to_img(px, py)
                hit = -1
                for i in range(len(self._boxes) - 1, -1, -1):
                    x, y, w, h = self._boxes[i]["bbox"]
                    if x <= ix <= x + w and y <= iy <= y + h:
                        hit = i
                        break
                self._selected_idx = hit
                self.update()
        elif ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.position()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drawing and self._draw_start is not None:
            ix, iy = self._widget_to_img(ev.position().x(), ev.position().y())
            self._draw_current = (ix, iy)
            self.update()
        elif self._panning and self._pan_start is not None:
            delta = ev.position() - self._pan_start
            self._offset += delta
            self._pan_start = ev.position()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton and self._drawing:
            if self._draw_start and self._draw_current:
                x0, y0 = self._draw_start
                x1, y1 = self._draw_current
                x = min(x0, x1); y = min(y0, y1)
                w = abs(x1 - x0); h = abs(y1 - y0)
                self._drawing = False
                self._draw_start = None
                self._draw_current = None
                if w > 2 and h > 2:
                    self._pending_rect = (x, y, w, h)
                    self._waiting_cat = True
                    self.update()
                    self.cat_pick_requested.emit(x, y, w, h)
                else:
                    self.update()
        elif ev.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None

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
                self.parent_window()._assign_pending_cat(int(txt))
            return

        if k in (Qt.Key_D, Qt.Key_Delete):
            if 0 <= self._selected_idx < len(self._boxes):
                ann_id = self._boxes[self._selected_idx]["id"]
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
    run_sam3_clicked = pyqtSignal()      # "Run SAM3 (all)" button
    toggle_masks_clicked = pyqtSignal()  # "Masks: on/off" button
    resegment_clicked = pyqtSignal()     # "Re-segment selected" button

    def __init__(self, coco: CocoState, parent=None):
        super().__init__(parent)
        self.coco = coco
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.cat_label = QLabel("Categories (click or press 0-9 when drawing):")
        layout.addWidget(self.cat_label)

        self.cat_list = QListWidget()
        layout.addWidget(self.cat_list, 1)
        self.cat_list.itemClicked.connect(self._on_cat_clicked)
        self._rebuild_cat_list()

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
        layout.addLayout(sam_layout)

        self.btn_masks = QPushButton("Masks: ON")
        self.btn_masks.setCheckable(True)
        self.btn_masks.setChecked(True)
        self.btn_masks.setToolTip("Toggle mask overlay visibility (M).")
        self.btn_masks.clicked.connect(self._on_masks_toggled)
        layout.addWidget(self.btn_masks)

        self.sam3_status = QLabel("SAM3: idle")
        layout.addWidget(self.sam3_status)

        self.frame_slider = QSlider(_QT_HORZ)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_moved.emit)
        layout.addWidget(self.frame_slider)

        self.info_label = QLabel("Frame: -\nTimestamp: -")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.help_label = QLabel(
            "<b>Keys (click canvas first):</b><br>"
            "D = delete sel &nbsp; A = draw &nbsp; N = next &nbsp; B = back<br>"
            "X = discard all &nbsp; S = save+quit &nbsp; Q = quit<br>"
            "0-9 = pick cat &nbsp; + / - = zoom &nbsp; 0 = fit<br>"
            "M = toggle masks &nbsp; R = re-seg selected"
        )
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

    def _on_masks_toggled(self) -> None:
        on = self.btn_masks.isChecked()
        self.btn_masks.setText("Masks: ON" if on else "Masks: OFF")
        self.toggle_masks_clicked.emit()

    def set_sam3_status(self, text: str) -> None:
        self.sam3_status.setText(text)

    def _on_cat_clicked(self, item: QListWidgetItem) -> None:
        cat_id = item.data(Qt.UserRole)
        if cat_id is not None:
            self.cat_clicked.emit(int(cat_id))

    def _rebuild_cat_list(self) -> None:
        self.cat_list.clear()
        for cat in sorted(self.coco.categories, key=lambda c: c["id"]):
            txt = f"{cat['id']} — {cat['name']}"
            it = QListWidgetItem(txt)
            it.setData(Qt.UserRole, cat["id"])
            self.cat_list.addItem(it)

    def set_slider_max(self, n: int) -> None:
        self.frame_slider.setMaximum(max(0, n - 1))

    def set_slider(self, idx: int) -> None:
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)

    def set_info(self, idx: int, total: int, ts_ns: int,
                 boxes: int) -> None:
        self.info_label.setText(
            f"Frame: {idx + 1}/{total}<br>"
            f"ts_ns: {ts_ns}<br>"
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
# Main window
# ---------------------------------------------------------------------------

class ReviewWindow(QMainWindow):

    def __init__(self, rrd_index: RrdFrameIndex, coco: CocoState,
                 grpc_uri: str, web_port: int,
                 sam3_model: Optional[str] = None,
                 sam3_device: str = "cuda",
                 sam3_conf: float = 0.25,
                 auto_segment: bool = False,
                 parent=None):
        super().__init__(parent)
        self.rrd_index = rrd_index
        self.coco = coco
        self.grpc_uri = grpc_uri
        self.web_port = web_port
        self.sam3_model = sam3_model
        self.sam3_device = sam3_device
        self.sam3_conf = sam3_conf
        self.auto_segment = auto_segment

        self.setWindowTitle("Rerun Label Review")
        self.resize(1600, 900)

        self._current_idx = coco.current_idx
        self._current_image_id: Optional[int] = None
        self._pending_cat_id: Optional[int] = None
        self._sam3_worker: Optional[SAM3Worker] = None
        # Tmp file path for the current frame's image (run_sam3 needs a path).
        self._tmp_image_path: Optional[str] = None

        # ---------- layout ----------
        splitter = QSplitter(_QT_HORZ)
        self.canvas = CanvasWidget()
        self.canvas.parent_window = self  # type: ignore[attr-defined]
        splitter.addWidget(self.canvas)

        self.side = SidePanel(coco)
        splitter.addWidget(self.side)
        splitter.setSizes([1200, 360])

        # Right column: web viewer below the canvas
        right_split = QSplitter(_QT_VERT)
        right_split.addWidget(splitter)

        self.web_view: Optional[QWebEngineView] = None
        self.bridge: Optional[TimeBridge] = None
        self.channel: Optional[QWebChannel] = None
        if _HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            right_split.addWidget(self.web_view)
            right_split.setSizes([700, 600])
            self._setup_web_bridge()
        else:
            placeholder = QLabel(
                "PyQt6-WebEngine not installed.\n"
                "Install with: pip install PyQt6-WebEngine\n"
                "Run `rerun` separately and scrub its timeline."
            )
            placeholder.setAlignment(Qt.AlignCenter)
            right_split.addWidget(placeholder)

        self.setCentralWidget(right_split)

        # ---------- signals ----------
        self.canvas.box_added.connect(self._on_box_added)
        self.canvas.box_deleted.connect(self._on_box_deleted)
        self.canvas.frame_nav.connect(self._on_frame_nav)
        self.canvas.save_quit.connect(self._on_save_quit)
        self.canvas.quit_request.connect(self._on_quit)
        self.canvas.discard_all.connect(self._on_discard_all)
        self.canvas.cat_pick_requested.connect(self._on_cat_pick_requested)
        self.canvas.toggle_masks.connect(self._on_toggle_masks)
        self.canvas.resegment_selected.connect(self._on_resegment_selected)
        self.side.cat_clicked.connect(self._on_side_cat_clicked)
        self.side.slider_moved.connect(self._on_slider_moved)
        self.side.run_sam3_clicked.connect(self._on_run_sam3_all)
        self.side.toggle_masks_clicked.connect(self._on_toggle_masks)
        self.side.resegment_clicked.connect(self._on_resegment_selected)

        if self.bridge is not None:
            self.bridge.time_changed.connect(self._on_viewer_time_changed)

        # Frame slider config
        self.side.set_slider_max(len(self.rrd_index))

        # Shortcuts that work even when canvas doesn't have focus
        for key, slot in [
            (Qt.Key_N, lambda: self._on_frame_nav(+1)),
            (Qt.Key_B, lambda: self._on_frame_nav(-1)),
            (Qt.Key_S, self._on_save_quit),
            (Qt.Key_Q, self._on_quit),
        ]:
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(slot)

        # Load first frame
        QTimer.singleShot(50, self._load_current)

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

        # Seed existing boxes from the .rrd on first visit.
        existing = frame.get("existing_boxes", [])
        already = self.coco.anns_for_image(image_id)
        if not already and existing:
            for (cx, cy, hw, hh, label) in existing:
                self.coco.seed_box(image_id, cx, cy, hw, hh, label)

        # Build box list for canvas.
        boxes = []
        for ann in self.coco.anns_for_image(image_id):
            x, y, bw, bh = ann["bbox"]
            boxes.append({
                "id": ann["id"],
                "bbox": [x, y, bw, bh],
                "cat_id": ann["category_id"],
                "cat_name": self.coco.cat_map.get(ann["category_id"], "?"),
            })
        self.canvas.set_image(arr)
        self.canvas.set_boxes(boxes)
        self.canvas.set_info(
            f"Frame {idx + 1}/{len(self.rrd_index)}  |  ts={frame['timestamp_ns']}"
        )
        self.side.set_slider(idx)
        self.side.set_info(idx, len(self.rrd_index),
                           frame["timestamp_ns"], len(boxes))
        # Update current index for save paths. We do NOT autosave on every
        # frame navigation (it would write the JSON every tick when the
        # embedded Rerun viewer autoplays the timeline). Progress is saved:
        #   - on explicit N/B/X keystrokes (see _on_frame_nav / _on_discard_all)
        #   - on quit (closeEvent, _on_save_quit, _on_quit)
        self.coco.current_idx = idx

    # ----------------------- event handlers ---------------------------- #

    def _on_frame_nav(self, delta: int) -> None:
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
        idx = self.rrd_index.find_idx_by_timestamp(ts_ns)
        if idx != self._current_idx and 0 <= idx < len(self.rrd_index):
            self._current_idx = idx
            self._load_current()

    def _on_box_deleted(self, ann_id: int) -> None:
        self.coco.remove_box(ann_id)
        # Refresh boxes on canvas.
        if self._current_image_id is not None:
            boxes = []
            for ann in self.coco.anns_for_image(self._current_image_id):
                x, y, bw, bh = ann["bbox"]
                boxes.append({
                    "id": ann["id"],
                    "bbox": [x, y, bw, bh],
                    "cat_id": ann["category_id"],
                    "cat_name": self.coco.cat_map.get(ann["category_id"], "?"),
                })
            self.canvas.set_boxes(boxes)
            self.side.set_info(self._current_idx, len(self.rrd_index),
                               self.rrd_index.frame_at(self._current_idx)["timestamp_ns"],
                               len(boxes))

    def _on_box_added(self, image_id: int, x: float, y: float,
                      w: float, h: float, cat_id: int) -> None:
        self.coco.add_box(image_id, x, y, w, h, cat_id)
        self._refresh_boxes()

    def _on_discard_all(self) -> None:
        if self._current_image_id is None:
            return
        for ann in self.coco.anns_for_image(self._current_image_id):
            self.coco.remove_box(ann["id"])
        self.canvas.set_boxes([])
        self._on_frame_nav(+1)

    def _on_save_quit(self) -> None:
        self.coco.save(is_final=True)
        QtWidgets.QApplication.quit()

    def _on_quit(self) -> None:
        self.coco.save(is_final=False)
        QtWidgets.QApplication.quit()

    # -------------- category assignment for pending rect -------------- #

    def _on_cat_pick_requested(self, x: float, y: float, w: float, h: float) -> None:
        print(f"📐 Drawn rect x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f} — pick category")
        print("  Categories:")
        for cid, name in sorted(self.coco.cat_map.items()):
            print(f"    {cid} -> {name}")

    def _assign_pending_cat(self, cat_id: int) -> None:
        if cat_id not in self.coco.cat_map:
            print(f"⚠️ Category {cat_id} not found")
            return
        rect = self.canvas.get_pending_rect()
        if rect is None or self._current_image_id is None:
            return
        x, y, w, h = rect
        new_ann_id = self.coco.add_box(self._current_image_id, x, y, w, h, cat_id)
        self.canvas.reset_state()
        self._refresh_boxes()
        print(f"✅ Added box cat={self.coco.cat_map[cat_id]} (ann_id={new_ann_id})")
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

    def _on_side_cat_clicked(self, cat_id: int) -> None:
        # If we have a pending rect, assign; else start a draw mode.
        if self.canvas.get_pending_rect() is not None:
            self._assign_pending_cat(cat_id)
        else:
            self.canvas._drawing = True
            self.canvas.update()
            print(f"✏️ Draw mode for cat_id={cat_id} "
                  f"({self.coco.cat_map.get(cat_id, '?')}) — drag on the image")
            # Remember which cat to use after the next draw.
            self._pending_cat_id = cat_id

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
            })
        self.canvas.set_boxes(boxes)
        self.side.set_info(self._current_idx, len(self.rrd_index),
                           self.rrd_index.frame_at(self._current_idx)["timestamp_ns"],
                           len(boxes))

    # ----------------------- SAM3 segmentation ------------------------- #

    def _write_tmp_image(self) -> Optional[str]:
        """Write the current frame's image blob to a tmp file (for run_sam3)."""
        if self._current_idx < 0 or self._current_idx >= len(self.rrd_index):
            return None
        frame = self.rrd_index.frame_at(self._current_idx)
        blob = frame.get("image_blob")
        if not blob:
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
        tmp_dir = Path(self.coco.output_json).parent / "_tmp_sam3_imgs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
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
        if self._sam3_worker is not None and self._sam3_worker.isRunning():
            print("⚠️ SAM3 already running, please wait…")
            return
        self.side.set_sam3_status(
            f"SAM3: running on {len(bboxes_xyxy)} box(es)…"
        )
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
        self._sam3_worker.start()

    def _on_sam3_finished(self, results: list) -> None:
        self.canvas.setEnabled(True)
        n_ok = sum(1 for r in results if r.get("success"))
        n_fail = len(results) - n_ok
        self.side.set_sam3_status(
            f"SAM3: done — {n_ok} mask(s), {n_fail} failed"
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

    def _on_sam3_failed(self, msg: str) -> None:
        self.canvas.setEnabled(True)
        self.side.set_sam3_status(f"SAM3: failed — {msg}")
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
        self.coco.save(is_final=False)
        super().closeEvent(ev)


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
                     seed_json: Optional[str],
                     db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build the category list. Priority:

    1. If --json is given, use its categories (and merge any labels found
       in the .rrd as new categories).
    2. Else if a SQLite db path is given (e.g. ``inspection_v2.db``),
       read categories from its ``categories`` table.
    3. Else scan the .rrd's Boxes2D:labels: if any label parses as a name
       (non-integer), use those as category names; otherwise fall back to
       numeric ids 0..N-1 with placeholder names "cat_0".."cat_N-1".
    """
    cats: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}

    if seed_json and os.path.exists(seed_json):
        with open(seed_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("categories", []):
            cats.append({"id": c["id"], "name": c["name"]})
            seen[c["name"]] = c["id"]

    if not cats and db_path and os.path.exists(db_path):
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute("SELECT id, name FROM categories ORDER BY id")
            for cid, cname in cur.fetchall():
                cats.append({"id": cid - 1, "name": cname})  # 0-based for COCO
                seen[cname] = cid - 1
            con.close()
            print(f"🏷️  Loaded {len(cats)} categories from {db_path}")
        except Exception as e:
            print(f"⚠️ Could not read categories from {db_path}: {e}")

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

    if not cats:
        # No labels found and no seed. Provide a single placeholder.
        cats = [{"id": 0, "name": "object"}]
    return cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rerun-based 2D bbox reviewer for an existing .rrd recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rrd", required=True, help="Path to the .rrd recording.")
    parser.add_argument("--output_json", required=True,
                        help="Output COCO JSON path (progress file is auto-saved).")
    parser.add_argument("--json", help="Seed COCO JSON for categories / existing labels.")
    parser.add_argument("--db", help="SQLite inspection DB (e.g. complete3/inspection_v2.db) "
                        "to read categories from when --json is not given.")
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
    args = parser.parse_args()

    rrd_path = os.path.abspath(args.rrd)
    if not os.path.exists(rrd_path):
        print(f"❌ .rrd not found: {rrd_path}")
        sys.exit(1)

    # ---------- 1. Index the .rrd ----------
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
    if len(rrd_index) == 0:
        print("❌ No image frames found.")
        sys.exit(1)

    # ---------- 2. Categories / COCO state ----------
    categories = _seed_categories(rrd_index, args.json, args.db)
    print(f"🏷️  Categories: {[c['name'] for c in categories]}")
    coco = CocoState(args.output_json, categories)
    coco.load_existing()
    coco.current_idx = coco.load_progress(len(rrd_index))

    # ---------- 3. Spawn the Rerun web viewer ----------
    proc = _spawn_rerun_web_viewer(rrd_path, args.web_port, args.grpc_port)

    # ---------- 4. Qt app ----------
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Rerun Label Review")

    # Clean up the spawned rerun process on exit.
    def _shutdown():
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    app.aboutToQuit.connect(_shutdown)
    signal.signal(signal.SIGINT, lambda *a: app.quit())

    win = ReviewWindow(rrd_index, coco,
                       grpc_uri=f"rerun+http://127.0.0.1:{args.grpc_port}/proxy",
                       web_port=args.web_port,
                       sam3_model=args.sam3_model,
                       sam3_device=args.sam3_device,
                       sam3_conf=args.sam3_conf,
                       auto_segment=args.auto_segment)
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
                # Write images to a temp dir from the .rrd blobs.
                tmp_img_dir = Path(args.output_yolo_dir) / "_src_images"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                for img in coco.images:
                    frame = rrd_index.frame_at(img["frame_idx"])
                    blob = frame["image_blob"]
                    if blob:
                        with open(tmp_img_dir / img["file_name"], "wb") as f:
                            f.write(blob)
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
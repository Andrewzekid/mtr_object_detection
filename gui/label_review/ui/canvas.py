"""Canvas widget: shows the image + bboxes, supports draw/select/delete."""

from ..qt_compat import Qt, QtCore, QtGui, pyqtSignal  # first: enum shims
from ..qt_compat import (  # noqa: F401
    QColor, QFont, QPainter, QPen, QPixmap, QSizePolicy, QWidget,
)

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

    def __init__(self, parent=None, side: str = "left"):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Stereo: which side this canvas shows ("left" / "right"; mono
        # sessions only ever have "left"). The window sets _image_id on
        # every frame load — box_added emits it instead of reaching into
        # parent_window._current_image_id.
        self.side: str = side
        self._image_id: Optional[int] = None

        self._pixmap: Optional[QPixmap] = None
        self._image_size: Tuple[int, int] = (0, 0)  # (w, h)
        self._boxes: List[Dict[str, Any]] = []  # see set_boxes
        self._masks_visible: bool = True
        self._mask_alpha: int = 120  # 0-255 overlay alpha for mask fill
        # Cached composed mask overlay (one QPixmap for all boxes), rebuilt
        # only when the box set or alpha changes — NOT on every repaint.
        # Building it per-repaint at full image resolution made slider
        # scrubbing laggy on 4K frames (150+ ms/paint with a few masks).
        self._boxes_rev: int = 0  # bumped by set_boxes
        self._mask_overlay_key: Optional[tuple] = None
        self._mask_overlay: Optional[QPixmap] = None
        # Whether "T<id>" track labels are drawn next to boxes (config:
        # tracking.show_ids; shown by default).
        self.show_track_ids: bool = True
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
        # False while the view is in auto-fit mode: resizeEvent then refits
        # so the whole frame stays visible (a canvas that received its
        # image before being laid out would otherwise stay stuck at
        # scale 1.0 and crop the frame). Any manual zoom/pan sets this.
        self._user_zoomed: bool = False
        # Max dimension (px) of the displayed pixmap; 0 = original
        # resolution. Downscaling is display-only: _image_size and all box
        # coordinates stay in original image pixels (config:
        # display.max_image_dim).
        self.display_max_dim: int = 0

        self._info_text = ""

    # ----------------------- public api -------------------------------- #

    def set_image(self, arr: np.ndarray) -> None:
        h, w, _ = arr.shape
        self._image_size = (w, h)
        qimg = QtGui.QImage(arr.data, w, h, 3 * w,
                            QtGui.QImage.Format_RGB888).copy()
        md = self.display_max_dim
        if md > 0 and max(w, h) > md:
            # Display-only downscale: the painter maps the pixmap onto the
            # logical (_image_size-based) target rect, so box/mask
            # coordinates are unaffected.
            qimg = qimg.scaled(md, md, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self._pixmap = QPixmap.fromImage(qimg)
        self._user_zoomed = False
        self._fit_to_view()
        self.update()

    def set_boxes(self, boxes: List[Dict[str, Any]]) -> None:
        """boxes: list of dicts with keys id, bbox=[x,y,w,h], cat_name, cat_id,
        optional mask (HxW bool array)."""
        self._boxes = list(boxes)
        self._boxes_rev += 1  # invalidates the cached mask overlay
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
        self._user_zoomed = False
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

    # Cap for the mask overlay working resolution (long side, px). Masks are
    # fuzzy tinted overlays — building them at full sensor resolution (4K+)
    # costs 100+ ms/frame for no visible difference once scaled to the
    # widget; the cached overlay is drawn scaled up by Qt instead.
    _MASK_OVERLAY_MAX_DIM = 1600

    def _build_mask_overlay(self, iw: int, ih: int) -> Optional[QPixmap]:
        """Compose all box masks into ONE RGBA overlay pixmap (cached).

        Working resolution is capped at _MASK_OVERLAY_MAX_DIM on the long
        side; mismatched mask sizes are resized to the overlay size, which
        also covers masks that don't match the image dimensions."""
        from PIL import Image as _PILImage
        s = min(1.0, self._MASK_OVERLAY_MAX_DIM / max(iw, ih))
        ow, oh = max(1, round(iw * s)), max(1, round(ih * s))
        rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
        any_mask = False
        for box in self._boxes:
            mask = box.get("mask")
            if mask is None or not isinstance(mask, np.ndarray) \
                    or mask.size == 0:
                continue
            h, w = mask.shape[:2]
            if (w, h) != (ow, oh):
                m_pil = _PILImage.fromarray(mask.astype(np.uint8) * 255,
                                            mode="L")
                m_pil = m_pil.resize((ow, oh), _PILImage.NEAREST)
                mask = np.array(m_pil) > 0
            if not mask.any():
                continue
            r, g, b = self._color_for_cat(box.get("cat_id", 0))
            # Later boxes win on overlap (same as stacked drawPixmap order).
            rgba[mask] = (r, g, b, self._mask_alpha)
            any_mask = True
        if not any_mask:
            return None
        qimg = QtGui.QImage(rgba.data, ow, oh, 4 * ow,
                            QtGui.QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg.copy())

    def _paint_masks(self, p: QPainter) -> None:
        """Paint the cached mask overlay, rebuilt only when boxes/alpha
        change — repaints from pan/zoom/resize reuse the cached pixmap."""
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0:
            return
        key = (self._boxes_rev, self._mask_alpha, iw, ih)
        if self._mask_overlay_key != key:
            self._mask_overlay = self._build_mask_overlay(iw, ih)
            self._mask_overlay_key = key
        if self._mask_overlay is None:
            return
        tl = self._img_to_widget(0, 0)
        br = self._img_to_widget(iw, ih)
        p.drawPixmap(QtCore.QRectF(tl, br), self._mask_overlay,
                     QtCore.QRectF(self._mask_overlay.rect()))

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
            ttxt = (f"T{tid} " if (tid is not None and self.show_track_ids)
                    else "")
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
            self._user_zoomed = True

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
                            self._image_id or 0,
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
        # Refit on resize while the view is in auto-fit mode — covers the
        # stereo case where a canvas got its image before its final layout
        # size (fit early-returned, leaving a cropping 1.0 scale), and
        # keeps the whole frame visible when the window/splitter resizes.
        # A manually zoomed/panned view is left alone.
        if self._pixmap is not None and not self._user_zoomed:
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
        self._user_zoomed = True
        self.update()

    # ----------------------- selection --------------------------------- #

    def select_all(self) -> None:
        """Select every box on the current frame (Ctrl+A)."""
        if not self._boxes:
            return
        self._multi_selected = set(range(len(self._boxes)))
        self._selected_idx = len(self._boxes) - 1
        self.selection_changed.emit(self._selected_idx)
        self.update()

    def clear_selection(self) -> None:
        """Drop the whole selection (Esc)."""
        self._selected_idx = -1
        self._multi_selected = set()
        self.selection_changed.emit(-1)
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
        elif k == Qt.Key_Q:
            self.quit_request.emit()
        elif k == Qt.Key_Escape:
            # Esc clears the current selection first; with nothing selected
            # it falls through to the usual quit request.
            if self._multi_selected or self._selected_idx >= 0:
                self.clear_selection()
            else:
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
            self._scale = min(40.0, self._scale * 1.2)
            self._user_zoomed = True
            self.update()
        elif k == Qt.Key_Minus:
            self._scale = max(0.05, self._scale / 1.2)
            self._user_zoomed = True
            self.update()
        elif k == Qt.Key_0:
            self._fit_to_view()
        else:
            super().keyPressEvent(ev)


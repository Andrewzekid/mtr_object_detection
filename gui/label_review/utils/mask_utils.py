"""Mask encoding / decoding helpers (PNG in-memory) + polygon conversion."""

import io
from typing import List, Optional

import numpy as np
from PIL import Image


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


def _mask_to_polygons(mask: np.ndarray,
                      min_area: float = 100.0) -> List[List[int]]:
    """Convert a boolean HxW mask to COCO polygon segmentation format —
    a list of flat [x1, y1, x2, y2, ...] outer contours in int pixels
    (same shape as the ``segmentation`` field in scripts/results.json).

    Contours whose area is below ``min_area`` (px²) are dropped — SAM3
    masks occasionally spawn scattered speck polygons far from the object.
    The largest contour is ALWAYS kept, even below ``min_area``, so a small
    object's mask is never silently dropped from the saved file. Returns []
    only when the mask is empty or no contour has ≥3 points."""
    import cv2  # lazy: heavy import, only needed when masks are saved
    if mask is None or mask.size == 0 or not mask.any():
        return []
    arr = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    best = None
    best_area = -1.0
    for c in contours:
        flat = c.reshape(-1).tolist()
        if len(flat) < 6:
            continue
        poly = [int(v) for v in flat]
        area = cv2.contourArea(c)
        if area >= min_area:
            polys.append(poly)
        if area > best_area:
            best_area, best = area, poly
    if not polys and best is not None:
        polys = [best]  # keep the main contour — never drop the whole mask
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

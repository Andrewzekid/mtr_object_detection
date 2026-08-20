"""Frame indexes — the frame source for the whole tool, backed by plain
image files (from --images or the File menu)."""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image-folder frame index — the frame source for the whole tool, backed by
# plain image files (from --images or the File menu).
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
    Interface: ``__len__``, ``frame_at``, ``decode_image``,
    ``find_idx_by_timestamp``, and a ``.frames`` list with
    timestamp_ns / log_time_ns / frame_idx / existing_boxes keys.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Mono source (see StereoIndex for the dual-folder variant).
    stereo = False

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
    stereo = False

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return 0

    def frame_at(self, idx: int) -> Dict[str, Any]:
        raise IndexError(idx)

    def decode_image(self, idx: int) -> np.ndarray:
        raise IndexError(idx)

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        return -1


# ---------------------------------------------------------------------------
# Stereo index — two image folders (left/right) paired positionally.
# ---------------------------------------------------------------------------

class StereoIndex:
    """Frame index over a stereo pair of image folders.

    Wraps two ``ImageFolderIndex`` instances; frame ``i`` is the pair
    ``(left[i], right[i])`` — pairing is positional, so the folders are
    expected to hold matching filenames in the same order (mismatches are
    tolerated, since lookup is purely positional). When the folders differ
    in length the shorter one wins and a warning is printed.

    Interface mirrors ``ImageFolderIndex`` with an added ``side`` keyword
    (``"left"`` / ``"right"``): ``__len__``, ``frame_at(idx, side)``,
    ``decode_image(idx, side)``, ``find_idx_by_timestamp`` (delegates to the
    left side), ``.files`` / ``.files_left`` / ``.files_right``,
    ``.timestamps_real``. ``side_index(side)`` returns a single-side view
    with the plain mono interface so existing workers run on one side
    unchanged.
    """

    stereo = True

    def __init__(self, left_paths: List[str], right_paths: List[str]):
        self.left = ImageFolderIndex(left_paths)
        self.right = ImageFolderIndex(right_paths)
        if len(self.left) != len(self.right):
            print(f"⚠️ Stereo folders differ in length "
                  f"(left={len(self.left)}, right={len(self.right)}) — "
                  f"using the first {min(len(self.left), len(self.right))} "
                  "pair(s)")
        self._len = min(len(self.left), len(self.right))
        self.timestamps_real = (self.left.timestamps_real
                                and self.right.timestamps_real)
        self.files_left = self.left.files[:self._len]
        self.files_right = self.right.files[:self._len]
        # `.files` (left side) keeps mono call sites (source label, YOLO
        # export) working unchanged.
        self.files = self.files_left

    def __len__(self) -> int:
        return self._len

    def _side(self, side: str) -> ImageFolderIndex:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self.left if side == "left" else self.right

    def frame_at(self, idx: int, side: str = "left") -> Dict[str, Any]:
        return self._side(side).frame_at(idx)

    def decode_image(self, idx: int, side: str = "left") -> np.ndarray:
        return self._side(side).decode_image(idx)

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        # Pairs share a timeline — the left side's lookup is authoritative.
        return self.left.find_idx_by_timestamp(ts_ns)

    def side_index(self, side: str) -> "StereoSideIndex":
        """A single-side view with the worker-facing mono interface."""
        return StereoSideIndex(self, side)


class StereoSideIndex:
    """One side of a ``StereoIndex`` with the plain mono index interface.

    Exposes exactly what the background workers need (``__len__``,
    ``frame_at`` — frames carry ``file_path``, ``decode_image``,
    ``find_idx_by_timestamp``, ``.files``) so batch jobs (interpolation,
    SAM3 ALL, autolabel ALL, propagate) can run on one side unchanged.
    """

    stereo = False  # looks mono to anything consuming it

    def __init__(self, parent: StereoIndex, side: str):
        parent._side(side)  # validates the side name
        self._parent = parent
        self.side = side
        self.timestamps_real = parent.timestamps_real
        self.files = (parent.files_left if side == "left"
                      else parent.files_right)

    def __len__(self) -> int:
        return len(self._parent)

    def frame_at(self, idx: int) -> Dict[str, Any]:
        return self._parent.frame_at(idx, self.side)

    def decode_image(self, idx: int) -> np.ndarray:
        return self._parent.decode_image(idx, self.side)

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        return self._parent.find_idx_by_timestamp(ts_ns)


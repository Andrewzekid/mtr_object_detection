"""Frame indexes — the frame source for the whole tool, backed by plain
image files (from --images or the File menu)."""

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# LRU decode cache — playback navigates one frame at a time and re-decodes
# the same image on every revisit (slider scrubbing, N/B back-and-forth).
# A small per-index LRU cache avoids the ~5-30 ms PIL open+convert per call,
# which was the dominant per-tick cost capping playback well below the
# configured speed. Sized small (8) so memory stays bounded even for 4K
# stereo folders: ~8 * 2 * (W*H*3) bytes.
# ---------------------------------------------------------------------------
_DECODE_CACHE_SIZE = 8


class _DecodeCache(OrderedDict):
    """LRU cache mapping file_path -> np.ndarray (RGB).

    Thread-safe: prefetch workers fill it from a background thread while
    the main thread reads during playback.
    """

    def __init__(self, capacity: int = _DECODE_CACHE_SIZE):
        super().__init__()
        self._cap = capacity
        self._lock = threading.Lock()

    def get_or_none(self, key):
        with self._lock:
            v = self.get(key)
            if v is not None:
                self.move_to_end(key)
            return v

    def put(self, key, value):
        with self._lock:
            self[key] = value
            self.move_to_end(key)
            while len(self) > self._cap:
                self.popitem(last=False)

    def contains(self, key) -> bool:
        with self._lock:
            return key in self


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
        # Per-index LRU decode cache (keyed by file_path). Avoids re-reading
        # + re-decoding the same JPEG/PNG on every playback tick / slider
        # scrub — the dominant per-tick cost that capped playback speed.
        self._decode_cache: _DecodeCache = _DecodeCache()
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
        fp = self.frames[idx]["file_path"]
        cached = self._decode_cache.get_or_none(fp)
        if cached is not None:
            return cached
        with Image.open(fp) as im:
            arr = np.array(im.convert("RGB"))
        self._decode_cache.put(fp, arr)
        return arr

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
# Stereo index — two image folders (left/right) paired by timestamp.
# ---------------------------------------------------------------------------

class StereoIndex:
    """Frame index over a stereo pair of image folders.

    Wraps two ``ImageFolderIndex`` instances. When both folders have
    timestamp filenames (every stem a bare integer), frame ``i`` is the
    pair ``(left[j], right[k])`` where both sides share the SAME timestamp
    — images without a matching timestamp on the other side are SKIPPED,
    so dropped frames on one camera never shift the pairing. When the
    filenames are not timestamps, pairing falls back to positional
    (``(left[i], right[i])``, shorter side wins, mismatched basenames
    reported via ``pairing_warning``).

    Interface mirrors ``ImageFolderIndex`` with an added ``side`` keyword
    (``"left"`` / ``"right"``): ``__len__``, ``frame_at(idx, side)``,
    ``decode_image(idx, side)``, ``find_idx_by_timestamp`` (the paired
    timeline, keyed on the left side's timestamps), ``.files`` /
    ``.files_left`` / ``.files_right``, ``.timestamps_real``.
    ``side_index(side)`` returns a single-side view with the plain mono
    interface so existing workers run on one side unchanged.
    """

    stereo = True

    def __init__(self, left_paths: List[str], right_paths: List[str]):
        self.left = ImageFolderIndex(left_paths)
        self.right = ImageFolderIndex(right_paths)
        self.timestamps_real = (self.left.timestamps_real
                                and self.right.timestamps_real)
        self.pairing_warning: Optional[str] = None
        if self.timestamps_real:
            self._pair_by_timestamp()
        else:
            self._pair_positionally()
        # `.files` (left side) keeps mono call sites (source label, YOLO
        # export) working unchanged.
        self.files = self.files_left

    def _pair_by_timestamp(self) -> None:
        """Pair frames whose timestamps match exactly; skip the rest."""
        def ts_map(side: ImageFolderIndex) -> Dict[int, int]:
            # First occurrence wins on duplicate timestamps (the extras
            # count as unpaired below).
            m: Dict[int, int] = {}
            for i, fr in enumerate(side.frames):
                m.setdefault(fr["timestamp_ns"], i)
            return m
        lmap, rmap = ts_map(self.left), ts_map(self.right)
        common = sorted(set(lmap) & set(rmap))
        self._pair_ts = common
        self._pair_map = {"left": [lmap[ts] for ts in common],
                          "right": [rmap[ts] for ts in common]}
        self._len = len(common)
        n_skip_l = len(self.left) - self._len
        n_skip_r = len(self.right) - self._len
        if n_skip_l or n_skip_r:
            print(f"⚠️ Stereo timestamp pairing: skipping "
                  f"{n_skip_l} left-only / {n_skip_r} right-only "
                  f"image(s) with no matching timestamp")
            self.pairing_warning = (
                f"Stereo: {n_skip_l} left-only / {n_skip_r} right-only "
                f"image(s) skipped — no matching timestamp on the other "
                f"side.")
        self.files_left = [self.left.files[i] for i in self._pair_map["left"]]
        self.files_right = [self.right.files[i]
                            for i in self._pair_map["right"]]

    def _pair_positionally(self) -> None:
        """Fallback for non-timestamp filenames: pair by sorted position."""
        if len(self.left) != len(self.right):
            print(f"⚠️ Stereo folders differ in length "
                  f"(left={len(self.left)}, right={len(self.right)}) — "
                  f"using the first {min(len(self.left), len(self.right))} "
                  "pair(s)")
        self._len = min(len(self.left), len(self.right))
        self._pair_ts = [self.left.frames[i]["timestamp_ns"]
                         for i in range(self._len)]
        self._pair_map = {"left": list(range(self._len)),
                          "right": list(range(self._len))}
        self.files_left = self.left.files[:self._len]
        self.files_right = self.right.files[:self._len]
        # Filename-equality guard: pairing is positional, but if the
        # basenames don't match the user has almost certainly loaded
        # mismatched folders and all downstream labels will be shifted.
        if self._len > 0:
            mismatches = [
                (i, os.path.basename(l), os.path.basename(r))
                for i, (l, r) in enumerate(zip(self.files_left,
                                               self.files_right))
                if os.path.basename(l) != os.path.basename(r)]
            if mismatches:
                shown = mismatches[:3]
                lines = ", ".join(
                    f"#{i}: {ln!r} vs {rn!r}" for i, ln, rn in shown)
                if len(mismatches) > 3:
                    lines += f" (+ {len(mismatches) - 3} more)"
                self.pairing_warning = (
                    f"Stereo filename mismatch: {lines}; pairing is "
                    f"positional — verify the folders are aligned!")

    def __len__(self) -> int:
        return self._len

    @property
    def paired_timestamps(self) -> List[int]:
        """The paired timeline: timestamps present on both sides, sorted
        earliest-first (synthetic 1 ms steps in positional fallback mode)."""
        return list(self._pair_ts)

    def _side(self, side: str) -> ImageFolderIndex:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self.left if side == "left" else self.right

    def frame_at(self, idx: int, side: str = "left") -> Dict[str, Any]:
        # Report the PAIR index as frame_idx (not the side folder's own
        # position — those diverge once unpaired frames are skipped), so
        # COCO image records, discard marks and (frame_idx, side) lookups
        # all speak the paired-timeline indexing the UI uses.
        frame = dict(self._side(side).frame_at(self._pair_map[side][idx]))
        frame["frame_idx"] = idx
        return frame

    def decode_image(self, idx: int, side: str = "left") -> np.ndarray:
        return self._side(side).decode_image(self._pair_map[side][idx])

    def find_idx_by_timestamp(self, ts_ns: int) -> int:
        # Both sides share the paired timeline; binary search the paired
        # timestamps and snap to the nearest pair.
        if not self._pair_ts:
            return -1
        lo, hi = 0, len(self._pair_ts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self._pair_ts[mid]
            if v == ts_ns:
                return mid
            elif v < ts_ns:
                lo = mid + 1
            else:
                hi = mid - 1
        if lo >= len(self._pair_ts):
            return len(self._pair_ts) - 1
        if hi < 0:
            return 0
        return lo if abs(self._pair_ts[lo] - ts_ns) < \
                    abs(self._pair_ts[hi] - ts_ns) else hi

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


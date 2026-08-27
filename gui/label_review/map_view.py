"""Pose database support for the Rerun view.

:class:`PoseDb` - Clio inspection DB lookup: per-image ``cam_tf`` /
lidar ``tf`` poses, matched to frames by ``filename`` column, numeric
filename stem as the ``id`` column, or nearest ``timestamp_ns`` (see
``PoseDb.MATCH_MODES``), used to place annotated frames on the map of an
opened .rrd recording (see rerun_logger.py).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

import numpy as np



# --------------------------------------------------------------------------- #
# Pose database
# --------------------------------------------------------------------------- #

class PoseDb:
    """Per-timestamp camera/lidar poses from a Clio inspection DB.

    Expects an ``images`` table with ``timestamp_ns``, ``is_left`` and
    either ``cam_tf_translation_{x,y,z}`` + ``cam_tf_rotation_{x,y,z,w}``
    (camera pose in the map frame) or the equivalent lidar ``tf_*`` columns
    (used as a fallback when the cam_tf row is empty).
    """

    _QUERY = (
        "SELECT timestamp_ns, "
        "cam_tf_translation_x, cam_tf_translation_y, cam_tf_translation_z, "
        "tf_translation_x, tf_translation_y, tf_translation_z{extra} "
        "FROM images"
    )

    # How a frame is matched to a DB row in pose_for:
    #   "auto"        — filename column, then numeric filename stem as the
    #                   `id` column, then nearest timestamp (guarded)
    #   "filename"    — exact match of the image file name against the
    #                   DB's `filename` column only
    #   "filename_id" — numeric filename stem (1042.jpg -> 1042) matched
    #                   against the DB's `id` column only
    #   "timestamp"   — nearest timestamp_ns only (guarded)
    MATCH_MODES = ("auto", "filename", "filename_id", "timestamp")

    # Timestamp fallback is only trusted within this window: image folders
    # with sequential names (1000.jpg, ...) get misparsed as nanosecond
    # timestamps, and snapping those to the "nearest" DB pose would pin
    # every frame to the first pose instead of failing loudly.
    MAX_TIMESTAMP_DT_NS = 10_000_000_000  # 10 s

    def __init__(self, db_path: str, match_mode: str = "auto"):
        if match_mode not in self.MATCH_MODES:
            raise ValueError(f"match_mode must be one of {self.MATCH_MODES}, "
                             f"got {match_mode!r}")
        self.match_mode = match_mode
        self.path = str(db_path)
        con = sqlite3.connect(self.path)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
            extra = [c for c in ("id", "filename") if c in cols]
            rows = con.execute(self._QUERY.format(
                extra=", " + ", ".join(extra) if extra else "")).fetchall()
        finally:
            con.close()
        if not rows:
            raise ValueError(f"{self.path}: 'images' table is empty - "
                             "no poses to place frames on the map")
        self._ts = np.array([r[0] for r in rows], dtype=np.int64)
        self._pos = np.array([[r[1], r[2], r[3]] for r in rows],
                             dtype=np.float64)
        self._pos_lidar = np.array([[r[4], r[5], r[6]] for r in rows],
                                    dtype=np.float64)
        self._valid = ~np.isnan(self._pos).any(axis=1)
        order = np.argsort(self._ts)
        self._ts = self._ts[order]
        self._pos = self._pos[order]
        self._pos_lidar = self._pos_lidar[order]
        self._valid = self._valid[order]
        # Exact-key -> sorted row index maps. DBs from the inspection
        # pipeline carry the image file name ('1042.jpg') and/or a numeric
        # `id` that exported image folders use as their file stem.
        self._by_filename = {}
        self._by_id = {}
        if extra:
            id_col = extra.index("id") if "id" in extra else None
            name_col = extra.index("filename") if "filename" in extra else None
            for old_i, r in enumerate(rows):
                new_i = int(np.flatnonzero(order == old_i)[0])
                if name_col is not None:
                    name = r[7 + name_col]
                    key = os.path.basename(str(name)) if name else None
                    if key and key not in self._by_filename:
                        self._by_filename[key] = new_i
                if id_col is not None:
                    row_id = r[7 + id_col]
                    if row_id is not None and int(row_id) not in self._by_id:
                        self._by_id[int(row_id)] = new_i
        if not self._valid.any():
            raise ValueError(f"{self.path}: no cam_tf poses found")

    def _pose_at_row(self, i: int) -> Optional[np.ndarray]:
        if self._valid[i]:
            return self._pos[i]
        lidar = self._pos_lidar[i]
        if not np.isnan(lidar).any():
            return lidar
        return None

    def _pose_by_timestamp(self, timestamp_ns: Optional[int]
                           ) -> Optional[np.ndarray]:
        """Nearest pose for a timestamp; None when further than
        MAX_TIMESTAMP_DT_NS (likely a misparsed sequential filename)."""
        if timestamp_ns is None:
            return None
        i = int(np.searchsorted(self._ts, int(timestamp_ns)))
        best: Optional[int] = None
        for j in (i - 1, i):
            if 0 <= j < len(self._ts) and (best is None or
                    abs(self._ts[j] - timestamp_ns) <
                    abs(self._ts[best] - timestamp_ns)):
                best = j
        if best is None:
            return None
        if abs(int(self._ts[best]) - int(timestamp_ns)) > \
                self.MAX_TIMESTAMP_DT_NS:
            return None
        return self._pose_at_row(best)

    def pose_for(self, file_name: Optional[str],
                 timestamp_ns: Optional[int]) -> Optional[np.ndarray]:
        """Pose for a frame, matched per ``match_mode`` (see MATCH_MODES)."""
        mode = self.match_mode
        if file_name and mode in ("auto", "filename"):
            i = self._by_filename.get(os.path.basename(str(file_name)))
            if i is not None:
                return self._pose_at_row(i)
            if mode == "filename":
                return None
        if file_name and mode in ("auto", "filename_id"):
            stem = os.path.splitext(os.path.basename(str(file_name)))[0]
            if stem.isdigit():
                i = self._by_id.get(int(stem))
                if i is not None:
                    return self._pose_at_row(i)
            if mode == "filename_id":
                return None
        if mode in ("auto", "timestamp"):
            return self._pose_by_timestamp(timestamp_ns)
        return None

    def pose_at(self, timestamp_ns: Optional[int]) -> Optional[np.ndarray]:
        """Nearest pose (3-vector) for a timestamp; None when out of range."""
        return self._pose_by_timestamp(timestamp_ns)

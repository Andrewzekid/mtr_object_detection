"""Point-cloud map + pose database support for the Rerun view.

* :func:`load_pcd` - minimal PCD reader (ascii + binary, ``x y z rgb``
  fields, packed float or uint rgb) returning positions + RGB colors.
* :class:`PoseDb` - Clio inspection DB lookup: per-image ``cam_tf`` /
  lidar ``tf`` poses keyed by ``timestamp_ns`` (with an ``is_left`` side
  column), used to place annotated frames on the colored map.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# PCD loading
# --------------------------------------------------------------------------- #

def load_pcd(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a .pcd into (positions Nx3 float32, colors Nx3 uint8).

    Supports the layouts our Clio exports use: FIELDS ``x y z rgb`` with
    DATA ascii or binary, rgb either packed into a float32 (bit-reinterpreted
    as uint32 RGB) or a plain uint32. Colors default to gray when absent.
    """
    p = Path(path)
    with p.open("rb") as f:
        header: dict[str, str] = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PCD header")
            text = line.decode("ascii", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break
        data_fmt = header.get("DATA", "").lower()
        if data_fmt not in ("ascii", "binary"):
            raise ValueError(f"{path}: unsupported PCD DATA {data_fmt!r} "
                             "(only ascii / binary)")

        fields = header.get("FIELDS", "").split()
        sizes = [int(v) for v in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts = [int(v) for v in header.get("COUNT", "").split()]
        n_points = int(header.get("POINTS", header.get("WIDTH", "0")))

        type_codes = {"F": "f", "U": "u", "I": "i"}
        dtype_list = []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            code = type_codes.get(typ, "u")
            dtype_list.append((name, f"<{code}{size}") if cnt == 1
                              else (name, f"<{code}{size}", (cnt,)))
        dtype = np.dtype(dtype_list)

        if data_fmt == "binary":
            raw = f.read(dtype.itemsize * n_points)
            arr = np.frombuffer(raw, dtype=dtype, count=n_points)
        else:
            arr = np.loadtxt(f, dtype=dtype, max_rows=n_points)

    if "x" not in arr.dtype.names:
        raise ValueError(f"{path}: PCD has no x/y/z fields "
                         f"({arr.dtype.names})")
    positions = np.stack([arr["x"], arr["y"], arr["z"]], axis=1) \
        .astype(np.float32)

    colors = np.full((n_points, 3), 128, dtype=np.uint8)
    for field in ("rgb", "rgba"):
        if field in (arr.dtype.names or ()):
            packed = arr[field]
            # Float-packed rgb (PCD convention): reinterpreting the float
            # bits as uint32 gives the packed color.
            if arr.dtype[field].kind == "f":
                packed = packed.view(np.uint32) \
                    if packed.ndim == 1 else None
            packed = np.asarray(packed, dtype=np.uint32)
            has_a = field == "rgba"
            r = (packed >> 16) & 0xFF if has_a else (packed >> 16) & 0xFF
            g = (packed >> 8) & 0xFF
            b = packed & 0xFF
            colors = np.stack([r, g, b], axis=1).astype(np.uint8)
            break

    return positions, colors


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
        "tf_translation_x, tf_translation_y, tf_translation_z "
        "FROM images"
    )

    def __init__(self, db_path: str):
        self.path = str(db_path)
        con = sqlite3.connect(self.path)
        try:
            rows = con.execute(self._QUERY).fetchall()
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
        if not self._valid.any():
            raise ValueError(f"{self.path}: no cam_tf poses found")

    def pose_at(self, timestamp_ns: Optional[int]) -> Optional[np.ndarray]:
        """Nearest pose (3-vector) for a timestamp; None when out of range."""
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
        if self._valid[best]:
            return self._pos[best]
        lidar = self._pos_lidar[best]
        if not np.isnan(lidar).any():
            return lidar
        return None

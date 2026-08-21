"""Rerun (.rrd) logging for the label review GUI.

When launched with ``--rrd <path>``, the app records what you label into a
Rerun recording so you can inspect it afterwards in the Rerun viewer
(``rerun <path>``). Every "Mark as annotated" (and every frame that already
carries annotations when it is marked) logs:

* the frame image,
* the 2D boxes (one color per category),
* one keypoint per box (its center), labeled with the category name - so
  the keypoint "map" shows at a glance what was labelled on each frame.

Entity layout (per camera side):

    world/<side>/image        - EncodedImage
    world/<side>/boxes        - Boxes2D (colored per category)
    world/<side>/keypoints    - Points2D (box centers, labeled)

The timeline is driven by the frame's ``timestamp_ns`` (falling back to the
frame index), so scrubbing the Rerun timeline walks the same frames as the
GUI slider.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import rerun as rr
    _HAS_RERUN = True
except Exception:  # pragma: no cover - rerun is optional
    rr = None  # type: ignore[assignment]
    _HAS_RERUN = False


# Distinct-ish colors, one per category id (cycled).
_CATEGORY_COLORS = [
    (255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 200, 60),
    (200, 120, 255), (255, 140, 200), (140, 255, 230), (255, 255, 120),
]


class RerunLogger:
    """Logs frames + annotation keypoints to a .rrd recording.

    A no-op (``enabled`` False) when rerun-sdk is missing or no ``--rrd``
    path was given, so call sites never need to guard.
    """

    def __init__(self, rrd_path: Optional[str] = None):
        self.enabled = False
        self.path = rrd_path
        self._stream = None
        self._logged_sides = set()
        if not rrd_path:
            return
        if not _HAS_RERUN:
            print("WARNING: --rrd given but rerun-sdk is not installed; "
                  "no Rerun recording will be written.")
            return
        try:
            rr.init("label_review", recording_id="label_review")
            self._stream = rr.get_global_data_recording()
            self._stream.save(rrd_path)
            self.enabled = True
        except Exception as exc:
            print(f"WARNING: could not open Rerun recording {rrd_path}: {exc}")
            self._stream = None

    # ------------------------------------------------------------------ #

    def log_frame(self, frame: Dict[str, Any],
                  image: Optional[np.ndarray],
                  anns: List[Dict[str, Any]],
                  cat_map: Dict[int, str],
                  side: str = "left") -> None:
        """Log one frame's image, boxes and box-center keypoints.

        ``frame`` is an ImageFolderIndex frame dict (needs ``timestamp_ns``
        and ``frame_idx``); ``anns`` are COCO annotations with xywh
        ``bbox``; ``cat_map`` maps category_id -> name.
        """
        if not self.enabled or self._stream is None:
            return

        ts = frame.get("timestamp_ns")
        if ts is not None:
            self._stream.set_time("timestamp",
                                  timestamp=np.datetime64(int(ts), "ns"))
        self._stream.set_time("frame_idx", sequence=int(
            frame.get("frame_idx", 0)))

        base = f"world/{side}"
        if image is not None and image.size:
            self._stream.log(f"{base}/image", rr.Image(image))

        if not anns:
            # Clear stale boxes/keypoints from any previous log at this
            # time point so the recording reflects the current state.
            self._stream.log(f"{base}/boxes", rr.Clear(recursive=False))
            self._stream.log(f"{base}/keypoints", rr.Clear(recursive=False))
            return

        names: List[str] = []
        colors: List[Any] = []
        rects: List[List[float]] = []
        centers: List[List[float]] = []
        for ann in anns:
            x, y, w, h = (float(v) for v in ann["bbox"])
            name = cat_map.get(ann["category_id"], str(ann["category_id"]))
            color = _CATEGORY_COLORS[int(ann["category_id"])
                                     % len(_CATEGORY_COLORS)]
            names.append(name)
            colors.append(color)
            rects.append([x, y, x + w, y + h])  # rerun boxes are xyxy
            centers.append([x + w / 2.0, y + h / 2.0])

        self._stream.log(
            f"{base}/boxes",
            rr.Boxes2D(array=rects, array_format=rr.Box2DFormat.XYXY,
                       colors=colors, labels=names),
        )
        self._stream.log(
            f"{base}/keypoints",
            rr.Points2D(centers, colors=colors, labels=names),
        )

    def flush(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

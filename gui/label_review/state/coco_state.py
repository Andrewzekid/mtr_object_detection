"""COCO state — in-memory COCO dataset, mirroring 08_click_review_coco.py's
schema, with undo-stack integration."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.mask_utils import (
    _decode_mask_png, _mask_to_polygons, _polygons_to_mask,
)
from .undo_stack import UndoStack


def _imported_mask(ann: Dict[str, Any], h: int, w: int):
    """Extract the in-memory mask from a loaded/imported annotation dict:
    polygon ``segmentation`` (current format) or legacy base64 ``mask``.
    Returns None when absent or undecodable (never raises)."""
    try:
        if ann.get("segmentation"):
            return _polygons_to_mask(ann["segmentation"], h, w)
        mask_b64 = ann.get("mask")
        if isinstance(mask_b64, str) and mask_b64:
            import base64
            return _decode_mask_png(base64.b64decode(mask_b64))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# COCO state
# ---------------------------------------------------------------------------

class _SideKeyedDict(dict):
    """Dict keyed by ``(key, side)`` where a bare (non-tuple) key means the
    ``"left"`` side.

    Stereo sessions key image-id lookups by ``(timestamp_ns, side)`` /
    ``(frame_idx, side)``; this shim keeps mono-era call sites (and tests)
    that pass a bare key working unchanged — a missing side always defaults
    to left, matching how old save files (no ``side`` field) are treated.
    """

    @staticmethod
    def _norm(k):
        return k if isinstance(k, tuple) else (k, "left")

    def __getitem__(self, k):
        return super().__getitem__(self._norm(k))

    def __setitem__(self, k, v):
        super().__setitem__(self._norm(k), v)

    def __contains__(self, k):
        return super().__contains__(self._norm(k))

    def get(self, k, default=None):
        return super().get(self._norm(k), default)


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
        # Frames the user explicitly marked as annotated ("Mark as
        # annotated" button) even though they have no boxes. Unioned into
        # the output JSON's `annotated_image_ids` on save (resolved from
        # frame_idx to image_id(s)); persisted in the .progress sidecar
        # like `reviewed`.
        self.annotated_marks: set = set()
        # Frames the user explicitly discarded ("Discard image" button) —
        # excluded from the FINAL json (images + their annotations +
        # annotated marks) but kept in the session and in the _tmp progress
        # file, so the toggle stays reversible until the final export.
        # Persisted in the .progress sidecar like `reviewed`.
        self.discarded_frames: set = set()
        # Track id assignment for newly drawn boxes. Default (False): every
        # new box gets a fresh id from the global auto-increment counter.
        # When True (config "tracking": {"sticky_ids": true}), the k-th box
        # drawn on a frame inherits the k-th track id from the nearest
        # earlier annotated frame (see _next_track_id) — what interpolation
        # pairing expects. Deleted ids are never recycled either way.
        self.sticky_track_ids: bool = False
        self._track_next: int = 1
        # Minimum contour area (px²) for a SAM3 mask contour to be saved as
        # a COCO polygon — drops the scattered speck polygons SAM3
        # occasionally produces. Config: "sam3": {"min_polygon_area": ...}.
        self.min_polygon_area: float = 100.0
        # Synced-timestamp filter for stereo sessions: the sorted set of
        # timestamps present in BOTH folders (set via
        # set_synced_timestamps). When set, save() writes only image
        # records whose timestamp is in this set — unsynced images
        # (e.g. stale records loaded from an older save file) are dropped
        # from the JSON. None disables filtering (mono sessions).
        self._synced_ts: Optional[set] = None
        # Image-id lookup tables, keyed by (timestamp_ns, side) and
        # (frame_idx, side) — stereo sessions have one image record per
        # side. _SideKeyedDict maps bare keys to the left side.
        self._img_id_by_ts: _SideKeyedDict = _SideKeyedDict()
        self._img_id_by_idx: _SideKeyedDict = _SideKeyedDict()
        self._ann_id_next = 1
        # Dirty flag — True when there are unsaved mutations since the last
        # successful save(). Cleared by save(). UI shows a "●" indicator.
        self.dirty: bool = False
        self.undo_stack: UndoStack = undo_stack or UndoStack()
        # --- Hot-path query caches (invalidated by _invalidate_ann_caches) ---
        # anns_for_image() and labeled_frame_idxs() are called on every
        # playback tick; without these they were O(total_annotations) /
        # O(annotations*images) per tick, capping playback well below the
        # configured speed. A dirty flag + lazy rebuild keeps them O(1) /
        # O(images) on the hot path.
        self._anns_cache_dirty: bool = True
        self._anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
        self._labeled_frame_idxs_cache: Optional[List[int]] = None
        self._labeled_frame_idxs_side_cache: Dict[str, List[int]] = {}
        # Image-record-by-id index so ensure_image's size-update path is
        # O(1) instead of scanning self.images linearly every tick.
        self._img_by_id: Dict[int, Dict[str, Any]] = {}

    # ------------------------- persistence ---------------------------- #

    def set_synced_timestamps(self, timestamps) -> None:
        """Restrict saved images to these timestamps (stereo sessions: the
        sorted timestamps present in both folders). Pass None to disable."""
        self._synced_ts = set(timestamps) if timestamps is not None else None

    def register_all_frames(self, frame_index) -> int:
        """Create an image record for every frame of the index (both sides
        in stereo), so the saved JSON lists every synced image — not just
        the frames actually visited. New records get width/height 0 until
        the frame is first decoded (``ensure_image`` fills real dims in on
        visit and never clobbers them with 0); downstream tools infer dims
        of unvisited images from the files. Returns records created."""
        stereo = getattr(frame_index, "stereo", False)
        sides = ("left", "right") if stereo else ("left",)
        created = 0
        for idx in range(len(frame_index)):
            for side in sides:
                frame = (frame_index.frame_at(idx, side) if stereo
                         else frame_index.frame_at(idx))
                before = len(self.images)
                self.ensure_image(frame, 0, 0, side=side)
                created += len(self.images) - before
        return created

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
                # (current format) or legacy base64 PNG ("mask"). A corrupt
                # entry must not kill the whole load — drop just the mask.
                h, w = img_dims.get(ann["image_id"], (0, 0))
                ann["_mask"] = _imported_mask(ann, h, w)
            for img in self.images:
                # Images saved before the stereo feature have no "side"
                # field — they are treated as left.
                side = img.get("side", "left")
                ts = img.get("timestamp_ns")
                if ts is not None:
                    self._img_id_by_ts[(ts, side)] = img["id"]
                self._img_id_by_idx[(img.get("frame_idx", 0), side)] = \
                    img["id"]
                self._img_by_id[img["id"]] = img
            print(f"📂 Loaded existing COCO: {path} "
                  f"({len(self.images)} imgs, {len(self.annotations)} anns)")
            self._invalidate_ann_caches()

    def load_progress(self, total_frames: int) -> int:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                idx = data.get("last_index", 0)
                self.reviewed = set(data.get("reviewed", []))
                self.keyframes = set(data.get("keyframes", []))
                self.annotated_marks = set(data.get("annotated_marks", []))
                self.discarded_frames = set(data.get("discarded", []))
                if 0 <= idx < total_frames:
                    print(f"⏳ Resuming from frame {idx + 1}/{total_frames}")
                    return idx
            except Exception:
                pass
        return 0

    def import_coco(self, data: Dict[str, Any],
                    file_to_frame: Dict[str, int],
                    frame_index) -> Tuple[int, int, int, int]:
        """Import annotations from another COCO dict (File → Load
        annotations…), matching images to the current source by file
        basename (and ``side``, in stereo sessions — imported images
        without a ``side`` field go to the left). Categories are merged by
        name; images that aren't in the
        current source are skipped silently. Masks are restored from polygon
        ``segmentation`` (or the legacy base64 ``mask``). Annotations that
        duplicate an existing live box (same image, category and bbox) are
        skipped so re-importing the same file is harmless — but when the
        existing box has no mask and the duplicate carries one, the mask is
        merged onto the existing box (this is how masks from an external
        SAM3 run reach boxes that were already drawn). Bulk action — no
        undo entry. Returns (frames_matched, anns_imported, anns_skipped,
        masks_merged)."""
        src_cat_to_dst = {}
        for cat in data.get("categories", []):
            src_cat_to_dst[int(cat["id"])] = self._resolve_cat_id(cat["name"])
        img_by_src_id = {int(i["id"]): i for i in data.get("images", [])}

        # Existing live boxes per image for duplicate detection + mask
        # merge (image_id -> {(cat_id, bbox-key): ann}).
        existing: Dict[int, Dict[tuple, Dict[str, Any]]] = {}
        for a in self.annotations:
            if a["id"] in self.removed_ids:
                continue
            key = (a["category_id"],
                   tuple(round(float(v), 1) for v in a["bbox"]))
            existing.setdefault(a["image_id"], {})[key] = a

        frames_matched: set = set()
        n_imported = n_skipped = n_merged = 0
        for ann in data.get("annotations", []):
            src_img = img_by_src_id.get(int(ann["image_id"]))
            cat_id = src_cat_to_dst.get(int(ann["category_id"]))
            if src_img is None or cat_id is None:
                n_skipped += 1  # dangling image/category reference
                continue
            # Match by (basename, side); annotations without a "side" field
            # go to the left side. Plain basename keys (mono sessions) are
            # accepted for backward compatibility.
            side = src_img.get("side", "left")
            basename = os.path.basename(src_img.get("file_name", ""))
            frame_idx = file_to_frame.get((basename, side))
            if frame_idx is None:
                frame_idx = file_to_frame.get(basename)
            if frame_idx is None:
                continue  # image not in the current source — not an error
            w = int(src_img.get("width", 0))
            h = int(src_img.get("height", 0))
            stereo = getattr(frame_index, "stereo", False)
            # Mask-bearing anns need real image dims to rasterize; when the
            # source record lacks them, decode the actual frame once.
            if (w <= 0 or h <= 0) and \
                    (ann.get("segmentation") or ann.get("mask")):
                try:
                    arr = (frame_index.decode_image(frame_idx, side)
                           if stereo else frame_index.decode_image(frame_idx))
                    h, w = arr.shape[:2]
                except Exception:
                    pass
            frame = (frame_index.frame_at(frame_idx, side)
                     if stereo else frame_index.frame_at(frame_idx))
            image_id = self.ensure_image(frame, w, h, side=side)
            bbox = [float(v) for v in ann["bbox"]]
            key = (cat_id, tuple(round(v, 1) for v in bbox))
            dup = existing.get(image_id, {}).get(key)
            if dup is not None:
                # Exact duplicate already in the session — don't re-add the
                # box, but DO merge the mask when the live box has none
                # (otherwise re-loading a SAM3-annotated file over existing
                # boxes silently drops every mask).
                if dup.get("_mask") is None:
                    mask = _imported_mask(ann, h, w)
                    if mask is not None:
                        dup["_mask"] = mask
                        n_merged += 1
                n_skipped += 1
                continue
            frames_matched.add(frame_idx)
            new_ann = {
                "id": self._ann_id_next,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": float(ann.get("area", bbox[2] * bbox[3])),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            new_ann["_mask"] = _imported_mask(ann, h, w)
            # Keep the source track id when it parses as an int; otherwise
            # assign a fresh one so tracking stays consistent.
            try:
                tid = int(ann["track_id"])
                new_ann["track_id"] = tid
                self._track_next = max(self._track_next, tid + 1)
            except (KeyError, TypeError, ValueError):
                new_ann["track_id"] = self._fresh_track_id()
            self.annotations.append(new_ann)
            existing.setdefault(image_id, {})[key] = new_ann
            self._ann_id_next += 1
            n_imported += 1
        if n_imported or n_merged:
            self.dirty = True
            self._invalidate_ann_caches()
        return len(frames_matched), n_imported, n_skipped, n_merged

    def save(self, is_final: bool) -> None:
        tmp_path = self.output_json.replace(".json", "_tmp.json")
        if not is_final and not self.dirty and os.path.exists(tmp_path):
            # Nothing changed since the last full write (pure navigation,
            # review marks, slider drags) — only refresh the tiny progress
            # file. The full dump re-polygonizes every mask and rewrites
            # the whole JSON, which made plain frame navigation laggy and
            # spammed "Saved progress" on every step.
            self._write_progress()
            return
        # Discarded frames are excluded from the FINAL export only: their
        # images, those images' annotations, and their annotated marks.
        # The _tmp file keeps everything so "Discard image" stays
        # reversible until the final save.
        drop_img_ids: set = set()
        if is_final and self.discarded_frames:
            drop_img_ids = {img["id"] for img in self.images
                            if img.get("frame_idx") in self.discarded_frames}
        # Unsynced image records (stereo sessions: timestamps present in
        # only one folder) are excluded from EVERY save — they can arrive
        # via stale records in a previously loaded JSON and must never be
        # written back out.
        if self._synced_ts is not None:
            unsynced = {img["id"] for img in self.images
                        if img.get("timestamp_ns") not in self._synced_ts}
            if unsynced:
                print(f"🧹 Excluding {len(unsynced)} unsynced image "
                      f"record(s) — no matching timestamp on the other side")
            drop_img_ids |= unsynced
        final_anns = []
        for ann in self.annotations:
            if ann["id"] in self.removed_ids:
                continue
            if ann["image_id"] in drop_img_ids:
                continue
            out = {k: v for k, v in ann.items() if not k.startswith("_")}
            # Legacy base64-PNG "mask" field is superseded by COCO polygon
            # "segmentation" (same shape as scripts/results.json).
            out.pop("mask", None)
            mask = ann.get("_mask")
            if mask is not None and isinstance(mask, np.ndarray) and mask.size:
                polys = _mask_to_polygons(mask, self.min_polygon_area)
                if polys:
                    out["segmentation"] = polys
            final_anns.append(out)
        # Convenience index: image ids that count as annotated — images with
        # at least one annotation, plus frames the user explicitly marked
        # with the "Mark as annotated" button (resolved to their image_id(s)
        # via the frame-idx → image-id lookup). Both sides' image records
        # share the same frame_idx in stereo, so a marked frame contributes
        # every side's image_id for that frame.
        annotated_ids = {a["image_id"] for a in final_anns}
        for frame_idx in self.annotated_marks:
            if frame_idx in self.discarded_frames:
                continue
            for (fi, _side), img_id in self._img_id_by_idx.items():
                if fi == frame_idx and img_id is not None \
                        and img_id not in drop_img_ids:
                    annotated_ids.add(img_id)
        out_images = ([img for img in self.images
                       if img["id"] not in drop_img_ids]
                      if drop_img_ids else self.images)
        # Deterministic order: by timestamp (earlier first), left before
        # right within a pair.
        out_images = sorted(
            out_images,
            key=lambda i: (i.get("timestamp_ns") or 0,
                           0 if i.get("side", "left") == "left" else 1))
        data = {
            "images": out_images,
            "annotations": final_anns,
            "categories": self.categories,
            "annotated_image_ids": sorted(annotated_ids),
        }
        path = self.output_json if is_final else tmp_path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._write_progress()
        self.dirty = False
        print(f"✅ Saved {'final' if is_final else 'progress'} → {path} "
              f"(idx {self.current_idx + 1})")

    def _write_progress(self) -> None:
        """Write only the small .progress sidecar (resume position,
        reviewed set, keyframes, annotated marks, discarded frames).
        Cheap — safe to call per navigation."""
        with open(self.progress_file, "w") as f:
            json.dump({"last_index": self.current_idx + 1,
                       "reviewed": sorted(self.reviewed),
                       "keyframes": sorted(self.keyframes),
                       "annotated_marks": sorted(self.annotated_marks),
                       "discarded": sorted(self.discarded_frames)}, f)

    # ------------------------- mutation -------------------------------- #

    def mark_reviewed(self, frame_idx: int) -> None:
        """Record that the user has reviewed frame `frame_idx` (0-based).

        Called on explicit forward navigation (N) and discard-all (X).
        Not a COCO mutation, so it does not set dirty or push undo — but it
        is persisted in the .progress sidecar on the next save().
        """
        self.reviewed.add(frame_idx)

    def ensure_image(self, frame: Dict[str, Any], width: int, height: int,
                     side: str = "left") -> int:
        # Stereo sessions hold one image record per (frame, side); the
        # timestamp alone is not unique across sides.
        ts = frame["timestamp_ns"]
        ts_key = (ts, side)
        if ts_key in self._img_id_by_ts:
            img_id = self._img_id_by_ts[ts_key]
            # update size if needed (O(1) via the by-id index instead of
            # scanning self.images linearly — that scan ran every tick).
            # Never clobber known dims with 0 (e.g. an import whose source
            # records lack width/height): zero dims silently drop every
            # mask on reload (polygons rasterize to nothing).
            img = self._img_by_id.get(img_id)
            if img is not None and width > 0 and height > 0:
                img["width"] = width
                img["height"] = height
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
            "side": side,
        }
        self.images.append(img_rec)
        self._img_by_id[img_id] = img_rec
        self._img_id_by_ts[ts_key] = img_id
        self._img_id_by_idx[(frame["frame_idx"], side)] = img_id
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
        self._invalidate_ann_caches()
        # No undo entry: seeding happens automatically on first frame visit,
        # so undoable seeds would (a) evict real user history from the
        # bounded stack and (b) let Ctrl+Z mutate frames the user isn't
        # looking at. Deleting a seeded box is itself undoable.

    def add_box(self, image_id: int, x: float, y: float,
                w: float, h: float, cat_id: int,
                track_id: Optional[int] = None) -> int:
        if track_id is None:
            track_id = (self._next_track_id(image_id, cat_id)
                        if self.sticky_track_ids
                        else self._fresh_track_id())
        ann = {
            "id": self._ann_id_next,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
            "track_id": track_id,
        }
        self.annotations.append(ann)
        self._ann_id_next += 1
        self.dirty = True
        self._invalidate_ann_caches()
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
        self._invalidate_ann_caches()
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

    def peek_fresh_track_id(self) -> int:
        """The id the next _fresh_track_id() would return, without
        consuming it (for previews, e.g. the propagate confirm dialog)."""
        return self._track_next

    def _reference_track_ids(self, image_id: int) -> List[Optional[int]]:
        """Track ids of the nearest earlier annotated frame, in that
        frame's creation order — the frame new boxes should inherit
        track ids from (this is what interpolation pairing expects)."""
        idx = None
        side = "left"
        for img in self.images:
            if img["id"] == image_id:
                idx = img.get("frame_idx")
                # Track ids are inherited within the same side only —
                # interpolation pairing is a per-side operation.
                side = img.get("side", "left")
                break
        if not idx:  # None or frame 0 → no earlier frame possible
            return []
        for fi in range(idx - 1, -1, -1):
            ref_img_id = self._img_id_by_idx.get((fi, side))
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

    def labeled_frame_idxs(self, side: Optional[str] = None) -> List[int]:
        """Sorted frame_idxs that have at least one live annotation.

        ``side=None`` means "either side" (progress/UX semantics — a frame
        counts as labeled when any side has boxes); pass ``"left"`` /
        ``"right"`` for side-specific queries (interpolation anchors)."""
        if side is None:
            if self._labeled_frame_idxs_cache is not None:
                return list(self._labeled_frame_idxs_cache)
        else:
            cached = self._labeled_frame_idxs_side_cache.get(side)
            if cached is not None:
                return list(cached)
        if self._anns_cache_dirty:
            self._rebuild_ann_caches()
        # Map image_id -> frame_idx (+ side) for the boxed images only.
        boxed_img_ids = set(self._anns_by_image.keys())
        if side is None:
            out = sorted({
                img.get("frame_idx", 0)
                for img in self.images
                if img["id"] in boxed_img_ids
            })
            self._labeled_frame_idxs_cache = out
        else:
            out = sorted({
                img.get("frame_idx", 0)
                for img in self.images
                if img["id"] in boxed_img_ids
                and img.get("side", "left") == side
            })
            self._labeled_frame_idxs_side_cache[side] = out
        return list(out)

    def frame_has_boxes(self, frame_idx: int,
                        side: Optional[str] = None) -> bool:
        """``side=None`` → "either side" (progress/UX); pass a side for
        side-specific checks (interpolation anchor detection)."""
        if side is not None:
            img_id = self._img_id_by_idx.get((frame_idx, side))
            if img_id is None:
                return False
            if self._anns_cache_dirty:
                self._rebuild_ann_caches()
            return bool(self._anns_by_image.get(img_id))
        # side=None: check both sides' image records for this frame.
        for s in ("left", "right"):
            img_id = self._img_id_by_idx.get((frame_idx, s))
            if img_id is not None:
                if self._anns_cache_dirty:
                    self._rebuild_ann_caches()
                if self._anns_by_image.get(img_id):
                    return True
        return False

    def anchor_candidates(self, side: Optional[str] = None) -> List[int]:
        """Sorted frame_idxs usable as interpolation anchors (have boxes).

        Keyframes (K) take priority: if any keyframe has boxes, only those
        are candidates — so the user can pin exactly which frames bound an
        interpolation span even when other frames also have boxes.
        ``side`` filters to one stereo side (anchors are per-side).
        """
        boxed = self.labeled_frame_idxs(side)
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
        self._invalidate_ann_caches()

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
        self._invalidate_ann_caches()

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
        self._invalidate_ann_caches()

    def _redo_remove(self, ann_id: int) -> None:
        self.removed_ids.add(ann_id)
        self.dirty = True
        self._invalidate_ann_caches()

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

    def _invalidate_ann_caches(self) -> None:
        """Mark the anns_for_image / labeled_frame_idxs caches stale.

        Called from every annotation mutation that changes WHICH anns
        exist or WHERE they belong (add/remove/move/import/undo/redo/
        load). Mask updates (set_mask) don't need this: the cache holds
        aliases to the same ann dicts, so in-place ``_mask`` changes are
        visible immediately. The caches rebuild lazily on the
        next query, so the cost is paid once per mutation, not per tick.
        """
        self._anns_cache_dirty = True
        self._labeled_frame_idxs_cache = None
        self._labeled_frame_idxs_side_cache.clear()

    def _rebuild_ann_caches(self) -> None:
        """Rebuild _anns_by_image from the current annotation list."""
        by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            if ann["id"] in self.removed_ids:
                continue
            by_img.setdefault(ann["image_id"], []).append(ann)
        self._anns_by_image = by_img
        self._anns_cache_dirty = False

    def anns_for_image(self, image_id: int) -> List[Dict[str, Any]]:
        if self._anns_cache_dirty:
            self._rebuild_ann_caches()
        # Return a shallow copy so callers can't mutate the cached list
        # (e.g. append/remove/sort) and silently corrupt the cache.
        return list(self._anns_by_image.get(image_id, []))

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


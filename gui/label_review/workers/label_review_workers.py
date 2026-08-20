"""Worker threads for 09_label_review.py.

Background QThread workers — SAM3 single-frame, SAM3 all-frames, and
optical-flow interpolation between labeled frames — plus their pure
helpers, extracted from 09_label_review.py to keep that file
UI-focused. Everything here is UI-agnostic: inputs are plain data and
results are reported via Qt signals.
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal


# SAM3 (optional — core.models_inference.run_sam3)
_SAM3_AVAILABLE = False
try:
    import sys as _sys
    _PROJ_ROOT = str(Path(__file__).resolve().parents[3])  # repo root
    if _PROJ_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJ_ROOT)
    from core.models_inference import run_sam3, sam3_video_propagate  # type: ignore[import-not-found]
    _SAM3_AVAILABLE = True
except Exception as _sam3_import_err:
    run_sam3 = None  # type: ignore[assignment]
    sam3_video_propagate = None  # type: ignore[assignment]
    print(f"⚠️ SAM3 not available ({_sam3_import_err}). "
          f"Segmentation features will be disabled.")

# ---------------------------------------------------------------------------
# SAM3Worker: runs run_sam3 on a worker thread so the UI doesn't freeze.
# ---------------------------------------------------------------------------

class SAM3Worker(QThread):
    """Asynchronous SAM3 inference thread.

    Inputs:
      image_path : str — path to the frame's image file on disk (run_sam3
                   needs a path).
      bboxes_xyxy: list of [x1,y1,x2,y2] pixel coords (one per region).
      concepts   : list of class names (one per bbox, used for labelling).
      model_path, device, conf : forwarded to run_sam3.

    Emits:
      finished_signal(list_of_dicts) where each dict is
        {ann_id: int|None, bbox_xyxy: [...], mask: HxW bool array|None,
         label: str, area: float, success: bool, error: str|None}
      progress_signal(done, total, concept) after each concept group.
      cancelled_signal() when cancel() was requested (results discarded).

    cancel() is cooperative: it is checked between concept groups, so a
    long-running in-flight run_sam3 call always completes first.
    """

    finished_signal = pyqtSignal(list)
    failed_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)  # done, total, concept
    cancelled_signal = pyqtSignal()

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
        self._cancel_requested = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current concept group."""
        self._cancel_requested = True

    def was_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit(
                "SAM3 is not installed. Install ultralytics + segment-anything "
                "and place model weights under core/sam3/models/sam3-model/sam3.pt"
            )
            return
        results, _device, cancelled = _segment_concepts(
            self.image_path, self.bboxes_xyxy, self.concepts, self.ann_ids,
            self.model_path, self.device, self.conf,
            cancel_check=lambda: self._cancel_requested,
            progress_cb=lambda d, t, c: self.progress_signal.emit(d, t, c),
        )
        if cancelled:
            self.cancelled_signal.emit()
            return
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


def _segment_concepts(image_path: str, bboxes_xyxy: list, concepts: list,
                      ann_ids: list, model_path: Optional[str], device: str,
                      conf: float, cancel_check=None, progress_cb=None
                      ) -> Tuple[List[Dict[str, Any]], str, bool]:
    """Run SAM3 on `bboxes_xyxy`, one run_sam3 call per unique concept.

    Shared by SAM3Worker (single frame) and SAM3BatchWorker (all frames).

    Returns (results, device, cancelled):
      * results — per-box dicts {ann_id, bbox_xyxy, mask, label, area,
        success, error}, one per input bbox.
      * device — the device actually in use at the end. On a CUDA
        out-of-memory error the call is retried on CPU and "cpu" is
        returned, so the caller can stay on CPU for subsequent frames.
      * cancelled — True when cancel_check() fired between concepts; the
        partial results are returned and the caller decides what to do.
    """
    per_concept: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(concepts):
        per_concept[c].append(i)

    results: List[Dict[str, Any]] = []
    total = len(per_concept)
    done = 0

    def _progress(concept: str) -> None:
        nonlocal done
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, concept)

    def _fail_all(idxs: List[int], concept: str, error: str) -> None:
        for i in idxs:
            results.append({
                "ann_id": ann_ids[i],
                "bbox_xyxy": bboxes_xyxy[i],
                "mask": None,
                "label": concept,
                "area": 0.0,
                "success": False,
                "error": error,
            })

    for concept, idxs in per_concept.items():
        if cancel_check is not None and cancel_check():
            return results, device, True
        bxs = [bboxes_xyxy[i] for i in idxs]
        try:
            res = run_sam3(
                image_path=image_path,
                bboxes=bxs,
                concepts=[concept],
                model_path=model_path,
                device=device,
                conf=conf,
            )
        except Exception as e:
            res = {"success": False, "error": str(e)}

        if not res.get("success") and device != "cpu":
            # Any CUDA failure (OOM — e.g. another process hogging the
            # GPU — or an ultralytics-internal error) falls back to CPU
            # for this concept and everything after it. run_sam3 logs the
            # full traceback, so the original CUDA error is not lost.
            err = str(res.get("error", ""))
            kind = "OOM" if "out of memory" in err.lower() else "error"
            print(f"⚠️ SAM3 CUDA {kind} — retrying on CPU "
                  "(and using CPU for the remaining concepts)")
            device = "cpu"
            try:
                res = run_sam3(
                    image_path=image_path,
                    bboxes=bxs,
                    concepts=[concept],
                    model_path=model_path,
                    device=device,
                    conf=conf,
                )
            except Exception as e2:
                res = {"success": False, "error": str(e2)}

        if not res.get("success"):
            _fail_all(idxs, concept, res.get("error", "SAM3 failed"))
            _progress(concept)
            continue

        masks = res.get("masks", []) or []
        dets = res.get("detections", []) or []
        # Pair each input bbox with the closest detection's mask.
        # dets[i].bbox is xyxy. We match by IoU.
        for k, i in enumerate(idxs):
            bx = bboxes_xyxy[i]
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
                "ann_id": ann_ids[i],
                "bbox_xyxy": bx,
                "mask": best_mask,
                "label": concept,
                "area": area,
                "success": best_mask is not None,
                "error": None if best_mask is not None else "no matching mask",
            })
        _progress(concept)

    return results, device, False


class SAM3BatchWorker(QThread):
    """Background SAM3 over many frames ("SAM3 ALL frames" button).

    jobs: list of dicts, one per frame:
        {frame_idx: int, bboxes_xyxy: [...], concepts: [...], ann_ids: [...]}
    Frames are decoded from the frame index inside this thread (decode_image
    is pure in-memory PIL decoding, so this is thread-safe) and written as
    tmp PNGs under tmp_dir.

    Emits frame_done_signal(frame_idx, results) after each frame so the UI
    can apply masks incrementally, plus progress/finished/cancelled signals.
    Cancel is cooperative: checked between frames and between concepts.
    """

    frame_done_signal = pyqtSignal(int, list)   # frame_idx, per-box results
    progress_signal = pyqtSignal(int, int)      # frames done, total frames
    finished_signal = pyqtSignal(int, int)      # masks assigned, failed
    failed_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, frame_index, jobs: List[Dict[str, Any]], tmp_dir: str,
                 model_path: Optional[str], device: str, conf: float,
                 parent=None):
        super().__init__(parent)
        self.frame_index = frame_index
        self.jobs = jobs
        self.tmp_dir = tmp_dir
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self._cancel_requested = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current frame."""
        self._cancel_requested = True

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit(
                "SAM3 is not installed. Install ultralytics + segment-anything "
                "and place model weights under core/sam3/models/sam3-model/sam3.pt"
            )
            return
        os.makedirs(self.tmp_dir, exist_ok=True)
        n_ok = 0
        n_fail = 0
        device = self.device
        for n, job in enumerate(self.jobs):
            if self._cancel_requested:
                self.cancelled_signal.emit()
                return
            arr = self.frame_index.decode_image(job["frame_idx"])
            img_path = os.path.join(self.tmp_dir,
                                    f"batch_{job['frame_idx']:06d}.png")
            Image.fromarray(arr).save(img_path)
            results, device, cancelled = _segment_concepts(
                img_path, job["bboxes_xyxy"], job["concepts"], job["ann_ids"],
                self.model_path, device, self.conf,
                cancel_check=lambda: self._cancel_requested,
            )
            if cancelled:
                # Discard the partial frame; frames already emitted stay.
                self.cancelled_signal.emit()
                return
            self.frame_done_signal.emit(job["frame_idx"], results)
            n_ok += sum(1 for r in results if r["success"])
            n_fail += sum(1 for r in results if not r["success"])
            self.progress_signal.emit(n + 1, len(self.jobs))
        self.finished_signal.emit(n_ok, n_fail)


# ---------------------------------------------------------------------------
# SAM3 autolabel: text-prompt detection (no user-drawn boxes needed).
# Uses SAM3SemanticPredictor directly — the high-level ultralytics SAM API
# loads sam3.pt as the *interactive* (bbox-exemplar-only) model and drops
# text prompts, so we drive the semantic predictor ourselves.
# ---------------------------------------------------------------------------

def _default_sam3_weights() -> str:
    return str(Path(__file__).resolve().parents[3]  # repo root
               / "core" / "sam3" / "models" / "sam3-model" / "sam3.pt")


def _load_sam3_semantic(model_path: Optional[str], device: str,
                        conf: float):
    """Build a SAM3SemanticPredictor ready for text-prompt detection."""
    from ultralytics.models.sam.predict import SAM3SemanticPredictor
    pred = SAM3SemanticPredictor(overrides=dict(
        model=model_path or _default_sam3_weights(),
        conf=conf, device=device, task="segment", mode="predict",
        save=False, verbose=False))
    pred.setup_model(model=None, verbose=False)
    return pred


def _autolabel_frame(pred, image_path: str,
                     concepts: List[str]) -> List[Dict[str, Any]]:
    """Run one text-prompt SAM3 detection on `image_path`.

    concepts: category names, one text prompt each (a single predict call
    covers all of them). Returns a list of detections:
        {label, bbox_xyxy, mask (HxW bool at original resolution), confidence}
    """
    import numpy as np
    pred.set_prompts({"text": list(concepts)})
    results = pred(source=str(image_path))
    r = results[0] if isinstance(results, list) else results
    dets: List[Dict[str, Any]] = []
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    cls = (boxes.cls.cpu().numpy().astype(int)
           if hasattr(boxes, "cls") else np.zeros(len(xyxy), int))
    confs = (boxes.conf.cpu().numpy()
             if hasattr(boxes, "conf") else np.ones(len(xyxy)))
    masks = getattr(r, "masks", None)
    mask_data = masks.data.cpu().numpy() if masks is not None else None
    for i in range(len(xyxy)):
        mask = None
        if mask_data is not None and i < len(mask_data):
            mask = mask_data[i].astype(bool)  # already at orig_shape
        label = concepts[int(cls[i])] if int(cls[i]) < len(concepts) \
            else "object"
        dets.append({
            "label": label,
            "bbox_xyxy": [float(v) for v in xyxy[i]],
            "mask": mask,
            "confidence": float(confs[i]),
        })
    return dets


def _autolabel_with_fallback(image_path: str, concepts: List[str],
                             model_path: Optional[str], device: str,
                             conf: float, pred=None
                             ) -> Tuple[List[Dict[str, Any]], str, Any]:
    """_autolabel_frame with model load + CUDA-OOM→CPU retry.

    If `pred` is None a predictor is built here; otherwise it is reused (the
    batch worker loads once and passes it in per frame). Returns
    (detections, device_in_use, predictor_in_use) so a caller can keep the
    (possibly CPU-fallback) predictor for the next frame."""
    if pred is None:
        pred = _load_sam3_semantic(model_path, device, conf)
    try:
        return _autolabel_frame(pred, image_path, concepts), device, pred
    except Exception as e:
        if device != "cpu" and "out of memory" in str(e).lower():
            print("⚠️ SAM3 CUDA OOM — retrying autolabel on CPU")
            pred = _load_sam3_semantic(model_path, "cpu", conf)
            return _autolabel_frame(pred, image_path, concepts), "cpu", pred
        raise


class SAM3AutolabelWorker(QThread):
    """Single-frame text-prompt autolabel ("Autolabel frame" buttons).

    Emits finished_signal(image_id, detections) — the image_id travels with
    the job so results apply to the right frame even if the user navigated
    while SAM3 ran. No cooperative cancel: one predict call per run.
    """

    finished_signal = pyqtSignal(int, list)  # image_id, detections
    failed_signal = pyqtSignal(str)

    def __init__(self, image_path: str, concepts: List[str],
                 cat_ids: List[int], image_id: int,
                 model_path: Optional[str], device: str, conf: float,
                 parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.concepts = concepts
        self.cat_ids = cat_ids
        self.image_id = image_id
        self.model_path = model_path
        self.device = device
        self.conf = conf

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit("SAM3 is not installed.")
            return
        try:
            dets, _dev, _pred = _autolabel_with_fallback(
                self.image_path, self.concepts, self.model_path,
                self.device, self.conf)
        except Exception as e:
            self.failed_signal.emit(str(e))
            return
        cat_by_name = dict(zip(self.concepts, self.cat_ids))
        for d in dets:
            d["cat_id"] = cat_by_name.get(d["label"])
        self.finished_signal.emit(self.image_id, dets)


class SAM3AutolabelBatchWorker(QThread):
    """Text-prompt autolabel over many frames ("Autolabel ALL frames").

    The predictor is loaded once and reused across frames. Emits
    frame_done_signal(frame_idx, detections) after each frame so the UI can
    add boxes incrementally. Cancel is cooperative (checked between frames).
    """

    frame_done_signal = pyqtSignal(int, list)  # frame_idx, detections
    progress_signal = pyqtSignal(int, int)     # frames done, total frames
    finished_signal = pyqtSignal(int)          # total detections
    failed_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, frame_index, frame_idxs: List[int],
                 concepts: List[str], cat_ids: List[int], tmp_dir: str,
                 model_path: Optional[str], device: str, conf: float,
                 parent=None):
        super().__init__(parent)
        self.frame_index = frame_index
        self.frame_idxs = frame_idxs
        self.concepts = concepts
        self.cat_ids = cat_ids
        self.tmp_dir = tmp_dir
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self._cancel_requested = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current frame."""
        self._cancel_requested = True

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit("SAM3 is not installed.")
            return
        os.makedirs(self.tmp_dir, exist_ok=True)
        cat_by_name = dict(zip(self.concepts, self.cat_ids))
        device = self.device
        # Load the predictor once and reuse it across frames; it is only
        # rebuilt if a CUDA OOM forces a CPU fallback mid-run. (Kept in a
        # try so a load failure reports failed_signal like the per-frame path.)
        try:
            pred = _load_sam3_semantic(self.model_path, device, self.conf)
        except Exception as e:
            self.failed_signal.emit(f"model load: {e}")
            return
        total_dets = 0
        for n, frame_idx in enumerate(self.frame_idxs):
            if self._cancel_requested:
                self.cancelled_signal.emit()
                return
            frame = self.frame_index.frame_at(frame_idx)
            img_path = frame.get("file_path")
            if not img_path or not os.path.exists(img_path):
                arr = self.frame_index.decode_image(frame_idx)
                img_path = os.path.join(self.tmp_dir,
                                        f"autolabel_{frame_idx:06d}.png")
                Image.fromarray(arr).save(img_path)
            try:
                dets, device, pred = _autolabel_with_fallback(
                    img_path, self.concepts, self.model_path, device,
                    self.conf, pred=pred)
            except Exception as e:
                self.failed_signal.emit(f"frame {frame_idx + 1}: {e}")
                return
            for d in dets:
                d["cat_id"] = cat_by_name.get(d["label"])
            total_dets += len(dets)
            self.frame_done_signal.emit(frame_idx, dets)
            self.progress_signal.emit(n + 1, len(self.frame_idxs))
        self.finished_signal.emit(total_dets)


# ---------------------------------------------------------------------------
# SAM3 track propagation: one memory-bank video session per run.
# ---------------------------------------------------------------------------

class SAM3PropagateWorker(QThread):
    """Propagate seeded boxes forward across frames ("Propagate →").

    All seeds (one per selected track on this side) are tracked together in
    a single SAM3VideoPredictor session: the frame range is written to a
    temp mp4, every seed box is registered as its own object on video frame
    0, and SAM3's memory bank follows each object — no per-frame
    re-detection, no IoU chaining. Emits frame_done_signal(frame_idx, dets)
    per frame with dets aligned with `seeds` (None = that object lost on
    this frame; the memory bank may recover it later). The seed frame's own
    result is consumed but not emitted (those boxes already exist). Cancel
    is cooperative, checked between frames; the temp mp4 is removed on
    every exit.
    """

    frame_done_signal = pyqtSignal(int, object)  # frame_idx, [det|None] *
    progress_signal = pyqtSignal(int, int)       # frames done, total frames
    stage_signal = pyqtSignal(str)               # free-text stage line
    finished_signal = pyqtSignal(int, object)    # boxes found, lost_map
                                                 # {seed idx: lost-at idx}
    failed_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, frame_index, start_frame_idx: int,
                 seeds: List[Dict[str, Any]], tmp_dir: str,
                 model_path: Optional[str], device: str, conf: float,
                 parent=None):
        super().__init__(parent)
        self.frame_index = frame_index
        self.start_frame_idx = start_frame_idx
        self.seeds = seeds
        self.tmp_dir = tmp_dir
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self._cancel_requested = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current frame."""
        self._cancel_requested = True

    def _build_clip(self, video_path: str, n_frames: int) -> bool:
        """Write frames start..end to `video_path` (mp4v). Returns False if
        cancelled; raises on unreadable frames / writer failures.

        The ultralytics video predictor only accepts a REAL video file as a
        video source (a frames folder loads as mode="image" and fails
        init_state's assert), so the range is materialized as an mp4 — the
        same thing scripts/track_sam3_video.py does.
        """
        import cv2  # lazy: only needed on this path
        writer = None
        total = n_frames - self.start_frame_idx
        try:
            for frame_idx in range(self.start_frame_idx, n_frames):
                if self._cancel_requested:
                    return False
                frame = self.frame_index.frame_at(frame_idx)
                img_path = frame.get("file_path")
                bgr = (cv2.imread(str(img_path))
                       if img_path and os.path.exists(img_path) else None)
                if bgr is None:
                    arr = self.frame_index.decode_image(frame_idx)  # RGB
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                        (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"could not open {video_path} for writing")
                writer.write(bgr)
                # Building the clip is the long silent phase — report it
                # (throttled: every 10 frames, plus first and last).
                done = frame_idx - self.start_frame_idx + 1
                if done == 1 or done == total or done % 10 == 0:
                    self.stage_signal.emit(f"building clip {done}/{total}…")
        finally:
            if writer is not None:
                writer.release()
        return True

    def run(self) -> None:  # noqa: D401 (QThread override)
        if not _SAM3_AVAILABLE:
            self.failed_signal.emit("SAM3 is not installed.")
            return
        os.makedirs(self.tmp_dir, exist_ok=True)
        n_frames = len(self.frame_index)
        total = n_frames - self.start_frame_idx - 1
        if total <= 0:
            self.finished_signal.emit(0, {})
            return
        video_path = os.path.join(self.tmp_dir, "propagate_clip.mp4")
        frame_idx = self.start_frame_idx
        try:
            if not self._build_clip(video_path, n_frames):
                self.cancelled_signal.emit()
                return
            self.stage_signal.emit(
                "loading model — propagation starts shortly…")
            n_found = 0
            # Per-seed last frame with a detection (for the lost report).
            last_seen = [self.start_frame_idx] * len(self.seeds)
            for k, per_seed in sam3_video_propagate(
                    video_path,
                    [list(s["bbox_xyxy"]) for s in self.seeds],
                    model_path=self.model_path, device=self.device,
                    conf=self.conf,
                    is_cancelled=lambda: self._cancel_requested):
                if self._cancel_requested:
                    self.cancelled_signal.emit()
                    return
                frame_idx = self.start_frame_idx + k
                if k == 0:
                    continue  # seed frame — those boxes already exist
                dets = []
                for i, (mask, bbox, score) in enumerate(per_seed):
                    if mask is None or bbox is None:
                        dets.append(None)
                        continue
                    last_seen[i] = frame_idx
                    dets.append({"bbox_xyxy": bbox, "mask": mask,
                                 "confidence": score})
                n_found += sum(d is not None for d in dets)
                self.frame_done_signal.emit(frame_idx, dets)
                self.progress_signal.emit(k, total)
            if self._cancel_requested:
                # The engine may have ended the generator silently via its
                # own is_cancelled check — never report that as "finished".
                self.cancelled_signal.emit()
                return
        except Exception as e:
            self.failed_signal.emit(f"frame {frame_idx + 1}: {e}")
            return
        finally:
            try:
                os.remove(video_path)
            except OSError:
                pass
        lost = {i: last_seen[i] + 1 for i in range(len(self.seeds))
                if last_seen[i] < n_frames - 1}
        self.finished_signal.emit(n_found, lost)


# ---------------------------------------------------------------------------
# 13_interpolate_tracks.py engine loader + interpolation worker
# ---------------------------------------------------------------------------

_interp13_mod: Optional[Any] = None


def _get_interp13():
    """Lazily import scripts/13_interpolate_tracks.py (cached).

    The file name starts with a digit, so a normal `import` is impossible;
    load it by path via importlib. Safe at import time: 13's top-level code
    only imports stdlib + cv2/numpy (+ scripts.tracking_utils, which is also
    dependency-light) and its main() is __main__-guarded.
    """
    global _interp13_mod
    if _interp13_mod is None:
        from importlib import util
        path = Path(__file__).resolve().parents[3] / "scripts" / \
            "13_interpolate_tracks.py"
        if not path.exists():
            # PyInstaller bundle: shipped as a data file in the bundle root.
            import sys
            path = Path(getattr(sys, "_MEIPASS", ".")) / \
                "13_interpolate_tracks.py"
        spec = util.spec_from_file_location("interpolate_tracks", path)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _interp13_mod = mod
    return _interp13_mod


class InterpBatchWorker(QThread):
    """Background optical-flow interpolation between labeled frames.

    jobs: list of dicts, one per matched (box_a, box_b) anchor pair:
        {a: int, b: int, box_a: boxdict, box_b: boxdict}
    where a < b are frame_idxs and box_a/box_b are 13-style box dicts
    (ann_id, category_id, track_id, xywh, xyxy, center).

    For each job the span's images are materialized into a fresh
    per-run tmp dir and 13's interpolate_span is called as-is (flow_method /
    camera_model passed through). Results for all jobs are collected and
    emitted together via finished_signal(list of (job, {p: result})) so the
    UI can apply them in a single undo group. Cancel is cooperative
    (checked between jobs).
    """

    progress_signal = pyqtSignal(int, int)   # jobs done, total jobs
    finished_signal = pyqtSignal(list)       # list of (job, pairs)
    failed_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self, frame_index, jobs: List[Dict[str, Any]],
                 base_tmp_dir: str, flow_method: str, camera_model: str,
                 parent=None):
        super().__init__(parent)
        self.frame_index = frame_index
        self.jobs = jobs
        self.base_tmp_dir = base_tmp_dir
        self.flow_method = flow_method
        self.camera_model = camera_model
        self._cancel_requested = False

    def cancel(self) -> None:
        """Ask the worker to stop after the current job."""
        self._cancel_requested = True

    @staticmethod
    def _frame_ext(frame: Dict[str, Any]) -> str:
        mt = frame.get("media_type") or "image/jpeg"
        if "png" in mt:
            return ".png"
        if "webp" in mt:
            return ".webp"
        if "bmp" in mt:
            return ".bmp"
        return ".jpg"

    def _write_span(self, run_dir: str, a: int, b: int) -> List[Optional[str]]:
        """Copy the span's source images into run_dir; return per-frame file
        names (length b+1, None before a). Raises RuntimeError on a frame
        with no backing file."""
        frames: List[Optional[str]] = [None] * (b + 1)
        for p in range(a, b + 1):
            frame = self.frame_index.frame_at(p)
            name = f"frame_{p:06d}{self._frame_ext(frame)}"
            frames[p] = name
            dst = os.path.join(run_dir, name)
            if frame.get("file_path"):
                # cv2.imread sniffs the format, so the .jpg name is fine for
                # any source type.
                shutil.copyfile(frame["file_path"], dst)
            else:
                raise RuntimeError(f"frame {p + 1}: no image data in source")
        return frames

    def run(self) -> None:  # noqa: D401 (QThread override)
        try:
            mod = _get_interp13()
        except Exception as e:
            self.failed_signal.emit(
                f"Cannot load 13_interpolate_tracks.py: {e}")
            return
        os.makedirs(self.base_tmp_dir, exist_ok=True)
        results_out: List[Tuple[Dict[str, Any], Dict[int, Any]]] = []
        for n, job in enumerate(self.jobs):
            if self._cancel_requested:
                self.cancelled_signal.emit()
                return
            a, b = job["a"], job["b"]
            run_dir = os.path.join(
                self.base_tmp_dir, f"run_{os.getpid()}_{id(self)}")
            os.makedirs(run_dir, exist_ok=True)
            try:
                frames = self._write_span(run_dir, a, b)
                pairs = mod.interpolate_span(
                    run_dir, frames, a, b, job["box_a"], job["box_b"],
                    flow_method=self.flow_method,
                    camera_model=self.camera_model)
                results_out.append((job, pairs))
            except Exception as e:
                self.failed_signal.emit(
                    f"Interpolation failed between frames {a + 1} and "
                    f"{b + 1}: {e}")
                return
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)
            self.progress_signal.emit(n + 1, len(self.jobs))
        self.finished_signal.emit(results_out)


# SAM3 Memory-Bank Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GUI's per-frame re-detect+IoU track propagation with SAM3VideoPredictor memory-bank tracking — one video session per side, seeding all of that side's selected boxes at once.

**Architecture:** A new generator `sam3_video_propagate()` in `core/models_inference.py` drives ultralytics `SAM3VideoPredictor` via its low-level session API (the recipe from `scripts/track_sam3_video.py` "reseed" mode) over a temporary mp4 built from the frame range, yielding obj_id-aligned per-seed masks per frame. `SAM3PropagateWorker` is rewritten around it (multi-seed, same Qt signal shape except `frame_done` now carries a per-seed list and `finished` carries a lost-map). The GUI groups selected seeds by side and runs ≤2 jobs through the existing SAM3 queue. Spec: `docs/superpowers/specs/2026-08-20-sam3-video-propagate-design.md`.

**Tech Stack:** Python 3.13, PyQt6, ultralytics 8.4.83 (`SAM3VideoPredictor`), torch, cv2, numpy, pytest.

## Global Constraints

- Repo root: `/home/wangyiming/code/object_detection_app`. GUI package: `gui/label_review/` (paths below are repo-relative).
- **No test ever runs real SAM3/torch inference** — ultralytics' predictor and the new engine are always faked/monkeypatched. `QT_QPA_PLATFORM=offscreen` is set by `tests/conftest.py`.
- Run tests from repo root: `python -m pytest tests/ -q`
- **Do NOT run any git mutations** (no `git add`/`commit`/etc.) — environment policy. Leave changes uncommitted; skip any "commit" step.
- `_iou_xyxy` in `gui/label_review/workers/label_review_workers.py` is still used by autolabel dedup in `main_window.py` — do not delete it.
- Engine constraint (verified against installed ultralytics source): video mode needs a REAL video file (a frames folder loads as `mode="image"` and fails `init_state`'s assert). Do not use the high-level `predictor(source=..., bboxes=..., stream=True)` API — its blank-mask filtering breaks seed↔mask positional alignment for multi-object. Use the low-level session API only.

---

### Task 1: Engine `sam3_video_propagate()` in `core/models_inference.py`

**Files:**
- Modify: `core/models_inference.py` (add after `run_sam3`, i.e. right before `def _find_sam3_checkpoint` near line 354; add `Iterator` to the `typing` import at line 66)
- Test: `tests/test_unit_models_inference.py` (new file)

**Interfaces:**
- Produces: `sam3_video_propagate(video_path: str | Path, seed_bboxes_xyxy: List[List[float]], model_path: Optional[str | Path] = None, device: str = "cuda", conf: float = 0.25, imgsz: int = 1024, quantize: Optional[int] = None, is_cancelled: Optional[Callable[[], bool]] = None)` — a generator yielding `(video_frame_idx: int, per_seed: List[Tuple[Optional[np.ndarray], Optional[List[float]], float]])` starting at video frame 0. `per_seed[i]` is `(mask_bool, bbox_xyxy, score)` for seed i, or `(None, None, 0.0)` when that object is lost on that frame. Alignment is by SAM3 `obj_id` (= seed index), never positional.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_unit_models_inference.py`:

```python
"""Unit tests for core/models_inference.sam3_video_propagate with a faked
ultralytics SAM3VideoPredictor — no real model, GPU, or video decoding."""

import numpy as np
import pytest


class _FakeDataset:
    """Minimal video dataset: a frames count plus one batch at frame 0."""

    def __init__(self, frames=4, shape=(80, 100, 3)):
        self.frames = frames
        self.frame = 0
        self._img = np.zeros(shape, np.uint8)

    def __iter__(self):
        return iter([(["f0"], [self._img.copy()], ["video"])])


class _FakeVideoPredictor:
    """Stands in for ultralytics.models.sam.SAM3VideoPredictor.

    Tracks 2 objects; object 1's mask is blank from frame `blank_from` on."""

    instances = []

    def __init__(self, overrides=None):
        self.overrides = overrides
        self.dataset = _FakeDataset()
        self.inference_state = {}
        self._bb_feat_sizes = [(20, 25)]
        self.seeded = []
        self.blank_from = 2
        _FakeVideoPredictor.instances.append(self)

    def setup_model(self):
        pass

    def setup_source(self, source):
        self.source = str(source)

    def init_state(self, predictor):
        predictor.inference_state["ready"] = True

    def preprocess(self, ims):
        return ims

    def inference(self, im, bboxes=None, **kw):
        import torch
        return torch.ones((len(bboxes), 80, 100)), None

    def add_new_prompts(self, obj_id, masks=None, frame_idx=None, **kw):
        self.seeded.append((obj_id, frame_idx))

    def propagate_in_video(self, state, frame_idx):
        import torch
        masks = torch.ones((2, 80, 100))
        if frame_idx >= self.blank_from:
            masks[1] = 0.0
        return [0, 1], masks, torch.zeros(2)


@pytest.fixture
def fake_video_predictor(monkeypatch):
    _FakeVideoPredictor.instances.clear()
    import ultralytics.models.sam as sam_mod
    monkeypatch.setattr(sam_mod, "SAM3VideoPredictor", _FakeVideoPredictor)
    return _FakeVideoPredictor


def test_video_propagate_obj_id_alignment(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    frames = list(sam3_video_propagate("/tmp/clip.mp4",
                                       [[1, 1, 9, 9], [20, 20, 30, 30]],
                                       model_path="m.pt", device="cpu"))
    assert [f for f, _ in frames] == [0, 1, 2, 3]
    # seed 0 found on every frame (all-ones 80x100 mask → full-frame box)
    assert all(per[0][1] == [0.0, 0.0, 99.0, 79.0] for _, per in frames)
    # seed 1 lost from frame 2 on — slot stays index 1, not shifted
    assert frames[0][1][1][1] is not None
    assert frames[1][1][1][1] is not None
    assert frames[2][1][1] == (None, None, 0.0)
    assert frames[3][1][1] == (None, None, 0.0)
    # both seeds registered as prompts on frame 0
    assert fake_video_predictor.instances[0].seeded == [(0, 0), (1, 0)]


def test_video_propagate_cancel_stops_stream(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    # one seed, but the fake returns obj 1 too — the out-of-range guard
    # must skip it without crashing
    frames = list(sam3_video_propagate("/tmp/clip.mp4", [[1, 1, 9, 9]],
                                       model_path="m.pt", device="cpu",
                                       is_cancelled=cancel))
    assert len(frames) == 1  # frame 0 yielded, frame 1 cancelled


def test_video_propagate_default_model_path(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    list(sam3_video_propagate("/tmp/clip.mp4", [[1, 1, 9, 9]], device="cpu"))
    pred = fake_video_predictor.instances[0]
    assert pred.overrides["model"].endswith("sam3.pt")
    assert pred.overrides["imgsz"] == 1024
    assert pred.overrides["device"] == "cpu"
    assert pred.source == "/tmp/clip.mp4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/test_unit_models_inference.py -q`
Expected: FAIL with `ImportError: cannot import name 'sam3_video_propagate'`

- [ ] **Step 3: Implement the engine**

In `core/models_inference.py`, extend the typing import (line 66) to include `Iterator`:

```python
from typing import Optional, Callable, Dict, Any, List, Tuple, Iterator
```

Add this function right after `run_sam3` ends (just before `def _find_sam3_checkpoint`):

```python
def sam3_video_propagate(
    video_path: str | Path,
    seed_bboxes_xyxy: List[List[float]],
    model_path: Optional[str | Path] = None,
    device: str = "cuda",
    conf: float = 0.25,
    imgsz: int = 1024,
    quantize: Optional[int] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Iterator[Tuple[int, List[Tuple[Optional[np.ndarray],
                                    Optional[List[float]], float]]]]:
    """Track seeded objects through a video with SAM3's memory bank.

    Uses ultralytics SAM3VideoPredictor's low-level session API (the recipe
    from scripts/track_sam3_video.py "reseed" mode): seed every bbox as its
    own object on video frame 0, then propagate frame-by-frame. The
    high-level streaming API is NOT used — its blank-mask filtering breaks
    the seed↔mask alignment for multi-object tracking.

    Yields (video_frame_idx, per_seed) starting at video frame 0 (the seed
    frame). per_seed[i] is (mask_bool, bbox_xyxy, score) for seed i, or
    (None, None, 0.0) when that object is lost on that frame — the memory
    bank may recover it on later frames. score is the sigmoid-squashed
    object score logit (0..1), for the box's confidence provenance field.
    """
    import torch
    from ultralytics.models.sam import SAM3VideoPredictor

    if model_path is None:
        model_path = (Path(__file__).parent / "sam3" / "models"
                      / "sam3-model" / "sam3.pt")
    overrides = dict(conf=conf, task="segment", mode="predict",
                     model=str(model_path), device=device, imgsz=imgsz)
    if quantize is not None:
        overrides["quantize"] = quantize
    predictor = SAM3VideoPredictor(overrides=overrides)
    n_seeds = len(seed_bboxes_xyxy)
    try:
        predictor.setup_model()
        predictor.setup_source(str(video_path))
        predictor.init_state(predictor)
        num_frames = predictor.dataset.frames

        # Seed frame (video frame 0): segment each exemplar box, then
        # register the downsampled masks as per-object prompts.
        predictor.dataset.frame = 0
        batch0 = next(iter(predictor.dataset))
        frame_h, frame_w = batch0[1][0].shape[:2]
        im0 = predictor.preprocess(batch0[1])
        seed_masks, _ = predictor.inference(
            im0, bboxes=[list(b) for b in seed_bboxes_xyxy])
        lr_h, lr_w = predictor._bb_feat_sizes[0]
        for i in range(n_seeds):
            low_res = torch.nn.functional.interpolate(
                seed_masks[[i]].unsqueeze(1).float(), size=(lr_h, lr_w),
                mode="bilinear", align_corners=False)
            predictor.add_new_prompts(obj_id=i,
                                      masks=(low_res > 0.5).float(),
                                      frame_idx=0)

        for frame_idx in range(num_frames):
            if is_cancelled is not None and is_cancelled():
                return
            obj_ids, pred_masks, obj_scores = \
                predictor.propagate_in_video(predictor.inference_state,
                                             frame_idx)
            masks_np = (pred_masks.cpu().numpy()
                        if hasattr(pred_masks, "cpu")
                        else np.asarray(pred_masks))
            scores_np = (obj_scores.detach().cpu().numpy()
                         if hasattr(obj_scores, "detach")
                         else np.asarray(obj_scores))
            per_seed: List[Tuple[Optional[np.ndarray],
                                 Optional[List[float]], float]] = \
                [(None, None, 0.0)] * n_seeds
            for j in range(len(obj_ids)):
                oid = int(obj_ids[j])
                if not (0 <= oid < n_seeds):
                    continue
                mask = masks_np[j].astype(bool)
                if mask.shape != (frame_h, frame_w):
                    mask = cv2.resize(mask.astype(np.uint8),
                                      (frame_w, frame_h),
                                      interpolation=cv2.INTER_NEAREST
                                      ).astype(bool)
                if not mask.any():
                    continue  # object lost on this frame (may recover)
                ys, xs = np.where(mask)
                bbox = [float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())]
                score = float(1.0 / (1.0 + np.exp(-float(scores_np[j]))))
                per_seed[oid] = (mask, bbox, score)
            yield frame_idx, per_seed
    finally:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/test_unit_models_inference.py -q`
Expected: 3 passed

---

### Task 2: Rewrite `SAM3PropagateWorker` (multi-seed, memory bank)

**Files:**
- Modify: `gui/label_review/workers/label_review_workers.py` (guarded import at line 29; delete `_PROPAGATE_MIN_IOU`/`_PROPAGATE_MIN_SEED_IOU`/`_propagate_step` at lines 515-602; rewrite `SAM3PropagateWorker` at lines 605-689)
- Modify: `tests/conftest.py` (`FakePropagateWorker`, lines 135-174)
- Test: `tests/test_unit_workers.py` (delete the five `_propagate_step` tests at lines 177-239; rewrite the two worker tests at lines 453-489)

**Interfaces:**
- Consumes: `sam3_video_propagate` from Task 1.
- Produces (relied on by Task 3's `main_window.py` changes):
  - `SAM3PropagateWorker(frame_index, start_frame_idx: int, seeds: List[Dict], tmp_dir: str, model_path: Optional[str], device: str, conf: float, parent=None)` where each seed is `{"bbox_xyxy": List[float], "track_id": int, "cat_id": int}`. No `concept`, `min_iou`, or `min_seed_iou`.
  - `frame_done_signal(int, object)` → `(frame_idx, dets)` with `dets: List[Optional[Dict]]` aligned with `seeds`; det = `{"bbox_xyxy", "mask", "confidence"}`.
  - `finished_signal(int, object)` → `(boxes_found_total, lost_map)` with `lost_map: Dict[int, int]` = seed index → first all-empty frame idx, only for seeds still lost at clip end.
  - `progress_signal(int, int)`, `failed_signal(str)`, `cancelled_signal()` unchanged.

- [ ] **Step 1: Rewrite the failing worker tests**

In `tests/test_unit_workers.py`:

a. Update the module docstring (line 2): replace `(_segment_concepts, _propagate_step, _autolabel_frame)` with `(_segment_concepts, _autolabel_frame)`.

b. Add `import os` to the imports (after `import numpy as np`).

c. Delete the five `_propagate_step` tests: `test_propagate_step_picks_best_iou`, `test_propagate_step_lost_on_empty`, `test_propagate_step_lost_below_min_iou`, `test_propagate_step_failure_raises`, `test_propagate_step_cuda_oom_retries_on_cpu` (lines 177-239), including their section-header comment.

d. Replace `test_propagate_worker_chains_and_stops_when_lost` and `test_propagate_worker_noop_at_last_frame` with:

```python
# SAM3PropagateWorker (memory-bank engine, monkeypatched)
# ---------------------------------------------------------------------------

def _fake_engine(scripted):
    """Fake sam3_video_propagate: scripted is one per_seed list per video
    frame. Records the video path + seeds it was called with."""

    def fake(video_path, seed_bboxes_xyxy, is_cancelled=None, **kw):
        fake.video_path = video_path
        fake.seeds = [list(b) for b in seed_bboxes_xyxy]
        assert os.path.exists(video_path)  # the mp4 was really built
        yield from scripted

    return fake


def test_propagate_worker_streams_multi_seed(sam3_on, workers, monkeypatch,
                                             tmp_path):
    m = np.ones((80, 100), bool)
    scripted = [
        [(m, [5, 5, 15, 15], 1.0), (m, [40, 40, 50, 50], 1.0)],  # seed frame
        [(m, [10, 10, 20, 20], 0.9), (m, [41, 41, 51, 51], 0.8)],
        [(m, [11, 11, 21, 21], 0.9), (None, None, 0.0)],  # seed 1 lost
        [(m, [12, 12, 22, 22], 0.9), (None, None, 0.0)],
    ]
    fake = _fake_engine(scripted)
    monkeypatch.setattr(workers, "sam3_video_propagate", fake)
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0},
             {"bbox_xyxy": [40, 40, 50, 50], "track_id": 2, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(4), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    frames, fin = [], []
    w.frame_done_signal.connect(lambda f, dets: frames.append((f, dets)))
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert [f for f, _ in frames] == [1, 2, 3]  # seed frame not emitted
    assert frames[0][1][0]["bbox_xyxy"] == [10, 10, 20, 20]
    assert frames[0][1][1]["bbox_xyxy"] == [41, 41, 51, 51]
    assert frames[1][1][1] is None              # lost seed → None slot
    assert fin == [(4, {1: 2})]  # 4 boxes found; seed 1 lost from frame 2
    assert fake.seeds == [[5, 5, 15, 15], [40, 40, 50, 50]]
    assert not os.path.exists(fake.video_path)  # temp mp4 cleaned up


def test_propagate_worker_cancel_mid_stream(sam3_on, workers, monkeypatch,
                                            tmp_path):
    m = np.ones((80, 100), bool)
    scripted = [[(m, [0, 0, 5, 5], 1.0)] for _ in range(5)]
    monkeypatch.setattr(workers, "sam3_video_propagate",
                        _fake_engine(scripted))
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(5), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    events = []

    def on_frame(f, dets):
        events.append(f)
        w.cancel()

    w.frame_done_signal.connect(on_frame)
    w.cancelled_signal.connect(lambda: events.append("cancelled"))
    w.run()
    assert events == [1, "cancelled"]
    assert not os.path.exists(
        os.path.join(str(tmp_path / "prop"), "propagate_clip.mp4"))


def test_propagate_worker_noop_at_last_frame(sam3_on, workers, tmp_path):
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(1), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    fin = []
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert fin == [(0, {})]
```

e. Update `tests/conftest.py` `FakePropagateWorker.__init__` (lines 141-146) to the new signature:

```python
    def __init__(self, frame_index, start_frame_idx, seeds, tmp_dir,
                 model_path, device, conf, parent=None):
        self.kw = dict(start_frame_idx=start_frame_idx, seeds=seeds)
```

(The rest of the fake — signals, start/cancel/stop/isRunning/wait/terminate — stays as is.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/test_unit_workers.py -q`
Expected: FAIL — `TypeError`/`AttributeError` from the old worker signature and the deleted `_propagate_step` references are gone from the test file, but the old worker still takes `(frame_index, start_frame_idx, seed_bbox_xyxy, concept, tmp_dir, ...)`.

- [ ] **Step 3: Rewrite the worker**

In `gui/label_review/workers/label_review_workers.py`:

a. Line 29: change the guarded import to

```python
    from core.models_inference import run_sam3, sam3_video_propagate  # type: ignore[import-not-found]
```

b. Delete lines 515-602: the section-header comment block ("SAM3 track propagation: seed with one box, re-detect frame-by-frame."), `_PROPAGATE_MIN_IOU`, `_PROPAGATE_MIN_SEED_IOU`, and all of `_propagate_step`. Replace with a new section header:

```python
# ---------------------------------------------------------------------------
# SAM3 track propagation: one memory-bank video session per run.
# ---------------------------------------------------------------------------
```

c. Replace the whole `SAM3PropagateWorker` class (lines 605-689) with:

```python
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
```

- [ ] **Step 4: Run worker + engine tests to verify they pass**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/test_unit_workers.py tests/test_unit_models_inference.py -q`
Expected: all passed (GUI tests using the old signature will still fail — fixed in Tasks 3-4)

---

### Task 3: GUI plumbing in `gui/label_review/ui/main_window.py`

**Files:**
- Modify: `gui/label_review/ui/main_window.py` — `__init__` propagate attrs (lines 64-67 and 104-112), `_apply_config` (lines 669-674), `_start_next_queued_sam3` propagate branch (lines 1744-1748), `_on_propagate_track` tail (seeds loop, ~lines 2390-2403), `_start_propagate_worker` (~lines 2383-2441), `_on_propagate_frame_done` (~lines 2443-2489), `_end_propagate` undo label (~lines 2500-2503), `_on_propagate_finished`/`_failed`/`_cancelled` (~lines 2512-2542)

**Interfaces:**
- Consumes: the Task 2 worker (`seeds` ctor arg, list-valued `frame_done`, `(int, dict)` `finished`).
- Produces: `_propagate_label(seeds, side) -> str` helper; `_propagate_meta` shape `{"seeds": List[Dict], "side": str, "added": int, "ann_ids": list, "anns": list}`; queue job shape `{"kind": "propagate", "start_frame_idx": int, "seeds": List[Dict], "side": str}`.

- [ ] **Step 1: Remove the dead IoU settings**

a. Delete the propagate-IoU attributes and their comment (lines 63-67 — the comment starting `# by >= min_seed_iou, otherwise the chain stops` may start a line or two earlier; delete the whole comment block plus):

```python
        self.sam3_propagate_min_iou: float = 0.3
        self.sam3_propagate_seed_iou: float = 0.2
```

b. Delete from `_apply_config` (lines 669-674):

```python
        if "propagate_min_iou" in sam3_cfg:
            self.sam3_propagate_min_iou = max(
                0.0, min(1.0, float(sam3_cfg["propagate_min_iou"])))
        if "propagate_min_seed_iou" in sam3_cfg:
            self.sam3_propagate_seed_iou = max(
                0.0, min(1.0, float(sam3_cfg["propagate_min_seed_iou"])))
```

c. In `__init__`, change the `_propagate_meta` initializer (lines 104-112) to:

```python
        # Meta for the running propagate job (seeds, side, count).
        self._propagate_meta: Dict[str, Any] = {
            "seeds": [], "side": "left",
            "added": 0, "ann_ids": [], "anns": []}
```

(Keep any surrounding comment lines that still apply; drop keys `track_id`/`cat_id`.)

- [ ] **Step 2: Update the queue-dispatch branch**

Replace lines 1744-1748 with:

```python
        elif job.get("kind") == "propagate":
            self._start_propagate_worker(
                job["start_frame_idx"], job["seeds"], side=job.get("side"))
```

- [ ] **Step 3: Group seeds by side in `_on_propagate_track`**

Replace the final seed loop (the `for side, box in seeds:` block that calls `_start_propagate_worker(..., side=side)`) with:

```python
        by_side: Dict[str, List[Dict[str, Any]]] = {}
        for side, box in seeds:
            tid = box.get("track_id")
            if tid is None:
                # Seed without a track id: assign a fresh one now that it's
                # confirmed (its own undoable entry).
                tid = self.coco._fresh_track_id()
                self.coco.set_track_id(box["id"], tid)
                box["track_id"] = tid
            x, y, w, h = box["bbox"]
            by_side.setdefault(side, []).append({
                "track_id": tid,
                "cat_id": box.get("cat_id", 0),
                "bbox_xyxy": [x, y, x + w, y + h],
            })
        # One memory-bank session per side covers ALL of that side's
        # seeds at once; the second side queues behind the first.
        for side, side_seeds in by_side.items():
            self._start_propagate_worker(self._current_idx, side_seeds, side)
```

Also update the method docstring's second paragraph to:

```
        With several boxes selected, each distinct track is propagated —
        one SAM3 video session per side covering ALL of that side's seeds
        (in stereo the two sides run back-to-back through the SAM3 queue).
        Boxes sharing a track id on the SAME side collapse into a single
        seed; boxes without a track id each seed a fresh track.
```

And in the confirm-dialog text, replace `Tracks run one " "at a time in the background` with `Each side runs as one background job` (keep the device mention).

- [ ] **Step 4: Rewrite `_start_propagate_worker` and add `_propagate_label`**

Replace the whole `_start_propagate_worker` method with:

```python
    def _propagate_label(self, seeds: List[Dict[str, Any]],
                         side: str) -> str:
        """Status/undo label for a propagate run."""
        if len(seeds) == 1:
            return f"T{seeds[0]['track_id']}"
        tag = f" [{side}]" if self._stereo else ""
        return f"{len(seeds)} tracks{tag}"

    def _start_propagate_worker(self, start_frame_idx: int,
                                seeds: List[Dict[str, Any]],
                                side: Optional[str] = None) -> None:
        if side is None:
            side = self._active_canvas.side
        if self._sam3_busy():
            self._sam3_queue.append({
                "kind": "propagate",
                "start_frame_idx": start_frame_idx,
                "seeds": seeds,
                "side": side,
            })
            self._refresh_sam3_status()
            self.statusBar().showMessage(
                f"SAM3 busy — propagate queued "
                f"({len(self._sam3_queue)} in queue)", 2500)
            return
        label = self._propagate_label(seeds, side)
        total = len(self.frame_index) - start_frame_idx - 1
        self._set_sam3_status(f"propagate {label}: 0/{total} frames…")
        self.side.set_sam3_running(True)
        self._propagate_meta = {"seeds": seeds, "side": side,
                                "added": 0, "ann_ids": [], "anns": []}
        tmp_dir = str(Path(self.coco.output_json).parent / "_tmp_sam3_imgs")
        self._sam3_propagate_worker = SAM3PropagateWorker(
            self._side_worker_index(side), start_frame_idx, seeds, tmp_dir,
            model_path=self.sam3_model,
            device=self.sam3_device,
            conf=self.sam3_conf,
            parent=self,
        )
        self._sam3_propagate_worker.frame_done_signal.connect(
            self._on_propagate_frame_done)
        self._sam3_propagate_worker.progress_signal.connect(
            lambda d, t: self._set_sam3_status(
                f"propagate {label}: {d}/{t} frames…"))
        self._sam3_propagate_worker.finished_signal.connect(
            self._on_propagate_finished)
        self._sam3_propagate_worker.failed_signal.connect(
            self._on_propagate_failed)
        self._sam3_propagate_worker.cancelled_signal.connect(
            self._on_propagate_cancelled)
        self._sam3_propagate_worker._lr_session = self._session_seq
        self._sam3_propagate_worker.start()
```

- [ ] **Step 5: Rewrite `_on_propagate_frame_done` for the per-seed list**

Replace the whole method with:

```python
    def _on_propagate_frame_done(self, frame_idx: int, dets) -> None:
        """Add propagated boxes (one per seed, keeping each seed's track
        id), or skip. `dets` is aligned with meta["seeds"]."""
        if self._stale_sender():
            return
        meta = self._propagate_meta
        side = meta.get("side", "left")
        for seed, det in zip(meta.get("seeds", []), dets or []):
            if det is None:
                continue
            frame = self._frame_at_side(frame_idx, side)
            mask = det.get("mask")
            if mask is not None:
                h, w = mask.shape
            else:
                arr = self._decode_side(frame_idx, side)
                h, w = arr.shape[:2]
            image_id = self.coco.ensure_image(frame, w, h, side=side)
            # Never overwrite a frame that already has this track id.
            have = any(a.get("track_id") == seed["track_id"]
                       for a in self.coco.anns_for_image(image_id))
            if have:
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            # Muted: per-frame pushes are dropped; _end_propagate
            # re-pushes the whole run as ONE undo entry, so user
            # edits made while the run is in flight stay separate.
            with self.coco.undo_stack.mute():
                # Pass the track id in — a separate add_box (which
                # consumes a fresh id) + set_track_id would leak the
                # global counter every frame.
                ann_id = self.coco.add_box(image_id, x1, y1,
                                           x2 - x1, y2 - y1,
                                           seed["cat_id"],
                                           track_id=seed["track_id"])
                ann = self.coco.get_box(ann_id)
                if ann is not None:
                    ann["propagated"] = True
                    ann["confidence"] = float(det.get("confidence", 1.0))
                if mask is not None:
                    self.coco.set_mask(ann_id, mask)
            meta["ann_ids"].append(ann_id)
            meta["anns"].append(dict(self.coco.get_box(ann_id)))
            meta["added"] += 1
        if frame_idx == self._current_idx:
            self._refresh_boxes()
        # Checkpoint every 10 frames so propagated boxes survive a crash.
        if (frame_idx + 1) % 10 == 0:
            self.coco.save(is_final=False)
```

- [ ] **Step 6: Update the run-end handlers**

a. In `_end_propagate`, replace the undo push block with:

```python
        if ids:
            # One composite undo entry for the whole run (the per-frame
            # pushes were muted).
            coco = self.coco
            seeds = meta.get("seeds", [])
            what = (f"track T{seeds[0]['track_id']}" if len(seeds) == 1
                    else f"{len(seeds)} tracks")
            coco.undo_stack.push(
                f"propagate {what} ({len(ids)} box(es))",
                undo=lambda: [coco._undo_remove(i) for i in reversed(ids)],
                redo=lambda: [coco._redo_add(a) for a in anns])
```

b. Replace `_on_propagate_finished`, `_on_propagate_failed`, and `_on_propagate_cancelled` with:

```python
    def _on_propagate_finished(self, n_found: int, lost_map) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta
        seeds = meta.get("seeds", [])
        label = self._propagate_label(seeds, meta.get("side", "left"))
        # n_found counts detections, but frames that already carried the
        # track id (or a sub-2px box) were skipped — report boxes actually
        # added.
        added = meta["added"]
        status = f"propagate {label}: done — {added} box(es) added"
        notes = [f"T{seeds[i]['track_id']} lost at frame {f + 1}"
                 for i, f in sorted((lost_map or {}).items())
                 if i < len(seeds)]
        if notes:
            status += f" ({', '.join(notes)})"
        self._end_propagate(status, "P" + status[1:])

    def _on_propagate_failed(self, err: str) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta
        label = self._propagate_label(meta.get("seeds", []),
                                      meta.get("side", "left"))
        print(f"❌ SAM3 propagate failed: {err}")
        self._end_propagate(f"propagate {label}: failed — {err}",
                            f"SAM3 propagate failed: {err}")

    def _on_propagate_cancelled(self) -> None:
        if self._stale_sender():
            return
        meta = self._propagate_meta
        label = self._propagate_label(meta.get("seeds", []),
                                      meta.get("side", "left"))
        added = meta["added"]
        self._end_propagate(
            f"propagate {label}: cancelled ({added} box(es) kept)",
            "Propagate cancelled — boxes already added were kept")
```

- [ ] **Step 7: Byte-compile check**

Run: `cd /home/wangyiming/code/object_detection_app && python -c "import ast; ast.parse(open('gui/label_review/ui/main_window.py').read())"`
Expected: no output, exit 0 (GUI tests are updated in Task 4)

---

### Task 4: Update the GUI integration/stereo tests

**Files:**
- Modify: `tests/test_integration_ui.py` (propagate section, lines 1288-1461)
- Modify: `tests/test_stereo.py` (`test_propagate_multiselect_both_sides`)

**Interfaces:**
- Consumes: Task 3's queue/meta/worker shapes and the updated `FakePropagateWorker` (`kw["seeds"]`).

- [ ] **Step 1: Update the propagate tests in `tests/test_integration_ui.py`**

a. `test_propagate_seed_starts_worker` — replace the assertions with:

```python
def test_propagate_seed_starts_worker(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    assert len(FakePropagateWorker.instances) == 1
    w = FakePropagateWorker.instances[-1]
    assert w.isRunning()
    assert w.kw["start_frame_idx"] == 0
    assert w.kw["seeds"] == [{"track_id": seed_tid, "cat_id": 0,
                              "bbox_xyxy": [10, 10, 40, 30]}]
    assert win._propagate_meta["seeds"][0]["track_id"] == seed_tid
    assert win.side.sam3_status.text().startswith(
        f"SAM3: propagate T{seed_tid}:")
```

b. `test_propagate_queues_behind_running_job` — replace the `_start_propagate_worker` call and queue assertion with:

```python
def test_propagate_queues_behind_running_job(lr, propagate_seeded):
    win, coco, seed_ann, seed_tid = propagate_seeded
    win._start_propagate_worker(
        1, [{"track_id": 99, "cat_id": 0, "bbox_xyxy": [0, 0, 5, 5]}],
        side="left")
    assert len(win._sam3_queue) == 1
    assert win._sam3_queue[0]["kind"] == "propagate"
    assert win._sam3_queue[0]["seeds"][0]["track_id"] == 99
    assert len(FakePropagateWorker.instances) == 1  # not started
    win._sam3_queue.clear()
```

c. `test_propagate_frame_results_apply_seed_track` — change the three `_on_propagate_frame_done` calls to list form and the finished call/assertion:

- `win._on_propagate_frame_done(1, det1)` → `win._on_propagate_frame_done(1, [det1])`
- `win._on_propagate_frame_done(2, det1)` → `win._on_propagate_frame_done(2, [det1])`
- `win._on_propagate_frame_done(3, None)` → `win._on_propagate_frame_done(3, [None])`
- `win._on_propagate_finished(1, 3)  # lost at frame 3` → `win._on_propagate_finished(1, {0: 3})  # lost from frame 3`
- the status assertion becomes:

```python
    assert f"propagate T{seed_tid}: done — 1 box(es) added " \
           f"(T{seed_tid} lost at frame 4)" in win.side.sam3_status.text()
```

d. `test_propagate_queued_dispatch_after_run_ends` — replace the queued dict and the trailing assertions:

```python
    win._sam3_queue.append({"kind": "propagate", "start_frame_idx": 2,
                            "seeds": [{"track_id": 7, "cat_id": 0,
                                       "bbox_xyxy": [0, 0, 5, 5]}],
                            "side": "left"})
    w3.stop()
    win._on_propagate_finished(0, {})  # ends the run, drain via direct call
    win._start_next_queued_sam3()
    w4 = FakePropagateWorker.instances[-1]
    assert w4 is not w3 and w4.isRunning()
    assert w4.kw["start_frame_idx"] == 2
    assert win._propagate_meta["seeds"][0]["track_id"] == 7
```

e. Replace `test_propagate_multi_select_queues_one_job_per_track` (multi-select now starts ONE job with all seeds) with:

```python
def test_propagate_multi_select_single_job_all_seeds(lr, propagate_win):
    """Selecting two boxes of different tracks propagates both in ONE
    memory-bank session (no queue)."""
    win, coco = propagate_win
    image_id = coco.ensure_image(FakeIdx().frame_at(0), 100, 80)
    coco.add_box(image_id, 10, 10, 30, 20, 0)
    coco.add_box(image_id, 50, 40, 20, 20, 0)
    win._refresh_boxes()
    win.canvas._multi_selected = {0, 1}
    win.canvas._selected_idx = 1
    win._on_propagate_track()
    assert len(FakePropagateWorker.instances) == 1
    w = FakePropagateWorker.instances[-1]
    assert w.isRunning()
    assert len(w.kw["seeds"]) == 2
    assert len(win._sam3_queue) == 0  # nothing queued — one job, two seeds
```

f. `test_propagate_multi_select_dedupes_shared_track` — keep the body, change the final assertions to:

```python
    win._on_propagate_track()
    assert len(FakePropagateWorker.instances) == 1
    assert len(FakePropagateWorker.instances[-1].kw["seeds"]) == 1
    assert len(win._sam3_queue) == 0
```

- [ ] **Step 2: Strengthen the stereo cross-side test**

In `tests/test_stereo.py` `test_propagate_multiselect_both_sides`, after the existing assertions add:

```python
    # each side's job carries exactly its own side's seed
    left_tid = coco.anns_for_image(win.canvas._image_id)[0]["track_id"]
    right_tid = coco.anns_for_image(win.canvas_right._image_id)[0]["track_id"]
    assert [s["track_id"] for s in
            FakePropagateWorker.instances[0].kw["seeds"]] == [left_tid]
    assert [s["track_id"] for s in
            win._sam3_queue[0]["seeds"]] == [right_tid]
```

- [ ] **Step 3: Run the GUI test files**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/test_integration_ui.py tests/test_stereo.py -q`
Expected: all passed

---

### Task 5: Dead-config cleanup + full verification

**Files:**
- Modify: `gui/label_review/ui/dialogs.py` (spin-box creation lines 142-161; prefill lines 278-279; `_prefill_from_config` lines 311-316; `_collect` lines 361-363)
- Modify: `scripts/config/label_review.example.json` (lines 15-16)
- Modify: `gui/label_review/main.py` (docstring lines 122-138)

- [ ] **Step 1: Remove the IoU spin boxes from the settings dialog**

In `gui/label_review/ui/dialogs.py` delete:

- the creation of `self.spin_propagate_iou` and `self.spin_propagate_seed_iou` including their `addRow` calls (lines 142-161),
- the two prefill lines in `_prefill_from_window`:

```python
        self.spin_propagate_iou.setValue(win.sam3_propagate_min_iou)
        self.spin_propagate_seed_iou.setValue(win.sam3_propagate_seed_iou)
```

- the two blocks in `_prefill_from_config` (lines 311-316),
- the two keys in `_collect` (`"propagate_min_iou": ...` and `"propagate_min_seed_iou": ...`, lines 361-363).

- [ ] **Step 2: Remove the keys from the example config**

In `scripts/config/label_review.example.json` delete:

```json
    "propagate_min_iou": 0.3,
    "propagate_min_seed_iou": 0.2
```

- [ ] **Step 3: Rewrite the docstring section in `gui/label_review/main.py`**

Replace the "SAM3 track propagation" section body (lines 124-138, keep the header/underline) with:

```
With one or more boxes selected, "Propagate →" tracks every selected
object through the following frames with SAM3's video memory bank
(SAM3VideoPredictor): the selected boxes seed ONE video session per
side (in stereo the left and right selections run as two back-to-back
jobs through the SAM3 queue), and each object keeps its seed's track id
and category. Object identity comes from the memory bank — there is no
per-frame re-detection or IoU chaining. A track that yields an empty
mask is reported as lost from that frame on (the memory bank may still
recover it on later frames); frames that already have a box with that
track id are skipped. Propagated boxes are ordinary editable boxes with
masks plus ``propagated: true`` / ``confidence`` provenance fields; each
side's run is one Ctrl+Z step, and edits you make while it runs stay
separate undo entries. Propagation shares the SAM3 queue and the Cancel
button.
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /home/wangyiming/code/object_detection_app && python -m pytest tests/ -q`
Expected: all passed (baseline before this plan: 141 passed + the new engine/worker tests)

- [ ] **Step 5: Grep for stragglers**

Run: `cd /home/wangyiming/code/object_detection_app && grep -rn "propagate_min_iou\|propagate_min_seed_iou\|_propagate_step\|PROPAGATE_MIN" --include="*.py" --include="*.json" gui scripts tests | grep -v docs/superpowers`
Expected: no matches

---

## Self-Review Notes

- Spec coverage: engine (Task 1), worker (Task 2), GUI plumbing incl. dead-config removal in main_window (Task 3), dialogs/example-json/main.py cleanup (Task 5), all test updates (Tasks 1/2/4). The spec's "both sides, one session per side" requirement is Task 3 Step 3 + Task 4 Step 2.
- Type consistency: `seeds` dicts carry `bbox_xyxy`/`track_id`/`cat_id` everywhere (worker ctor, queue dict, meta, tests). `finished_signal(int, object)` — old GUI handler signature `(self, n_found, lost_at)` replaced everywhere. `frame_done_signal(int, object)` second arg is a list from Task 2 onward; only `_on_propagate_frame_done` consumes it (updated Task 3 Step 5).
- No git commits anywhere in this plan (environment policy overrides the skill's commit steps).

# Design: SAM3 memory-bank tracking for "Propagate →"

Date: 2026-08-20 (revised same day: multi-seed per side)
Status: approved-in-auto-mode (presented in chat; user refined scope:
one session per side covering ALL of that side's selected boxes)

## Problem

The GUI's track propagation (`SAM3PropagateWorker`) re-detects the object on
every frame with single-image SAM3, chaining the previous frame's box as the
next exemplar and gating on IoU (`propagate_min_iou` vs previous box,
`propagate_min_seed_iou` vs seed box). This ignores SAM3's video memory bank,
drifts easily, and stops permanently on the first weak frame.

Additionally, selected tracks are propagated one at a time (queued single-box
jobs). SAM3VideoPredictor accepts multiple seed boxes in one session, so one
run should cover all selected boxes of a side.

## Decision

Use ultralytics `SAM3VideoPredictor` (video memory bank) instead of per-frame
re-detection. Pressing "Propagate →" starts **one job per side** (left /
right), each seeding **all selected boxes of that side** in a single video
session. In mono: one job with all selected boxes.

Rejected alternatives:

- Raw `sam3` package (`build_sam3_video_predictor`, frames-folder sessions):
  the `sam3` package is not installed; ultralytics (installed, 8.4.83) wraps
  the same model. No new dependency.
- High-level `predictor(source=video, bboxes=..., stream=True)` streaming
  API: `SAM2VideoPredictor.inference()` filters blank masks and
  `postprocess()` re-indexes `cls` after filtering, so Results do NOT carry
  stable object ids — a lost object shifts the positional mask↔seed
  alignment for the rest of the frame. Unusable for multi-object.
  → Use the low-level session API instead, mirroring the high-level
  driver's per-frame flow but taking outputs from `propagate_in_video`,
  which returns `(obj_ids, pred_masks, obj_scores)` with masks aligned to
  `obj_ids` and no blank-mask filtering.
- Keeping IoU chaining: this is the behavior being replaced.

## Constraints discovered (from installed ultralytics source)

- Video mode requires a real **video file** source. A directory of frame
  images loads with `dataset.mode == "image"` and
  `SAM2VideoPredictor.init_state` asserts `mode == "video"`. So each
  propagate run builds a temporary mp4 from the frame range (exactly what
  `track_sam3_video.py` does).
- `init_state()` returns early when `inference_state` is non-empty, so
  predictor instances do **not** reset between runs. v1 constructs a fresh
  `SAM3VideoPredictor` per run (correctness over model-load latency); caching
  with a manual `inference_state = {}` reset is a future optimization.
- Low-level flow (verified against ultralytics 8.4.83 on GPU):
  `setup_model()` → `setup_source(video)` → `init_state(predictor)` → per
  frame: take the next dataset batch (the loader advances `dataset.frame`
  after each read, so predictor-internal frame indices are **1-based**),
  set `predictor.batch = batch` (inference/prompt prep reads it),
  `preprocess`, store into `inference_state["im"]` so the tracker extracts
  features from the CURRENT frame → on the first frame register each seed
  box via `_prepare_prompts` + `add_new_prompts(obj_id=i, points=...,
  frame_idx=frame)` (the box-prompt path the high-level API uses) →
  `propagate_in_video_preflight()` → `propagate_in_video(inference_state,
  frame)` → `(obj_ids, pred_masks, obj_scores)`, masks possibly at internal
  resolution (resize NEAREST to frame size). pred_masks are raw **logits** —
  binarize with `> model.mask_threshold` (0.0); a plain bool cast treats
  negative logits as foreground. An all-zero mask = object lost
  on that frame (the memory bank may recover it on later frames — a strict
  improvement over today's permanent stop).
- The `track_sam3_video.py` "reseed"-mode mask-prompt path (downsample seed
  masks to `_bb_feat_sizes[0]`, `add_new_prompts(masks=...)`) is broken in
  ultralytics 8.4.83 — `_consolidate_temp_output_across_obj` rejects the
  mask size. Box prompts alone suffice.
- `obj_scores` are raw logits; store `sigmoid(logit)` as the box's
  `confidence` provenance field so it stays in 0..1.

## Components

### 1. Engine: `sam3_video_propagate()` in `core/models_inference.py`

Generator:

```python
def sam3_video_propagate(
    video_path: str | Path,
    seed_bboxes_xyxy: List[List[float]],
    model_path: Optional[str | Path] = None,   # default: core/sam3/models/sam3-model/sam3.pt
    device: str = "cuda",
    conf: float = 0.25,
    imgsz: int = 1024,
    quantize: Optional[int] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Iterator[Tuple[int, List[Tuple[Optional[np.ndarray], Optional[List[float]], float]]]]
```

Yields `(video_frame_idx, per_seed)` from frame 0 (the seed frame), where
`per_seed[i]` is `(mask_bool|None, bbox_xyxy|None, score)` for seed i —
identity preserved via `obj_ids`, never positionally. Implements the
low-level recipe above. Checks `is_cancelled()` between frames (generator
returns; `finally` cleanup runs). `try/finally`: `del predictor` +
`torch.cuda.empty_cache()` when CUDA is available (mirrors `process_chunk`
in track_sam3_video.py). Ultralytics/torch imports stay lazy (inside the
function), matching `_get_sam_predictor`'s pattern so `_SAM3_AVAILABLE`
semantics are unchanged.

### 2. Worker rewrite: `SAM3PropagateWorker`

File: `gui/label_review/workers/label_review_workers.py`

- Constructor: `(frame_index, start_frame_idx, seeds, tmp_dir, model_path,
  device, conf, parent=None)` where `seeds` is
  `List[Dict]` with keys `bbox_xyxy` (xyxy floats), `track_id` (int),
  `cat_id` (int). Drops `concept`, `min_iou`, `min_seed_iou`.
- Signals:
  - `frame_done_signal(int, object)` — `(frame_idx, dets)` where `dets` is a
    list aligned with `seeds`, each `{"bbox_xyxy", "mask", "confidence"}`
    or `None` (seed lost on that frame).
  - `progress_signal(int, int)` — unchanged.
  - `finished_signal(int, object)` — `(boxes_found_total,
    lost_map)` where `lost_map: Dict[int, int]` maps seed index → first
    all-empty frame idx, but only for seeds that were still lost at the end
    of the clip (recovered objects are not reported).
  - `failed_signal(str)`, `cancelled_signal()` — unchanged.
- `run()`:
  1. Materialize frames `start_frame_idx .. len-1` into
     `tmp_dir/propagate_clip.mp4` (queue serializes runs, so the fixed name
     cannot clobber): `cv2.imread(frame.file_path)` when present, else
     `decode_image(idx)` (RGB → `cv2.cvtColor(..., COLOR_RGB2BGR)`).
     VideoWriter `mp4v`, fps 30, size from the first frame. Cooperative
     cancel per frame.
  2. Stream `sam3_video_propagate(video, [s["bbox_xyxy"] for s in seeds],
     ...)`; video frame k → `frame_idx = start_frame_idx + k`. k == 0 is the
     seed frame: consumed, not emitted (seed boxes already exist).
  3. k ≥ 1: emit `frame_done_signal(frame_idx, dets)` + progress; track
     per-seed last-nonempty frame for `lost_map`; count found boxes.
  4. End → `finished_signal(n_found, lost_map)`; cancel →
     `cancelled_signal()`; exception → `failed_signal(f"frame {idx+1}: {e}")`.
  5. All exits: release writer, remove the temp mp4.
- `_propagate_step`, `_PROPAGATE_MIN_IOU`, `_PROPAGATE_MIN_SEED_IOU` are
  deleted. `_iou_xyxy` stays (used by autolabel dedup in main_window).

### 3. GUI plumbing: `gui/label_review/ui/main_window.py`

- `_on_propagate_track`: seeds are collected from BOTH sides (existing
  `_selected_boxes_all_sides`), deduped per `(side, track_id)`, fresh track
  ids assigned after confirmation (unchanged), then **grouped by side**: one
  `_start_propagate_worker(self._current_idx, seed_dicts, side)` call per
  side. The second side queues behind the first via the existing busy-queue.
  Confirm dialog unchanged (already lists all seeds with `[side]` tags).
- `_start_propagate_worker(start_frame_idx, seeds, side)`: new signature;
  queue dict becomes `{"kind": "propagate", "start_frame_idx", "seeds",
  "side"}`. `_propagate_meta` becomes `{"side", "seeds", "added",
  "ann_ids", "anns"}`. Status: one seed → `propagate T{id}: x/y frames…`
  (unchanged); several → `propagate {n} tracks [{side}]: x/y frames…`
  (side tag only in stereo).
- `_on_propagate_frame_done(frame_idx, dets)`: loops `zip(meta["seeds"],
  dets)`; per-det logic unchanged (skip when the frame already has that
  track id, muted `add_box(track_id=...)`, `propagated`/`confidence` fields,
  `set_mask`, 10-frame checkpoint, current-frame refresh).
- `_on_propagate_finished(n_found, lost_map)`: status `propagate: done —
  N box(es) added` plus `T{id} lost at frame {f+1}` notes for lost-map
  entries. `_on_propagate_failed` / `_on_propagate_cancelled` lose the
  single-`T{id}` phrasing (use the same run label helper).
- `_end_propagate` undo label: one seed → unchanged
  `propagate track T{id} (N box(es))`; several →
  `propagate {m} tracks (N box(es))` (one Ctrl+Z step per side-run).
- Dead config: remove `sam3_propagate_min_iou` / `sam3_propagate_seed_iou`
  attrs, their `_apply_config` entries, and the worker kwargs.

### 4. Dead-config cleanup elsewhere

- `dialogs.py`: remove `spin_propagate_iou` / `spin_propagate_seed_iou`
  (creation, prefill, `_prefill_from_config`, `_collect`).
- `scripts/config/label_review.example.json`: remove the two keys.
- `main.py`: rewrite the "SAM3 track propagation" docstring section
  (memory-bank tracking, one session per side, lost-track reporting,
  no IoU settings).

## Data flow (per propagate run)

```
selected boxes (both sides, _selected_boxes_all_sides)
  → group by side → one _start_propagate_worker per side (queued)
  → SAM3PropagateWorker (one SAM3VideoPredictor session per side)
      frames[start..end] → tmp mp4
      → sam3_video_propagate(): obj_id-aligned masks per frame
      → frame_done_signal(frame_idx, dets aligned with seeds)
      → _on_propagate_frame_done: add_box per seed's track_id + set_mask
```

## Error handling

- SAM3 unavailable → `failed_signal("SAM3 is not installed.")` (unchanged).
- Unreadable frame during mp4 build / VideoWriter open failure →
  `failed_signal("preparing video: ...")`.
- Engine exception → `failed_signal(f"frame {frame_idx + 1}: {e}")`.
- Cancel during mp4 build or streaming → `cancelled_signal`, temp mp4
  removed.
- No CUDA-OOM→CPU retry in v1 (the old `_propagate_step` had one); an OOM
  surfaces as a failed run with the error message. Follow-up if needed.

## Testing

Repo policy: no test runs real SAM3/torch inference.

- New `tests/test_unit_models_inference.py`: engine driven with a fake
  `ultralytics.models.sam.SAM3VideoPredictor` (monkeypatched) — obj_id
  alignment, blank-mask → `(None, None, 0.0)`, lost-object recovery keeps
  seed positions, cancel stops the generator, overrides/model-path defaults.
- `tests/test_unit_workers.py`: delete the five `_propagate_step` tests;
  rewrite the worker tests against a monkeypatched `sam3_video_propagate`
  (frame mapping, seed-frame skip, lost_map, cancel, mp4 cleanup).
  `test_iou_xyxy` stays.
- `tests/conftest.py`: `FakePropagateWorker` ctor matches the new signature
  (`seeds` list; no `min_iou`/`min_seed_iou`).
- `tests/test_integration_ui.py`: propagate tests updated to the new
  contract (seeds list, list-valued frame_done, finished lost_map);
  multi-select on one side now starts ONE job with both seeds (no queue).
- `tests/test_stereo.py`: cross-side multi-select starts two jobs — left
  runs, right queued — each with its own side's seeds.

## Out of scope

- Predictor instance caching across runs (needs on-GPU verification of
  state reset; follow-up).
- Attaching the seed frame's masks to the seed boxes.
- Per-track undo entries within a side-run (the run is one Ctrl+Z step).

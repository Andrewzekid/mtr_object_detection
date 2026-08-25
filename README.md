# Object Detection Application

A desktop application and CLI pipeline for object detection / segmentation
training and inference, built with PyQt6 and Ultralytics YOLO.

---

## Quickstart — one complete run, start to finish

Everything below is paste-able top to bottom. It takes raw footage to a
trained segmentation model + tracking results in a single pipeline run.

### Step 0 — install

```bash
pip install -r requirements.txt
# SAM3 weights for GUI segmentation/autolabel (one-time):
#   place sam3.pt at core/sam3/models/sam3-model/sam3.pt
```

### Step 1 — start the Qwen VLM server (seed labels)

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
    --mmproj Qwen3.8-mmproj-F16.gguf \
    --image-min-tokens 2048 --port 8089
```

### Step 2 — run the orchestrator (blocks at the GUI)

From a Metacam rosbag, stereo:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
  --camera both \
  --sample-size 1000 --keyframe-stride 10 \
  --llamacpp-url http://127.0.0.1:8089 \
  --ratios 0.7 0.15 0.15 --split-seed 42 \
  --augmentations flip_horizontal rotate brightness --multiplier 2 \
  --epochs 100 --batch-size 16 --model-type yolo26n --task segment --imgsz 768 \
  --eval-splits test train --tracker deepocsort --device 0
```

Or from an existing image folder (mono; for stereo point `--images` at the
parent containing `left/` + `right/` and pass `--camera both`):

```bash
python scripts/orchestrate_pipeline.py --images Datasets/HKU_GH/HKU_GH_left
```

The pipeline runs automatically through:
`undistort → sample → keyframes → stats → qwen → qwen_coco` and then opens
the **label-review GUI** on the keyframes.

### Step 3 — review labels in the GUI (the only human step)

Fix/remove wrong boxes, segment with SAM3 (`Re-segment`, point prompt 🎯,
Propagate), autolabel missing categories, discard bad frames, then **save**
and close. The pipeline resumes by itself.

If you closed without saving, re-run just the GUI stage:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag .../2026-08-20_22-06-52 --camera both --stage gui
```

After a review session you can also re-run only everything downstream:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag .../2026-08-20_22-06-52 --camera both \
  --stage yolo --stage split --stage augment --stage assemble \
  --stage train --stage evaluate --stage tracking
```

### Step 4 — collect the outputs

Under `<input>_pipeline/`:

| Path | Contents |
|------|----------|
| `reviewed/labels_coco.json` | human-reviewed COCO annotations |
| `dataset/final/` (+ `dataset.yaml`) | final train/val/test dataset |
| `dataset/dataset_statistics.csv` | per-class instance statistics |
| `training/yolo_training/weights/best.pt` | trained model |
| `evaluation/metrics.csv` | per-class P/R/F1/AP metrics |
| `tracking/<cam>/` | tracked frames, `results.json`, video |

---

## Overview
Label review and segmentation:
```
python -m gui.label_review.main
# stereo dual-view (left/right folders, frames paired by timestamp filename;
# frames without a matching timestamp on the other side are skipped).
# The saved COCO JSON lists every timestamp-synced image (sorted earliest
# first) — not just the frames you viewed or annotated.
python -m gui.label_review.main --images /path/to/left --images-right /path/to/right
```
Launches interactive GUI for complete label review and segmentation. Point it
at the keyframe folder of a pipeline run to review the combined Qwen output
(it auto-loads a `labels_coco.json` found next to the images):

```bash
python -m gui.label_review.main --images output/<run>/keyframes/left
```

### Autolabel backends

The GUI can pre-fill boxes/masks for the session categories
(**Autolabel frame** / **Autolabel ALL frames**). Pick the backend under
Settings → Autolabel (persisted as `autolabel.detector` in the GUI config):

| Backend | Output | Notes |
|---|---|---|
| `sam3` | boxes + masks | text prompts, local Ultralytics SAM3 checkpoint |
| `owlv2` | boxes | zero-shot, `google/owlv2-large-patch14-ensemble` (HF) |
| `owlv2_exemplar` | boxes | 1-shot: select an existing box first — its crop becomes the visual query (`image_guided_detection`) |
| `grounding_dino` | boxes | zero-shot; default `IDEA-Research/grounding-dino-base` (the `-large` checkpoint is not published in transformers format) |
| `florence2` | boxes | phrase grounding per category via `microsoft/Florence-2-large`; no confidence scores (all 1.0) |
| `falcon` | boxes + masks | `tiiuae/Falcon-Perception`; free-form text query per category |

Per-backend config keys (Settings dialog or config JSON): `owlv2_model` /
`owlv2_conf` (default 0.3), `gdino_model` / `gdino_conf` (0.35, maps to the
box threshold), `florence2_model`, `falcon_model`. First use downloads the
checkpoint from Hugging Face; models are cached per session.

Two compatibility notes for the HF backends (handled automatically in
`core/`): Falcon's pre-compiled flex-attention kernels exceed the shared
memory of consumer GPUs (RTX 4090), so they are recompiled with smaller
blocks at load; Florence-2 runs its 2023-era remote code under
transformers 5.x via shims plus a local beam search (`model.generate` is
unusable there — see `core/detectors.py`).

### Labelling assist features (when and how to use them)

The review GUI is built so a human never labels every frame by hand. Typical
session: let Qwen seed boxes on the keyframes, open the GUI, then combine
the assistants below depending on the footage.

- **Autolabel (frame / ALL frames)** — open-set detectors fill in boxes (+
  masks for SAM3/Falcon) for the session categories. Highlighting 2+
  categories in the list restricts the run to just those. Use it to add a
  category you forgot to prompt Qwen for, or to re-label with a stronger
  open-set model. `owlv2_exemplar` is the 1-shot variant: draw/fix ONE good
  box by hand, select it, then autolabel — best for odd objects text
  prompts miss.
- **Keyframes** — for near-static cameras (corridors, platforms) frames are
  near-duplicates, so label every Nth (`--keyframe-stride N`, N≈10) and let
  interpolation/propagation cover the gaps. Small N for fast motion, large
  N for static scenes. The GUI's ★ Keyframe toggle (K) marks frames that
  anchor interpolation.
- **Interpolate (I)** — linearly/optical-flow interpolates boxes of the
  same track between two labelled frames. Use when motion is **smooth and
  in a constant direction** (walking past a static camera, slow pans).
  Workflow: label a frame, jump ~10–30 frames, label the same objects again,
  select the boxes, Interpolate. Bad under occlusion, direction reversals,
  or objects appearing/disappearing — check the in-between frames after
  running it.
- **Propagate → (SAM3 tracking)** — select a box and let SAM3's video
  memory bank follow that instance forward frame by frame, producing masks.
  Use for deformable/rotating objects or longer ranges where interpolation
  drifts. Two methods in Settings → SAM3: `memory` (default; one video
  session, can recover after temporary loss) and `chain` (re-detect each
  frame from the previous box, IoU-gated; stops permanently at the first
  miss). Stops being reliable under large camera jumps — re-seed from a
  later frame.
- **Re-segment selected** — re-runs SAM3 on a box you moved/resized to get
  a fresh mask. Use after every manual box edit if masks matter.
- **Point segment (🎯 Add points + ▶ Segment points)** — SAM2-style point
  prompting in two steps. With **Add points** toggled ON, left-click adds a
  positive point, right-click a negative point — clicks only accumulate,
  nothing runs yet, so you can place as many points as you want. Press
  **▶ Segment points** to run SAM3 once with the full point set; press it
  again after adding more points to refine. Enter accepts the object,
  Esc cancels it. The selected category is sent to SAM3 as a **text
  prompt**: it detects all instances of that category and your points pick
  which one to keep (pure point prompting is the fallback when the text
  finds nothing there). Pick a category row first — points added before a
  category is picked are kept and used when you press Segment. Use for
  precise single-object masks where a drag-box would include too much
  background.
- **Discard image (🚫)** — drops the frame from the saved COCO entirely.
  Use for blurry/irrelevant frames so they never reach training.
- **Stereo sync** — left/right frames pair by timestamp filename; unmatched
  frames are skipped everywhere (display and save), so a saved stereo COCO
  only contains synced pairs, sorted earliest first.

End-to-end pipeline that takes **raw images** and produces a **trained YOLO
detection/segmentation model** plus **tracking results**, with a
human-in-the-loop review step in the middle. It targets Hong Kong MTR and
industrial (IW) station imagery but is dataset-agnostic.
For a quick segmentation run for the YOLO seg  + Tracking, try the following command:
```
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```
There are two labeling flows:

**Keyframe pipeline** (MAIN — label keyframes in the GUI with Qwen seed
labels, use the GUI's labelling assistants to cover the rest):

```
data folder (or rosbag) → undistort → 00 sample → 12 keyframes → 01a stats
   → 07 Qwen seed labels on keyframes → 08c combine to COCO
   → GUI review (fix boxes, SAM3-segment, autolabel, interpolate/propagate,
     discard bad frames) → 01b COCO→YOLO-seg → 03 train/test/val split
   → 02 augment (train split only, mask-aware) → 04 YOLO seg train
   → 05 evaluate (per-class P/R/F1 CSV) → 11 tracking on raw frames
```

The whole chain runs from one command — see "Orchestrated pipeline" below.

**Full pipeline** (dense labeling — every frame goes through the VLM; slower,
no keyframe step):

```
raw images → 07 Qwen VLM auto-label → 08c combine per-image results into
            labels_coco.json → GUI review (same as above)
            → 01b → 03 split → 02 augment train → 04 train → 05 evaluate
            → 11 tracking
```

**GUI segmentation pipeline** (annotate + segment in the label-review GUI, no
Roboflow round-trip — masks come from SAM3 in the GUI):

```
GUI annotate (SAM3 masks) → COCO labels_coco.json (discard unwanted frames
   in the GUI) → 01b COCO→YOLO-seg → 02 augment → 03 train/test/val split
   → 04 YOLO seg train
```

See "Segmentation dataset pipeline (GUI → YOLO seg)" below for commands, or
run all post-GUI steps at once with `scripts/run_seg_dataset_pipeline.py`.

**Orchestrated pipeline** — the whole keyframe workflow with one command.
Input is either a Metacam rosbag (adds the fisheye undistortion stage) or a
plain image folder:

```bash
# 1. Start the llama.cpp server with the Qwen GGUF + mmproj:
#    llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj Qwen3.8-mmproj-F16.gguf --port 8089

# 2. Run the orchestrator (blocks at the GUI stage; resumes when you close
#    the GUI after saving):
python scripts/orchestrate_pipeline.py \
  --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
  --camera both \
  --sample-size 1000 --keyframe-stride 10 \
  --llamacpp-url http://127.0.0.1:8089 \
  --qwen-model Qwen3.8-27B-Q4_K_M.gguf --qwen-mmproj Qwen3.8-mmproj-F16.gguf \
  --ratios 0.7 0.15 0.15 --split-seed 42 \
  --augmentations flip_horizontal rotate brightness --multiplier 2 \
  --epochs 100 --batch-size 16 --model-type yolo26n --task segment --imgsz 768 \
  --eval-splits test train --tracker deepocsort

# From an existing image folder instead of a rosbag (mono; for stereo point
# --images at the parent of left/ + right/ and pass --camera both):
python scripts/orchestrate_pipeline.py --images Datasets/HKU_GH --camera both

# Or via the shell wrapper (env overrides: LLAMACPP_URL, QWEN_MODEL,
# QWEN_MMPROJ; all extra args pass through):
./scripts/run_pipeline.sh 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 --camera both
```

Stages: `undistort`, `sample`, `keyframes`, `stats`, `qwen`, `qwen_coco`,
`gui`, `yolo`, `split`, `augment`, `assemble`, `train`, `evaluate`,
`tracking`. Run a subset with `--stage <name>` (repeatable) and skip with
`--skip-stage <name>`; completed stages write `stage_completed.json` markers
so re-runs resume where they stopped (`--force` ignores markers). The `gui`
stage launches the label-review GUI on the keyframes (`--gui-on all` loads
every sampled frame instead) and waits — the pipeline continues after you
save and close. Useful per-stage knobs:

- llama.cpp / Qwen: `--llamacpp-url`, `--qwen-model`, `--qwen-mmproj`,
  `--prompt` (class list is parsed from it; or `--classes`), `--resume-from`
- sampling/keyframes: `--sample-size` (stereo: right side is synced to the
  sampled left timestamps), `--sample-seed`, `--keyframe-stride`
- dataset: `--ratios TRAIN TEST VAL`, `--split-seed`, `--bbox-as-rect`
- augmentation (train split only): `--augmentations` (flip_horizontal,
  flip_vertical, rotate, brightness, contrast, hue, blur, resize, mosaic),
  `--multiplier`, `--rotation-range`, `--brightness-range`, `--hue-range`,
  `--blur-range`, `--resize W H`, `--skip-augment`
- training: `--epochs`, `--batch-size`, `--model-type`, `--task`,
  `--imgsz`, `--pretrained`, `--lr0`, `--device`
- evaluation: `--eval-splits`, `--eval-conf`, `--eval-iou`
- tracking: `--tracker`, `--track-conf`, `--track-iou`, `--track-imgsz`,
  `--track-fps`, `--track-max-frames`

Outputs under `<input>_pipeline/`: `undistorted/`, `sampled/`, `keyframes/`,
`qwen/`, `reviewed/labels_coco.json`, `dataset/{yolo_flat,split,
train_augmented,final}` (+ `dataset.yaml` and `dataset_statistics.csv`),
`training/yolo_training/weights/best.pt`, `evaluation/metrics.csv`,
`tracking/<camera>/`.


### Directory layout

| Path | Contents |
|------|----------|
| `scripts/` | Numbered CLI entry points (pipeline stages) + helpers. |
| `core/` | Reusable OOP classes / headless inference wrappers the scripts build on. |
The numbered `scripts/` are thin CLI wrappers over `core/` classes
(`ModelTrainer`, `ModelEvaluator`, `DataProcessor`, `DatasetCreator`,
`ModelVisualizer`) and inference wrappers (`run_qwen`, `run_qwen_api`,
`run_sam3`, `run_gemini` in `core/models_inference.py`). A full per-script
reference with inputs/outputs is at the bottom of this file.

---

## Prerequisites

```bash
pip install -r requirements.txt        # loose bounds
# or exactly reproduce the dev machine:
pip install -r requirements-lock.txt
```

### Standalone bundle (no Python needed on the target)

The label review app can be frozen into a single-folder executable with
PyInstaller (Linux "exe"):

```bash
pyinstaller scripts/label_review.spec --distpath dist --workpath build --noconfirm
# result: dist/label-review/label-review
```

> **Note:** the spec is stale — it still targets the old monolithic
> `scripts/09_label_review.py`. It needs updating for the new
> `gui/label_review/` package layout (entry point
> `python -m gui.label_review.main`) before this works again.

Copy the `dist/label-review/` folder to the target machine (e.g.
`tar czf label-review.tar.gz -C dist label-review`) and run:

```bash
./label-review --images /path/to/frames --output_json out/coco.json \
    --sam3-model /path/to/sam3.pt
```

Notes:

- SAM3 weights are not bundled (3.4 GB) — pass `--sam3-model` or place
  `core/sam3/models/sam3-model/sam3.pt` under the working directory.
- The bundle inherits this machine's torch build: a CUDA torch bundle needs
  a compatible NVIDIA driver on the target; build in a CPU-torch env for
  CPU-only targets.
- On a fresh Linux install PyQt6 needs system GL/X libraries:
  `sudo apt install libgl1 libegl1 libxkbcommon0 libdbus-1-3`.


- **Qwen3.6 labeling** can run locally with Ollama or via DashScope API.
  The examples below use the DashScope API. Set your key first:

  ```bash
  export API_KEY=your_dashscope_key
  ```

- **SAM3 segmentation** needs the Ultralytics SAM checkpoint. Place it at
  `core/sam3/models/sam3-model/sam3.pt` (or pass `--model` explicitly).

  ```

---

## Complete pipeline example: `Datasets/MTR/MTR_new_1k`

The commands below run the whole workflow from raw images in
`Datasets/MTR/MTR_new_1k` to a trained model and tracked results.
You can paste each block into the terminal from top to bottom.

### 0. (Optional) select new unlabeled images

Skip this if you already have a folder of images to label.

```bash
python scripts/00_sample_from_dataset.py \
  --out-dir Datasets/MTR/MTR_new_1k

#Example with only 10 images
python scripts/00_sample_from_dataset.py \
  --source-dir Datasets/MTR/MTR_new_1k --out-dir Datasets/MTR/MTR_new_10_images -n 10 --copy
```

### 1. Auto-label with Qwen3.6

```bash
python scripts/07_run_qwen.py \
  --use-api --api-key-env API_KEY --api-model qwen3.6-flash \
  --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions." \
  --template object_detection --format json \
  --image-folder Datasets/MTR/MTR_new_1k \
  --output Datasets/MTR/MTR_new_1k/qwen_results \
  --vis-output output/MTR_new_1k/qwen_vis

#Example with only 10 images
python scripts/07_run_qwen.py \
  --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions." \
  --template object_detection --format json \
  --image-folder Datasets/MTR/MTR_new_10_images \
  --output Datasets/MTR/MTR_new_10_images_annotations \
  --vis-output output/MTR_new_10_images/qwen_vis

#Per image labels
python scripts/07_run_qwen.py   --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions."   --template object_detection --format json   --image-folder Datasets/MTR/MTR_new_10_images   --output Datasets/MTR/MTR_new_10_images_annotations   --vis-output output/MTR_new_10_images/qwen_vis --per-class --conditioning-images ./ref_images --per-image-labels 

python scripts/07_run_qwen.py   --prompt "Detect all: Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions."   --template object_detection --format json   --image-folder Datasets/MTR/MTR_new_10_images   --output Datasets/MTR/MTR_new_10_images_annotations   --vis-output output/MTR_new_10_images/qwen_vis --per-class --conditioning-images ./ref_images --per-image-labels 

```

Output:

- `Datasets/MTR/MTR_new_1k/qwen_results/<image_stem>_result.json`
- `Datasets/MTR/MTR_new_1k/qwen_results/summary.json`
- `output/MTR_new_1k/qwen_vis/`

> `08_click_review_coco.py` auto-detects both the default flat `*_result.json`
> layout and the `--split-by-class` per-image/per-class layout, so either Qwen
> output format works in the review step.

### 2. Review and clean the boxes with `08_click_review_coco.py`

```
python scripts/08_click_review_coco.py \
  --qwen-annotations-dir output/annotations/MTR_4k \
  --img_dir Datasets/MTR/MTR_4k_dataset_exit_signs \
  --output_json output/MTR_4k/reviewed/coco_reviewed.json \
  --output-yolo-dir output/MTR_4k/reviewed/yolo_reviewed \
  --data-yaml Datasets/MTR/MTR_4k/detect/train_yolo_detection/data.yaml

python scripts/08_click_review_coco.py \
  --qwen-annotations-dir Datasets/MTR/MTR_new_10_images_annotations \
  --img_dir Datasets/MTR/MTR_new_10_images \
  --output_json output/MTR_new_10_images/reviewed/coco_reviewed.json \
  --output-yolo-dir output/MTR_new_10_images/reviewed/yolo_reviewed \
  --data-yaml Datasets/MTR/detect/train_yolo_detection/data.yaml

```

Interactive controls in the matplotlib window:

| Key | Action |
|-----|--------|
| click | select a box |
| `D` | delete selected box |
| `A` | add a new box (draw, then press digit for class) |
| `N` | next image |
| `B` | previous image |
| `S` | save and quit |
| `Q` | quit (progress is saved in `.progress`) |

Output:

- `output/MTR_new_1k/reviewed/coco_reviewed.json`
- `output/MTR_new_1k/reviewed/yolo_reviewed/` (images, labels, data.yaml)


### 3. Split the reviewed dataset into train/val/test

`09_create_seg_dataset.py` expects a split layout (`images/{train,val,test}/`,
`labels/{train,val,test}/`), but `08_click_review_coco.py` emits a flat
`yolo_reviewed/` dataset. This intermediate step splits it and writes a
`data.yaml` carrying the class names from the reviewed dataset.

```bash
python scripts/08b_split_reviewed_dataset.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_reviewed \
  --output-dir output/MTR_new_1k/reviewed/yolo_split \
  --ratios 0.7 0.15 0.15 --seed 42

python scripts/08b_split_reviewed_dataset.py \
  --input-dir output/MTR_new_10_images/reviewed/yolo_reviewed \
  --output-dir output/MTR_new_10_images/reviewed/yolo_split \
  --ratios 0.7 0.15 0.15 --seed 42
```

Output layout:

```
output/MTR_new_10_images/reviewed/yolo_split/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

### 4. Convert detection boxes to SAM3 segmentation masks

```bash
python scripts/09_create_seg_dataset.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_split \
  --output-dir output/MTR_new_1k/reviewed/yolo_seg \
  --model core/sam3/models/sam3-model/sam3.pt \
  --conf 0.4 --device cuda

python scripts/09_create_seg_dataset.py \
  --input-dir output/MTR_new_10_images/reviewed/yolo_split \
  --output-dir output/MTR_new_10_images/reviewed/yolo_seg \
  --model core/sam3/models/sam3-model/sam3.pt \
  --conf 0.4 --device cuda
```

Output layout:

```
output/MTR_new_1k/reviewed/yolo_seg/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
└── creation_summary.json
```

### 5. Upload to Roboflow (optional)

Upload to roboflow and clean the segmentation masks. Afterwards, create a new dataset on roboflow and do data augmentation. Download the dataset and use for training model

### 6. Train a YOLO segmentation model

```bash
python scripts/04_train_model.py \
  --config output/MTR_new_10_images/reviewed/yolo_seg/data.yaml \
  --model-type yolo26l --task segment --loss-type focal \
  --epochs 1000 --batch-size 32 --device 0 --imgsz 640
```

Trained weights are written to
`runs/segment/output/training/yolo_training/weights/best.pt` by default.

### 7. Evaluate the trained model

```bash
python scripts/05_evaluate_model.py \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data output/MTR_new_1k/reviewed/yolo_seg/data.yaml \
  --split test train --conf 0.5 --iou 0.5 \
  --csv output/evaluation/per_class_metrics.csv
```

Evaluates each split (default: `test train`) and reports per-class
precision / recall / F1 / AP50 / AP50-95 (box and mask for seg models);
`--csv` writes one row per split × class plus an `all` summary row.
For dataset statistics (images, per-class instance counts and
percentages) use `scripts/01a_dataset_statistics.py`:

```bash
python scripts/01a_dataset_statistics.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_seg \
  --csv dataset_statistics.csv
```

### 8. Run tracking on the original images
If data is from metcam, need to first undistort the result. Undistortion example:
```bash
python scripts/undistort_rosbag.py --images-root Datasets/iw/tracking/IW_run2/camera/left --output-root Datasets/iw/tracking/IW_run2_left_undistorted --calibration Datasets/iw/tracking/IW_run2/info/calibration.json --camera-name left

python scripts/undistort_rosbag.py --images-root Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag/camera/left --output-root Datasets/MTR/tracking/MTR_left_undistorted --calibration Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag/info/calibration.json --camera-name left

```
```bash
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/MTR_new_1k \
  --output output/MTR_new_1k/tracking \
  --conf 0.5 --device 0

#Examples:
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/MTR_segmentation/weights/best.pt \
  --data Datasets/MTR/tracking/MTR_left_undistorted \
  --conf 0.4 --device 0 \
  --warmup-frames 3 \
  --output output/tracking/MTR/metacam/left/tracking1

python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```

Tracking output:

- `output/MTR_new_1k/tracking/tracked_*.jpg`
- `output/MTR_new_1k/tracking/results.json` — COCO-style tracking results
- `output/MTR_new_1k/tracking/tracking_result.mp4` — summary video

---

### Benchmark mode (no output files, reports timing)

Run YOLO tracking + segmentation without saving images, JSON, or video, and
measure the time split between the model+tracker call and the result
post-processing.

```bash
python scripts/11_run_tracking.py \
  --tracker benchmark \
  --benchmark-tracker botsort \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/MTR_new_1k \
  --conf 0.5 --device 0 \
  --warmup-frames 3

#Examples:
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/MTR/tracking/MTR_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/MTR/metacam/left/tracking1

python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```

It prints mean / min / max / p50 / p95 / p99 latencies and FPS for:

- `YOLO model + tracker time`
- `Tracking result processing time`
- `Combined per-frame time`

Speed options for benchmark mode:

- `--half` — FP16 inference on CUDA.
- `--no-masks` — skip reading segmentation masks from results.
- `--no-cmc` — disable BoT-SORT camera-motion compensation.
- `--imgsz 320` — smaller input resolution (trades accuracy for speed).
- `--trt` — export the model to a TensorRT engine once (FP16, at `--imgsz`) and
  track with it; the cached `.engine` next to the checkpoint is reused on later
  runs. It only accelerates the model-inference stage, so the end-to-end gain
  depends on how inference-bound the run is (measured ~7% wall-clock on
  HKU_GH with yolo26l-seg @ 768 in light scenes; larger when detections are
  dense). Requires `pip install tensorrt onnx`. Delete the `.engine`
  after changing `--imgsz`. Also applies to standard tracking and
  `detect-then-sam3` modes (previously a separate `15_export_trt.py` step).

Other speed/accuracy knobs (all modes): `--nms-iou` (pre-tracker class-agnostic
NMS, default 0.5; `1.0` disables), `--postprocess-workers` (CPU post-processing
threads overlapping GPU inference, default 4), `--detect-batch N` (one batched
GPU forward per N frames — tracking stays sequential/stateful and results match
batch 1; try 4–8), `--mask-max-dim` (contour resolution cap; masks are
pre-scaled on the GPU before transfer), `--no-masks`, `--no-vis`.

---

## Segmentation dataset pipeline (GUI → YOLO seg)

Post-annotation chain that turns the label-review GUI's COCO output into a
YOLO segmentation dataset ready for training — no Roboflow round-trip:

```
GUI annotate (SAM3 masks on every frame, discard bad frames with the
   discard button) → labels_coco.json
   → 01b_coco_to_yolo_seg.py   COCO → flat YOLO-seg (images/ + labels/ + classes.txt)
   → 02_augment_data.py        augmented flat YOLO-seg dataset
   → 03_split_dataset.py       train/test/val split + dataset.yaml
   → 04_train_model.py --task segment
```

### One-command runner

Runs all three post-GUI steps and validates the dataset format after each
step (image/label pairing, label syntax, coordinate range, class ids), so a
format mismatch fails at the step that introduced it:

```bash
python scripts/run_seg_dataset_pipeline.py \
  --coco-json /data/run/labels_coco.json \
  --images-dir /data/run/camera \
  --output-dir output/my_dataset \
  --augmentations flip_horizontal rotate brightness hue blur \
  --multiplier 2 --ratios 0.7 0.15 0.15 --seed 42

# skip augmentation entirely:
python scripts/run_seg_dataset_pipeline.py \
  --coco-json /data/run/labels_coco.json --images-dir /data/run/camera \
  --output-dir output/my_dataset --skip-augment
```

`--images-dir` is the plain image folder for mono sessions, or the parent
folder containing `left/` + `right/` for stereo sessions (the GUI writes a
`side` field per image; output filenames get `left_`/`right_` prefixes so the
identical timestamp names don't collide). The runner prints the exact
`04_train_model.py --config .../dataset.yaml --task segment` command at the end.

Output layout:

```
output/my_dataset/
├── yolo_flat/      # images/ + labels/ + classes.txt + conversion_summary.json
├── augmented/      # same layout, originals + augmented copies
└── dataset/        # train/ test/ val/ (each images/ + labels/) + dataset.yaml
```

### Step by step

```bash
# 1. COCO → flat YOLO-seg (mask-less boxes are skipped by default;
#    --bbox-as-rect emits them as rectangle polygons)
python scripts/01b_coco_to_yolo_seg.py \
  --coco-json /data/run/labels_coco.json \
  --images-dir /data/run/camera \
  --output-dir output/my_dataset/yolo_flat

# 2. Augment (masks are transformed with the image: flip/rotate/mosaic move
#    every polygon point; brightness/contrast/hue/blur/resize leave the
#    normalized labels untouched; classes.txt is passed through)
python scripts/02_augment_data.py \
  --input-dir output/my_dataset/yolo_flat \
  --output-dir output/my_dataset/augmented \
  --augmentations flip_horizontal rotate brightness hue blur \
  --multiplier 2 --hue-range -15 15 --blur-range 3 9
# add the 'resize' augmentation with: --resize WIDTH HEIGHT

# 3. Split + dataset.yaml (class names are read from classes.txt when
#    --class-names is not given)
python scripts/03_split_dataset.py \
  --input-dir output/my_dataset/augmented \
  --output-dir output/my_dataset/dataset \
  --ratios 0.7 0.15 0.15 --seed 42 --generate-yaml

# 4. Train
python scripts/04_train_model.py \
  --config output/my_dataset/dataset/dataset.yaml --task segment \
  --epochs 100 --device 0
```

> Caveat: because augmentation runs before the split, augmented copies of one
> source image can land in both train and val. Use `--skip-augment` (or split
> first and augment the splits separately) if you need strict separation.

---

## Script reference

Every entry point in `scripts/`. "Key inputs" lists the important CLI flags
(not exhaustive — run `python scripts/<name>.py --help` for the full set).

### Pipeline stages

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `00_sample_from_dataset.py` | Pick N images from a source folder that aren't already in the labeled YOLO set, ready for labeling. | `--source-dir`, `--labeled-dir`, `--out-dir`, `-n`, `--copy`/`--symlink`, `--dry-run` | New image folder (copied or symlinked) |
| `07_run_qwen.py` | Auto-label images with a VLM: Qwen via Ollama or DashScope API, or Gemma/Gemini. Supports conditioning refs, per-class, per-image labels. | `--prompt`, `--image`/`--image-folder`, `--template`, `--model`, `--conditioning-images`, `--per-class`, `--per-image-labels`, `--bbox-order`, `--dedup-iou`, `--use-api` | `<stem>_result.json` per image, `summary.json`, optional `--vis-output` |
| `08_click_review_coco.py` | **Superseded by the label-review GUI** (moved to `tests/`). Old matplotlib reviewer for Qwen boxes. | — | — |
| `08b_split_reviewed_dataset.py` | **Superseded by `03_split_dataset.py`** (moved to `tests/`). | — | — |
| `09_create_seg_dataset.py` | **Superseded by GUI SAM3 segmentation + `01b`** (moved to `tests/`). | — | — |
| `04_train_model.py` | Train a YOLO detect/segment model (Ultralytics). | `--config` (data.yaml), `--model-type`, `--task`, `--epochs`, `--batch-size`, `--device`, `--loss-type` | `runs/.../weights/best.pt` |
| `05_evaluate_model.py` | Evaluate a trained model on one or more splits (default `test train`) with per-class precision/recall/F1/AP50/AP50-95 (box + mask), or compare pred vs GT COCO JSON. | `--model`, `--data`, `--split`, `--conf`, `--iou`, `--csv` (or `--pred-json`/`--gt-json`) | Metrics report (stdout) + per-class CSV |
| `11_run_tracking.py` | YOLO tracking (ByteTrack / BoT-SORT / Deep OC-SORT / detect-then-SAM3) + a no-output benchmark mode. Pre-tracker class-agnostic NMS; optional TensorRT engine export (`--trt`, replaces the standalone `15_export_trt.py` step). | `--tracker`, `--model`, `--data`, `--output`, `--conf`, `--device`, `--warmup-frames`, `--nms-iou`, `--trt`, `--no-masks`, `--no-vis`, `--postprocess-workers` | `tracked_*.jpg`, `results.json`, `tracking_result.mp4` |
| `orchestrate_pipeline.py` | End-to-end keyframe pipeline: undistort → sample → keyframes → stats → Qwen → COCO combine → GUI review → 01b COCO→YOLO-seg → 03 split → 02 augment (train only) → assemble final dataset + stats CSV → 04 train → 05 evaluate → 11 tracking. Stage markers make it resumable. | `--rosbag`/`--images`, `--camera`, `--stage`, `--skip-stage`, `--force`, per-stage args (`--sample-size`, `--ratios`, `--augmentations`, `--epochs`, `--tracker`, ...) | `<input>_pipeline/` output tree |
| `run_pipeline.sh` | Shell wrapper around the orchestrator with env-var overrides (`LLAMACPP_URL`, `QWEN_MODEL`, `QWEN_MMPROJ`). | `<rosbag_path>` + any orchestrator args | same as orchestrator |

### Keyframe pipeline

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `12_extract_keyframes.py` | Select every Nth frame as a keyframe + write a manifest for the interpolator. | `--image-folder`/`--video`, `--output-dir`, `--every`, `--mode` | Keyframe images + `keyframe_manifest.json` |
| `13_interpolate_tracks.py` | Propagate reviewed keyframe boxes to every frame via anchored optical flow. An optional per-frame RANSAC camera model (`--camera-model global`) can absorb non-linear camera shake (tracking/anchoring on the residual, KLT dropouts re-seeded); off by default — accuracy testing on the re-reviewed MTR 4k frames showed no net gain on this fisheye camera. New objects at a keyframe are back-tracked. Each output box carries a `source`/`confidence` provenance field. | `--keyframes-coco`, `--manifest`, `--image-folder`, `--output-coco`, `--match-max-dist`, `--flow-method`, `--interp-method`, `--camera-model` | COCO annotations for every frame (+ optional vis) |
| `scripts/08c_qwen_results_to_coco.py` | Combine per-image Qwen `*_result.json` files into one COCO `labels_coco.json` the label review GUI can open (COCO xywh bboxes, categories from Qwen labels, `annotated_image_ids`). | `--qwen-results-dir`, `--output`, `--side` | `labels_coco.json` |

### Data prep & conversion

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `01_verify_labels.py` | Validate YOLO labels: missing files, bad formats, class stats. | `--input-dir`, `--images-subdir`, `--labels-subdir`, `--class-names`, `--fix` | Report (stdout / `--output`); optional in-place fixes |
| `01a_dataset_statistics.py` | Dataset statistics for a YOLO dataset (single `images/`+`labels/`, `<split>/{images,labels}` or `images/<split>`+`labels/<split>` layouts): image/label/background counts, per-class instances, % of instances, % of images, avg instances/img. Class names from `classes.txt` or `dataset.yaml`. | `--input-dir`, `--csv` | Report (stdout) + optional per-class CSV |
| `01b_coco_to_yolo_seg.py` | Convert label-review GUI COCO output into a flat YOLO segmentation dataset (input for `02`). Stereo-aware (`left_`/`right_` prefixes); mask-less boxes skipped unless `--bbox-as-rect`. | `--coco-json`, `--images-dir`, `--output-dir`, `--bbox-as-rect`, `--symlink` | `images/`, `labels/`, `classes.txt`, `conversion_summary.json` |
| `02_augment_data.py` | Augment a labeled YOLO dataset (flip/rotate/brightness/contrast/hue/blur/resize/mosaic); polygon labels are transformed with the image. | `--input-dir`, `--output-dir`, `--augmentations`, `--multiplier`, `--hue-range`, `--blur-range`, `--resize` | Augmented images + labels (+ `classes.txt` passthrough) |
| `03_split_dataset.py` | Split a labeled YOLO dataset into train/val/test + `data.yaml`. Class names fall back to `classes.txt` when `--class-names` is omitted. | `--input-dir`, `--output-dir`, `--ratios`, `--generate-yaml` | `images/labels/{train,val,test}/`, `data.yaml` |
| `run_seg_dataset_pipeline.py` | One-command chain: GUI COCO → `01b` convert → `02` augment → `03` split, with dataset-format validation after each step. | `--coco-json`, `--images-dir`, `--output-dir`, `--augmentations`, `--ratios`, `--skip-augment` | `<out>/{yolo_flat,augmented,dataset}/` + `dataset.yaml` |
| `10_qwen_json_to_yolo.py` | Convert `07 --split-by-class` JSON into a YOLO detection dataset. | `--annotations-dir`, `--image-folder`, `--output-dir`, `--data-yaml` | YOLO detect dataset (`images/`, `labels/`, `data.yaml`) |

### Standalone tools

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `06_run_sam3.py` | Run SAM3 segmentation on an image/folder, optionally with bbox exemplars. | `--image`/`--image-folder`, `--bbox`/`--bbox-json`, `--concept`, `--model` | Masks / overlay images (`--output`, `--save-overlay`) |
| `15_export_trt.py` | Standalone TensorRT engine export from a `.pt` checkpoint (FP16, fixed imgsz; also available inline via `11_run_tracking.py --trt`). | `--model`, `--imgsz`, `--workspace`, `--device` | `<model>.engine` next to the checkpoint |
| `undistort_rosbag.py` | Fisheye-undistort a folder of images using a calibration JSON. | `--images-root`, `--output-root`, `--calibration`, `--camera-name` | Undistorted images |
| `visualize.py` | Visualize Qwen annotations / YOLO detect / YOLO seg / model predictions. | `--mode`, `--dataset`/`--annotations-folder`, `--output`, `--model` | Annotated images |
| `tracking_utils.py` | Shared helpers (tracker YAML, IoU, mask→polygon, summary video). | — (library, not a CLI) | — |

> `scripts/12_upload_to_roboflow.py` is referenced in step 5 but is not present
> in the repo (only a stale `.pyc`); the `12_*` slot is now `12_extract_keyframes.py`.

### `core/` modules

These back the scripts above; not normally invoked directly.

| Module | Role |
|--------|------|
| `models_inference.py` | Headless VLM/SAM3 inference wrappers: `run_qwen` (Ollama), `run_qwen_api` (DashScope), `run_gemini` (HKU proxy), `run_sam3`; prompt templates + output parsing. |
| `model_trainer.py` | `ModelTrainer` — YOLO training pipeline (backs `04`). |
| `model_evaluator.py` | `ModelEvaluator` — evaluation + pred/GT comparison (backs `05`). |
| `data_processor.py` | `DataProcessor` — augmentation + dataset stats (backs `02`). |
| `dataset_creator.py` | `DatasetCreator` — undistortion, random selection, split (backs `00`/`03`). |
| `model_visualizer.py` / `visualizer.py` / `visualization.py` | Visualization helpers (back `visualize.py`). |
| `hyperparameter_search.py` | Search-space + grid/random search strategies for training. |
| `sam3_test.py` | SAM3 scratch/test harness (not a pipeline stage). |



---

## Label-review GUI — complete user guide

The interactive review app (`gui/label_review/`) is where the human step of
the pipeline happens: correct the seeded boxes, create masks with SAM3, and
discard bad frames. Launch it standalone or let the orchestrator open it for
you.

### Launching

```bash
# mono session
python -m gui.label_review.main --images /path/to/left --output_json out/coco.json

# stereo dual-view (frames paired by timestamp filename; unsynced frames skipped)
python -m gui.label_review.main \
    --images /path/to/left --images-right /path/to/right

# seed from an existing COCO (e.g. the qwen_coco stage output) + SAM3 options
python -m gui.label_review.main --images /path/to/keyframes \
    --json output/<run>/reviewed/labels_coco.json \
    --sam3-model core/sam3/models/sam3-model/sam3.pt --sam3-device cuda \
    --auto-segment
```

All flags: `--images`, `--images-right`, `--output_json`, `--json` (seed
COCO), `--sam3-model/--sam3-device/--sam3-conf`, `--auto-segment`
(SAM3 after every drawn box), `--interp-flow-method {dis,klt,farneback}`,
`--interp-camera-model {none,global}`, `--output-yolo-dir` (also export YOLO
on exit), `--rrd` (Rerun recording), `--pose-db` (Clio poses for map view),
`--data-yaml` (class order), `--config FILE`. Omitting `--images` starts in
idle mode — pick a source from the File menu. If a `labels_coco.json` sits
next to the images it is auto-loaded.

### File menu

| Action | Shortcut | Notes |
|---|---|---|
| Open image file(s)… | `Ctrl+O` | |
| Open folder… | `Ctrl+Shift+O` | |
| Open stereo folders… | — | pick left then right folder |
| Load annotations file… | `Ctrl+I` | import COCO JSON |
| Save / Save as… | `Ctrl+S` / `Ctrl+Shift+S` | |
| Config settings… | `Ctrl+G` | same as the ⚙ button |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `N` / `B` (or `→` / `←`) | next / previous frame (forward nav marks reviewed) |
| `U` | jump to next unlabeled frame |
| `Space` | play/pause playback |
| `S` | save and quit · `Q` quit (progress kept) |
| `A` | toggle draw-box mode |
| `D` / `Del` | delete selected box(es) |
| `X` | discard all boxes on current frame, mark reviewed, advance |
| `0–9` | assign category to a pending new box |
| `R` | re-segment selected box(es) with SAM3 (fresh masks) |
| `I` | interpolate between anchors |
| `K` | toggle ★ keyframe on current frame |
| `M` | show/hide mask overlays |
| `Z` | zoom to selected box · `F` / `0` fit view · `+` / `-` zoom steps |
| `Ctrl+A` | select all boxes on frame |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo (every operation is undoable) |
| `Esc` | clear selection / cancel point-segment or pending box |

### Mouse on the canvas

- **Left-drag in draw mode** (`A`) draws a box; release adds it if a category
  is preselected, otherwise pick one with digit keys or a category click.
- **Left-click** a box to select + drag-move; drag **corner handles** to
  resize. **Shift-click** toggles multi-selection.
- **Middle-drag** pans; **wheel** zooms around the cursor.
- **Point-segment mode (🎯 Add points)**: left-click = positive point (+),
  right-click = negative point (−); clicks only accumulate. Press
  **▶ Segment points** to run SAM3 with all placed points; add more and
  press again to refine. `Enter` accepts the object, `Esc` cancels it.

### Side panel buttons

| Button | What it does |
|---|---|
| **Add / Rename / Delete** (+ name field) | manage categories |
| **▶ Play / ⏸ Pause** + speed combo | playback at 0.25x–10x |
| **−10 −5 +5 +10** | jump buttons |
| **✔ Mark as annotated** | count frame as done without boxes |
| **🚫 Discard image** | drop the frame from the saved COCO entirely (blurry/irrelevant frames never reach training) |
| **★ Keyframe** (`K`) | mark anchor frame for interpolation |
| **Interpolate (I)** + **Stop** | optical-flow-fill boxes between two labeled/keyframe anchors; use for smooth, constant-direction motion |
| **Run SAM3 (all)** / **Re-seg sel (R)** / **Cancel** | segment all boxes on the frame / re-mask the selection |
| **SAM3 ALL frames** | background segmentation of every frame (checkpoints every 10 frames) |
| **🎯 Add points** + **▶ Segment points** | point-prompt mode: accumulate +/− points on clicks, then run SAM3 once with all of them (see mouse section) |
| **Propagate →** | follow the selected instance(s) forward with SAM3 video tracking, producing masks per frame (`memory` or `chain` method, Settings → SAM3); use for deformable objects / longer ranges than interpolation |
| **Autolabel frame** / **Autolabel ALL frames** | open-vocabulary pre-labeling with the configured backend; highlight 2+ categories to restrict the run |
| **Masks ON/OFF** + opacity slider | mask overlay visibility/transparency |

The **Boxes on this frame** list selects boxes by clicking rows;
**Cat of selected** / **Track of selected** fields reassign them by typing an
id + Enter (`C` / `T` focus the fields). The progress bar counts annotated
frames; the slider scrubs the sequence (progress saved on release).

### Recommended labeling workflow

1. Qwen seeds boxes on keyframes → GUI opens on them.
2. Per frame: fix/remove wrong boxes (`D`), add missed ones (`A`),
   re-segment edited boxes (`R`) so masks match the moved boxes.
3. Sparse footage? Label every ~10th keyframe, then `I` interpolate between
   anchors, or select a good box and `Propagate →` for masks over long gaps.
4. Missing a whole category? Highlight it in the category list and
   **Autolabel ALL frames** with SAM3/Falcon.
5. 🚫 Discard blurry/broken frames; ✔ mark empty-but-checked ones.
6. `Ctrl+S` save (or `S` save-and-quit). Discarded frames are excluded from
   the final COCO; stereo sessions save only timestamp-synced pairs.

### Settings dialog (⚙ Config / Ctrl+G)

- **Hide UI elements** — hide panel groups you don't use (`ui.hide`).
- **Advanced settings** — enables Interpolation/Tracking sections.
  - Interpolation: flow method (`dis`/`klt`/`farneback`), camera model,
    match distance, mismatch confirmation (`interpolation.*`).
  - Tracking: sticky track ids, show ids (`tracking.*`).
- **SAM3**: device, model path, confidence, auto-segment-on-add, min polygon
  area, autolabel NMS IoU, propagate method (`memory`/`chain`) + thresholds
  (`sam3.*`).
- **Autolabel detector**: `sam3`, `owlv2`, `owlv2_exemplar` (1-shot — select
  an existing box first; its crop becomes the visual query),
  `grounding_dino`, `florence2`, `falcon`, plus per-backend model/conf keys
  (`autolabel.*`). First use downloads HF checkpoints.
- **Masks / Display**: overlay opacity (`ui.mask_opacity`), max image size
  (`display.max_image_dim`, 0 = original).

Load/Apply/Save buttons live at the bottom; example config:
`scripts/config/label_review.example.json`.

### Files written by the GUI

- `<output_json>` — the reviewed COCO: polygon `segmentation` masks,
  `"side"` + `timestamp_ns` per image, `annotated_image_ids`; discarded
  frames excluded. Stereo saves only synced pairs, sorted earliest first.
- `<output>.progress` sidecar — current index, reviewed/annotated/discard
  marks, keyframes (so you can quit and resume anytime).
- Optional `.rrd` Rerun recording when launched with `--rrd`.
